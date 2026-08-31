"""Lambda entrypoint: poll reMarkable, transcribe changed pages, commit to the vault.

Sequence::

    1. load state.json                  <- S3
    2. list items                       <- reMarkable Cloud (creds <- SSM)
    3. resolve scope, diff vs state     -> changed pages only
    4. render page -> PNG               (blank pages skipped before any model call)
    5. one vision call: OCR + tags      -> configured provider
    6. commit .md                       -> GitHub Contents API (PAT <- SSM)
    7. write state.json                 -> S3, ONLY after the commit succeeded

Step 7 is the ordering that matters most: writing state before a successful
commit means a failed commit is never retried and the note is silently lost.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import boto3

from . import batch as batch_mod
from . import providers
from . import state as state_mod
from .auth import get_secret, get_user_token
from .batch import BedrockBatch, QueuedPage
from .commit import (
    GitHubVault,
    attachment_path,
    commit_message,
    disambiguate_path,
    note_path,
)
from .config import Config
from .enrich import compose_note, sanitize_path_component
from .fetch import diff_pages, download_pages, select_documents
from .remarkable import RemarkableClient
from .render import render_page
from .scope import resolve_scope_map

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_root = logging.getLogger()
if _root.handlers:
    # The Lambda runtime installs a root handler before this module is imported,
    # which makes logging.basicConfig() a silent no-op. Without setting the level
    # directly, every INFO line - including RUN_SUMMARY and the token counts that
    # make cost attributable from logs alone - is dropped.
    _root.setLevel(_LOG_LEVEL)
else:
    logging.basicConfig(level=_LOG_LEVEL, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rmsync")
logger.setLevel(_LOG_LEVEL)
for _noisy in ("botocore", "urllib3", "boto3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def stage_png(bucket: str, key: str, png: bytes) -> None:
    """Park a rendered page in S3 so a later poll can still reach it.

    Batch results arrive up to a day after the run that rendered them, long
    after this container is gone. The staging/ prefix expires after 7 days.
    """
    _s3_client().put_object(Bucket=bucket, Key=key, Body=png, ContentType="image/png")


def load_png(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def build_batch_record(page) -> dict[str, Any]:
    """One JSONL line for a Bedrock batch job.

    Batch `modelInput` uses the model's native InvokeModel body, not the
    Converse envelope - this is the Anthropic Messages shape. A non-Anthropic
    model needs its own body here.
    """
    import base64

    from .providers.base import MAX_OUTPUT_TOKENS, TEMPERATURE, build_prompt

    png = load_png(os.environ["STATE_BUCKET"], page.s3_key)
    return {
        "recordId": page.page_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(png).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": build_prompt([])},
                    ],
                }
            ],
        },
    }


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stats: dict[str, int] = {
            "pagesConsidered": 0,
            "pagesRendered": 0,
            "blankSkipped": 0,
            "modelCalls": 0,
            "commits": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "pullRequests": 0,
            "batchQueued": 0,
            "batchReconciled": 0,
            "errors": 0,
        }
        self.committed_notes: list[str] = []
        self._vault: GitHubVault | None = None
        self._providers: dict[str, providers.VisionProvider] = {}
        self._provider: providers.VisionProvider | None = None

    # ------------------------------------------------------------ lazy deps --

    @property
    def vault(self) -> GitHubVault:
        if self._vault is None:
            self._vault = GitHubVault(
                self.cfg.github_repo,
                get_secret("github-pat", prefix=self.cfg.ssm_prefix),
                self.cfg.github_branch,
            )
        return self._vault

    def provider_for(self, link_mode: str = "") -> providers.VisionProvider:
        """Provider bound to a specific link mode.

        The link mode is baked into the prompt, so a run that mixes routes -
        book notes linking, journals not - needs one provider per mode. They are
        cached per invocation; construction is cheap but not free.
        """
        mode = link_mode or self.cfg.link_mode
        if mode not in self._providers:
            if self._provider is not None and mode == self.cfg.link_mode:
                # Honour a provider injected by tests.
                self._providers[mode] = self._provider
            else:
                api_key = ""
                if self.cfg.ai_provider == "direct":
                    api_key = get_secret("ai-api-key", prefix=self.cfg.ssm_prefix)
                self._providers[mode] = providers.get(
                    self.cfg.ai_provider,
                    self.cfg.ai_model_id,
                    api_key=api_key,
                    base_url=self.cfg.ai_base_url,
                    link_mode=mode,
                )
        return self._providers[mode]

    @property
    def provider(self) -> providers.VisionProvider:
        if self._provider is None:
            api_key = ""
            if self.cfg.ai_provider == "direct":
                api_key = get_secret("ai-api-key", prefix=self.cfg.ssm_prefix)
            self._provider = providers.get(
                self.cfg.ai_provider,
                self.cfg.ai_model_id,
                api_key=api_key,
                base_url=self.cfg.ai_base_url,
                link_mode=self.cfg.link_mode,
            )
        return self._provider

    # -------------------------------------------------------------- commit --

    def commit_note(
        self,
        st: dict[str, Any],
        *,
        doc_id: str,
        doc_hash: str,
        notebook: str,
        page_id: str,
        page_hash: str,
        page_index: int,
        vault_dir: str,
        note,
        png: bytes | None,
    ) -> None:
        """Commit a note (and its attachment), then record it in state.

        The existing-path check runs regardless of which path produced the
        content, so a batch fallback cannot double-commit the same page.
        """
        rec = state_mod.page_record(st, doc_id, page_id)
        existing = rec.get("notePath") if rec else None
        if (
            rec
            and rec.get("hash") == page_hash
            and rec.get("status") == state_mod.STATUS_COMMITTED
        ):
            logger.info("Page %s already committed; skipping", page_id[:8])
            return

        path = existing or disambiguate_path(
            note_path(vault_dir, note.title),
            state_mod.claimed_paths(st, excluding=(doc_id, page_id)),
            f"p{page_index + 1}",
        )

        if self.cfg.dry_run:
            logger.info("DRY_RUN - would commit %s:\n%s", path, note.body)
        else:
            if note.attach_png and png:
                apath = attachment_path(vault_dir, note.title)
                self.vault.put_file(apath, png, commit_message(notebook, doc_id))
            self.vault.put_file(
                path, note.body.encode("utf-8"), commit_message(notebook, doc_id)
            )

        self.stats["commits"] += 1
        self.committed_notes.append(path)
        state_mod.record_page(
            st,
            doc_id=doc_id,
            doc_hash=doc_hash,
            notebook=notebook,
            page_id=page_id,
            page_hash=page_hash,
            note_path=path,
            status=state_mod.STATUS_COMMITTED,
            timestamp=_now_iso(),
        )
        state_mod.learn_vocabulary(st, note.tags, note.title)

    # ------------------------------------------------------- sync pipeline --

    def process_sync(self, st: dict[str, Any], pages: list) -> None:
        for page in pages:
            self.stats["pagesConsidered"] += 1
            try:
                png = render_page(
                    page.data,
                    width=self.cfg.render_width,
                    blank_threshold=self.cfg.blank_page_threshold,
                )
                if png is None:
                    self.stats["blankSkipped"] += 1
                    # A blank page is genuinely done: record it so it is never
                    # rendered again, but never commit an empty note.
                    state_mod.record_page(
                        st,
                        doc_id=page.doc_id,
                        doc_hash=page.doc_hash,
                        notebook=page.notebook,
                        page_id=page.page_id,
                        page_hash=page.page_hash,
                        note_path="",
                        status=state_mod.STATUS_COMMITTED,
                        timestamp=_now_iso(),
                    )
                    continue

                self.stats["pagesRendered"] += 1
                attachment = sanitize_path_component(
                    f"{page.notebook} p{page.page_index + 1}", fallback="page"
                )
                result = self.provider_for(page.link_mode).extract_and_tag(
                    png, st.get("tagVocabulary", []), st.get("noteTitles", [])
                )
                self.stats["modelCalls"] += 1
                note = compose_note(
                    result,
                    doc_id=page.doc_id,
                    notebook=page.notebook,
                    page_index=page.page_index,
                    min_text_length=self.cfg.min_text_length,
                    attachment_name=f"{attachment}.png",
                    link_mode=page.link_mode or self.cfg.link_mode,
                    known_titles=st.get("noteTitles", []),
                    link_notebook=self.cfg.link_notebook,
                    title_strip_pattern=self.cfg.title_strip_pattern,
                )
                self.stats["inputTokens"] += note.input_tokens
                self.stats["outputTokens"] += note.output_tokens

                self.commit_note(
                    st,
                    doc_id=page.doc_id,
                    doc_hash=page.doc_hash,
                    notebook=page.notebook,
                    page_id=page.page_id,
                    page_hash=page.page_hash,
                    page_index=page.page_index,
                    vault_dir=page.vault_dir,
                    note=note,
                    png=png,
                )
            except Exception:  # noqa: BLE001 - one bad page must not lose the rest
                self.stats["errors"] += 1
                logger.exception(
                    "Failed page %s of %s; leaving it unrecorded for the next poll",
                    page.page_id[:8],
                    page.notebook,
                )

    # ------------------------------------------------------- pull requests --

    def open_branch(self, stamp: str) -> str:
        """Start a run branch and point the vault at it."""
        branch = f"rm-sync/{stamp}"
        self.vault.create_branch(branch, self.cfg.github_branch)
        self.vault.branch = branch
        return branch

    def finish_pull_request(self, branch: str, notes: list[str]) -> None:
        """Open and squash-merge the run's PR.

        Raises on failure so the caller skips the state write: until this merges,
        the notes are not in the vault and the pages must be retried.
        """
        listed = "\n".join(f"- `{path}`" for path in sorted(notes))
        body = (
            f"Synced {len(notes)} page(s) from reMarkable.\n\n{listed}\n\n"
            "_Opened automatically by inkpath._"
        )
        title = f"rm-sync: {len(notes)} note(s)"
        number = self.vault.create_pull_request(
            head=branch, base=self.cfg.github_branch, title=title, body=body
        )
        self.vault.merge_pull_request(number, message=title)
        self.vault.delete_branch(branch)
        self.stats["pullRequests"] += 1

    # ------------------------------------------------------ batch pipeline --

    def _batch(self) -> BedrockBatch:
        return BedrockBatch(
            model_id=self.cfg.ai_model_id,
            bucket=self.cfg.state_bucket,
            role_arn=self.cfg.batch_role_arn,
        )

    def _commit_from_text(self, st: dict[str, Any], record: dict, raw_text: str) -> None:
        """Shared tail for batch results: parse -> compose -> commit."""
        from .providers.base import fallback_result, parse_response

        try:
            result = parse_response(raw_text)
        except ValueError:
            result = fallback_result(raw_text)

        png = None
        try:
            png = load_png(self.cfg.state_bucket, record["s3_key"])
        except Exception:  # noqa: BLE001 - the staged render may have expired
            logger.warning("Staged PNG %s unavailable; committing without it", record["s3_key"])

        attachment = sanitize_path_component(
            f"{record['notebook']} p{record['page_index'] + 1}", fallback="page"
        )
        note = compose_note(
            result,
            doc_id=record["doc_id"],
            notebook=record["notebook"],
            page_index=record["page_index"],
            min_text_length=self.cfg.min_text_length,
            attachment_name=f"{attachment}.png",
            link_mode=self.cfg.link_mode,
            known_titles=st.get("noteTitles", []),
            link_notebook=self.cfg.link_notebook,
            title_strip_pattern=self.cfg.title_strip_pattern,
        )
        self.commit_note(
            st,
            doc_id=record["doc_id"],
            doc_hash=record["doc_hash"],
            notebook=record["notebook"],
            page_id=record["page_id"],
            page_hash=record["page_hash"],
            page_index=record["page_index"],
            vault_dir=record.get("vault_dir", ""),
            note=note,
            png=png,
        )

    def reconcile_batches(self, st: dict[str, Any]) -> None:
        """Step 1 of every batch-mode poll: land whatever finished."""
        if not st.get("batch", {}).get("pendingJobs"):
            return
        batch = self._batch()
        completed, failed, running = batch.check_pending(st)
        logger.info(
            "Batch reconcile: %d completed, %d failed, %d still running",
            len(completed), len(failed), len(running),
        )

        for job in completed:
            try:
                results = batch.retrieve_results(job)
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                logger.exception("Could not read results for %s; leaving pending", job["jobArn"])
                continue
            for record in job.get("records", []):
                raw = results.get(record["page_id"])
                if raw is None:
                    logger.warning("No batch result for page %s", record["page_id"][:8])
                    continue
                try:
                    self._commit_from_text(st, record, raw)
                    self.stats["batchReconciled"] += 1
                except Exception:  # noqa: BLE001
                    self.stats["errors"] += 1
                    logger.exception("Failed to commit batch result for %s", record["page_id"][:8])
            batch.clear_job(st, job)

        # A failed or expired job must not lose its pages: re-run them on-demand.
        for job in failed:
            self._process_records_synchronously(st, job.get("records", []))
            batch.clear_job(st, job)

    def _process_records_synchronously(self, st: dict[str, Any], records: list[dict]) -> None:
        """On-demand fallback for pages a batch job could not deliver."""
        for record in records:
            try:
                png = load_png(self.cfg.state_bucket, record["s3_key"])
            except Exception:  # noqa: BLE001 - re-fetched from reMarkable next poll
                self.stats["errors"] += 1
                logger.exception(
                    "Staged PNG for %s is gone; clearing state so it re-syncs",
                    record["page_id"][:8],
                )
                st.get("docs", {}).get(record["doc_id"], {}).get("pages", {}).pop(
                    record["page_id"], None
                )
                continue
            try:
                result = self.provider_for(record.get("link_mode", "")).extract_and_tag(
                    png, st.get("tagVocabulary", []), st.get("noteTitles", [])
                )
                self.stats["modelCalls"] += 1
                attachment = sanitize_path_component(
                    f"{record['notebook']} p{record['page_index'] + 1}", fallback="page"
                )
                note = compose_note(
                    result,
                    doc_id=record["doc_id"],
                    notebook=record["notebook"],
                    page_index=record["page_index"],
                    min_text_length=self.cfg.min_text_length,
                    attachment_name=f"{attachment}.png",
                )
                self.commit_note(
                    st,
                    doc_id=record["doc_id"],
                    doc_hash=record["doc_hash"],
                    notebook=record["notebook"],
                    page_id=record["page_id"],
                    page_hash=record["page_hash"],
                    page_index=record["page_index"],
                    vault_dir=record.get("vault_dir", ""),
                    note=note,
                    png=png,
                )
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                logger.exception("Sync fallback failed for %s", record["page_id"][:8])

    def process_batch(self, st: dict[str, Any], pages: list) -> None:
        """Step 2 of a batch-mode poll: render, stage, queue, and maybe submit."""
        queued: list[QueuedPage] = []
        for page in pages:
            self.stats["pagesConsidered"] += 1
            try:
                png = render_page(
                    page.data,
                    width=self.cfg.render_width,
                    blank_threshold=self.cfg.blank_page_threshold,
                )
                if png is None:
                    self.stats["blankSkipped"] += 1
                    state_mod.record_page(
                        st, doc_id=page.doc_id, doc_hash=page.doc_hash, notebook=page.notebook,
                        page_id=page.page_id, page_hash=page.page_hash, note_path="",
                        status=state_mod.STATUS_COMMITTED, timestamp=_now_iso(),
                    )
                    continue
                self.stats["pagesRendered"] += 1
                key = f"staging/{page.doc_id}/{page.page_id}.png"
                stage_png(self.cfg.state_bucket, key, png)
                queued.append(
                    QueuedPage(
                        doc_id=page.doc_id, doc_hash=page.doc_hash, notebook=page.notebook,
                        page_id=page.page_id, page_hash=page.page_hash,
                        page_index=page.page_index, s3_key=key, queued_at=_now_iso(),
                        vault_dir=page.vault_dir,
                    )
                )
                # Mark pending so the next poll neither re-queues nor re-fetches it.
                state_mod.record_page(
                    st, doc_id=page.doc_id, doc_hash=page.doc_hash, notebook=page.notebook,
                    page_id=page.page_id, page_hash=page.page_hash, note_path="",
                    status=state_mod.STATUS_PENDING_BATCH, timestamp=_now_iso(),
                )
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                logger.exception("Failed to stage page %s", page.page_id[:8])

        batch_mod.queue_pages(st, queued)
        self.stats["batchQueued"] += len(queued)

        # Force-flush anything that has waited too long. Bedrock rejects a
        # sub-minimum job outright, so overdue pages go to the synchronous path.
        overdue = batch_mod.overdue_pages(st, self.cfg.batch_max_wait_days)
        if overdue:
            logger.warning(
                "%d page(s) exceeded BatchMaxWaitDays=%d; processing synchronously",
                len(overdue), self.cfg.batch_max_wait_days,
            )
            self._process_records_synchronously(st, [p.to_dict() for p in overdue])
            batch_mod.drop_from_queue(st, overdue)

        if self.cfg.batch_mode == "bedrock-batch":
            if batch_mod.should_submit(st, self.cfg.batch_min_records, self.cfg.batch_mode):
                pending = [
                    QueuedPage.from_dict(i) for i in st.get("batch", {}).get("queue", [])
                ]
                try:
                    self._batch().submit(
                        st, pending,
                        min_records=self.cfg.batch_min_records,
                        build_record=build_batch_record,
                    )
                    batch_mod.drop_from_queue(st, pending)
                except Exception:  # noqa: BLE001 - queue survives for the next poll
                    self.stats["errors"] += 1
                    logger.exception("Batch submission failed; queue retained")
            else:
                logger.info(
                    "Queue holds %d of %d records needed to submit",
                    len(st.get("batch", {}).get("queue", [])),
                    self.cfg.batch_min_records,
                )


def run(event: dict[str, Any], cfg: Config | None = None) -> dict[str, int]:
    started = time.time()
    cfg = cfg or Config.from_env()
    runner = Runner(cfg)

    st = state_mod.load_state(cfg.state_bucket)

    # Reconcile first: land any finished batch results before queueing more.
    if cfg.batch_mode != "none":
        runner.reconcile_batches(st)

    device_token = get_secret("remarkable-token", prefix=cfg.ssm_prefix)
    client = RemarkableClient(get_user_token(device_token))

    items, meta_cache = client.list_items(st.get("metadataCache"))
    st["metadataCache"] = meta_cache

    scope_map = resolve_scope_map(
        items, folder_names=cfg.watch_folders, folder_ids=cfg.watch_folder_ids
    )
    docs = select_documents(items, scope_map, cfg)
    pending, _pending_batch = diff_pages(client, docs, st, cfg, scope_map)

    if pending:
        pages = download_pages(client, pending, cfg)

        # A branch is only cut once there is genuinely something to commit, so a
        # quiet poll leaves no branches and no empty pull requests behind.
        branch = ""
        use_pr = cfg.commit_mode == "pull-request" and not cfg.dry_run
        if use_pr:
            branch = runner.open_branch(_now_iso().replace(":", "").replace("-", ""))

        if cfg.batch_mode == "none":
            runner.process_sync(st, pages)
        else:
            runner.process_batch(st, pages)

        if use_pr:
            if runner.committed_notes:
                # Raises on failure, which skips the state write below so the
                # pages are retried: until this merges they are not in the vault.
                runner.finish_pull_request(branch, runner.committed_notes)
            else:
                logger.info("Nothing was committed; discarding branch %s", branch)
                runner.vault.delete_branch(branch)
    else:
        logger.info("No changed pages: zero model calls, zero commits")

    # CRITICAL: state is written last, after the commits it records succeeded.
    # A dry run must not persist anything: recording pages it only pretended to
    # commit would make the next real run believe the work was already done and
    # skip every page.
    if cfg.dry_run:
        logger.info("DRY_RUN - state not persisted (%d page(s) simulated)", runner.stats["commits"])
    else:
        state_mod.save_state(st, cfg.state_bucket)

    runner.stats["durationMs"] = int((time.time() - started) * 1000)
    # Log usage so cost is attributable from logs alone.
    logger.info("RUN_SUMMARY %s", json.dumps(runner.stats, sort_keys=True))
    return runner.stats


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    try:
        return {"statusCode": 200, "body": json.dumps(run(event or {}))}
    except Exception:
        logger.exception("rm-sync run failed")
        raise
