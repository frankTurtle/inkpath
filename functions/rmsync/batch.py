"""Optional batch inference: ~50% cheaper, up to 24h slower.

A scheduled Lambda cannot block for hours waiting on a batch job, so batch mode
splits the work across polls. Every invocation does two things, in this order:

  1. Reconcile: check jobs recorded as pending. Completed -> pull results and run
     them through the normal sanitize -> frontmatter -> commit path.
     Failed/Expired -> route those pages back to the synchronous path rather
     than losing them.
  2. Submit: queue this run's new pages, and submit only when legal.

The 100-record Bedrock minimum is the whole reason this is not automatic: at
~100 pages/month no single poll produces 100 changed pages, so pages accumulate.
Anything older than BATCH_MAX_WAIT_DAYS is force-flushed to the synchronous path
instead - Bedrock rejects a sub-100 job outright, so this is the correctness
fallback, not an optimisation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3

logger = logging.getLogger(__name__)

JOB_COMPLETED = {"Completed"}
JOB_FAILED = {"Failed", "Expired", "Stopped", "PartiallyCompleted"}

# Bedrock's minimum timeout is 24h - you cannot ask for less, even for a job you
# expect to finish quickly.
MIN_TIMEOUT_HOURS = 24
# A 24h SLA is a ceiling, not a target. Only treat a job as stuck well past it.
STUCK_AFTER_HOURS = 36


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(raw: str) -> datetime:
    try:
        return datetime.fromisoformat((raw or "").replace("Z", "+00:00"))
    except ValueError:
        return _now()


@dataclass
class QueuedPage:
    doc_id: str
    doc_hash: str
    notebook: str
    page_id: str
    page_hash: str
    page_index: int
    s3_key: str
    queued_at: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueuedPage:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def queue_pages(state: dict[str, Any], pages: list[QueuedPage]) -> None:
    """Append pages to the accumulation queue, skipping any already queued."""
    queue = state.setdefault("batch", {}).setdefault("queue", [])
    known = {(i["doc_id"], i["page_id"], i["page_hash"]) for i in queue}
    for page in pages:
        key = (page.doc_id, page.page_id, page.page_hash)
        if key not in known:
            queue.append(page.to_dict())
            known.add(key)


def overdue_pages(state: dict[str, Any], max_wait_days: int) -> list[QueuedPage]:
    """Queued pages older than the force-flush threshold."""
    cutoff = _now() - timedelta(days=max_wait_days)
    return [
        QueuedPage.from_dict(item)
        for item in state.get("batch", {}).get("queue", [])
        if _parse_iso(item.get("queued_at", "")) < cutoff
    ]


def drop_from_queue(state: dict[str, Any], pages: list[QueuedPage]) -> None:
    keys = {(p.doc_id, p.page_id, p.page_hash) for p in pages}
    queue = state.setdefault("batch", {}).setdefault("queue", [])
    state["batch"]["queue"] = [
        i for i in queue if (i["doc_id"], i["page_id"], i["page_hash"]) not in keys
    ]


def should_submit(state: dict[str, Any], min_records: int, batch_mode: str) -> bool:
    """Whether a batch job may legally be submitted right now.

    CRITICAL: a Bedrock job below the minimum fails at submission, not at
    completion. Never call CreateModelInvocationJob speculatively.
    """
    queue = state.get("batch", {}).get("queue", [])
    if batch_mode == "direct-batch":
        return bool(queue)
    return len(queue) >= min_records


class BedrockBatch:
    """Submit/poll/retrieve mechanics for Bedrock batch inference."""

    def __init__(
        self, *, model_id: str, bucket: str, role_arn: str, client=None, s3_client=None
    ) -> None:
        self.model_id = model_id
        self.bucket = bucket
        self.role_arn = role_arn
        self._bedrock = client or boto3.client("bedrock")
        self._s3 = s3_client or boto3.client("s3")

    # ------------------------------------------------------------- submit --

    def submit(
        self,
        state: dict[str, Any],
        pages: list[QueuedPage],
        *,
        min_records: int,
        build_record,
    ) -> str | None:
        """Write a JSONL manifest and create the job. Returns the job ARN."""
        if len(pages) < min_records:
            logger.info(
                "Not submitting: %d queued page(s) is below the %d-record minimum",
                len(pages),
                min_records,
            )
            return None

        stamp = _iso(_now()).replace(":", "").replace("-", "")
        input_key = f"batch/{stamp}/input.jsonl"
        output_prefix = f"batch/{stamp}/output/"

        lines = [json.dumps(build_record(page)) for page in pages]
        self._s3.put_object(
            Bucket=self.bucket,
            Key=input_key,
            Body=("\n".join(lines) + "\n").encode("utf-8"),
            ContentType="application/jsonl",
        )

        resp = self._bedrock.create_model_invocation_job(
            jobName=f"inkpath-{stamp}",
            roleArn=self.role_arn,
            modelId=self.model_id,
            inputDataConfig={
                "s3InputDataConfig": {"s3Uri": f"s3://{self.bucket}/{input_key}"}
            },
            outputDataConfig={
                "s3OutputDataConfig": {"s3Uri": f"s3://{self.bucket}/{output_prefix}"}
            },
            timeoutDurationInHours=MIN_TIMEOUT_HOURS,
        )
        job_arn = resp["jobArn"]
        state.setdefault("batch", {}).setdefault("pendingJobs", []).append(
            {
                "jobArn": job_arn,
                "submittedAt": _iso(_now()),
                "outputPrefix": output_prefix,
                "records": [p.to_dict() for p in pages],
            }
        )
        logger.info("Submitted batch job %s with %d record(s)", job_arn, len(pages))
        return job_arn

    # -------------------------------------------------------------- poll ---

    def check_pending(
        self, state: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Classify pending jobs into (completed, failed, still_running)."""
        completed: list[dict] = []
        failed: list[dict] = []
        running: list[dict] = []
        for job in state.get("batch", {}).get("pendingJobs", []):
            try:
                status = self._bedrock.get_model_invocation_job(
                    jobIdentifier=job["jobArn"]
                )["status"]
            except Exception:  # noqa: BLE001 - a poll failure must not lose the job
                logger.exception("Could not poll %s; leaving pending", job["jobArn"])
                running.append(job)
                continue
            job["status"] = status
            if status in JOB_COMPLETED:
                completed.append(job)
            elif status in JOB_FAILED:
                logger.error(
                    "Batch job %s ended as %s; falling back to sync",
                    job["jobArn"],
                    status,
                )
                failed.append(job)
            else:
                age = _now() - _parse_iso(job.get("submittedAt", ""))
                if age > timedelta(hours=STUCK_AFTER_HOURS):
                    logger.error(
                        "Batch job %s still %s after %.1fh - past the 24h SLA ceiling",
                        job["jobArn"],
                        status,
                        age.total_seconds() / 3600,
                    )
                running.append(job)
        return completed, failed, running

    # ---------------------------------------------------------- retrieve ---

    def retrieve_results(self, job: dict) -> dict[str, str]:
        """Map recordId -> raw model output text for a completed job."""
        results: dict[str, str] = {}
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=job["outputPrefix"]):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith((".jsonl.out", ".jsonl")):
                    continue
                body = self._s3.get_object(Bucket=self.bucket, Key=obj["Key"])[
                    "Body"
                ].read()
                for line in body.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    record_id = record.get("recordId")
                    output = record.get("modelOutput") or {}
                    blocks = (
                        output.get("output", {}).get("message", {}).get("content", [])
                        or output.get("content", [])
                    )
                    text = "".join(
                        b.get("text", "") for b in blocks if isinstance(b, dict)
                    )
                    if record_id:
                        results[record_id] = text
        return results

    def clear_job(self, state: dict[str, Any], job: dict) -> None:
        pending = state.setdefault("batch", {}).setdefault("pendingJobs", [])
        state["batch"]["pendingJobs"] = [
            j for j in pending if j["jobArn"] != job["jobArn"]
        ]
