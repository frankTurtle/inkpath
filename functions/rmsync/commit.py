"""Commit notes to the GitHub-backed Obsidian vault via the Contents API."""

from __future__ import annotations

import base64
import logging
import time

import requests

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
HTTP_TIMEOUT = 30
MAX_SHA_RETRIES = 3


class CommitError(RuntimeError):
    """A commit failed. State must not advance past this."""


class GitHubVault:
    def __init__(self, repo: str, token: str, branch: str = "main") -> None:
        if "/" not in repo:
            raise CommitError(f"GitHubRepo must be 'owner/repo', got {repo!r}")
        self.repo = repo
        self.branch = branch
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "inkpath",
            }
        )

    def _url(self, path: str) -> str:
        clean = "/".join(part for part in path.split("/") if part)
        return f"{API_ROOT}/repos/{self.repo}/contents/{clean}"

    def get_sha(self, path: str) -> str | None:
        """Current blob SHA for a path, or None if the file does not exist."""
        resp = self._session.get(
            self._url(path), params={"ref": self.branch}, timeout=HTTP_TIMEOUT
        )
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise CommitError(
                f"Could not read {path} ({resp.status_code}): {resp.text[:200]}"
            )
        data = resp.json()
        if isinstance(data, list):
            raise CommitError(f"{path} is a directory, not a file")
        return data.get("sha")

    def put_file(self, path: str, content: bytes, message: str) -> str:
        """Create or update a file, retrying on a stale-SHA conflict.

        The Contents API requires the current SHA to update an existing file, and
        a concurrent write invalidates it - so the SHA is re-read on each retry
        rather than reused.
        """
        encoded = base64.b64encode(content).decode("ascii")
        last_error = ""
        for attempt in range(1, MAX_SHA_RETRIES + 1):
            payload: dict[str, object] = {
                "message": message,
                "content": encoded,
                "branch": self.branch,
            }
            sha = self.get_sha(path)
            if sha:
                payload["sha"] = sha
            resp = self._session.put(self._url(path), json=payload, timeout=HTTP_TIMEOUT)
            if resp.ok:
                commit = resp.json().get("commit", {}).get("sha", "")
                logger.info("Committed %s (%s)", path, commit[:8])
                return commit
            if resp.status_code in (409, 422):
                last_error = f"{resp.status_code}: {resp.text[:200]}"
                logger.warning(
                    "SHA conflict committing %s (attempt %d/%d); re-reading SHA",
                    path,
                    attempt,
                    MAX_SHA_RETRIES,
                )
                time.sleep(0.5 * attempt)
                continue
            raise CommitError(
                f"Commit of {path} failed ({resp.status_code}): {resp.text[:200]}"
            )
        raise CommitError(
            f"Commit of {path} failed after {MAX_SHA_RETRIES} attempts. {last_error}"
        )


    # ------------------------------------------------------- branches / PRs --

    def default_branch_sha(self, base: str) -> str:
        """Head commit SHA of the base branch."""
        resp = self._session.get(
            f"{API_ROOT}/repos/{self.repo}/git/ref/heads/{base}", timeout=HTTP_TIMEOUT
        )
        if not resp.ok:
            raise CommitError(
                f"Could not read branch {base!r} ({resp.status_code}): {resp.text[:200]}"
            )
        return resp.json()["object"]["sha"]

    def create_branch(self, name: str, base: str) -> None:
        """Branch off `base`. Tolerates the branch already existing."""
        sha = self.default_branch_sha(base)
        resp = self._session.post(
            f"{API_ROOT}/repos/{self.repo}/git/refs",
            json={"ref": f"refs/heads/{name}", "sha": sha},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 422 and "already exists" in resp.text:
            logger.info("Branch %s already exists; reusing it", name)
            return
        if not resp.ok:
            raise CommitError(
                f"Could not create branch {name!r} ({resp.status_code}): {resp.text[:200]}"
            )
        logger.info("Created branch %s from %s@%s", name, base, sha[:8])

    def create_pull_request(self, *, head: str, base: str, title: str, body: str) -> int:
        resp = self._session.post(
            f"{API_ROOT}/repos/{self.repo}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 403:
            raise CommitError(
                "Creating a pull request was forbidden. A fine-grained PAT needs "
                "'Pull requests: Read and write' in addition to 'Contents: Read and "
                f"write'. ({resp.text[:160]})"
            )
        if not resp.ok:
            raise CommitError(
                f"Could not open a pull request ({resp.status_code}): {resp.text[:200]}"
            )
        number = int(resp.json()["number"])
        logger.info("Opened pull request #%d (%s -> %s)", number, head, base)
        return number

    def merge_pull_request(self, number: int, *, message: str) -> bool:
        resp = self._session.put(
            f"{API_ROOT}/repos/{self.repo}/pulls/{number}/merge",
            json={"merge_method": "squash", "commit_title": message},
            timeout=HTTP_TIMEOUT,
        )
        if resp.ok:
            logger.info("Merged pull request #%d", number)
            return True
        # 405 means not mergeable (e.g. required reviews); leave it open for a human.
        raise CommitError(
            f"Could not merge pull request #{number} ({resp.status_code}): "
            f"{resp.text[:200]}"
        )

    def delete_branch(self, name: str) -> None:
        """Best effort - a leftover branch is untidy, never harmful."""
        resp = self._session.delete(
            f"{API_ROOT}/repos/{self.repo}/git/refs/heads/{name}", timeout=HTTP_TIMEOUT
        )
        if not resp.ok:
            logger.warning("Could not delete branch %s (%s)", name, resp.status_code)


def note_path(vault_note_path: str, notebook: str, title: str) -> str:
    from .enrich import sanitize_path_component

    folder = sanitize_path_component(notebook, fallback="reMarkable")
    name = sanitize_path_component(title, fallback="untitled")
    return f"{vault_note_path}/{folder}/{name}.md"


def attachment_path(vault_note_path: str, notebook: str, title: str) -> str:
    from .enrich import sanitize_path_component

    folder = sanitize_path_component(notebook, fallback="reMarkable")
    name = sanitize_path_component(title, fallback="untitled")
    return f"{vault_note_path}/{folder}/attachments/{name}.png"


def disambiguate_path(path: str, taken: set[str], discriminator: str) -> str:
    """Return a path that no other page has already claimed.

    Two pages of the same notebook can easily yield the same title - a book's
    notes often repeat a theme - and both would resolve to one path, silently
    overwriting each other. The page id is a stable, deterministic
    discriminator, so the same page keeps the same filename across runs.
    """
    if path not in taken:
        return path
    stem, _, ext = path.rpartition(".")
    candidate = f"{stem} ({discriminator}).{ext}" if stem else f"{path} ({discriminator})"
    suffix = 2
    while candidate in taken:
        candidate = f"{stem} ({discriminator}-{suffix}).{ext}"
        suffix += 1
    return candidate


def commit_message(notebook: str, doc_id: str) -> str:
    return f"rm-sync: {notebook} ({doc_id[:8]})"
