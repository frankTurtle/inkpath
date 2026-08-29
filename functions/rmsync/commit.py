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


def commit_message(notebook: str, doc_id: str) -> str:
    return f"rm-sync: {notebook} ({doc_id[:8]})"
