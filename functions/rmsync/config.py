"""Environment-driven configuration.

Every tunable is an env var set by template.yaml, so behaviour changes are a
`sam deploy --parameter-overrides`, never a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _csv(name: str) -> list[str]:
    """Comma-separated env var -> stripped, non-empty list."""
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


class ConfigError(RuntimeError):
    """Raised for a configuration mistake that must fail fast and loudly."""


@dataclass(frozen=True)
class Config:
    state_bucket: str = ""
    github_repo: str = ""
    github_branch: str = "main"
    vault_note_path: str = "Inbox/reMarkable"
    commit_mode: str = "direct"

    watch_folder: str = ""
    watch_folder_id: str = ""
    watch_folders: list[str] = field(default_factory=list)
    watch_folder_ids: list[str] = field(default_factory=list)
    include_notebooks: list[str] = field(default_factory=list)
    exclude_notebooks: list[str] = field(default_factory=list)

    ai_provider: str = "bedrock"
    ai_model_id: str = ""
    ai_base_url: str = ""
    link_mode: str = "related"

    batch_mode: str = "none"
    batch_min_records: int = 100
    batch_max_wait_days: int = 14
    batch_role_arn: str = ""

    max_pages_per_run: int = 20
    render_width: int = 1400
    blank_page_threshold: int = 3
    min_text_length: int = 20
    dry_run: bool = False
    ssm_prefix: str = "/rmsync"

    def __post_init__(self) -> None:
        """Let the singular WatchFolder/WatchFolderId stand in for the list form.

        Keeps a one-folder config - and every existing deployment - working
        unchanged now that several folders can be watched at once.
        """
        if not self.watch_folders and self.watch_folder:
            object.__setattr__(self, "watch_folders", [self.watch_folder])
        if not self.watch_folder_ids and self.watch_folder_id:
            object.__setattr__(self, "watch_folder_ids", [self.watch_folder_id])


    @classmethod
    def from_env(cls) -> Config:
        cfg = cls(
            state_bucket=os.environ.get("STATE_BUCKET", ""),
            github_repo=os.environ.get("GITHUB_REPO", ""),
            github_branch=os.environ.get("GITHUB_BRANCH", "main"),
            vault_note_path=os.environ.get("VAULT_NOTE_PATH", "Inbox/reMarkable").strip("/"),
            commit_mode=os.environ.get("COMMIT_MODE", "direct").strip(),
            watch_folder=os.environ.get("WATCH_FOLDER", "").strip(),
            watch_folder_id=os.environ.get("WATCH_FOLDER_ID", "").strip(),
            watch_folders=_csv("WATCH_FOLDER"),
            watch_folder_ids=_csv("WATCH_FOLDER_ID"),
            include_notebooks=_csv("INCLUDE_NOTEBOOKS"),
            exclude_notebooks=_csv("EXCLUDE_NOTEBOOKS"),
            ai_provider=os.environ.get("AI_PROVIDER", "bedrock").strip(),
            ai_model_id=os.environ.get("AI_MODEL_ID", "").strip(),
            ai_base_url=os.environ.get("AI_BASE_URL", "").strip(),
            link_mode=os.environ.get("LINK_MODE", "related").strip(),
            batch_mode=os.environ.get("BATCH_MODE", "none").strip(),
            batch_min_records=_int("BATCH_MIN_RECORDS", 100),
            batch_max_wait_days=_int("BATCH_MAX_WAIT_DAYS", 14),
            batch_role_arn=os.environ.get("BATCH_ROLE_ARN", "").strip(),
            max_pages_per_run=_int("MAX_PAGES_PER_RUN", 20),
            render_width=_int("RENDER_WIDTH", 1400),
            blank_page_threshold=_int("BLANK_PAGE_THRESHOLD", 3),
            min_text_length=_int("MIN_TEXT_LENGTH", 20),
            dry_run=_bool("DRY_RUN", False),
            ssm_prefix=os.environ.get("SSM_PREFIX", "/rmsync"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Fail fast at the top of the handler rather than halfway through a run."""
        # PRP: mutually exclusive - never silently pick one.
        if self.include_notebooks and self.exclude_notebooks:
            raise ConfigError(
                "IncludeNotebooks and ExcludeNotebooks are mutually exclusive; "
                f"got include={self.include_notebooks!r} exclude={self.exclude_notebooks!r}. "
                "Set at most one."
            )
        # CRITICAL: without a folder scope the first run OCRs the entire library.
        if not self.watch_folders and not self.watch_folder_ids:
            raise ConfigError(
                "One of WatchFolder or WatchFolderId is required. Running unscoped "
                "would OCR your entire reMarkable library in a single execution."
            )
        if self.link_mode not in {"related", "inline", "both"}:
            raise ConfigError(
                f"Unknown LinkMode {self.link_mode!r}; expected related, inline or both"
            )
        if self.commit_mode not in {"direct", "pull-request"}:
            raise ConfigError(
                f"Unknown CommitMode {self.commit_mode!r}; expected 'direct' or 'pull-request'"
            )
        if self.batch_mode not in {"none", "bedrock-batch", "direct-batch"}:
            raise ConfigError(f"Unknown BatchMode {self.batch_mode!r}")
        if self.ai_provider not in {"bedrock", "direct"}:
            raise ConfigError(f"Unknown AiProvider {self.ai_provider!r}")
        if not self.dry_run and not self.github_repo:
            raise ConfigError("GitHubRepo is required unless DRY_RUN=true")
