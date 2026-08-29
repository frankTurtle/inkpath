from __future__ import annotations

import pytest

from rmsync import auth
from rmsync.config import Config, ConfigError
from rmsync.providers import get as get_provider

ENV = {
    "STATE_BUCKET": "bucket",
    "GITHUB_REPO": "owner/vault",
    "WATCH_FOLDER": "Reading",
    "AI_MODEL_ID": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def _env(monkeypatch, **extra):
    for k in list(ENV) + [
        "INCLUDE_NOTEBOOKS", "EXCLUDE_NOTEBOOKS", "MAX_PAGES_PER_RUN", "DRY_RUN",
        "BATCH_MODE", "WATCH_FOLDER_ID", "AI_PROVIDER", "RENDER_WIDTH",
    ]:
        monkeypatch.delenv(k, raising=False)
    for k, v in {**ENV, **extra}.items():
        monkeypatch.setenv(k, v)


def test_config_from_env_reads_defaults(monkeypatch):
    _env(monkeypatch)
    cfg = Config.from_env()
    assert cfg.state_bucket == "bucket"
    assert cfg.max_pages_per_run == 20
    assert cfg.render_width == 1400
    assert cfg.dry_run is False
    assert cfg.batch_mode == "none"


def test_config_parses_csv_lists(monkeypatch):
    _env(monkeypatch, INCLUDE_NOTEBOOKS=" A , B ,, C ")
    assert Config.from_env().include_notebooks == ["A", "B", "C"]


def test_config_bad_int_falls_back_to_default(monkeypatch):
    _env(monkeypatch, MAX_PAGES_PER_RUN="not-a-number")
    assert Config.from_env().max_pages_per_run == 20


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("no", False), ("", False)])
def test_config_bool_parsing(monkeypatch, raw, expected):
    _env(monkeypatch, DRY_RUN=raw)
    assert Config.from_env().dry_run is expected


def test_config_strips_vault_path_slashes(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("VAULT_NOTE_PATH", "/Inbox/reMarkable/")
    assert Config.from_env().vault_note_path == "Inbox/reMarkable"


def test_config_from_env_enforces_mutual_exclusion(monkeypatch):
    _env(monkeypatch, INCLUDE_NOTEBOOKS="A", EXCLUDE_NOTEBOOKS="B")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        Config.from_env()


def test_config_dry_run_does_not_require_repo():
    Config(state_bucket="b", watch_folder="R", dry_run=True).validate()


# -------------------------------------------------------------- providers ---


def test_provider_registry_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown AiProvider"):
        get_provider("nope", "model")


def test_provider_registry_returns_bedrock():
    from rmsync.providers.bedrock import BedrockProvider

    p = get_provider("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert isinstance(p, BedrockProvider)


# ------------------------------------------------------------------ auth ----


class FakeSsm:
    def __init__(self, params):
        self.params = params
        self.calls = 0

    def get_parameter(self, Name, WithDecryption):  # noqa: N803
        self.calls += 1
        if Name not in self.params:
            raise RuntimeError("ParameterNotFound")
        return {"Parameter": {"Value": self.params[Name]}}


@pytest.fixture(autouse=True)
def _clear_caches():
    auth.reset_caches()
    yield
    auth.reset_caches()


def test_get_secret_reads_and_caches(monkeypatch):
    fake = FakeSsm({"/rmsync/github-pat": "ghp_x"})
    monkeypatch.setattr(auth, "_client", lambda: fake)
    assert auth.get_secret("github-pat") == "ghp_x"
    assert auth.get_secret("github-pat") == "ghp_x"
    assert fake.calls == 1          # cached per container


def test_get_secret_missing_raises_with_remediation(monkeypatch):
    monkeypatch.setattr(auth, "_client", lambda: FakeSsm({}))
    with pytest.raises(auth.AuthError, match="put-parameter"):
        auth.get_secret("remarkable-token")


def test_get_secret_optional_returns_empty(monkeypatch):
    monkeypatch.setattr(auth, "_client", lambda: FakeSsm({}))
    assert auth.get_secret("ai-api-key", required=False) == ""


def test_register_device_rejects_bad_code_length():
    with pytest.raises(auth.AuthError, match="8 characters"):
        auth.register_device("short")


def test_get_user_token_caches_and_reports_revocation(monkeypatch):
    calls = {"n": 0}

    class Resp:
        ok = True
        status_code = 200
        reason = "OK"
        text = "usertoken"

    monkeypatch.setattr(auth.requests, "post", lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), Resp())[1])
    assert auth.get_user_token("devtok") == "usertoken"
    assert auth.get_user_token("devtok") == "usertoken"
    assert calls["n"] == 1

    class Bad:
        ok = False
        status_code = 401
        reason = "Unauthorized"
        text = "nope"

    auth.reset_caches()
    monkeypatch.setattr(auth.requests, "post", lambda *a, **kw: Bad())
    with pytest.raises(auth.AuthError, match="revoked"):
        auth.get_user_token("devtok")


def test_bedrock_provider_construction_needs_no_aws_credentials(monkeypatch):
    """Constructing a provider must not require a region or credentials.

    Only an actual call should. Regression test: eager boto3 client creation
    made the provider registry unusable anywhere AWS was not configured.
    """
    from rmsync.providers.bedrock import BedrockProvider

    for var in ("AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)

    p = BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert p.model_id.startswith("us.")
    assert p._explicit_client is None      # nothing built yet
