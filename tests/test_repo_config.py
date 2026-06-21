"""Tests for per-repo .github/retinue.yml schema parsing and validation.

Presence of the file is opt-in; absence and malformed content both yield None
(an observable skip) rather than raising and crashing the worker.
"""

from __future__ import annotations

import logging

import pytest

from retinue.repo_config import RepoConfig, load_repo_config

FULL_CONFIG = """
staging_branch: integration
retry_cap: 5
max_parallel: 4
cron: "0 */6 * * *"
models:
  planner: claude-opus-4
  coder: claude-sonnet-4
secrets:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  refs:
    - vault://team/retinue/github-token
"""


def test_full_config_parses_every_field() -> None:
    """A complete config exposes every documented field downstream."""
    config = load_repo_config(FULL_CONFIG)
    assert config is not None
    assert config.staging_branch == "integration"
    assert config.retry_cap == 5
    assert config.max_parallel == 4
    assert config.cron == "0 */6 * * *"
    assert config.models == {"planner": "claude-opus-4", "coder": "claude-sonnet-4"}
    assert config.secrets.values == {"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"}
    assert config.secrets.refs == ["vault://team/retinue/github-token"]


def test_minimal_config_applies_defaults() -> None:
    """An empty mapping is valid; documented defaults fill the gaps."""
    config = load_repo_config("{}")
    assert config is not None
    assert config.staging_branch == "staging"
    assert config.retry_cap == 3
    assert config.max_parallel is None
    assert config.cron is None
    assert config.models == {}
    assert config.secrets.values == {}
    assert config.secrets.refs == []


def test_empty_file_applies_defaults() -> None:
    """An empty file parses to an all-defaults config rather than failing."""
    config = load_repo_config("")
    assert config is not None
    assert config.staging_branch == "staging"


def test_validates_staging_branch_type() -> None:
    """A non-string staging_branch is rejected (config skipped)."""
    assert load_repo_config("staging_branch: 123\n") is None


def test_validates_retry_cap_is_non_negative_int() -> None:
    """retry_cap must be a non-negative integer."""
    assert load_repo_config("retry_cap: -1\n") is None
    assert load_repo_config("retry_cap: notanint\n") is None


def test_validates_max_parallel_is_positive() -> None:
    """max_parallel, when present, must be a positive integer."""
    assert load_repo_config("max_parallel: 0\n") is None


def test_validates_cron_cadence() -> None:
    """A cron expression with the wrong field count is rejected."""
    assert load_repo_config('cron: "not a cron"\n') is None
    assert load_repo_config('cron: "* * * *"\n') is None


def test_validates_model_overrides_are_strings() -> None:
    """Model overrides must map role names to string model ids."""
    assert load_repo_config("models:\n  planner: 42\n") is None


def test_validates_secrets_block_shape() -> None:
    """A secrets block with a non-list refs entry is rejected."""
    assert load_repo_config("secrets:\n  refs: not-a-list\n") is None


def test_rejects_unknown_top_level_key() -> None:
    """An unknown top-level key is a typo'd config and is skipped, not ignored."""
    assert load_repo_config("staging_brnch: main\n") is None


def test_malformed_yaml_is_skipped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unparseable YAML yields None and an explanatory log, never an exception."""
    with caplog.at_level(logging.WARNING, logger="retinue.repo_config"):
        config = load_repo_config("staging_branch: [unclosed\n")
    assert config is None
    assert "retinue.yml" in caplog.text.lower()


def test_non_mapping_document_is_skipped() -> None:
    """A YAML scalar or list at the top level is not a valid config."""
    assert load_repo_config("- just\n- a\n- list\n") is None
    assert load_repo_config("plain string\n") is None


def test_repo_config_is_constructible_directly() -> None:
    """RepoConfig can be built in code with all defaults for downstream use."""
    config = RepoConfig()
    assert config.staging_branch == "staging"
    assert config.retry_cap == 3
