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


ROUTING_CONFIG = """
routing:
  default: standard
  classifier:
    model: claude-haiku-4
    effort: low
  levels:
    trivial:
      description: Typo fixes, docs-only changes, and other one-line diffs.
      roles:
        implementer:
          model: claude-haiku-4
    standard:
      description: Ordinary feature work and bug fixes of moderate scope.
      roles:
        implementer:
          model: claude-sonnet-4-6
        reviewer:
          model: claude-opus-4-8
          effort: high
    high-risk:
      description: Cross-module migrations, concurrency refactors, schema changes.
      roles:
        implementer:
          model: claude-opus-4-8
          effort: xhigh
        reviewer:
          model: claude-opus-4-8
          effort: max
"""


def test_routing_block_parses_full_shape() -> None:
    """Three levels, partial role maps, default, and a classifier override all parse."""
    config = load_repo_config(ROUTING_CONFIG)
    assert config is not None
    assert config.routing is not None
    routing = config.routing
    assert routing.default == "standard"
    assert routing.classifier is not None
    assert routing.classifier.model == "claude-haiku-4"
    assert routing.classifier.effort == "low"
    assert set(routing.levels) == {"trivial", "standard", "high-risk"}
    trivial = routing.levels["trivial"]
    assert trivial.description == "Typo fixes, docs-only changes, and other one-line diffs."
    assert set(trivial.roles) == {"implementer"}
    assert trivial.roles["implementer"].model == "claude-haiku-4"
    assert trivial.roles["implementer"].effort is None
    assert routing.levels["standard"].roles["reviewer"].effort == "high"
    high_risk = routing.levels["high-risk"]
    assert high_risk.roles["implementer"].effort == "xhigh"
    assert high_risk.roles["reviewer"].effort == "max"


def test_routing_absent_is_valid_and_reports_none() -> None:
    """A config with no routing: block validates, with routing simply off."""
    config = load_repo_config("staging_branch: staging\n")
    assert config is not None
    assert config.routing is None


def test_routing_rejects_bad_level_slug() -> None:
    """An uppercase or otherwise non-slug level name is rejected."""
    bad = "routing:\n  default: Trivial\n  levels:\n    Trivial:\n      description: bad name\n"
    assert load_repo_config(bad) is None


def test_routing_rejects_bad_effort_value() -> None:
    """An effort tier outside the known five-tier set is rejected."""
    bad = """
routing:
  default: standard
  levels:
    standard:
      description: ok
      roles:
        implementer:
          model: claude-sonnet-4-6
          effort: extreme
"""
    assert load_repo_config(bad) is None


def test_routing_rejects_unknown_role_key() -> None:
    """A role key not in the role registry is rejected."""
    bad = """
routing:
  default: standard
  levels:
    standard:
      description: ok
      roles:
        wizard:
          model: claude-sonnet-4-6
"""
    assert load_repo_config(bad) is None


def test_routing_rejects_default_naming_missing_level() -> None:
    """default: naming a level that isn't declared is rejected."""
    bad = "routing:\n  default: nonexistent\n  levels:\n    standard:\n      description: ok\n"
    assert load_repo_config(bad) is None


def test_routing_rejects_zero_levels() -> None:
    """A routing: block with an empty levels map is invalid."""
    bad = "routing:\n  default: standard\n  levels: {}\n"
    assert load_repo_config(bad) is None


def test_routing_rejects_unknown_key_in_block() -> None:
    """A typo'd or unknown key in the routing: block is rejected."""
    bad = """
routing:
  default: standard
  levels:
    standard:
      description: ok
  extra_key: true
"""
    assert load_repo_config(bad) is None


def test_routing_violation_logs_warning_not_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A routing violation degrades to a logged warning, never an exception."""
    bad = "routing:\n  default: standard\n  levels: {}\n"
    with caplog.at_level(logging.WARNING, logger="retinue.repo_config"):
        config = load_repo_config(bad)
    assert config is None
    assert "retinue.yml" in caplog.text.lower()
