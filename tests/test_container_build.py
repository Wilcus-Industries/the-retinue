"""Tests for the shared container-build helpers (:mod:`retinue.container_build`).

The lifecycle itself (:func:`retinue.container_build.build_issue_in_container`) is
exercised through its two lanes — ``tests/test_orchestrator.py`` drives it via
``build_slice`` and ``tests/test_adhoc_build.py`` via ``build_adhoc_issue`` — so this
module covers only the pure, lane-agnostic building blocks: the credential env routing
and the byte-exact in-container file writer.
"""

from __future__ import annotations

import base64

from retinue.container_build import implement_env, write_file_command


def test_implement_env_api_key_mode_uses_anthropic_api_key() -> None:
    """api_key mode threads the credential to the CLI as ANTHROPIC_API_KEY."""
    env = implement_env("sk-ant-api03-secret", "api_key")

    assert env == {"ANTHROPIC_API_KEY": "sk-ant-api03-secret"}
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_implement_env_subscription_mode_uses_oauth_token() -> None:
    """subscription mode threads the credential as CLAUDE_CODE_OAUTH_TOKEN."""
    env = implement_env("sk-ant-oat01-secret", "subscription")

    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-secret"}
    assert "ANTHROPIC_API_KEY" not in env


def test_write_file_command_round_trips_arbitrary_content() -> None:
    """The write-file argv carries content base64-encoded as a positional arg, byte-exact."""
    content = '<<<<<<< ours\n"x" & $y\n=======\nz\n>>>>>>> theirs\n'
    argv = write_file_command("dir/f.py", content)

    # No shell interpolation: the body is a base64 positional arg, the path another.
    assert argv[0] == "sh" and argv[-1] == "dir/f.py"
    blob = argv[-2]
    assert base64.b64decode(blob).decode() == content
