# Single image shared by both services (web + worker). Compose sets the per-service
# command; this image carries the full runtime and toolchain for either.
FROM python:3.12-slim

# uv: copy the static binary from the pinned upstream image (pin the tag, never :latest).
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /usr/local/bin/

# git: the worker clones/operates on repos. docker CLI: the worker drives the mounted
# host docker.sock to spin up disposable done-check containers as sibling containers.
# gh: the worker shells out to the GitHub CLI on the host (not in the runner) to list
# ready-for-agent issues, file fix/backlog issues, and open staging PRs — installed from
# GitHub's own apt repo (the slim base has no `gh`). curl/gnupg/ca-certificates bootstrap
# that repo's signing key.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git docker.io curl gnupg ca-certificates \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Put the project venv on PATH so `uvicorn` / `retinue-worker` resolve without `uv run`.
ENV PATH="/app/.venv/bin:${PATH}"
# Bytecode-compile on install; never write .pyc back into the read-only layers at runtime.
ENV UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Layer-cache dependencies: resolve from the lockfile before the source is present, so
# a source-only change does not invalidate the (expensive) dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project itself into the same venv.
COPY . .
RUN uv sync --frozen --no-dev

# No CMD: docker-compose.yml supplies the command for each service.
