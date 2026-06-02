default:
    @just --list

# Sync the dev environment with all optional extras (diarize + summarize).
# The package ships transcription-only by default; contributors run the full
# suite, so the dev env always carries every extra.
sync:
    uv sync --all-extras

lint:
    uv run --all-extras ruff check .
    uv run --all-extras ruff format --check .

format:
    uv run --all-extras ruff format .
    uv run --all-extras ruff check --fix .

typecheck:
    uv run --all-extras pyright

test:
    uv run --all-extras pytest

all: lint typecheck test
