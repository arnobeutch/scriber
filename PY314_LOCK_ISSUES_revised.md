# Python 3.14: committed `uv.lock` is not installable — process Windows-side

**Status:** **decided 2026-06-24 (Windows-side): defer 3.14, target 3.11–3.13.**
See _Feedback & plan_ at the bottom for the agreed fix. Originally diagnosed on
the Linux box 2026-06-23 against commit `e224fb9`.
**Action owner:** Linux box (it has `python3.14`/`3.13` and can both regenerate
the lock — `uv lock` is platform-agnostic — and validate the sync; Windows is on
3.11.9 and can't validate 3.14 anyway). Windows re-checks after the push.
**Constraint:** uv only — no `pip` / `uv pip` in the fix. Regenerate via `uv lock`, install via `uv sync`.

## TL;DR

The committed `uv.lock` resolves **single** versions of several compiled
transitive deps that have **no cp314 wheel** (and, for `onnxruntime`, no sdist
either). A lock-based `uv sync` onto CPython 3.14.5 therefore fails. The dev box
is pinned to 3.11.9 via `.python-version`, so this never shows up locally — but
the lock as committed cannot produce a 3.14 environment.

If 3.14 works on the Windows box, its **working `uv.lock` differs from the
committed one** — check whether Windows has uncommitted lock upgrades, or whether
it was installed with `--upgrade`. The lock pulled to Linux does not install on
3.14.

## What *is* verified good

- **3.11.9** (the `.python-version` pin): `rm -rf .venv && uv sync --all-extras`
  rebuilds clean; `just all` green (ruff-ALL, pyright-strict, 368 tests); the full
  ML stack imports live (torch 2.12.0+cu130, pyannote 4.0.4, whisper).
- The lock *intends* to cover 3.14 — it carries `python_full_version >= '3.14'`
  resolution-markers and forks some deps by Python version (e.g. grpcio
  1.71 `<3.14` / 1.81 `>=3.14`). The blockers below are deps it *didn't* fork.

## The 3.14 blockers (stale single-version pins, no cp314 wheel)

| Package | Locked | Pulled in by | Problem on cp314 | cp314-capable from |
| --- | --- | --- | --- | --- |
| `onnxruntime` | 1.22.0 | `chromadb` (summarize/RAG extra) | wheels cp311–313 only, **no sdist** → hard resolver error | `>=1.24.1` |
| `tiktoken` | 0.9.0 | `openai-whisper` (base) | wheels cp311–313, falls back to Rust sdist build → fails | `>=0.12.0` |
| `watchfiles` | 1.0.5 | `chromadb` → `uvicorn[standard]` (summarize extra) | no 3.14-compatible wheel → Rust sdist build → fails | `>=1.1.0` |

`numba` (0.65.1) and `torch` (2.12.0) already ship cp314 wheels — they are **not**
blockers. The list above is the set found by walking `uv sync --all-extras
--python 3.14` failures one at a time; **more may surface in the chromadb /
uvicorn\[standard] chain** once these three are bumped (only `onnxruntime` +
`tiktoken` were cleared before `watchfiles` was hit).

## Do NOT "just `uv lock --upgrade`"

A blind full upgrade is the wrong fix. Tested it: it **downgrades `numba`
0.65.1 → 0.53.1** (which only supports Python `<3.10`), because numpy
backtracking drags numba down — breaking *every* Python. It also makes a large,
risky sweep (chromadb 1.0.9→1.5.9, `huggingface-hub` 0.31→1.20 major bump, etc.)
that needs its own validation.

## Recommended fix (Windows-side)

Targeted bumps of the cp314-blocking transitive pins only, keeping `numba 0.65.1`:

```sh
uv lock --upgrade-package onnxruntime \
        --upgrade-package tiktoken \
        --upgrade-package watchfiles
# then iterate: each `uv sync --all-extras` on 3.14 may surface the next stale
# pin in the chromadb/uvicorn chain — add it to the --upgrade-package list and
# re-lock. Stop when a 3.14 all-extras sync installs + imports cleanly.
uv sync --all-extras            # on the 3.14 box
```

Verify after: confirm `numba` is still `0.65.1` (not 0.53.x) in the new lock, and
that 3.11/3.13 still resolve (the lock is universal — one regen must keep the
whole `>=3.11` range working).

## Docs that currently overstate 3.14 support

Both claim 3.14 is verified; reconcile once the lock is fixed (or downgrade the
claim to 3.11–3.13 until then):

- `CLAUDE.md` → "3.11–3.14 verified: resolve + cp314 wheels + live import on 3.14.5".
- `pyproject.toml` → the cp314 comment on the `pyannote-audio>=4.0` / torchaudio note.

## How this was tested (reproducible, uv-only)

```sh
# lock-based 3.14 sync into a throwaway env — never touches ./.venv:
UV_PROJECT_ENVIRONMENT=/tmp/v314 uv sync --all-extras --python /usr/bin/python3.14
```

No `pip` / `uv pip` involved. cp314 wheel availability cross-checked against the
PyPI JSON API.

---

## Feedback & plan (Windows-side, 2026-06-24)

**Decision: defer 3.14, target 3.11–3.13. Cap `requires-python = ">=3.11,<3.14"`.**
The maintainer never actually required 3.14 — it was assumed to be a dep
constraint, which it is not. So we relax it.

## Analysis — agreement + one reframe

The diagnosis is correct and the **numba 0.65.1 → 0.53.1** trap on a blind
`uv lock --upgrade` is a great catch. But there's a reframe that **collapses
most of the work**:

- **All three blockers are cp314-only.** Per this doc's own table, `onnxruntime`
  1.22, `tiktoken` 0.9, and `watchfiles` 1.0.5 each ship **cp311–313 wheels**.
  Within 3.11–3.13 they install **off-the-shelf — prebuilt wheels, no sdist, no
  Rust build** — already proven by the clean 3.11.9 `uv sync --all-extras` (368
  tests green). The "no sdist → hard error" only bites at cp314, where no wheel
  exists. So **3.11–3.13 incl. `summarize` is OTS today.**
- **Therefore the `--upgrade-package onnxruntime/tiktoken/watchfiles` chase is
  not needed.** It existed only to *reach* 3.14. Dropping 3.14 turns this from a
  fragile source-build chase into: cap the upper bound, re-lock, validate. And
  there's **no numba risk** — that was a `--upgrade` artifact; a plain `uv lock`
  within the narrower range keeps numba (and everything else) pinned as-is.
- **"3.11–3.14 verified" was overstated.** `tiktoken` is a **base** dep (via
  `openai-whisper`), so no clean `uv sync --all-extras` on 3.14 ever passed with
  this lock. The honest, demonstrable range is **3.11–3.13**.

## Plan (Linux-side, uv-only)

### 1. Cap the range — `pyproject.toml`

```toml
# was: requires-python = ">=3.11"
requires-python = ">=3.11,<3.14"  # 3.14 deferred: summarize chain (chromadb/uvicorn → onnxruntime, watchfiles) + base tiktoken lack cp314 wheels
```

Leave ruff `target-version = "py311"` and pyright `pythonVersion = "3.11"` — the
3.11 floor for back-compat is still correct.

### 2. Doc corrections (drop the 3.14 claim)

- **`pyproject.toml`**, the `diarize` extra comment — remove the cp314 framing
  (the torchaudio requirement is about pyannote 4.x, not Python):

  ```
  # pyannote 4.x is required: torchaudio 2.11 removed torchaudio.AudioMetaData,
  # which every pyannote 3.x imports, so 3.x can't run on modern torchaudio.
  # 4.x targets the new torchaudio/torchcodec audio backend.
  ```

- **`CLAUDE.md`** (Tech-stack Python bullet, ~line 9) — replace the
  `3.11–3.14 verified` sentence with:

  ```
  - Python `>=3.11,<3.14` (3.11–3.13; 3.11.9 verified clean, 368 tests). 3.14 is
    deferred: the `summarize` extra's `chromadb → uvicorn[standard]` chain
    (`onnxruntime`, `watchfiles`) and base `tiktoken` ship no cp314 wheels yet.
    **Diarization requires `pyannote-audio>=4.0`** (4.x dropped the
    `torchaudio.AudioMetaData` pyannote 3.x imports → needs the new
    torchaudio/torchcodec backend); uses `token=` (not `use_auth_token=`).
  ```

- **`docs/DESIGN.md` §12** — currently reads `>=3.11 (3.11–3.14 verified)`
  (introduced in the doc-audit commit `e224fb9`; it now also overstates). Set to:

  ```
  - **Python `>=3.11,<3.14`** (3.11–3.13). 3.14 deferred — `summarize`'s
    chromadb/uvicorn chain + `tiktoken` lack cp314 wheels. Code stays
    3.11-compatible (`ruff`/`pyright` target 3.11).
  ```

- **`PY314_LOCK_ISSUES.md`** — `git rm` this file once the fix lands; the
  rationale lives in the commit message + the comments above.

### 3. Re-lock (no upgrade flags)

```sh
uv lock
```

Capping the upper bound makes uv drop the 3.14 forks → clean 3.11–3.13 universal
lock; 3.11–3.13 resolutions stay put.

### 4. Validate (throwaway envs — never touches `./.venv`)

```sh
grep -c "python_full_version >= '3.14'" uv.lock        # expect 0
grep -A2 'name = "numba"' uv.lock | grep version       # expect 0.65.1

for py in 3.11 3.12 3.13; do
  echo "=== $py ===" ; UV_PROJECT_ENVIRONMENT=/tmp/scriber-$py uv sync --all-extras --python "$py" || echo "FAILED $py"
done

UV_PROJECT_ENVIRONMENT=/tmp/scriber-3.12 uv run --no-sync python -c "import whisper, chromadb, pyannote.audio, openai, textblob; print('imports ok')"
UV_PROJECT_ENVIRONMENT=/tmp/scriber-3.12 uv run --no-sync pytest -q

uv sync --all-extras && just all   # dev env (3.11.9) still green
```

### 5. Commit + push (Linux)

```sh
git add pyproject.toml uv.lock CLAUDE.md docs/DESIGN.md
git rm PY314_LOCK_ISSUES.md
git commit -m "build: cap requires-python <3.14 (defer 3.14); docs to 3.11-13"
git push
```

### 6. Windows re-check (after push)

`git pull`, then `rm -rf .venv && uv sync --all-extras && just all` on 3.11.9.

## Fallback (only if step 4 surprises us)

If `uv sync --all-extras --python 3.13` fails on some *other* dep lacking a cp313
wheel (unlikely — the three known laggards all have cp313 wheels), drop to
`requires-python = ">=3.11,<3.13"` (3.11–3.12, the safe "3.12 compromise"),
re-lock, and set docs to 3.11–3.12.
