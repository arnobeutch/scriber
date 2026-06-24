# Pull note: scriber → BDC OS (2026-06-24)

**Audience:** the bdc_os maintainer / session.
**Author:** Claude, from a scriber session.
**Pull target:** scriber `main` **at or after `38d65ad`** (`build: cap
requires-python <3.14`). Pulling an earlier ref reintroduces the silent
3.14-build-failure behaviour described below.

This note is the reverse of `BDCOS_INTEGRATION.md` (which was bdc_os → scriber).
It tells the bdc_os side what changed on scriber and what to reconcile. The
scriber side will **not** edit `BDCOS_INTEGRATION.md` — that file is bdc_os-owned.

---

## TL;DR

1. **Python is now capped `>=3.11,<3.14`.** Provision a 3.11–3.13 interpreter for
   the scriber tool env.
2. **pyannote 3.x → 4.x re-gates the Hugging Face model** (`speaker-diarization-3.1`
   → `speaker-diarization-community-1`). One-time HF access re-grant for `--diarize`.
3. **`docs/BDCOS_INSTALL.md` (scriber-side mirror) was updated** — a new
   **Prerequisites** section. The canonical bdc_os copy + BDC OS's general prereqs
   need the same change.
4. **`BDCOS_INTEGRATION.md` has stale facts** (listed below). Recommend updating or
   retiring it bdc_os-side; canonical truth is `BDCOS_INSTALL.md` + scriber's
   `pyproject.toml`.

---

## 1. What changed in scriber that affects the pull

| # | Change | Impact on BDC OS | Action |
| --- | --- | --- | --- |
| 1 | `requires-python = ">=3.11,<3.14"` (3.14 deferred — `tiktoken` + the summarize chain lack cp314 wheels) | `uv tool install` needs a 3.11–3.13 interpreter. uv usually provisions a managed one; a **system-only 3.14** env fails to resolve. Before the cap, 3.14 failed anyway (at `tiktoken` build) — 3.14 was never actually working. | Ensure a 3.11–3.13 interpreter is reachable on the host. |
| 2 | pyannote 3.x → 4.x: gated model is now `pyannote/speaker-diarization-community-1` | HF accounts with access to the old `speaker-diarization-3.1` must request the new repo, or `--diarize` fails with `GatedRepoError`. | Re-grant HF access once, per account. |
| 3 | `torchcodec` now pulled (pyannote 4.x / torchaudio 2.11) | Adds install weight; scriber sidesteps it at runtime (in-memory waveform), so no FFmpeg-DLL runtime gotcha. The `~2.1 GB` diarize footprint is likely now a bit low. | Re-measure footprint on the BDC OS box if it's budgeted. |
| 4 | New `--suggest-primer` flag + `primer.py` | None — transcribe-only callers unaffected; the `scriber transcribe <input> --language <fr\|en> [--diarize]` contract is unchanged. | None. |

---

## 2. `docs/BDCOS_INSTALL.md` was updated (scriber-side mirror)

Added a **`## Prerequisites`** section consolidating what the host must have
*before* install (none are pulled by the install):

- **`uv`** — installs/runs scriber as an isolated tool.
- **Python 3.11–3.13** — the new cap (was previously unstated).
- **`ffmpeg`** system binary — whisper decodes audio through `ffmpeg-python`; no
  wheel ships it. **Previously undocumented** — a host without `ffmpeg` installs
  fine then fails at first transcription.
- **`HUGGINGFACE_TOKEN`** + gated-model access — `--diarize` only (already
  documented; now also listed in the consolidated table).

**bdc_os-side actions:**
1. Sync the canonical copy `bdc_os/docs/scriber-install.md` with the new
   Prerequisites section.
2. **Fold Python 3.11–3.13 and `ffmpeg` into BDC OS's *general* prerequisites
   list** — they were missing there, which is the gap that prompted this. The HF
   token already lives with the other secrets (`.bdcos.env`); just ensure the
   gated-model access step is noted.

---

## 3. `BDCOS_INTEGRATION.md` — what's no longer valid + what to change

It is self-flagged as a historical snapshot, so most drift is pre-disclaimed.
Concretely stale (verified against scriber `main` @ `38d65ad`):

| Where | No longer valid | Change to |
| --- | --- | --- |
| Embedded `pyproject` (`diarize = [...]`) | `pyannote-audio>=3.3.2` | `pyannote-audio>=4.0`; the `diarize` extra also pulls `torchcodec`. |
| Embedded `[tool.uv.sources]` block | unconditional `torch = { index = "pytorch-cpu" }` | **Rejected design** (the note's own deviation #1 says so). Reality: torch is *not* CPU-pinned upstream; CPU is per-install via `UV_TORCH_BACKEND=cpu`. The only committed `[tool.uv.sources]` pin is the **win32-only CUDA** index. |
| Verify section | "full `just all` (271 tests)" | **368 tests**. |
| "The measurement" §| "6.3 GB full `uv sync`" + heavy-hitter table | Point-in-time, pre-4.x / pre-torchcodec. Re-measure or drop the figure. |
| (absent) | no Python version ceiling mentioned | Add `>=3.11,<3.14`. |
| §1–4 "TODO" framing | reads as a plan to execute | The work is **implemented** (status header already says so) — reframe to past tense or retire. |

**Recommendation:** rather than maintain two overlapping docs, **retire
`BDCOS_INTEGRATION.md` to an archive** (or add a hard "superseded — see
`BDCOS_INSTALL.md` + scriber `pyproject.toml`" banner at the top). It has served
its handoff purpose; keeping it live invites exactly this drift.

---

## 4. Still valid — no change needed

- The **transcribe-only contract**: `scriber transcribe <input> --language <fr|en> [--diarize]`, subprocess, no shared env, no fixed path.
- **Token sourcing**: caller exports `HUGGINGFACE_TOKEN` into scriber's process
  env; `settings.py` also seeds a cwd `.env` via `setdefault` (real env var wins).
- **CPU-torch-per-install** via `UV_TORCH_BACKEND=cpu` (tested on uv 0.11; current
  scriber dev box is uv 0.11.19).
- **whisper engine + tuning** (`docs/WHISPER_SETUP.md`) — unchanged.
