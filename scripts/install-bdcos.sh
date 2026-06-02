#!/usr/bin/env bash
# Install scriber as a lean, CPU-only transcription tool (e.g. for BDC OS).
#
# Base install is transcription-only: whisper on a CPU-only torch wheel, with
# no diarization and no summarization dependencies. Pass --diarize to add
# speaker attribution (pulls pyannote + torchaudio, ~+0.5 GB, and needs
# HUGGINGFACE_TOKEN at runtime for the gated pyannote models).
#
# Usage:
#   ./install-bdcos.sh                 # transcription-only, CPU torch
#   ./install-bdcos.sh --diarize       # + speaker diarization extra
#
# scriber itself does NOT pin torch to CPU (it stays GPU-capable for its own
# users) — we select the CPU backend here, at install time, via UV_TORCH_BACKEND.
set -euo pipefail

REPO="git+https://github.com/arnobeutch/scriber.git"
EXTRAS=""

for arg in "$@"; do
  case "$arg" in
    --diarize) EXTRAS="diarize" ;;
    -h | --help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (expected --diarize)" >&2
      exit 2
      ;;
  esac
done

# CPU-only torch. Honored by uv's resolver (uv >= 0.5; tested on 0.11). If a
# future uv ignores it for `tool install`, fall back to an explicit CPU index:
#   uv tool install --index https://download.pytorch.org/whl/cpu "scriber @ $REPO"
export UV_TORCH_BACKEND=cpu

if [[ -n "$EXTRAS" ]]; then
  echo "Installing scriber[$EXTRAS] (CPU torch) ..."
  uv tool install "scriber[$EXTRAS] @ $REPO"
  echo "Diarization enabled — export HUGGINGFACE_TOKEN before running --diarize."
else
  echo "Installing scriber (transcription-only, CPU torch) ..."
  uv tool install "scriber @ $REPO"
fi

echo "Done. Try: scriber transcribe <input> --language <fr|en> [--diarize]"
