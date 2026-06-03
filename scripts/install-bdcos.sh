#!/usr/bin/env bash
# Install scriber as BDC OS's transcription provider.
#
# BDC OS owns this integration (canonical copy lives in the bdc_os repo); this is
# scriber's in-sync reference copy. Installs scriber from GitHub as an isolated
# `uv` tool — no local scriber checkout is read.
#
# Default install is the "reduced" scriber: CPU-only torch (sheds the ~2.7 GB
# CUDA nvidia/* stack), speaker diarization INCLUDED, summarization backends
# EXCLUDED (BDC OS summarizes downstream with its own skills).
#
#   default footprint ~2.1 GB  = base transcription + diarize stack (measured)
#   --no-diarize     ~1.3 GB   = transcription only (leaner; no speaker labels)
#
# Diarization needs HUGGINGFACE_TOKEN at runtime (gated pyannote models); the
# install itself does not.
#
# Usage:
#   ./scripts/install-bdcos.sh                # CPU, base + diarization (default)
#   ./scripts/install-bdcos.sh --no-diarize   # CPU, transcription only
#
# scriber stays GPU-capable for its own users; the CPU choice is made HERE, at
# install time, via UV_TORCH_BACKEND — scriber does not pin it upstream.
set -euo pipefail

REPO="git+https://github.com/arnobeutch/scriber.git"
SPEC="scriber[diarize]"
DIARIZE=1

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --no-diarize)
      SPEC="scriber"
      DIARIZE=0
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (expected --no-diarize)" >&2
      exit 2
      ;;
  esac
done

# CPU-only torch. Honored by uv's resolver (tested on uv 0.11). If a future uv
# ignores it for `tool install`, fall back to an explicit CPU index:
#   uv tool install --index https://download.pytorch.org/whl/cpu "$SPEC @ $REPO"
export UV_TORCH_BACKEND=cpu

echo "Installing ${SPEC} (CPU torch) from GitHub ..."
uv tool install "${SPEC} @ ${REPO}"

if [[ "${DIARIZE}" -eq 1 ]]; then
  echo "Diarization included — export HUGGINGFACE_TOKEN before running with --diarize."
fi
echo "Done. Try: scriber transcribe <input> --language <fr|en> [--diarize]"
