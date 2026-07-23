#!/usr/bin/env bash
# Auto-detect CUDA and install this repo with a matching torch build.
# The wheel index is chosen from the driver's max CUDA version; pip picks the
# CPU arch (x86_64 / aarch64) automatically. For CUDA 11.x drivers, install
# manually with --extra-index-url .../cu118 (see README).
set -euo pipefail
cd "$(dirname "$0")"

# Newer drivers (R610+) print "CUDA UMD Version:"; older ones "CUDA Version:"
cuda=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9]+' | grep -oE '[0-9]+$' || true)

if [ -z "${cuda:-}" ]; then
  idx=cpu                       # no NVIDIA GPU
elif [ "$cuda" -ge 13 ]; then
  idx=cu130                     # Blackwell / CUDA-13 drivers
else
  idx=cu126                     # CUDA 12.x drivers (default)
fi

echo ">> detected CUDA=${cuda:-none} -> installing torch build: $idx"
pip install -e . --extra-index-url "https://download.pytorch.org/whl/$idx"
