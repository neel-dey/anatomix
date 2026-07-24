#!/usr/bin/env bash
# Install the FireANTs registration backend for anatomix.
#
# FireANTs (upstream: https://github.com/rohitrango/FireANTs) is the GPU
# diffeomorphic registration library that powers ``anatomix-register.py``. This
# script installs the anatomix author's fork
# (https://github.com/neel-dey/FireANTs) as a gitignored, editable clone next to
# this file, so the backend is never committed into anatomix. FireANTs is
# distributed under its own license -- the custom "FireANTs License v1.0" (see
# ``fireants/LICENSE`` after cloning), which is more restrictive than Apache-2.0.
#
# Usage:
#   bash install_fireants.sh                 # full install, WITH fused-ops (recommended)
#   bash install_fireants.sh --no-fused-ops  # skip the fused-ops CUDA extension
#
# The fused-ops CUDA extension (module: fireants_fused_ops) is built by default
# and REQUIRED FOR NOW for correct results: the current fork's pure-PyTorch
# fallback has a multi-resolution FFT-downsampling issue that degrades accuracy
# and introduces folds. Building it needs a CUDA toolkit whose version matches
# your PyTorch build (e.g. for a cu130 torch, point CUDA_HOME at a CUDA 13.x
# toolkit and set TORCH_CUDA_ARCH_LIST to your GPU arch, e.g. 12.0 for Blackwell).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE="${HERE}/fireants"
REPO_URL="https://github.com/neel-dey/FireANTs"

WITH_FUSED_OPS=1
if [ "${1:-}" = "--no-fused-ops" ]; then
    WITH_FUSED_OPS=0
fi

# 1. Clone the fork (skip if already present).
if [ ! -d "${CLONE}/.git" ]; then
    echo ">> Cloning FireANTs into ${CLONE}"
    git clone "${REPO_URL}" "${CLONE}"
else
    echo ">> FireANTs clone already present at ${CLONE} (skipping clone)"
fi

# 2. Editable install WITHOUT dependencies. This protects the environment's
#    existing (e.g. Blackwell / CUDA-specific) torch build: FireANTs only
#    requires torch>=2.3.0, which the anatomix environment already satisfies.
echo ">> pip install -e (--no-deps) FireANTs"
python -m pip install -e "${CLONE}" --no-deps

# 3. Install FireANTs' runtime dependencies, excluding torch and numpy (already
#    present) and the defunct upstream 'typing' backport.
echo ">> Installing FireANTs runtime dependencies"
python -m pip install \
    "SimpleITK>=2.2.1" nibabel scipy scikit-image matplotlib tqdm pandas hydra-core

# 4. Build the fused-ops CUDA extension (module: fireants_fused_ops) by default.
if [ "${WITH_FUSED_OPS}" = "1" ]; then
    echo ">> Building fused-ops CUDA extension (required for correct results)"
    if ! ( cd "${CLONE}/fused_ops" && python setup.py build_ext && python setup.py install ); then
        echo "!! WARNING: fused-ops build failed. FireANTs will fall back to its"
        echo "!! pure-PyTorch path, which on the current fork is accuracy-degraded"
        echo "!! (fold-heavy) -- SOTA reproduction requires the fused-ops kernels."
        echo "!! Ensure a CUDA toolkit matching your torch build is available"
        echo "!! (CUDA_HOME / TORCH_CUDA_ARCH_LIST) and re-run, or pass"
        echo "!! --no-fused-ops to acknowledge the degraded fallback."
    fi
else
    echo ">> Skipping fused-ops extension (--no-fused-ops)."
    echo ">> NOTE: the pure-PyTorch fallback on the current fork is accuracy-"
    echo ">> degraded (fold-heavy); the fused-ops kernels are required for now"
    echo ">> for correct/SOTA results."
fi

# 5. Smoke-import.
echo ">> Verifying: import fireants"
python -c "import fireants; print('FireANTs OK:', fireants.__file__)"
if [ "${WITH_FUSED_OPS}" = "1" ]; then
    python -c "import torch, fireants_fused_ops; print('fused-ops OK')" \
        || echo "!! fused-ops not importable; see the warning above."
fi
echo ">> Done."
