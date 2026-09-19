#!/usr/bin/env bash
# Install the FireANTs registration library (a minimally modified fork) as an
# editable clone next to this script.
# Usage: bash install_fireants.sh [--no-fused-ops]
# The fused CUDA kernels need a CUDA toolkit that matches the installed PyTorch
# build (set CUDA_HOME and TORCH_CUDA_ARCH_LIST if the build cannot find it).
# FireANTs has its own license; see fireants/LICENSE after cloning.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE="${HERE}/fireants"
REPO_URL="https://github.com/neel-dey/FireANTs"
# Pinned fork revision. An existing clone is moved to it unless it has local changes.
FIREANTS_REV="171f1527ea97e87661e60a09958e38a2e8d2d269"

WITH_FUSED_OPS=1
if [ "${1:-}" = "--no-fused-ops" ]; then
    WITH_FUSED_OPS=0
fi

if [ ! -d "${CLONE}/.git" ]; then
    echo ">> Cloning FireANTs into ${CLONE}"
    git clone "${REPO_URL}" "${CLONE}"
    git -C "${CLONE}" checkout --quiet "${FIREANTS_REV}"
else
    echo ">> FireANTs clone already present at ${CLONE}"
    CURRENT_REV="$(git -C "${CLONE}" rev-parse HEAD)"
    if [ "${CURRENT_REV}" != "${FIREANTS_REV}" ]; then
        if [ -n "$(git -C "${CLONE}" status --porcelain)" ]; then
            echo "!! It is at ${CURRENT_REV}, not the pinned revision, and has local changes."
            echo "!! Commit or stash them and re-run, or check out ${FIREANTS_REV} by hand."
            exit 1
        fi
        echo ">> Updating it to the pinned ${FIREANTS_REV}"
        git -C "${CLONE}" fetch --quiet origin
        git -C "${CLONE}" checkout --quiet "${FIREANTS_REV}"
    fi
fi

# --no-deps keeps the environment's PyTorch build.
echo ">> pip install -e FireANTs"
python -m pip install -e "${CLONE}" --no-deps
python -m pip install \
    "SimpleITK>=2.2.1" nibabel scipy scikit-image matplotlib tqdm pandas hydra-core

if [ "${WITH_FUSED_OPS}" = "1" ]; then
    echo ">> Building the fused-ops CUDA extension"
    if ! ( cd "${CLONE}/fused_ops" && python setup.py build_ext && python setup.py install ); then
        echo "!! fused-ops build failed; FireANTs will use its slower pure-PyTorch path."
        echo "!! Make a CUDA toolkit matching your torch build visible (CUDA_HOME,"
        echo "!! TORCH_CUDA_ARCH_LIST) and re-run to get the speedup."
    fi
else
    echo ">> Skipping the fused-ops extension (--no-fused-ops)."
fi

echo ">> Verifying the installation"
python -c "import fireants; print('FireANTs OK:', fireants.__file__)"
if [ "${WITH_FUSED_OPS}" = "1" ]; then
    python -c "import torch, fireants_fused_ops; print('fused-ops OK')" \
        || echo "!! fused-ops not importable; see the warning above."
fi
echo ">> Done."
