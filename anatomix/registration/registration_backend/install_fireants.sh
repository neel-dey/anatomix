#!/usr/bin/env bash
# Install the FireANTs registration backend for anatomix.
#
# FireANTs (https://github.com/rohitrango/FireANTs) is the GPU diffeomorphic
# registration library that powers ``anatomix-register.py``. This script installs
# the anatomix author's fork (https://github.com/neel-dey/FireANTs) as a
# gitignored, editable clone next to this file, so the backend is never committed
# into anatomix and is installed under its own (Apache-2.0) license.
#
# Usage:
#   bash install_fireants.sh                    # core install (recommended)
#   bash install_fireants.sh --with-fused-ops   # also build the optional CUDA
#                                               # fused-ops extension
#
# The optional fused-ops extension only accelerates FireANTs (fused CC/MI losses
# and interpolation); ordinary single-GPU registration works without it through
# FireANTs' built-in PyTorch fallbacks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE="${HERE}/fireants"
REPO_URL="https://github.com/neel-dey/FireANTs"

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

# 4. Optionally build the fused-ops CUDA extension (module: fireants_fused_ops).
if [ "${1:-}" = "--with-fused-ops" ]; then
    echo ">> Building optional fused-ops CUDA extension"
    ( cd "${CLONE}/fused_ops" && python setup.py build_ext && python setup.py install )
else
    echo ">> Skipping optional fused-ops extension (re-run with --with-fused-ops to build it)."
fi

# 5. Smoke-import.
echo ">> Verifying: import fireants"
python -c "import fireants; print('FireANTs OK:', fireants.__file__)"
echo ">> Done."
