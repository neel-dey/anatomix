"""Whether FireANTs runs on its compiled CUDA kernels.

FireANTs picks its kernels at import time: the interpolator reads ``USE_FFO``,
while the Adam update and the FFT downsample simply try to import
``fireants_fused_ops``. Hiding that module therefore switches all three to their
PyTorch paths at once, which is what ``--fused-ops off`` does. It has to happen
before FireANTs is imported, so ``configure`` reads the flag straight off the
command line rather than waiting for argparse.
"""
import os
import sys

_MODULE = "fireants_fused_ops"


class _Blocker:
    """Raise on ``import fireants_fused_ops`` so FireANTs takes its fallbacks."""

    def find_spec(self, name, path=None, target=None):
        if name == _MODULE:
            raise ImportError("the compiled kernels are off (--fused-ops off)")
        return None


def disable():
    """Hide the compiled kernels; False if FireANTs already imported them."""
    if _MODULE in sys.modules:
        return False
    sys.meta_path.insert(0, _Blocker())
    os.environ["USE_FFO"] = "False"
    return True


def configure(argv):
    """Apply ``--fused-ops on|off`` before FireANTs is imported; returns the choice.

    An unrecognized value is left alone for argparse to report.
    """
    choice = "off"
    for index, argument in enumerate(argv):
        if argument == "--fused-ops" and index + 1 < len(argv):
            choice = argv[index + 1]
        elif argument.startswith("--fused-ops="):
            choice = argument.split("=", 1)[1]
    if choice.lower() == "off":
        disable()
    return choice
