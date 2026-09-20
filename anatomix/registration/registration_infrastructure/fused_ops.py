"""Whether FireANTs runs on its compiled CUDA kernels.

FireANTs picks its kernels at import time: the interpolator reads ``USE_FFO``,
while the Adam update and the FFT downsample simply try to import
``fireants_fused_ops``. Hiding that module therefore switches all three to their
PyTorch paths at once, which is what ``--fused-ops off`` does; ``--fused-ops on``
sets ``USE_FFO`` itself, so a value inherited from the environment cannot leave
the interpolator out of step with the other two. It all has to happen before
FireANTs is imported, so ``configure`` reads the flag straight off the command
line rather than waiting for argparse, accepting the same abbreviations argparse
would.
"""
import os
import sys

_MODULE = "fireants_fused_ops"
_FLAG = "--fused-ops"


class _Blocker:
    """Raise on ``import fireants_fused_ops`` so FireANTs takes its fallbacks.

    ``ModuleNotFoundError``, not a bare ``ImportError``: callers that test for the
    kernels, `pytest.importorskip` among them, read the two differently.
    """

    def find_spec(self, name, path=None, target=None):
        if name == _MODULE:
            raise ModuleNotFoundError(
                "the compiled kernels are off (--fused-ops off)", name=_MODULE)
        return None


def disable():
    """Hide the compiled kernels; False if FireANTs already imported them."""
    if _MODULE in sys.modules:
        return False
    sys.meta_path.insert(0, _Blocker())
    os.environ["USE_FFO"] = "False"
    return True


def enable():
    """Ask for the compiled kernels, overriding a ``USE_FFO`` set in the environment."""
    os.environ["USE_FFO"] = "True"


def _names_the_flag(word):
    """True for ``--fused-ops`` and for the abbreviations argparse accepts for it."""
    return len(word) > 2 and _FLAG.startswith(word)


def configure(argv):
    """Apply ``--fused-ops on|off`` before FireANTs is imported; returns the choice.

    An unrecognized value is left alone for argparse to report.
    """
    choice = "off"
    for index, argument in enumerate(argv):
        word, separator, inline = argument.partition("=")
        if not _names_the_flag(word):
            continue
        if separator:
            choice = inline
        elif index + 1 < len(argv):
            choice = argv[index + 1]
    if choice.lower() == "off":
        disable()
    elif choice.lower() == "on":
        enable()
    return choice
