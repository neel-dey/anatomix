"""Registration backends for anatomix.

This package holds self-contained registration backends:

- :mod:`~anatomix.registration.registration_backend.convexadam` -- the original
  ConvexAdam network-feature + MIND-SSC backend (ICLR'25 results), retained and
  importable but not wired into the new FireANTs command-line interface.
- ``fireants`` -- a gitignored, editable clone of the FireANTs library installed
  by ``install_fireants.sh``. It is imported as the top-level ``fireants``
  package (not through this path) once installed.
"""
