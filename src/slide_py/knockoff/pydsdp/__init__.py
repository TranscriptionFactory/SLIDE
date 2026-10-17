"""
Local copy of pydsdp with bug fixes for knockoff SDP solving.

Fixes applied:
- convert.py: Added str() conversion for float values in SDPA file output
- convert.py: Use COO format for consistent row/col/data ordering in sparse matrices

The C extension (pydsdp5) is still imported from the system pydsdp package.
"""

from .dsdp5 import dsdp, dsdp_readsdpa

__all__ = ['dsdp', 'dsdp_readsdpa']
