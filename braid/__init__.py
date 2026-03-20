"""Public BRAID wrapper package.

The implementation still lives under the legacy ``rapidsplice`` module path for
backward compatibility. New code should prefer the public ``braid`` package and
CLI entry point.
"""

from rapidsplice import __version__

__all__ = ["__version__"]
