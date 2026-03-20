"""Allow running BRAID as ``python -m braid``."""

from braid.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
