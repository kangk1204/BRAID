"""Safety checks for loading legacy BRAID pickle artifacts."""

from __future__ import annotations

import hashlib
import os
import stat


def trusted_pickle_roots(project_root: str | None = None) -> list[str]:
    """Return real trusted roots for legacy pickle artifacts."""
    env = os.environ.get("BRAID_TRUSTED_PKL_ROOT")
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    candidates = (
        [p for p in env.split(os.pathsep) if p]
        if env
        else [
            os.path.join(project_root, "benchmarks", "results"),
            os.path.join(project_root, "benchmark_results"),
        ]
    )

    roots: list[str] = []
    for candidate in candidates:
        root = os.path.realpath(candidate)
        if not os.path.isdir(root):
            if env:
                raise ValueError(f"Trusted pickle root does not exist: {candidate!r}")
            continue
        if os.stat(root).st_mode & stat.S_IWOTH:
            raise ValueError(f"Trusted pickle root is world-writable: {root!r}")
        roots.append(root)
    return roots


def file_sha256(path: str) -> str:
    """Compute a file's SHA256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pickle_allowlist_hint(path: str) -> str:
    """Return a shell-ready hint for allow-listing a trusted pickle artifact."""
    digest = file_sha256(path)
    return f"export BRAID_TRUSTED_PKL_SHA256={digest}"


def _allowed_pickle_hashes() -> set[str]:
    env = os.environ.get("BRAID_TRUSTED_PKL_SHA256", "")
    normalized = env.replace(",", os.pathsep).replace(" ", os.pathsep)
    return {
        token.strip().lower()
        for token in normalized.split(os.pathsep)
        if token.strip()
    }


def ensure_trusted_pickle(path: str, project_root: str | None = None) -> None:
    """Require a trusted root and explicit SHA256 allowlist before unpickling."""
    rp = os.path.realpath(path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"Bootstrap results file not found: {path}")

    roots = trusted_pickle_roots(project_root)
    if not roots:
        raise ValueError("No trusted pickle root directories exist.")
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        raise ValueError(
            f"Refusing to unpickle {path!r}: pickle can execute arbitrary code "
            f"on load, so BRAID only reads .pkl files under a trusted root "
            f"({', '.join(roots)}). Move the file there, or set "
            f"BRAID_TRUSTED_PKL_ROOT to a directory you trust."
        )

    digest = file_sha256(rp)
    allowed = _allowed_pickle_hashes()
    if digest not in allowed:
        raise ValueError(
            f"Refusing to unpickle {path!r}: SHA256 {digest} is not allow-listed. "
            f"Set BRAID_TRUSTED_PKL_SHA256 to the expected digest for this "
            f"BRAID-generated artifact, for example: "
            f"`export BRAID_TRUSTED_PKL_SHA256={digest}`."
        )
