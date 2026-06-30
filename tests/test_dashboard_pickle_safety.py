from __future__ import annotations

import os

import pytest

from braid.dashboard.pickle_safety import (
    ensure_trusted_pickle,
    file_sha256,
    pickle_allowlist_hint,
    trusted_pickle_roots,
)


def test_pickle_requires_sha256_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    pkl = root / "artifact.pkl"
    pkl.write_bytes(b"not really a pickle")
    monkeypatch.setenv("BRAID_TRUSTED_PKL_ROOT", str(root))
    monkeypatch.delenv("BRAID_TRUSTED_PKL_SHA256", raising=False)

    with pytest.raises(ValueError, match="not allow-listed"):
        ensure_trusted_pickle(str(pkl), str(tmp_path))

    hint = pickle_allowlist_hint(str(pkl))
    assert hint == f"export BRAID_TRUSTED_PKL_SHA256={file_sha256(str(pkl))}"


def test_pickle_accepts_matching_sha256_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    pkl = root / "artifact.pkl"
    pkl.write_bytes(b"trusted BRAID pickle bytes")
    monkeypatch.setenv("BRAID_TRUSTED_PKL_ROOT", str(root))
    monkeypatch.setenv("BRAID_TRUSTED_PKL_SHA256", file_sha256(str(pkl)))

    ensure_trusted_pickle(str(pkl), str(tmp_path))


def test_pickle_rejects_world_writable_trusted_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    root.chmod(0o777)
    monkeypatch.setenv("BRAID_TRUSTED_PKL_ROOT", str(root))

    try:
        with pytest.raises(ValueError, match="world-writable"):
            trusted_pickle_roots(str(tmp_path))
    finally:
        root.chmod(0o700)


def test_pickle_rejects_path_outside_trusted_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"outside")
    monkeypatch.setenv("BRAID_TRUSTED_PKL_ROOT", str(root))
    monkeypatch.setenv("BRAID_TRUSTED_PKL_SHA256", file_sha256(str(outside)))

    with pytest.raises(ValueError, match="under a trusted root"):
        ensure_trusted_pickle(os.path.join(root, "..", "outside.pkl"), str(tmp_path))
