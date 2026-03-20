"""Tests for the BRAID install diagnostics."""

from __future__ import annotations

from braid import doctor


def test_build_install_report_core_ready(monkeypatch) -> None:
    """Core packages should be enough to mark the install as ready."""

    def fake_check_package(spec: dict[str, str]) -> dict[str, object]:
        return {
            "label": spec["label"],
            "module": spec["module"],
            "distribution": spec["distribution"],
            "installed": True,
            "version": "1.0",
            "path": "/tmp/fake.py",
        }

    def fake_check_command(spec: dict[str, str]) -> dict[str, object]:
        return {
            "label": spec["label"],
            "command": spec["command"],
            "installed": False,
            "path": None,
            "note": spec.get("note"),
        }

    monkeypatch.setattr(doctor, "_check_python_package", fake_check_package)
    monkeypatch.setattr(doctor, "_check_command", fake_check_command)

    report = doctor.build_install_report(strict=False)
    assert report["summary"]["core_ready"] is True
    assert report["summary"]["status"] == "ok"
    assert "samtools" in report["summary"]["missing_optional_commands"]


def test_build_install_report_strict_requires_optional_tools(monkeypatch) -> None:
    """Strict mode should fail when BRAID benchmark dependencies are missing."""

    def fake_check_package(spec: dict[str, str]) -> dict[str, object]:
        installed = spec["label"] != "pyliftover"
        return {
            "label": spec["label"],
            "module": spec["module"],
            "distribution": spec["distribution"],
            "installed": installed,
            "version": "1.0" if installed else None,
            "path": "/tmp/fake.py" if installed else None,
            "error": None if installed else "No module named 'pyliftover'",
        }

    def fake_check_command(spec: dict[str, str]) -> dict[str, object]:
        installed = spec["label"] not in {"samtools", "stringtie", "rmats.py"}
        return {
            "label": spec["label"],
            "command": spec["command"],
            "installed": installed,
            "path": f"/usr/bin/{spec['command']}" if installed else None,
            "note": spec.get("note"),
        }

    monkeypatch.setattr(doctor, "_check_python_package", fake_check_package)
    monkeypatch.setattr(doctor, "_check_command", fake_check_command)

    report = doctor.build_install_report(strict=True)
    assert report["summary"]["core_ready"] is True
    assert report["summary"]["benchmark_packages_ready"] is False
    assert report["summary"]["optional_commands_ready"] is False
    assert report["summary"]["status"] == "needs_attention"
    assert "pyliftover" in report["summary"]["missing_benchmark_packages"]
    assert "samtools" in report["summary"]["missing_optional_commands"]


def test_build_install_report_gpu_requires_cupy(monkeypatch) -> None:
    """GPU mode should fail cleanly when cupy is missing."""

    def fake_check_package(spec: dict[str, str]) -> dict[str, object]:
        installed = spec["label"] != "cupy"
        return {
            "label": spec["label"],
            "module": spec["module"],
            "distribution": spec["distribution"],
            "installed": installed,
            "version": "1.0" if installed else None,
            "path": "/tmp/fake.py" if installed else None,
            "note": spec.get("note"),
            "error": None if installed else "libcuda.so not found",
        }

    def fake_check_command(spec: dict[str, str]) -> dict[str, object]:
        return {
            "label": spec["label"],
            "command": spec["command"],
            "installed": True,
            "path": f"/usr/bin/{spec['command']}",
            "note": spec.get("note"),
        }

    monkeypatch.setattr(doctor, "_check_python_package", fake_check_package)
    monkeypatch.setattr(doctor, "_check_command", fake_check_command)

    report = doctor.build_install_report(gpu=True)
    assert report["summary"]["gpu_ready"] is False
    assert report["summary"]["status"] == "needs_attention"
    assert report["summary"]["missing_gpu_packages"] == ["cupy"]


def test_format_install_report_includes_import_error_and_gpu_hint() -> None:
    """Human-readable output should show package import errors."""

    report = {
        "platform": {"system": "Linux", "release": "6.0", "machine": "x86_64"},
        "python": {"version": "3.10", "executable": "/usr/bin/python"},
        "core_packages": [],
        "benchmark_packages": [
            {
                "label": "pyliftover",
                "installed": False,
                "error": "No module named 'pyliftover'",
            }
        ],
        "optional_packages": [
            {
                "label": "cupy",
                "installed": False,
                "error": "libcuda.so not found",
            }
        ],
        "optional_commands": [],
        "summary": {
            "strict": True,
            "gpu": True,
            "core_ready": True,
            "benchmark_packages_ready": False,
            "optional_packages_ready": False,
            "optional_commands_ready": True,
            "gpu_ready": False,
            "status": "needs_attention",
            "missing_core_packages": [],
            "missing_benchmark_packages": ["pyliftover"],
            "missing_optional_packages": ["cupy"],
            "missing_gpu_packages": ["cupy"],
            "missing_optional_commands": [],
        },
    }

    text = doctor.format_install_report(report)
    assert "No module named 'pyliftover'" in text
    assert "libcuda.so not found" in text
    assert "Missing GPU Python packages: cupy" in text
