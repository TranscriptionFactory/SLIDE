"""Packaging must fail on unresolved libraries or existing artifacts."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "package_rust_linux", Path(__file__).parents[2] / "scripts/package-rust-linux.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


@pytest.mark.parametrize("output", ["", "libmissing.so\n", "/missing/slide-library.so\n"])
def test_reject_unresolved_dependencies(monkeypatch, output):
    monkeypatch.setattr(packager.subprocess, "check_output", lambda *args, **kwargs: output)
    with pytest.raises(RuntimeError):
        packager.dependencies(Path("/unused/slide"))


def test_existing_archive_is_not_overwritten(tmp_path):
    destination = tmp_path / "slide.1.0"
    archive = Path(str(destination) + ".tar.gz")
    archive.write_bytes(b"previous artifact")
    with pytest.raises(FileExistsError):
        packager.package(tmp_path / "missing-binary", destination)
    assert archive.read_bytes() == b"previous artifact"
    assert not destination.exists()


@pytest.mark.parametrize("versions,allowed", [
    ("GLIBC_2.2.5 GLIBC_2.35", True),
    ("GLIBC_2.9 GLIBC_2.34", True),
    ("GLIBC_2.36 GLIBC_2.3", False),
])
def test_glibc_floor_is_numeric(monkeypatch, versions, allowed):
    monkeypatch.setattr(packager.subprocess, "check_output", lambda *args, **kwargs: versions)
    if allowed:
        packager.check_glibc_floor(Path("unused"))
    else:
        with pytest.raises(RuntimeError, match="floor is 2.35"):
            packager.check_glibc_floor(Path("unused"))
