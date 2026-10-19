"""Pinned provider preparation must preserve inputs and reject unsafe archives."""

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "windows_openblas", Path(__file__).parents[2] / "scripts/prepare-windows-openblas.py")
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)


def asset(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name in provider.REQUIRED:
            content = "placeholder"
            if name.endswith("openblas.pc"):
                content = ("version=0.3.34\nopenblas_config= USE_64BITINT= NO_LAPACK= "
                           "NO_LAPACKE= DYNAMIC_ARCH=1 DYNAMIC_OLDER=1 GENERIC\n")
            elif name.endswith("openblas_config.h"):
                content = '#define OPENBLAS_ARCH_X86_64 1\n#define OPENBLAS_VERSION "OpenBLAS 0.3.34"\n'
            output.writestr(name, content)
    return archive


def test_checksum_rejection_does_not_extract(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not the pinned archive")
    destination = tmp_path / "provider"
    with pytest.raises(ValueError, match="SHA256"):
        provider.prepare(destination, archive)
    assert not destination.exists()
    assert archive.read_bytes() == b"not the pinned archive"


def test_existing_destination_refused_before_archive_access(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError):
        provider.prepare(destination, tmp_path / "missing.zip")
    assert sentinel.read_text() == "preserve me"


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "C:/absolute", "C:relative", "a/../../escaped", r"..\escaped", "//server/share", "name:stream", "a/.. /escaped", "a./file", "NUL.txt", "CON/file"])
def test_unsafe_zip_entries_rejected_before_any_extraction(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("otherwise-safe", "must not extract")
        output.writestr(name, "bad")
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive) as source, pytest.raises(ValueError, match="Unsafe"):
        provider.safe_extract(source, destination)
    assert not destination.exists()


def test_symlink_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    entry = zipfile.ZipInfo("link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(entry, "../outside")
    with zipfile.ZipFile(archive) as source, pytest.raises(ValueError, match="Unsafe"):
        provider.safe_extract(source, tmp_path / "output")


def test_preparation_preserves_archive_and_spaces(tmp_path, monkeypatch):
    archive = asset(tmp_path)
    monkeypatch.setattr(provider, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    destination = tmp_path / "provider with spaces"
    provider.prepare(destination, archive)
    assert (destination / "OpenBLAS-0.3.34-x64.zip").read_bytes() == archive.read_bytes()
    assert (destination / "archive/openblas.pc").read_text().startswith("version=")
    shim = (destination / "lib/pkgconfig/openblas.pc").read_text()
    assert 'Libs: -L"${libdir}" -llibopenblas' in shim
    assert destination.as_posix() in shim
    assert (destination / "licenses/OpenBLAS-LICENSE.txt").is_file()
    assert "review remains" in (destination / "licenses/REDISTRIBUTION-REVIEW.txt").read_text()
    if shutil.which("pkg-config"):
        env = dict(os.environ, PKG_CONFIG_PATH=str(destination / "lib/pkgconfig"))
        result = subprocess.check_output(["pkg-config", "--libs", "openblas"], text=True, env=env)
        assert shlex.split(result) == [f"-L{destination.as_posix()}/lib", "-llibopenblas"]


@pytest.mark.parametrize("old,new", [("USE_64BITINT= ", "USE_64BITINT=1 "),
    ("NO_LAPACK= ", "NO_LAPACK=1 "), ("DYNAMIC_ARCH=1", "DYNAMIC_ARCH="),
    ("version=0.3.34", "version=0.3.35")])
def test_metadata_guard_preserves_upstream_pc(tmp_path, old, new):
    archive = asset(tmp_path)
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive) as source:
        provider.safe_extract(source, destination)
    pc = destination / "lib/pkgconfig/openblas.pc"
    pc.write_text(pc.read_text().replace(old, new))
    before = pc.read_bytes()
    with pytest.raises(ValueError, match="metadata"):
        provider.validate_metadata(destination)
    assert pc.read_bytes() == before
    assert not (destination / "archive").exists()


def test_header_ilp64_and_missing_library_rejected(tmp_path):
    archive = asset(tmp_path)
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive) as source:
        provider.safe_extract(source, destination)
    header = destination / "include/openblas_config.h"
    header.write_text(header.read_text() + "#define OPENBLAS_USE64BITINT 1\n")
    with pytest.raises(ValueError, match="metadata"):
        provider.validate_metadata(destination)
    (destination / "lib/libopenblas.lib").rename(destination / "lib/retired.lib")
    with pytest.raises(ValueError, match="Missing OpenBLAS input"):
        provider.validate_metadata(destination)


def test_download_checksum_failure_preserves_zip_without_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(provider.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad download"))
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="SHA256"):
        provider.prepare(destination)
    assert (destination / "OpenBLAS-0.3.34-x64.zip").read_bytes() == b"bad download"
    assert not (destination / "lib").exists()


def test_dangling_destination_symlink_refused(tmp_path):
    destination = tmp_path / "existing-link"
    try:
        destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable")
    with pytest.raises(FileExistsError):
        provider.prepare(destination, tmp_path / "missing.zip")
    assert destination.is_symlink()
    assert not (tmp_path / "missing-target").exists()
