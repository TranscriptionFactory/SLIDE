"""Dependency parsers and closure guards must fail closed before native packaging."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), Path(__file__).parents[2] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


macos = load("package-rust-macos")
windows = load("package-rust-windows")
smoke = load("slide_artifact_smoke")


def test_macho_load_commands_preserve_spaces(monkeypatch):
    output = """Mach header
Load command 1
          cmd LC_LOAD_DYLIB
      cmdsize 64
         name @rpath/path with spaces.dylib (offset 24)
Load command 2
          cmd LC_RPATH
         path @loader_path/../lib (offset 12)
Load command 3
          cmd LC_ID_DYLIB
         name /brew/lib/libfoo.dylib (offset 24)
"""
    monkeypatch.setattr(macos.subprocess, "check_output", lambda *a, **k: output)
    assert macos.load_commands(Path("unused")) == (
        ["@rpath/path with spaces.dylib"], ["@loader_path/../lib"], "/brew/lib/libfoo.dylib")


def test_macho_rpath_resolution_and_missing(tmp_path):
    library = tmp_path / "spaced lib" / "libfoo.dylib"
    library.parent.mkdir()
    library.touch()
    binary = tmp_path / "slide"
    assert macos.resolve_dependency("@rpath/libfoo.dylib", binary, binary, [str(library.parent)]) == library
    assert macos.resolve_dependency("/usr/lib/libSystem.B.dylib", binary, binary, []) is None
    with pytest.raises(RuntimeError, match="Unresolved"):
        macos.resolve_dependency("@rpath/missing.dylib", binary, binary, [str(library.parent)])


def test_macho_recursive_closure_and_duplicates(tmp_path, monkeypatch):
    binary, a, b = [tmp_path / name for name in ["slide", "liba.dylib", "libb.dylib"]]
    for path in (binary, a, b):
        path.touch()
    metadata = {binary: ([str(a)], [], None), a: (["@loader_path/libb.dylib"], [], "a"), b: ([str(a)], [], "b")}
    monkeypatch.setattr(macos, "load_commands", lambda path: metadata[path])
    assert set(macos.closure(binary)) == {binary, a, b}
    metadata[b] = (["@rpath/missing.dylib"], [], "b")
    with pytest.raises(RuntimeError, match="Unresolved"):
        macos.closure(binary)


def test_pe_parser_includes_delayed_imports():
    assert windows.parse_imports("""File Type: EXECUTABLE IMAGE
  Image has the following dependencies:
    KERNEL32.dll
    cairo-2.dll
  Image has the following delay load dependencies:
    VCRUNTIME140.dll
  Summary
""") == ["KERNEL32.dll", "cairo-2.dll", "VCRUNTIME140.dll"]
    with pytest.raises(RuntimeError):
        windows.parse_imports("garbage")
    assert windows.system_library("API-MS-WIN-CORE-file-l1-1-0.dll")
    assert not windows.system_library("VCRUNTIME140.dll")
    assert not windows.system_library("libopenblas.dll")


def test_pe_recursive_case_insensitive_closure(tmp_path, monkeypatch):
    binary, a, b = [tmp_path / name for name in ["slide.exe", "CAIRO.dll", "VCRUNTIME140.dll"]]
    for path in (binary, a, b):
        path.touch()
    graph = {binary: ["cairo.DLL", "KERNEL32.dll"], a: ["vcruntime140.dll"], b: []}
    monkeypatch.setattr(windows, "dependencies", lambda path: graph[path])
    assert windows.closure(binary, [tmp_path]) == {binary, a, b}
    graph[b] = ["missing.dll"]
    with pytest.raises(RuntimeError, match="Unresolved"):
        windows.closure(binary, [tmp_path])
    graph[b] = []
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "cairo.dll").touch()
    with pytest.raises(RuntimeError, match="ambiguous"):
        windows.closure(binary, [tmp_path, duplicate])


@pytest.mark.parametrize("module,suffix", [(macos, ".tar.gz"), (windows, ".zip")])
def test_existing_archives_are_preserved(tmp_path, module, suffix):
    destination = tmp_path / "candidate"
    archive = Path(str(destination) + suffix)
    archive.write_bytes(b"existing")
    args = ([], tmp_path, tmp_path, []) if module is windows else (tmp_path, tmp_path, [])
    with pytest.raises(FileExistsError):
        module.package(tmp_path / "missing", destination, *args)
    assert archive.read_bytes() == b"existing"
    assert not destination.exists()


def test_shared_smoke_rejects_incomplete_checkpoint_and_unembedded_pdf(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"final_cv": {"folds": [1]}, "evaluations": [{"love": {"k": 2}}]}))
    pdf = tmp_path / "ControlPerformancePlot.pdf"
    pdf.write_bytes(b"%PDF-1.5 /FontFile2")
    smoke.validate(checkpoint, tmp_path)
    pdf.write_bytes(b"%PDF-1.5")
    with pytest.raises(RuntimeError, match="embedded font"):
        smoke.validate(checkpoint, tmp_path)
    checkpoint.write_text("{}")
    with pytest.raises(RuntimeError, match="final_cv"):
        smoke.validate(checkpoint)


def test_macho_inherited_rpath_uses_ancestor_loader(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "slide"
    a = tmp_path / "lib" / "nested" / "liba.dylib"
    b = tmp_path / "lib" / "libb.dylib"
    for path in (binary, a, b):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    metadata = {
        binary: (["@rpath/nested/liba.dylib"], ["@loader_path/../lib"], None),
        a: (["@rpath/libb.dylib"], [], "a"),
        b: ([], [], "b"),
    }
    monkeypatch.setattr(macos, "load_commands", lambda path: metadata[path])
    assert set(macos.closure(binary)) == {binary, a, b}
    duplicate = tmp_path / "other" / "libb.dylib"
    duplicate.parent.mkdir()
    duplicate.touch()
    metadata[b] = ([str(duplicate)], [], "b")
    with pytest.raises(RuntimeError, match="Duplicate dylib"):
        macos.closure(binary)
