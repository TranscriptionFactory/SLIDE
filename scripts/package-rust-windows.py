#!/usr/bin/env python3
"""Bundle an MSVC-built executable and its recursive non-system PE DLL closure."""

import argparse
import logging
from pathlib import Path
import re
import runpy
import shutil
import subprocess


bundle_assets = runpy.run_path(str(Path(__file__).with_name("package-rust-linux.py")))["bundle_assets"]
# OS DLLs only: MSVC redistributables must be supplied explicitly and bundled.
SYSTEM_DLLS = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll", "gdi32.dll",
    "advapi32.dll", "ole32.dll", "oleaut32.dll", "shell32.dll", "shlwapi.dll",
    "ws2_32.dll", "bcrypt.dll", "crypt32.dll", "secur32.dll", "rpcrt4.dll",
    "comdlg32.dll", "comctl32.dll", "version.dll", "winmm.dll", "imm32.dll",
    "setupapi.dll", "cfgmgr32.dll", "dwrite.dll", "d2d1.dll", "dwmapi.dll",
    "usp10.dll", "windowscodecs.dll", "normaliz.dll", "ucrtbase.dll", "msvcrt.dll",
    "iphlpapi.dll", "dnsapi.dll", "userenv.dll", "psapi.dll", "dbghelp.dll",
    "winspool.drv", "winhttp.dll", "wininet.dll", "wtsapi32.dll", "synchronization.dll",
}


def system_library(name):
    return name.lower() in SYSTEM_DLLS or name.lower().startswith(("api-ms-win-", "ext-ms-win-"))


def parse_imports(output):
    # dumpbin /DEPENDENTS includes normal and delay-load dependency sections.
    if "dependencies" not in output.lower():
        raise RuntimeError("No PE dependency information; run English-language dumpbin /DEPENDENTS")
    return list(dict.fromkeys(line.strip() for line in output.splitlines()
                             if re.fullmatch(r"\s*[^\\/:\s]+\.(?:dll|drv)\s*", line, re.IGNORECASE)))


def dependencies(binary):
    output = subprocess.check_output(["dumpbin", "/NOLOGO", "/DEPENDENTS", str(binary)], text=True)
    return parse_imports(output)


def closure(binary, library_dirs):
    index = {}
    for directory in library_dirs:
        if not directory.is_dir():
            raise RuntimeError(f"Missing DLL directory: {directory}")
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".dll", ".drv"}:
                index.setdefault(path.name.lower(), set()).add(path.resolve())
    found = set()

    def visit(source):
        source = source.resolve()
        if source in found:
            return
        found.add(source)
        for name in dependencies(source):
            if system_library(name):
                continue
            matches = index.get(name.lower(), set())
            if len(matches) != 1:
                raise RuntimeError(f"Unresolved or ambiguous PE dependency {name} in {source}: {sorted(map(str, matches))}")
            visit(next(iter(matches)))
    visit(binary)
    return found


def package(binary, destination, library_dirs, font_dir, font_notice, notice_dirs):
    archive = Path(str(destination) + ".zip")
    if archive.exists() or destination.exists():
        raise FileExistsError(archive if archive.exists() else destination)
    sources = closure(binary, [binary.parent] + library_dirs)
    executable_dir = destination / "bin"
    executable_dir.mkdir(parents=True)
    bundle_assets(destination, font_dir, font_notice)
    for index, directory in enumerate(notice_dirs):
        shutil.copytree(directory, destination / "licenses" / f"native-{index}")
    for source in sources:
        shutil.copy2(source, executable_dir / ("slide.exe" if source == binary else source.name))
    for path in closure(executable_dir / "slide.exe", [executable_dir]):
        if not path.is_relative_to(destination):
            raise RuntimeError(f"Unbundled PE dependency: {path}")
    (destination / "slide.cmd").write_text(
        '@echo off\nsetlocal\nset "FONTCONFIG_FILE=%~dp0fonts.conf"\n'
        '"%~dp0bin\\slide.exe" %*\nexit /b %ERRORLEVEL%\n')
    (destination / "README.md").write_text(
        "# SLIDE Windows x86-64 MSVC build candidate\n\nKeep the entire directory together; run `slide.cmd --help`.\n"
        "Native Windows Server 2022 CI target; desktop minimum and terminal startup require testing.\n"
        "The submit command requires Unix/SLURM. R-backed execution requires external R.\n"
        "Bundled notices are in `licenses/`; redistribution/legal review remains pending. Not a published release.\n")
    shutil.make_archive(str(destination), "zip", destination.parent, destination.name)
    logging.info("Created %s", archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dll-dir", type=Path, action="append", required=True, help="Search only these explicit DLL directories (repeat, including MSVC CRT redist)")
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--font-notice", type=Path, required=True)
    parser.add_argument("--notice-dir", type=Path, action="append", required=True, help="Native dependency notices; repeat for vcpkg share and CRT notices")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    package(args.binary.resolve(), args.destination.resolve(), args.dll_dir, args.font_dir, args.font_notice, args.notice_dir)
