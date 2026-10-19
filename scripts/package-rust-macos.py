#!/usr/bin/env python3
"""Bundle a native macOS executable, rewrite its Mach-O closure, then ad-hoc sign."""

import argparse
import logging
from pathlib import Path
import re
import runpy
import shutil
import subprocess


bundle_assets = runpy.run_path(str(Path(__file__).with_name("package-rust-linux.py")))["bundle_assets"]


def system_library(name):
    return name.startswith(("/usr/lib/", "/System/Library/"))


def load_commands(binary):
    output = subprocess.check_output(["otool", "-l", str(binary)], text=True)
    dependencies, rpaths, identity = [], [], None
    command = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("cmd "):
            command = line[4:]
        elif command == "LC_RPATH" and line.startswith("path "):
            rpaths.append(line[5:].rsplit(" (offset ", 1)[0])
        elif command in {"LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB", "LC_LOAD_UPWARD_DYLIB", "LC_ID_DYLIB"} and line.startswith("name "):
            name = line[5:].rsplit(" (offset ", 1)[0]
            if command == "LC_ID_DYLIB":
                identity = name
            else:
                dependencies.append(name)
    if not re.search(r"\bcmd LC_", output):
        raise RuntimeError(f"No Mach-O load commands for {binary}")
    return dependencies, rpaths, identity


def expand_path(name, loader, executable):
    return Path(name.replace("@loader_path", str(loader.parent)).replace("@executable_path", str(executable.parent)))


def resolve_dependency(name, loader, executable, rpaths):
    if system_library(name):
        return None
    if name.startswith("@rpath/"):
        candidates = [expand_path(base, loader, executable) / name[7:] for base in rpaths]
    else:
        candidates = [expand_path(name, loader, executable)]
    matches = {path.resolve() for path in candidates if path.is_absolute() and path.is_file()}
    if len(matches) != 1:
        raise RuntimeError(f"Unresolved or ambiguous Mach-O dependency {name} in {loader}: {sorted(map(str, matches))}")
    return matches.pop()


def closure(binary):
    found, names = {}, {}

    def visit(source, inherited=()):
        source = source.resolve()
        if source in found:
            return
        deps, rpaths, identity = load_commands(source)
        # Resolve each ancestor's loader token in that ancestor's context.
        search = [str(expand_path(base, source, binary)) for base in rpaths] + list(inherited)
        edges = []
        found[source] = (edges, identity)
        for name in deps:
            target = resolve_dependency(name, source, binary, search)
            if target is None:
                continue
            if target.name in names and names[target.name] != target:
                raise RuntimeError(f"Duplicate dylib name: {target.name}")
            names[target.name] = target
            edges.append((name, target))
            visit(target, search)
    visit(binary)
    return found


def package(binary, destination, font_dir, font_notice, notice_dirs):
    archive = Path(str(destination) + ".tar.gz")
    if archive.exists() or destination.exists():
        raise FileExistsError(archive if archive.exists() else destination)
    graph = closure(binary)
    (destination / "bin").mkdir(parents=True)
    (destination / "lib").mkdir()
    bundle_assets(destination, font_dir, font_notice)
    for index, directory in enumerate(notice_dirs):
        shutil.copytree(directory, destination / "licenses" / f"native-{index}")
    targets = {source: destination / ("bin/slide" if source == binary else f"lib/{source.name}") for source in graph}
    for source, target in targets.items():
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | 0o200)
    for source, (edges, identity) in graph.items():
        target = targets[source]
        for name, dependency in edges:
            relative = f"@loader_path/{'../lib/' if source == binary else ''}{dependency.name}"
            subprocess.run(["install_name_tool", "-change", name, relative, str(target)], check=True)
        if identity is not None:
            subprocess.run(["install_name_tool", "-id", f"@rpath/{target.name}", str(target)], check=True)
        _, rpaths, _ = load_commands(target)
        for rpath in rpaths:
            subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(target)], check=True)
    # Recheck the copied closure, never silently retaining Homebrew/@rpath imports.
    for source, (edges, _) in closure(destination / "bin/slide").items():
        if not source.is_relative_to(destination):
            raise RuntimeError(f"Unbundled Mach-O dependency: {source}")
    for target in targets.values():
        subprocess.run(["codesign", "--force", "--sign", "-", str(target)], check=True)
        subprocess.run(["codesign", "--verify", "--strict", str(target)], check=True)
    launcher = destination / "slide"
    launcher.write_text('#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                        'export FONTCONFIG_FILE="$root/fonts.conf"\nexec "$root/bin/slide" "$@"\n')
    launcher.chmod(0o755)
    (destination / "README.md").write_text(
        "# SLIDE macOS build candidate\n\nmacOS 15+; native build architecture only. Keep the entire directory together.\n"
        "Run `./slide --help`. Ad-hoc signed, not notarized. R-backed execution requires external R.\n"
        "Bundled notices are in `licenses/`; redistribution/legal review remains pending. Not a published release.\n")
    shutil.make_archive(str(destination), "gztar", destination.parent, destination.name)
    logging.info("Created %s", archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--font-notice", type=Path, required=True)
    parser.add_argument("--notice-dir", type=Path, action="append", required=True, help="Installed native-library license directory; repeat for the closure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    package(args.binary.resolve(), args.destination.resolve(), args.font_dir, args.font_notice, args.notice_dir)
