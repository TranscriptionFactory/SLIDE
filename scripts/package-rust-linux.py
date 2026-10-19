#!/usr/bin/env python3
"""Bundle an Ubuntu-built slide executable and its non-glibc shared libraries."""

import argparse
import logging
from pathlib import Path
import re
import shutil
import subprocess


GLIBC_LIBRARIES = {
    "ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libdl.so.2",
    "libpthread.so.0", "librt.so.1", "libresolv.so.2", "libutil.so.1",
}


def dependencies(binary):
    # lddtree inspects ELF metadata rather than executing the input binary.
    output = subprocess.check_output(
        ["/usr/bin/python3", "/usr/bin/lddtree", "-l", str(binary)], text=True
    )
    paths = [Path(line) for line in output.splitlines()]
    if not paths:
        raise RuntimeError(f"No ELF dependency information for {binary}")
    for path in paths:
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"Unresolved ELF dependency: {path}")
    return paths


def check_glibc_floor(binary):
    output = subprocess.check_output(["readelf", "--version-info", str(binary)], text=True)
    versions = [tuple(map(int, match)) for match in re.findall(r"GLIBC_(\d+)\.(\d+)", output)]
    if versions and max(versions) > (2, 35):
        raise RuntimeError(f"{binary} requires glibc {'.'.join(map(str, max(versions)))}; candidate floor is 2.35")


def bundle_assets(destination, font_dir, font_notice):
    """Copy the same font/config assets on every platform."""
    notices = destination / "licenses"
    notices.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", notices / "SLIDE.txt")
    fonts = destination / "fonts"
    fonts.mkdir()
    for name in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"]:
        shutil.copy2(font_dir / name, fonts / name)
    shutil.copy2(font_notice, notices / "DejaVu.txt")
    (destination / "fonts.conf").write_text(
        '<?xml version="1.0"?>\n<fontconfig>\n'
        '  <dir prefix="relative">fonts</dir>\n'
        '  <cachedir prefix="xdg">fontconfig</cachedir>\n'
        '</fontconfig>\n'
    )


def package(binary, destination):
    archive = Path(str(destination) + ".tar.gz")
    if archive.exists() or destination.exists():
        raise FileExistsError(archive if archive.exists() else destination)
    paths = dependencies(binary)
    for path in paths:
        if path.name not in GLIBC_LIBRARIES:
            check_glibc_floor(path)
    destination.mkdir(parents=True, exist_ok=False)
    library_dir = destination / "lib"
    library_dir.mkdir()
    notices = destination / "licenses"
    notices.mkdir()
    (destination / "bin").mkdir()
    executable = destination / "bin" / "slide"
    shutil.copy2(binary, executable)
    for source in paths:
        if source.resolve() == binary.resolve() or source.name in GLIBC_LIBRARIES:
            continue
        target = library_dir / source.name
        if target.exists():
            raise RuntimeError(f"Duplicate library name: {source.name}")
        shutil.copy2(source, target)
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(target)], check=True)
        # Ubuntu's merged /usr can expose either spelling to dpkg-query.
        resolved = str(source.resolve())
        candidates = [resolved, resolved.removeprefix("/usr")]
        owner = None
        for candidate in candidates:
            result = subprocess.run(
                ["dpkg-query", "-S", candidate], capture_output=True, text=True
            )
            if result.returncode == 0:
                owner = result.stdout.split(": ", 1)[0].split(":", 1)[0]
                break
        if owner is None:
            raise RuntimeError(f"Cannot find package copyright for {source}")
        shutil.copy2(Path("/usr/share/doc") / owner / "copyright", notices / f"{owner}.txt")
    subprocess.run(["patchelf", "--set-rpath", "$ORIGIN/../lib", str(executable)], check=True)
    # Fail before archiving if any non-glibc dependency still resolves to the host.
    for path in dependencies(executable):
        if path.name not in GLIBC_LIBRARIES and not path.resolve().is_relative_to(destination):
            raise RuntimeError(f"Unbundled dependency: {path}")
    bundle_assets(destination, Path("/usr/share/fonts/truetype/dejavu"),
                  Path("/usr/share/doc/fonts-dejavu-core/copyright"))
    launcher = destination / "slide"
    launcher.write_text(
        '#!/bin/sh\nset -eu\n'
        'root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'export FONTCONFIG_FILE="$root/fonts.conf"\n'
        'exec "$root/bin/slide" "$@"\n'
    )
    launcher.chmod(0o755)
    (destination / "README.md").write_text(
        "# SLIDE Linux x86-64 build candidate\n\n"
        "CI target: Ubuntu 22.04 / Linux x86-64 (glibc >=2.35).\n"
        "Local builds inherit their build host runtime requirements.\n"
        "Keep the entire directory together; run `./slide --help`. No compiler is needed.\n"
        "R-backed execution still requires Rscript and R packages.\n"
        "Not for Alpine/musl, macOS, Windows, or ARM.\n\n"
        "Includes SLIDE and bundled system-library notices in `licenses/`.\n"
        "Redistribution review, including Rust dependency licenses, remains pending.\n"
        "This candidate is not a published release.\n"
    )
    shutil.make_archive(str(destination), "gztar", destination.parent, destination.name)
    logging.info("Created %s", archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path, help="New bundle directory (also creates .tar.gz)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    package(args.binary.resolve(), args.destination.resolve())
