#!/usr/bin/env python3
"""Prepare only the pinned LP64 Windows x64 OpenBLAS DLL and MSVC import library."""

import argparse
import hashlib
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import urllib.request
import zipfile


VERSION = "0.3.34"
URL = f"https://github.com/OpenMathLib/OpenBLAS/releases/download/v{VERSION}/OpenBLAS-{VERSION}-x64.zip"
SHA256 = "e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b"
REQUIRED = ("bin/libopenblas.dll", "lib/libopenblas.lib", "include/openblas_config.h",
            "include/cblas.h", "include/lapack.h", "lib/pkgconfig/openblas.pc")


def verify_checksum(archive):
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"OpenBLAS SHA256 mismatch: {archive}")


def safe_extract(archive, destination):
    for entry in archive.infolist():
        name = entry.filename.replace("\\", "/")
        path = PurePosixPath(name)
        mode = entry.external_attr >> 16
        if (path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts
                or ":" in name or any(part.endswith((" ", ".")) for part in path.parts)
                or any(PureWindowsPath(part).is_reserved() for part in path.parts) or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
            raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
    archive.extractall(destination)


def validate_metadata(prefix):
    for name in REQUIRED:
        path = prefix / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing OpenBLAS input: {path}")
    pc = (prefix / "lib/pkgconfig/openblas.pc").read_text(encoding="utf-8")
    fields = dict(line.split("=", 1) for line in pc.splitlines() if "=" in line and not line.startswith(("Libs", "Cflags")))
    config = fields.get("openblas_config", "").split()
    required = {"USE_64BITINT=", "NO_LAPACK=", "NO_LAPACKE=", "DYNAMIC_ARCH=1", "DYNAMIC_OLDER=1", "GENERIC"}
    header = (prefix / "include/openblas_config.h").read_text(encoding="utf-8")
    if (fields.get("version") != VERSION or not required.issubset(config)
            or re.search(r"^\s*#\s*define\s+OPENBLAS_USE64BITINT\b", header, re.MULTILINE)
            or not re.search(r"^#define OPENBLAS_ARCH_X86_64 1$", header, re.MULTILINE)
            or f"OpenBLAS {VERSION}" not in header):
        raise ValueError("OpenBLAS metadata differs from pinned LP64/LAPACK/dynamic-arch configuration")


def write_shim(prefix):
    # Quotes belong in the flags, not the variable values, for paths with spaces.
    value = prefix.as_posix()
    if any(char in value for char in '\n\r\x00$"'):
        raise ValueError("OpenBLAS prefix must not contain newlines, dollar signs, or quotes")
    pc = prefix / "lib/pkgconfig/openblas.pc"
    archived = prefix / "archive"
    archived.mkdir()
    shutil.move(str(pc), archived / "openblas.pc")
    pc.write_text(
        f"prefix={value}\nlibdir=${{prefix}}/lib\nincludedir=${{prefix}}/include\n\n"
        f"Name: openblas\nDescription: OpenBLAS LP64 Windows MSVC import library\nVersion: {VERSION}\n"
        'Libs: -L"${libdir}" -llibopenblas\nCflags: -I"${includedir}"\n', encoding="utf-8")


def prepare(destination, archive=None):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    if archive is not None:
        verify_checksum(archive)
    destination.mkdir(parents=True)
    saved = destination / f"OpenBLAS-{VERSION}-x64.zip"
    if archive is None:
        with urllib.request.urlopen(URL, timeout=120) as response, saved.open("xb") as output:
            shutil.copyfileobj(response, output)
        verify_checksum(saved)
    else:
        shutil.copy2(archive, saved)
    with zipfile.ZipFile(saved) as source:
        safe_extract(source, destination)
    validate_metadata(destination)
    write_shim(destination)
    notices = destination / "licenses"
    notices.mkdir()
    for name, content in NOTICES.items():
        (notices / name).write_text(content, encoding="utf-8")
    warning = (
        "OpenBLAS and Reference-LAPACK notices below come from OpenBLAS v0.3.34.\n"
        "The DLL asset contains no notices. Embedded GCC/libgfortran/libquadmath and\n"
        "MinGW runtime notices are not complete here. Redistribution review remains\n"
        "pending. These files do not establish legal clearance.\n"
        "Source: https://github.com/OpenMathLib/OpenBLAS/tree/v0.3.34\n")
    (notices / "REDISTRIBUTION-REVIEW.txt").write_text(warning, encoding="utf-8")
    logging.warning("Embedded compiler/runtime notices incomplete; redistribution review pending")
    for name, value in {
        "BLAS_LIB_DIR": destination / "lib", "LAPACK_LIB_DIR": destination / "lib",
        "BLAS_LIB": "libopenblas", "LAPACK_LIB": "libopenblas",
        "PKG_CONFIG_PATH": destination / "lib/pkgconfig",
    }.items():
        logging.info("%s=%s%s", name, value, " (prepend)" if name == "PKG_CONFIG_PATH" else "")


# Verbatim notices from the pinned OpenBLAS v0.3.34 source archive.
NOTICES = {
    'OpenBLAS-LICENSE.txt': """Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of\x20
      its contributors may be used to endorse or promote products\x20
      derived from this software without specific prior written\x20
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
""",
    'Reference-LAPACK-LICENSE.txt': """Copyright (c) 1992-2023 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
Copyright (c) 2000-2023 The University of California Berkeley. All
                        rights reserved.
Copyright (c) 2006-2023 The University of Colorado Denver.  All rights
                        reserved.

$COPYRIGHT$

Additional copyrights may follow

$HEADER$

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

The copyright holders provide no reassurances that the source code
provided does not infringe any patent, copyright, or any other
intellectual property rights of third parties.  The copyright holders
disclaim any liability to any recipient for claims brought against
recipient by any third party for infringement of that parties
intellectual property rights.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
""",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New output prefix; use a dated directory")
    parser.add_argument("--archive", type=Path, help="Use this pinned ZIP without network access")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare(args.destination, args.archive.resolve() if args.archive else None)
