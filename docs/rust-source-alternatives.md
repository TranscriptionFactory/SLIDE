# Manual Rust source builds

Use the [Pixi source-build recipe](../README.md#source-checkout-and-platform-coverage)
unless you need system packages or cluster modules. Run the commands below from
the repository root.

Source builds support stable Rust ≥1.88 (`rust-version`); release CI pins
Rust 1.99.1. You need C/C++ compilers, CMake, libclang, pkg-config, system
OpenBLAS/LAPACK with a Fortran runtime, Cairo, fontconfig and FreeType. HiGHS
builds C++ code and uses libclang to generate Rust bindings. Install rustup first,
then install the pinned toolchain with `rustup toolchain install 1.99.1 --profile minimal`.
On Debian/Ubuntu:

```bash
sudo apt-get install build-essential libopenblas-dev liblapack-dev gfortran cmake clang libclang-dev libcairo2-dev libfontconfig1-dev libfreetype6-dev pkg-config
cargo +1.99.1 install --locked --path crates/slide-cli
# Or use installed stable Rust >=1.88:
# cargo install --locked --path crates/slide-cli
# Or build without installing:
# cargo +1.99.1 build --release --locked          # target/release/slide
```

On Pitt CRC (RHEL 9), use the setup wrapper; no manual module loads are needed:

```bash
scripts/cargo-rust-blas +1.99.1 build --release --locked
```

The wrapper loads LLVM, locates libclang, selects system GCC/G++ and binutils
instead of Linuxbrew tools, and checks OpenBLAS and Cairo availability. It keeps
compiler temporary files in `.tmpfiles/` and retains the Rust toolchain from your
original PATH. On other clusters, load the required modules first and adapt
`scripts/with-rust-blas-env.sh` for local library paths. To install on CRC, use
`scripts/cargo-rust-blas +1.99.1 install --locked --path crates/slide-cli`.

The following macOS 15+ recipe is **untested on native macOS**. Install Xcode
Command Line Tools first; use native Homebrew for the selected architecture.

```bash
brew install openblas libomp cairo fontconfig freetype pkg-config cmake llvm
export MACOSX_DEPLOYMENT_TARGET=15.0
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig:$(brew --prefix cairo)/lib/pkgconfig"
cargo +1.99.1 build --release --locked
```

The Windows x86-64 MSVC recipe is **awaiting native CI**, not a verified install.
Use a Visual Studio 2022 x64 Developer PowerShell with C++ tools and Windows SDK,
Python 3.11, CMake, LLVM/libclang, rustup, and a bootstrapped vcpkg installation.
Set `VCPKG_ROOT` to that installation. Do not substitute vcpkg OpenBLAS.

```powershell
$ErrorActionPreference = 'Stop'
$prefix = Join-Path $PWD 'build/OpenBLAS-provider'
python scripts/prepare-windows-openblas.py "$prefix"
if ($LASTEXITCODE -ne 0) { throw 'OpenBLAS preparation failed' }
& "$env:VCPKG_ROOT/vcpkg.exe" install "cairo[fontconfig,freetype]:x64-windows" "pkgconf:x64-windows"
if ($LASTEXITCODE -ne 0) { throw 'vcpkg installation failed' }
$installed = "$env:VCPKG_ROOT/installed/x64-windows"
$env:VCPKGRS_DYNAMIC = '1'
$env:PKG_CONFIG = "$installed/tools/pkgconf/pkgconf.exe"
$env:PKG_CONFIG_PATH = "$prefix/lib/pkgconfig;$installed/lib/pkgconfig;$installed/share/pkgconfig"
$env:LIBCLANG_PATH = 'C:/Program Files/LLVM/bin'
$env:PATH = "$prefix/bin;$installed/bin;$env:PATH"
rustup toolchain install 1.99.1 --profile minimal
if ($LASTEXITCODE -ne 0) { throw 'Rust toolchain installation failed' }
cargo +1.99.1 build --release --locked --target x86_64-pc-windows-msvc
if ($LASTEXITCODE -ne 0) { throw 'Rust build failed' }
```

Windows desktop terminal behavior remains unverified. For Windows Python, use
the experimental Pixi source recipe in the README; no verified native installation is available.

The executable still needs shared libraries at runtime, including OpenBLAS,
Cairo and compiler runtimes. Keep the required libraries available on compute
nodes; copying the executable alone does not make a portable installation.
