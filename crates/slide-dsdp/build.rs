fn main() {
    let root = std::path::Path::new("../../src/slide_py/dsdp_solver/dsdp/C");
    let mut sources: Vec<_> = std::fs::read_dir(root.join("allc"))
        .expect("vendored DSDP sources")
        .map(|entry| entry.expect("DSDP source entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "c"))
        .collect();
    sources.sort();
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed=src/diagonal.c");
    cc::Build::new()
        .files(sources)
        .file("src/diagonal.c")
        .include(root.join("allinclude"))
        .warnings(false)
        .compile("slide_dsdp");
    println!("cargo:rustc-link-lib=openblas");
}
