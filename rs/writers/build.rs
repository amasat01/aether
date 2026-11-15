//! Links the IR core's static library into this crate's test binaries when RAPTOR_IR_LIB names it (the path of libraptor_ir.a),
//! and the wing analyses' static library (libraptor_wings.a, the same folder) when it is there: the writers declare the
//! analyses (`rw_*`) and a test binary has no final binary around it to provide them.
fn main() {
    println!("cargo:rerun-if-env-changed=RAPTOR_IR_LIB");
    println!("cargo:rustc-check-cfg=cfg(raptor_ir_lib)");
    let Ok(p) = std::env::var("RAPTOR_IR_LIB") else {
        return;
    };
    let path = std::path::Path::new(&p);
    let (Some(dir), Some(stem)) = (path.parent(), path.file_stem().and_then(|s| s.to_str())) else {
        return;
    };
    println!("cargo:rustc-link-search=native={}", dir.display());
    // the wings come first: a GNU linker takes an archive's members for the names unresolved so far, and the wings need the core
    let wings = dir.join("libraptor_wings.a");
    println!("cargo:rerun-if-changed={}", wings.display());
    if wings.is_file() {
        println!("cargo:rustc-link-lib=static=raptor_wings");
    } else if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|o| o == "macos") {
        // without the wings library the analyses stay unresolved: the macOS linker binds them lazily (no test calls one)
        println!("cargo:rustc-link-arg-tests=-Wl,-undefined,dynamic_lookup");
    }
    println!(
        "cargo:rustc-link-lib=static={}",
        stem.trim_start_matches("lib")
    );
    println!("cargo:rustc-cfg=raptor_ir_lib");
}
