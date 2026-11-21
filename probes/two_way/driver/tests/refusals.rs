//! The shapes the reader leaves by name stop the raise at the syntax gate, and the error names
//! the function, the author's file and line and the way out. The compiler is the one that
//! refuses (a left statement with a traced condition does not compile); the driver connects its
//! first error to the construct that was left. `CXX`, `RAPTOR_IR` and `LDLIBS` as in `lines.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use raise::pipeline::run;
use raise::request::Request;
use raise::RaiseError;

fn setting(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run these tests")))
}

/// Raises `m::k` of a header `m.h` holding `kernel` and returns the error it stops with.
fn refused(name: &str, kernel: &str) -> (RaiseError, PathBuf) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("inc")).unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let dir = fs::canonicalize(dir).unwrap();
    let header = format!("#pragma once\n#include \"aether_probe/real.h\"\nnamespace m {{\nusing aether::Mutable;\nusing aether::PerSample;\nusing aether::Real;\n{kernel}}}\n");
    fs::write(dir.join("inc/m.h"), header).unwrap();
    fs::write(dir.join("entry.cpp"), "#include <m.h>\n").unwrap();
    let req = Request {
        entry: dir.join("entry.cpp"),
        kernels: vec!["m::k".to_string()],
        roots: vec![dir.clone()],
        includes: vec![dir.join("inc")],
        defines: vec![],
        cxx: setting("CXX"),
        std: "c++20".into(),
        trace_include: Path::new(env!("CARGO_MANIFEST_DIR")).join("../include"),
        ir: setting("RAPTOR_IR"),
        link: std::env::var("LDLIBS")
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect(),
        depfile: None,
        out: dir.join("out"),
        keep: false,
    };
    (run(&req).unwrap_err(), dir.join("inc/m.h"))
}

/// The error is a refusal at the syntax gate that names the kernel, the function, the author's
/// file and line, the kind and the way out, on one line.
fn assert_names(err: &RaiseError, file: &Path, line: usize, function: &str, kind: &str, way: &str) {
    assert_eq!(err.step, "syntax gate");
    let place = format!("{}:{line}: {kind} in {function}", file.display());
    for wanted in ["while raising m::k", place.as_str(), way] {
        assert!(err.message.contains(wanted), "{wanted}\n{err}");
    }
    assert!(!err.message.contains('\n'));
}

#[test]
fn a_ternary_used_as_an_assignable_value_stops_the_raise_by_name() {
    let kernel = "\
inline void k(PerSample<Real> v, Mutable<Real> y) {
    Real a = v;
    Real b = v * 2.0;
    (v > 1.0 ? a : b) = 3.0;
    y = a + b;
}
";
    let (err, file) = refused("refused_lvalue", kernel);
    assert_names(
        &err,
        &file,
        10,
        "m::k",
        "ternary-as-lvalue",
        "assigned to, addressed or bound to a reference",
    );
}

#[test]
fn an_early_return_in_a_function_returning_a_reference_stops_the_raise_by_name() {
    let kernel = "\
inline Real& pick(Real& a, Real& b, Real c) {
    if (c > 1.0) {
        return a;
    }
    return b;
}
inline void k(PerSample<Real> v, Mutable<Real> y) {
    Real a = v;
    Real b = v * 2.0;
    y = pick(a, b, v);
}
";
    let (err, file) = refused("refused_reference", kernel);
    assert_names(
        &err,
        &file,
        9,
        "m::pick",
        "return-in-branch",
        "or a function that returns a reference",
    );
}

#[test]
fn a_statement_cut_by_a_preprocessor_conditional_stops_the_raise_by_name() {
    let kernel = "\
inline void k(PerSample<Real> v, Mutable<Real> y) {
    Real r = v;
    if (r > 1.0) {
        r = r * 2.0;
    }
#if defined(MORE)
    else {
        r = r * 3.0;
    }
#endif
    y = r;
}
";
    let (err, file) = refused("refused_cut", kernel);
    assert_names(
        &err,
        &file,
        9,
        "m::k",
        "cut-by-preprocessor",
        "keep each `#if` branch around whole statements",
    );
}
