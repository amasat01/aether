//! A compiler diagnostic anywhere in a shadow file names the author's line, even after a rewrite
//! that changed the number of lines; and an overloaded kernel name stops at the `kernels` step.
//! These tests run the real compiler: `CXX` names it, `RAPTOR_IR` names the raptor checkout with
//! the built IR library, `LDLIBS` holds extra link arguments; the trace headers are the folder
//! `include` beside this crate.

use std::fs;
use std::path::{Path, PathBuf};

use raise::closure::Closure;
use raise::pipeline::{find_kernels, run};
use raise::request::Request;

fn setting(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run these tests")))
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("inc")).unwrap();
    fs::canonicalize(dir).unwrap()
}

fn request(dir: &Path, kernels: &[&str]) -> Request {
    Request {
        entry: dir.join("entry.cpp"),
        kernels: kernels.iter().map(|k| (*k).to_string()).collect(),
        roots: vec![dir.to_path_buf()],
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
    }
}

#[test]
fn an_error_after_a_multi_line_rewrite_is_reported_at_the_authors_line() {
    let dir = scratch("author_line");
    let kernel = "\
#pragma once
#include \"aether_probe/real.h\"
namespace m {
inline void k(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> y) {
    aether::Real r = v;
    if (r > 1.0)
    {
        r = r * 2.0;
    }
    else
    {
        r = r * 3.0;
    }
    r = undeclared_name;
    y = r;
}
}
";
    fs::write(dir.join("inc/m.h"), kernel).unwrap();
    fs::write(dir.join("entry.cpp"), "#include <m.h>\n").unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let err = run(&request(&dir, &["m::k"])).unwrap_err();
    assert_eq!(err.step, "syntax gate");
    let wanted = format!("{}:14:", dir.join("inc/m.h").display());
    assert!(err.message.contains(&wanted), "{err}");
}

#[test]
fn a_kernel_name_that_matches_two_declarations_names_each_of_them() {
    let dir = scratch("overloaded");
    let header = "namespace n {\ninline void k(int a) { }\ninline void k(double b) { }\n}\n";
    let file = dir.join("inc/o.h");
    fs::write(&file, header).unwrap();
    let files = Closure {
        all: vec![file.clone()],
        project: vec![file.clone()],
    };
    let err = find_kernels(&request(&dir, &["n::k"]), &files).unwrap_err();
    assert_eq!(err.step, "kernels");
    for line in [2, 3] {
        let place = format!("{}:{line}", file.display());
        assert!(err.message.contains(&place), "{err}");
    }
    assert!(!err.message.contains('\n'));
}

#[test]
fn a_project_alias_that_hides_the_role_is_refused_by_name() {
    let dir = scratch("alias");
    let header = "\
#pragma once
#include \"aether_probe/real.h\"
namespace m {
using Speed = aether::PerSample<aether::Real>;
inline void k(Speed v, aether::Mutable<aether::Real> y) { y = v; }
}
";
    let file = dir.join("inc/m.h");
    fs::write(&file, header).unwrap();
    let files = Closure {
        all: vec![file.clone()],
        project: vec![file.clone()],
    };
    let err = find_kernels(&request(&dir, &["m::k"]), &files).unwrap_err();
    assert_eq!(err.step, "kernels");
    let place = format!("{}:5:", file.display());
    for wanted in [
        place.as_str(),
        "kernel m::k",
        "parameter v",
        "`Speed`",
        "aether::PerSample<T>",
    ] {
        assert!(err.message.contains(wanted), "{wanted}: {err}");
    }
    assert!(!err.message.contains('\n'));
}

#[test]
fn a_kernel_that_depends_on_a_global_counter_is_refused_after_the_second_recording() {
    let dir = scratch("counter");
    let kernel = "\
#pragma once
#include \"aether_probe/real.h\"
namespace m {
inline int gCalls = 0;
inline void k(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> x) {
    x = v * aether::Real(static_cast<double>(++gCalls));
}
}
";
    fs::write(dir.join("inc/m.h"), kernel).unwrap();
    fs::write(dir.join("entry.cpp"), "#include <m.h>\n").unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let err = run(&request(&dir, &["m::k"])).unwrap_err();
    assert_eq!(err.step, "run");
    assert!(
        err.message
            .contains("kernel m::k records differently when recorded again"),
        "{err}"
    );
}

#[test]
fn a_role_that_does_not_match_how_the_parameter_is_passed_stops_the_run() {
    // A by-value parameter spelled `Mutable` cannot come from an alias-form header: here the
    // by-reference alias is used for a read-only parameter instead.
    let dir = scratch("mismatch");
    let kernel = "\
#pragma once
#include \"aether_probe/real.h\"
namespace m {
inline void k(aether::Uniform<aether::Real>& v, aether::Mutable<aether::Real> x) { x = v; }
}
";
    fs::write(dir.join("inc/m.h"), kernel).unwrap();
    fs::write(dir.join("entry.cpp"), "#include <m.h>\n").unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let err = run(&request(&dir, &["m::k"])).unwrap_err();
    assert_eq!(err.step, "run");
    assert!(
        err.message
            .contains("parameter v is taken by reference and has the role uniform"),
        "{err}"
    );
}
