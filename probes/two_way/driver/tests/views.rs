//! Kernels that take views: what the trace arm refuses, by name, and what the reader cannot read
//! as a role. `CXX`, `RAPTOR_IR` and `LDLIBS` as in `lines.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use raise::closure::Closure;
use raise::pipeline::{find_kernels, run};
use raise::request::Request;
use raise::RaiseError;

fn setting(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run these tests")))
}

/// The preamble of every kernel header and the request that raises `m::k` from it.
fn project(name: &str, kernel: &str) -> (Request, PathBuf) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("inc")).unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let dir = fs::canonicalize(dir).unwrap();
    let header = format!(
        "#pragma once\n#include <cstdint>\n#include \"aether/expr/Assign.h\"\n#include \"aether/expr/nodes/Constant.h\"\n#include \"aether/index/SampleIndex.h\"\n#include \"aether/view/View.h\"\n#include \"aether_probe/real.h\"\nnamespace m {{\nusing aether::dyn;\nusing aether::extents;\nusing aether::Real;\nusing aether::SampleIndex;\nusing aether::View;\n{kernel}}}\n"
    );
    let file = dir.join("inc/m.h");
    fs::write(&file, header).unwrap();
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
    (req, file)
}

fn refused(name: &str, kernel: &str) -> RaiseError {
    run(&project(name, kernel).0).unwrap_err()
}

#[test]
fn a_store_at_a_computed_index_inside_a_branch_is_refused_by_name() {
    let err = refused(
        "views_branch",
        "inline void k(const View<const std::int32_t, extents<dyn>> slot, const View<const double, extents<dyn>> v, View<double, extents<dyn>> out, SampleIndex i) {\n    if (v[i].eval<>() > 0.5) {\n        out(slot[i].eval<>()) = v[i].eval<>();\n    }\n}\n",
    );
    assert_eq!(err.step, "run");
    assert!(
        err.message
            .contains("a store at a computed index on plane out sits in a branch or a loop body"),
        "{err}"
    );
    assert!(err.message.contains("write it outside"), "{err}");
}

#[test]
fn writing_a_const_view_is_refused_by_name() {
    let err = refused(
        "views_const_write",
        "inline void k(const View<const double, extents<dyn>> v, const View<const double, extents<dyn>> out, SampleIndex i) {\n    out[i] = v[i].eval<>();\n}\n",
    );
    assert_eq!(err.step, "run");
    assert!(
        err.message
            .contains("plane out is read-only (a const view) and the kernel writes it"),
        "{err}"
    );
}

#[test]
fn a_plane_used_as_a_sample_element_and_as_a_table_is_refused_by_name() {
    let err = refused(
        "views_two_ways",
        "inline void k(const View<const std::int32_t, extents<dyn>> row, const View<const double, extents<dyn>> t, View<double, extents<dyn>> out, SampleIndex i) {\n    Real a = t[i].eval<>();\n    Real b = t(row[i].eval<>());\n    out[i] = a + b;\n}\n",
    );
    assert_eq!(err.step, "run");
    assert!(
        err.message.contains("plane t is used in more than one way"),
        "{err}"
    );
}

#[test]
fn updating_a_sample_element_before_it_was_assigned_is_refused_by_name() {
    let err = refused(
        "views_update_first",
        "inline void k(const View<const double, extents<dyn>> v, View<double, extents<dyn>> out, SampleIndex i) {\n    out[i] += v[i].eval<>();\n}\n",
    );
    assert_eq!(err.step, "run");
    assert!(
        err.message
            .contains("plane out is updated with += or -= before it was assigned"),
        "{err}"
    );
}

#[test]
fn a_project_alias_that_hides_a_view_is_refused_at_the_kernels_step() {
    let (req, file) = project(
        "views_alias",
        "using Vec3In = const View<const double, extents<3, dyn>>;\ninline void k(Vec3In v, View<double, extents<dyn>> out, SampleIndex i) {\n    out[i] = v[i].eval<0>();\n}\n",
    );
    let files = Closure {
        all: vec![file.clone()],
        project: vec![file.clone()],
    };
    let err = find_kernels(&req, &files).unwrap_err();
    assert_eq!(err.step, "kernels");
    for wanted in ["parameter v", "`Vec3In`", "spells no role"] {
        assert!(err.message.contains(wanted), "{wanted}: {err}");
    }
}
