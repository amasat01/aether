//! A switch whose cases return values, rewritten into a chain of choices, gives the values of the
//! source it came from. The oracle is the unmodified source built as ordinary C++ (the normal
//! arm); the rewritten copy is built against plain-`bool` stand-ins of the recording calls, and
//! each selector is also raised through the driver. `CXX`, `RAPTOR_IR` and `LDLIBS` as in
//! `lines.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use raise::pipeline::run;
use raise::request::Request;

/// Helpers with every shape of the rewrite: braced and braceless cases, a fall-through group, a
/// `default`, and a switch without `default` whose following statements return.
const HELPERS: &str = "\
inline double withDefault(int k, double v) {
    switch (k) {
    case 0: { return v * 2.0; }
    case 1:
    case 2: return v + 1.0;
    case 3: {
        double w = v - 1.0;
        return w * w;
    }
    default: { return -v; }
    }
}

inline double withoutDefault(int k, double v) {
    switch (k) {
    case 0: return v;
    case 5: { return v * 10.0; }
    }
    return 7.0;
}
";

const STUB: &str = "\
#include <type_traits>
namespace aether::trace::detail {
template <class T, class E>
constexpr auto choose(bool c, T&& a, E&& b) -> std::common_type_t<decltype(a()), decltype(b())> {
    if (c) return a();
    return b();
}
}
";

const PROGRAM: &str = "\
#include <cstdio>
#include <initializer_list>
#include \"helpers.h\"
int main() {
    for (int k = -1; k <= 6; ++k)
        for (double v : {-2.5, 0.0, 1.5, 4.0})
            std::printf(\"%d %g %.17g %.17g\\n\", k, v, withDefault(k, v), withoutDefault(k, v));
}
";

fn env(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run these tests")))
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("inc")).unwrap();
    fs::canonicalize(dir).unwrap()
}

/// Builds `main.cpp` of `dir` with `extra` arguments and returns what it prints.
fn build_and_print(dir: &Path, extra: &[&str]) -> String {
    let binary = dir.join("program");
    let built = Command::new(env("CXX"))
        .args(["-std=c++20", "-Wall", "-Wextra"])
        .args(extra)
        .arg(dir.join("main.cpp"))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&binary).output().unwrap();
    assert!(ran.status.success());
    String::from_utf8(ran.stdout).unwrap()
}

#[test]
fn the_rewritten_switches_give_the_values_of_the_source_for_every_selector() {
    let original = scratch("switch_original");
    fs::write(original.join("helpers.h"), HELPERS).unwrap();
    fs::write(original.join("main.cpp"), PROGRAM).unwrap();
    let expected = build_and_print(&original, &[]);

    let rewritten = scratch("switch_rewritten");
    let analysis = reader::analyse(HELPERS);
    assert_eq!(analysis.rewritten_count("switch"), 2);
    assert_eq!(analysis.reports, []);
    fs::write(
        rewritten.join("helpers.h"),
        format!("{STUB}{}", analysis.output),
    )
    .unwrap();
    fs::write(rewritten.join("main.cpp"), PROGRAM).unwrap();
    assert_eq!(build_and_print(&rewritten, &[]), expected);
    assert_eq!(expected.lines().count(), 32);
}

/// A request over the scratch folder `dir`, raising `kernels`.
fn request(dir: &Path, kernels: &[&str]) -> Request {
    Request {
        entry: dir.join("entry.cpp"),
        kernels: kernels.iter().map(|k| (*k).to_string()).collect(),
        roots: vec![dir.to_path_buf()],
        includes: vec![dir.join("inc")],
        defines: vec![],
        cxx: env("CXX"),
        std: "c++20".into(),
        trace_include: Path::new(env!("CARGO_MANIFEST_DIR")).join("../include"),
        ir: env("RAPTOR_IR"),
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
fn raised_through_the_driver_each_selector_gives_the_ir_of_the_case_written_by_hand() {
    let dir = scratch("switch_raised");
    let kernels = "\
#pragma once
#include \"aether_probe/real.h\"
namespace m {
using aether::Mutable;
using aether::PerSample;
using aether::Real;
inline Real pick(int k, Real v) {
    switch (k) {
    case 0: { return v * 2.0; }
    case 1:
    case 2: return v + 1.0;
    default: { return -v; }
    }
}
inline void viaTwo(PerSample<Real> v, Mutable<Real> y) { y = pick(2, v); }
inline void byHandTwo(PerSample<Real> v, Mutable<Real> y) { y = v + 1.0; }
inline void viaZero(PerSample<Real> v, Mutable<Real> y) { y = pick(0, v); }
inline void byHandZero(PerSample<Real> v, Mutable<Real> y) { y = v * 2.0; }
inline void viaOther(PerSample<Real> v, Mutable<Real> y) { y = pick(9, v); }
inline void byHandOther(PerSample<Real> v, Mutable<Real> y) { y = -v; }
}
";
    fs::write(dir.join("inc/m.h"), kernels).unwrap();
    fs::write(dir.join("entry.cpp"), "#include <m.h>\n").unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    let names = [
        "m::viaTwo",
        "m::byHandTwo",
        "m::viaZero",
        "m::byHandZero",
        "m::viaOther",
        "m::byHandOther",
    ];
    let mut req = request(&dir, &names);
    req.keep = true;
    run(&req).unwrap();
    let shadow = fs::read_to_string(dir.join("out/shadow/0/m.h")).unwrap();
    assert!(
        shadow.contains("detail::choose(k == 0"),
        "the switch was not rewritten"
    );
    for (via, by_hand) in [
        ("viaTwo", "byHandTwo"),
        ("viaZero", "byHandZero"),
        ("viaOther", "byHandOther"),
    ] {
        let a = fs::read_to_string(dir.join(format!("out/m_{via}.ir"))).unwrap();
        let b = fs::read_to_string(dir.join(format!("out/m_{by_hand}.ir"))).unwrap();
        assert_eq!(a, b, "{via}");
    }
}
