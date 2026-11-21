//! The pieces of the driver that need no compiler: the request, the make-rule parser, the mirror
//! mapping and the helpers of the include check.

use std::fs;
use std::path::{Path, PathBuf};

use raise::closure::parse_make_rule;
use raise::mirror::Mirror;
use raise::pipeline::{first_error, normalize, quoted_includes};
use raise::request::{expand_response, parse};

/// A fresh scratch folder for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn a_response_file_is_expanded_in_place_without_blank_or_comment_lines() {
    let dir = scratch("response_in_place");
    let file = dir.join("a.rsp");
    fs::write(&file, "# a comment\n--kernel\n\n  k1  \n#another\nk2\n").unwrap();
    let args = strings(&[
        "--entry",
        "e.cpp",
        &format!("@{}", file.display()),
        "--keep",
    ]);
    assert_eq!(
        expand_response(&args).unwrap(),
        strings(&["--entry", "e.cpp", "--kernel", "k1", "k2", "--keep"])
    );
}

#[test]
fn a_response_file_may_name_another() {
    let dir = scratch("response_nested");
    fs::write(dir.join("inner.rsp"), "-DX=1\n").unwrap();
    fs::write(
        dir.join("outer.rsp"),
        format!("--keep\n@{}\n", dir.join("inner.rsp").display()),
    )
    .unwrap();
    let got = expand_response(&[format!("@{}", dir.join("outer.rsp").display())]).unwrap();
    assert_eq!(got, strings(&["--keep", "-DX=1"]));
}

#[test]
fn a_missing_response_file_is_one_line_naming_the_file() {
    let err = expand_response(&strings(&["@/no/such/file.rsp"])).unwrap_err();
    assert_eq!(err.step, "request");
    assert!(err.to_string().contains("/no/such/file.rsp"));
    assert!(!err.to_string().contains('\n'));
}

/// The arguments of a complete request over folders made in `dir`.
fn complete(dir: &Path) -> Vec<String> {
    for sub in ["root", "inc", "trace", "ir"] {
        fs::create_dir_all(dir.join(sub)).unwrap();
    }
    fs::write(dir.join("entry.cpp"), "").unwrap();
    let p = |s: &str| dir.join(s).display().to_string();
    strings(&[
        "--entry",
        &p("entry.cpp"),
        "--kernel",
        "a::k1",
        "b::k2",
        "--root",
        &p("root"),
        "-I",
        &p("inc"),
        "-DM",
        "-D",
        "N=2",
        "--cxx",
        "clang++",
        "--trace-include",
        &p("trace"),
        "--ir",
        &p("ir"),
        "--out",
        &p("out"),
        "--keep",
    ])
}

#[test]
fn a_request_reads_repeated_and_attached_values() {
    let dir = scratch("request_ok");
    let req = parse(&complete(&dir)).unwrap();
    assert_eq!(req.kernels, ["a::k1", "b::k2"]);
    assert_eq!(req.defines, ["M", "N=2"]);
    assert_eq!(req.includes.len(), 1);
    assert_eq!(req.std, "c++20");
    assert!(req.keep);
    assert!(req.out.is_dir());
}

#[test]
fn a_request_without_a_kernel_or_with_an_unknown_flag_is_refused_by_name() {
    let dir = scratch("request_bad");
    let mut args = complete(&dir);
    args.push("--frobnicate".into());
    assert!(parse(&args)
        .unwrap_err()
        .to_string()
        .contains("--frobnicate"));
    let without: Vec<String> = complete(&dir)
        .into_iter()
        .filter(|a| a != "a::k1" && a != "b::k2" && a != "--kernel")
        .collect();
    assert!(parse(&without)
        .unwrap_err()
        .to_string()
        .contains("--kernel"));
}

#[test]
fn a_path_that_does_not_exist_names_its_flag() {
    let dir = scratch("request_path");
    let mut args = complete(&dir);
    let at = args.iter().position(|a| a == "--ir").unwrap();
    args[at + 1] = "/no/such/ir".into();
    let msg = parse(&args).unwrap_err().to_string();
    assert!(msg.contains("--ir") && msg.contains("/no/such/ir"), "{msg}");
}

#[test]
fn the_make_rule_joins_continuation_lines_and_drops_the_target() {
    let rule = "main.o: src/a.cpp \\\n  inc/b.h \\\n  inc/c.h\n";
    assert_eq!(parse_make_rule(rule), ["src/a.cpp", "inc/b.h", "inc/c.h"]);
}

#[test]
fn the_make_rule_keeps_escaped_spaces_and_dollars_inside_a_name() {
    let rule = "t.o: my\\ dir/a.h cost$$.h plain.h\n";
    assert_eq!(parse_make_rule(rule), ["my dir/a.h", "cost$.h", "plain.h"]);
}

#[test]
fn a_target_with_an_escaped_space_does_not_leak_into_the_prerequisites() {
    assert_eq!(parse_make_rule("a\\ b.o: x.h\n"), ["x.h"]);
}

#[test]
fn an_empty_rule_has_no_prerequisites() {
    assert_eq!(parse_make_rule("t.o:\n"), Vec::<String>::new());
}

fn mirror() -> Mirror {
    let includes = [
        PathBuf::from("/p/inc"),
        PathBuf::from("/p/inc/sub"),
        PathBuf::from("/q/inc"),
    ];
    Mirror::new(&includes, &[PathBuf::from("/p"), PathBuf::from("/q")])
}

#[test]
fn a_file_is_placed_under_the_include_directory_that_holds_it() {
    let got = mirror().placements(Path::new("/q/inc/x/y.h"));
    assert_eq!(got, [(2, PathBuf::from("x/y.h"))]);
}

#[test]
fn a_file_under_nested_include_directories_is_placed_under_each_spelling() {
    let got = mirror().placements(Path::new("/p/inc/sub/z.h"));
    assert_eq!(
        got,
        [(0, PathBuf::from("sub/z.h")), (1, PathBuf::from("z.h"))]
    );
}

#[test]
fn a_file_in_no_include_directory_goes_under_its_root() {
    let got = mirror().placements(Path::new("/p/kernels.cpp"));
    assert_eq!(got, [(3, PathBuf::from("kernels.cpp"))]);
}

#[test]
fn a_file_outside_every_base_has_no_placement() {
    assert_eq!(mirror().placements(Path::new("/usr/include/vector")), []);
}

#[test]
fn the_shadow_path_is_the_index_folder_then_the_relative_path() {
    let path = Mirror::shadow_path(Path::new("/out/shadow"), &(1, PathBuf::from("a/b.h")));
    assert_eq!(path, Path::new("/out/shadow/1/a/b.h"));
}

#[test]
fn each_shadow_directory_comes_before_the_directory_it_mirrors() {
    let flags = mirror().include_flags(Path::new("/out/shadow"));
    assert_eq!(
        flags,
        [
            "-I/out/shadow/0",
            "-I/p/inc",
            "-I/out/shadow/1",
            "-I/p/inc/sub",
            "-I/out/shadow/2",
            "-I/q/inc"
        ]
    );
}

#[test]
fn quoted_includes_are_found_and_angle_includes_are_not() {
    let text = "#include <a.h>\n  #  include \"b/c.h\"\n#include \"d.h\" // x\nint x; // #include \"e.h\"\n";
    assert_eq!(quoted_includes(text), ["b/c.h", "d.h"]);
}

#[test]
fn a_relative_path_is_normalised_without_the_file_system() {
    assert_eq!(normalize(Path::new("a/./b/../c/d.h")), Path::new("a/c/d.h"));
}

#[test]
fn the_first_compiler_error_line_is_chosen_from_the_diagnostics() {
    let stderr =
        "In file included from x.cpp:1:\nbad.h:9:12: error: oops\nbad.h:10:1: error: more\n";
    assert_eq!(first_error(stderr), "bad.h:9:12: error: oops");
    assert_eq!(first_error(""), "the compiler failed without a message");
}

#[test]
fn a_dependency_file_names_the_target_then_every_file_with_spaces_and_dollars_escaped() {
    let text = raise::pipeline::depfile_text(
        Path::new("/out/k.ir"),
        &[PathBuf::from("/p/a.h"), PathBuf::from("/my dir/b$.h")],
    );
    assert_eq!(text, "/out/k.ir: /p/a.h /my\\ dir/b$$.h\n");
}
