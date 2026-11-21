//! The request: the command line, in which `@FILE` stands for the arguments of a response file
//! (one per line), exactly as compilers read them, so that `CMake` can write the request.

use std::fs;
use std::path::{Path, PathBuf};

use crate::{RaiseError, Res};

/// Everything the driver is asked for. Paths are absolute and exist (except `out`, which is made).
#[derive(Debug)]
pub struct Request {
    /// The translation unit that includes the kernels.
    pub entry: PathBuf,
    /// Qualified names of the kernels to raise.
    pub kernels: Vec<String>,
    /// Only files under a root are project files and are rewritten.
    pub roots: Vec<PathBuf>,
    /// Include directories, in the order the author's build gives them.
    pub includes: Vec<PathBuf>,
    /// Definitions, `NAME` or `NAME=VALUE`.
    pub defines: Vec<String>,
    /// The C++ compiler.
    pub cxx: PathBuf,
    /// The language standard, such as `c++20`.
    pub std: String,
    /// The folder that holds `aether_probe/`.
    pub trace_include: PathBuf,
    /// A raptor checkout with `rs/ir/include` and `target/release/libraptor_ir.a`.
    pub ir: PathBuf,
    /// Extra arguments of the link of the raise program, such as `-lpthread`.
    pub link: Vec<String>,
    /// Where to write the make-style dependency file: the first output, then every project file
    /// of the closure.
    pub depfile: Option<PathBuf>,
    /// Where the IR, the plane lists and the left-over list are written.
    pub out: PathBuf,
    /// Keep the shadow tree and the generated main after a green run.
    pub keep: bool,
}

/// Replaces each `@FILE` by the lines of that file, in place; blank lines and lines that start
/// with `#` are dropped.
///
/// # Errors
/// A response file that cannot be read.
pub fn expand_response(args: &[String]) -> Res<Vec<String>> {
    let mut out = Vec::new();
    for arg in args {
        let Some(file) = arg.strip_prefix('@') else {
            out.push(arg.clone());
            continue;
        };
        let text = fs::read_to_string(file).map_err(|e| {
            RaiseError::new(
                "request",
                format!("response file {file}: {e}; check the path after `@`"),
            )
        })?;
        let lines: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect();
        out.extend(expand_response(&lines)?);
    }
    Ok(out)
}

/// The flags that take one or more values.
const MULTI: [&str; 3] = ["--kernel", "--root", "--entry"];

/// Reads a request from expanded arguments.
///
/// # Errors
/// An unknown flag, a missing value or required flag, or a path that does not exist.
pub fn parse(args: &[String]) -> Res<Request> {
    let mut values: Vec<(String, String)> = Vec::new();
    let mut keep = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        i += 1;
        if arg == "--keep" {
            keep = true;
        } else if let Some(value) = attached(arg) {
            values.push((arg[..2].to_string(), value.to_string()));
        } else if matches!(arg, "-I" | "-D") || arg.starts_with("--") {
            let known = [
                "-I",
                "-D",
                "--entry",
                "--kernel",
                "--root",
                "--cxx",
                "--std",
                "--trace-include",
                "--ir",
                "--link",
                "--depfile",
                "--out",
            ];
            if !known.contains(&arg) {
                return Err(RaiseError::new(
                    "request",
                    format!("unknown argument {arg}; see the interface"),
                ));
            }
            let take = if MULTI.contains(&arg) { usize::MAX } else { 1 };
            let mut taken = 0;
            while taken < take && i < args.len() && !is_flag(&args[i]) {
                values.push((arg.to_string(), args[i].clone()));
                i += 1;
                taken += 1;
            }
            if taken == 0 {
                return Err(RaiseError::new("request", format!("{arg} needs a value")));
            }
        } else {
            return Err(RaiseError::new(
                "request",
                format!("unexpected argument {arg}"),
            ));
        }
    }
    build(&values, keep)
}

/// `-IDIR` and `-DNAME` carry their value in the same argument.
fn attached(arg: &str) -> Option<&str> {
    let rest = arg.strip_prefix("-I").or_else(|| arg.strip_prefix("-D"))?;
    (!rest.is_empty()).then_some(rest)
}

fn is_flag(arg: &str) -> bool {
    arg.starts_with("--") || matches!(arg, "-I" | "-D") || attached(arg).is_some()
}

fn build(values: &[(String, String)], keep: bool) -> Res<Request> {
    let all = |flag: &str| -> Vec<&str> {
        values
            .iter()
            .filter(|(f, _)| f == flag)
            .map(|(_, v)| v.as_str())
            .collect()
    };
    let one = |flag: &str| -> Res<&str> {
        all(flag).last().copied().ok_or_else(|| {
            RaiseError::new(
                "request",
                format!("{flag} is required; add it to the request"),
            )
        })
    };
    let kernels: Vec<String> = all("--kernel").into_iter().map(str::to_string).collect();
    if kernels.is_empty() {
        return Err(RaiseError::new(
            "request",
            "--kernel is required; name at least one kernel",
        ));
    }
    let exists = |flag: &str, path: &str| -> Res<PathBuf> {
        fs::canonicalize(path)
            .map_err(|e| RaiseError::new("request", format!("{flag} {path}: {e}; check the path")))
    };
    let out = Path::new(one("--out")?);
    fs::create_dir_all(out)
        .map_err(|e| RaiseError::new("request", format!("--out {}: {e}", out.display())))?;
    Ok(Request {
        entry: exists("--entry", one("--entry")?)?,
        kernels,
        roots: all("--root")
            .iter()
            .map(|r| exists("--root", r))
            .collect::<Res<_>>()?,
        includes: all("-I")
            .iter()
            .map(|d| exists("-I", d))
            .collect::<Res<_>>()?,
        defines: all("-D").into_iter().map(str::to_string).collect(),
        cxx: PathBuf::from(one("--cxx")?),
        std: all("--std").last().map_or("c++20", |s| *s).to_string(),
        trace_include: exists("--trace-include", one("--trace-include")?)?,
        ir: exists("--ir", one("--ir")?)?,
        link: all("--link").into_iter().map(str::to_string).collect(),
        depfile: all("--depfile")
            .last()
            .map(|d| std::path::absolute(d).unwrap_or_else(|_| PathBuf::from(d))),
        out: exists("--out", one("--out")?)?,
        keep,
    })
}
