//! The steps of a raise, in order: closure, rewrite, parameters, generated main, syntax gate,
//! build and run. Each step is one function; each failure is one `RaiseError`.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use crate::closure::{closure, Closure};
use crate::mirror::Mirror;
use crate::request::Request;
use crate::{RaiseError, Res};

/// The trace-arm header the generated main calls, written next to the shadow tree.
const RAISE_HEADER: &str = include_str!("../include/aether_probe/raise.h");

/// What a green run did.
#[derive(Debug)]
pub struct Summary {
    /// Files in the closure.
    pub closure: usize,
    /// Project files in the closure.
    pub project: usize,
    /// Statements the reader left untouched.
    pub left: usize,
    /// Kernels raised.
    pub kernels: usize,
}

/// A statement the reader left as written, with the file it is in.
#[derive(Debug)]
pub struct Left {
    /// The author's file.
    pub file: PathBuf,
    /// What was left, where, in which function, and the way out.
    pub report: reader::Report,
}

/// A kernel with the parameter names read from its definition.
#[derive(Debug, PartialEq, Eq)]
pub struct Kernel {
    /// Qualified name.
    pub name: String,
    /// Name and role of each parameter, in order.
    pub params: Vec<(String, &'static str)>,
}

/// The first line of a compiler's diagnostics that names an error, or its first line.
#[must_use]
pub fn first_error(stderr: &str) -> String {
    stderr
        .lines()
        .find(|l| l.contains("error:"))
        .or_else(|| stderr.lines().find(|l| !l.trim().is_empty()))
        .unwrap_or("the compiler failed without a message")
        .trim()
        .to_string()
}

/// Runs the whole pipeline.
///
/// # Errors
/// The first step that fails.
pub fn run(req: &Request) -> Res<Summary> {
    let shadow = req.out.join("shadow");
    let mirror = Mirror::new(&req.includes, &req.roots);
    let mut flags = base_flags(req);
    flags.extend(req.includes.iter().map(|d| format!("-I{}", d.display())));
    flags.extend(arm_includes(req));
    let files = closure(req, &flags)?;
    write_depfile(req, &files)?;
    println!(
        "closure: {} files, {} project files",
        files.all.len(),
        files.project.len()
    );
    let left = rewrite_project(req, &files, &mirror, &shadow)?;
    check_quotes(req, &files, &mirror)?;
    let kernels = find_kernels(req, &files)?;
    let main = generate(req, &mirror, &shadow, &kernels)?;
    let mut build_flags = base_flags(req);
    build_flags.extend(mirror.include_flags(&shadow));
    build_flags.push(format!("-I{}", req.out.join("include").display()));
    build_flags.extend(arm_includes(req));
    syntax_gate(req, &build_flags, &main, &left)?;
    build_and_run(req, &build_flags, &main)?;
    for k in &kernels {
        for ext in ["ir", "planes"] {
            let file = req.out.join(format!("{}.{ext}", k.name.replace("::", "_")));
            if !file.exists() {
                return Err(RaiseError::new(
                    "run",
                    format!(
                        "{} was not written; the kernel run ended early",
                        file.display()
                    ),
                ));
            }
        }
    }
    if !req.keep {
        cleanup(req);
    }
    Ok(Summary {
        closure: files.all.len(),
        project: files.project.len(),
        left: left.len(),
        kernels: kernels.len(),
    })
}

/// Standard, trace arm and definitions: the same for every compile of the driver.
fn base_flags(req: &Request) -> Vec<String> {
    let mut flags = vec![format!("-std={}", req.std), "-DAETHER_TRACE".to_string()];
    flags.extend(req.defines.iter().map(|d| format!("-D{d}")));
    flags
}

/// The trace arm's headers and the IR library's header.
fn arm_includes(req: &Request) -> Vec<String> {
    vec![
        format!("-I{}", req.trace_include.display()),
        format!("-I{}", req.ir.join("rs/ir/include").display()),
    ]
}

/// A name as a make dependency file spells it: a space is `\\ ` and a dollar `$$`.
#[must_use]
pub fn depfile_name(path: &Path) -> String {
    path.display()
        .to_string()
        .replace(' ', "\\ ")
        .replace('$', "$$")
}

/// The text of a make-style dependency file: `target: file file ...`.
#[must_use]
pub fn depfile_text(target: &Path, files: &[PathBuf]) -> String {
    let names: Vec<String> = files.iter().map(|f| depfile_name(f)).collect();
    format!("{}: {}\n", depfile_name(target), names.join(" "))
}

/// Writes the dependency file when the request asks for one, so that a build system reruns the
/// raise when any project file of the closure changes.
fn write_depfile(req: &Request, files: &Closure) -> Res<()> {
    let Some(depfile) = &req.depfile else {
        return Ok(());
    };
    let first = req
        .kernels
        .first()
        .map_or("raise", |k| k.as_str())
        .replace("::", "_");
    let target = req.out.join(format!("{first}.ir"));
    write_file(depfile, &depfile_text(&target, &files.project), "closure")
}

/// Step 2: writes every project file of the closure, rewritten, into the shadow tree, and the
/// statements the reader left into `left.txt`. Returns how many were left.
///
/// # Errors
/// A file that cannot be read or written.
pub fn rewrite_project(
    req: &Request,
    files: &Closure,
    mirror: &Mirror,
    shadow: &Path,
) -> Res<Vec<Left>> {
    // A shadow tree of an earlier run could hold a different file under the same index.
    let _ = fs::remove_dir_all(shadow);
    let mut left = Vec::new();
    for file in &files.project {
        let text = fs::read_to_string(file).map_err(|e| {
            RaiseError::new(
                "rewrite",
                format!("{}: {e}; the file must be UTF-8 text", file.display()),
            )
        })?;
        let analysis = reader::analyse(&text);
        // The directive makes the compiler name the author's file in every diagnostic.
        let body = format!(
            "#line 1 \"{}\"\n{}",
            cpp_escape(&file.display().to_string()),
            analysis.output
        );
        let placements = mirror.placements(file);
        if placements.is_empty() {
            return Err(RaiseError::new(
                "rewrite",
                format!("{} is under no root or include directory", file.display()),
            ));
        }
        for placement in &placements {
            let target = Mirror::shadow_path(shadow, placement);
            write_file(&target, &body, "rewrite")?;
        }
        left.extend(analysis.reports.into_iter().map(|report| Left {
            file: file.clone(),
            report,
        }));
    }
    let lines: Vec<String> = left.iter().map(describe).collect();
    let listing = lines
        .iter()
        .fold(String::new(), |all, line| all + line + "\n");
    write_file(&req.out.join("left.txt"), &listing, "rewrite")?;
    Ok(left)
}

/// One line about a left statement: where, what, in which function, and the way out.
fn describe(left: &Left) -> String {
    let r = &left.report;
    let place = format!("{}:{}", left.file.display(), r.line);
    let function = if r.function.is_empty() {
        String::new()
    } else {
        format!(" in {}", r.function)
    };
    format!("{place}: {}{function}: {}", r.kind, r.way_out())
}

/// The file and line of a compiler diagnostic `path:line:column: error: ...`.
fn error_location(line: &str) -> Option<(&str, usize)> {
    let head = &line[..line.find(": error:")?];
    let mut parts = head.rsplitn(3, ':');
    let _column = parts.next()?;
    let number = parts.next()?.parse().ok()?;
    Some((parts.next()?, number))
}

fn write_file(path: &Path, text: &str, step: &'static str) -> Res<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)
            .map_err(|e| RaiseError::new(step, format!("{}: {e}", dir.display())))?;
    }
    fs::write(path, text).map_err(|e| RaiseError::new(step, format!("{}: {e}", path.display())))
}

fn cpp_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The quoted includes of a source: `#include "name"`.
#[must_use]
pub fn quoted_includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let rest = line
                .trim_start()
                .strip_prefix('#')?
                .trim_start()
                .strip_prefix("include")?;
            let rest = rest.trim_start().strip_prefix('"')?;
            Some(rest[..rest.find('"')?].to_string())
        })
        .collect()
}

/// `a/./b/../c` as `a/c`, without touching the file system.
#[must_use]
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// A quoted include is looked up first beside the including file. In the shadow tree that works
/// only for files mirrored at the same relative place; any other target must be found through an
/// include directory. Anything else would silently read an unrewritten file, so it stops here.
///
/// # Errors
/// A quoted include that the shadow tree would not resolve to the same file.
pub fn check_quotes(req: &Request, files: &Closure, mirror: &Mirror) -> Res<()> {
    for file in &files.project {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        for quoted in quoted_includes(&text) {
            let Some(dir) = file.parent() else { continue };
            let Ok(target) = fs::canonicalize(dir.join(&quoted)) else {
                continue;
            };
            if !files.all.contains(&target) {
                continue; // a branch the build does not take
            }
            let by_include = req
                .includes
                .iter()
                .any(|x| fs::canonicalize(x.join(&quoted)).is_ok_and(|p| p == target));
            if by_include {
                continue;
            }
            let there = mirror.placements(&target);
            let consistent = files.project.contains(&target)
                && mirror.placements(file).iter().all(|(i, rel)| {
                    let wanted = normalize(&rel.parent().unwrap_or(Path::new("")).join(&quoted));
                    there.iter().any(|(j, r)| j == i && *r == wanted)
                });
            if !consistent {
                return Err(RaiseError::new(
                    "mirror",
                    format!(
                        "{}: #include \"{quoted}\" reaches {} beside the including file, where the shadow tree has no matching copy; add its folder as -I (or put both files under one -I directory)",
                        file.display(),
                        target.display()
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Step 3: the parameter names of each requested kernel, from its one definition in the closure.
///
/// # Errors
/// A kernel found no time, or more than once, or with an unnamed parameter.
pub fn find_kernels(req: &Request, files: &Closure) -> Res<Vec<Kernel>> {
    let mut found: Vec<(PathBuf, reader::Signature)> = Vec::new();
    for file in &files.project {
        let text = fs::read_to_string(file).unwrap_or_default();
        for sig in reader::signatures(&text) {
            if req.kernels.contains(&sig.name) {
                found.push((file.clone(), sig));
            }
        }
    }
    let mut kernels = Vec::new();
    for name in &req.kernels {
        let hits: Vec<_> = found.iter().filter(|(_, s)| &s.name == name).collect();
        match hits.as_slice() {
            [] => {
                return Err(RaiseError::new(
                    "kernels",
                    format!("kernel {name} is not defined in a project file of the closure; check the qualified name, --root and --entry"),
                ))
            }
            [(file, sig)] => {
                let params = kernel_params(file, name, sig)?;
                kernels.push(Kernel { name: name.clone(), params });
            }
            many => {
                let places: Vec<String> =
                    many.iter().map(|(f, s)| format!("{}:{}", f.display(), s.line)).collect();
                return Err(RaiseError::new(
                    "kernels",
                    format!("kernel {name} is defined {} times ({}); raise one definition", many.len(), places.join(", ")),
                ));
            }
        }
    }
    Ok(kernels)
}

/// The name and role of each parameter of a kernel definition.
fn kernel_params(
    file: &Path,
    kernel: &str,
    sig: &reader::Signature,
) -> Res<Vec<(String, &'static str)>> {
    let mut params = Vec::new();
    for p in &sig.params {
        let place = format!("{}:{}", file.display(), p.line);
        if p.name.is_empty() {
            return Err(RaiseError::new(
                "kernels",
                format!("{place}: kernel {kernel} has an unnamed parameter of type {}; name every parameter", p.type_text),
            ));
        }
        let Some(role) = p.role else {
            return Err(RaiseError::new(
                "kernels",
                format!("{place}: kernel {kernel} parameter {} has type `{}`, which spells no role; spell the role type in the declaration: aether::PerSample<T>, aether::Uniform<T> or aether::Mutable<T>", p.name, p.type_text),
            ));
        };
        params.push((p.name.clone(), role));
    }
    Ok(params)
}

/// Step 4: writes the trace header and `raise_main.cpp`, which includes the entry (its rewritten
/// copy when it is a project file) and raises each kernel. Returns the main's path.
///
/// # Errors
/// A file that cannot be written.
pub fn generate(req: &Request, mirror: &Mirror, shadow: &Path, kernels: &[Kernel]) -> Res<PathBuf> {
    write_file(
        &req.out.join("include/aether_probe/raise.h"),
        RAISE_HEADER,
        "generate",
    )?;
    let entry = mirror
        .placements(&req.entry)
        .first()
        .map_or_else(|| req.entry.clone(), |p| Mirror::shadow_path(shadow, p));
    let mut text = format!(
        "// Generated by the raise driver.\n#include \"{}\"\n#include \"aether_probe/raise.h\"\n\nint main() {{\n",
        cpp_escape(&entry.display().to_string())
    );
    for k in kernels {
        let names: Vec<String> = k
            .params
            .iter()
            .map(|(name, role)| format!("{{\"{name}\", \"{role}\"}}"))
            .collect();
        let call = format!(
            "    aether::trace::raise(\"{0}\", &{0}, {{{{{1}}}}}, \"{2}\");\n",
            k.name,
            names.join(", "),
            cpp_escape(&req.out.display().to_string())
        );
        text.push_str(&call);
    }
    text.push_str("    return 0;\n}\n");
    let main = req.out.join("raise_main.cpp");
    write_file(&main, &text, "generate")?;
    Ok(main)
}

fn compiler(req: &Request, flags: &[String], more: &[&str]) -> Res<std::process::Output> {
    Command::new(&req.cxx)
        .args(flags)
        .args(more)
        .output()
        .map_err(|e| {
            RaiseError::new(
                "build",
                format!(
                    "cannot run {}: {e}; pass --cxx a compiler",
                    req.cxx.display()
                ),
            )
        })
}

/// Step 5: the generated main must compile against the shadow tree.
///
/// # Errors
/// The compiler's first error line. When it lies in code the reader left as written, the error
/// names that construct, its function and the way out; otherwise it points at `left.txt`.
pub fn syntax_gate(req: &Request, flags: &[String], main: &Path, left: &[Left]) -> Res<()> {
    let output = compiler(req, flags, &["-fsyntax-only", &main.display().to_string()])?;
    if output.status.success() {
        return Ok(());
    }
    let first = first_error(&String::from_utf8_lossy(&output.stderr));
    let culprit = error_location(&first).and_then(|(file, line)| {
        left.iter().find(|l| {
            l.file.display().to_string() == file
                && (l.report.span.0..=l.report.span.1).contains(&line)
        })
    });
    let advice = match culprit {
        Some(l) => format!(
            "while raising {}, the reader left a construct as written: {}",
            req.kernels.join(", "),
            describe(l)
        ),
        None => format!(
            "the file named is the author's source as rewritten (statements the reader left are in {}/left.txt); rewrite that construct in a form the reader knows",
            req.out.display()
        ),
    };
    Err(RaiseError::new("syntax gate", format!("{first}; {advice}")))
}

/// Step 6: builds the main against the IR library and runs it.
///
/// # Errors
/// The compiler's or the program's failure.
pub fn build_and_run(req: &Request, flags: &[String], main: &Path) -> Res<()> {
    let binary = req.out.join("raise_bin");
    let lib = req.ir.join("target/release/libraptor_ir.a");
    let (main, lib, binary_text) = (
        main.display().to_string(),
        lib.display().to_string(),
        binary.display().to_string(),
    );
    let mut arguments = vec!["-O1", main.as_str(), lib.as_str()];
    arguments.extend(req.link.iter().map(String::as_str));
    arguments.extend(["-o", binary_text.as_str()]);
    let output = compiler(req, flags, &arguments)?;
    if !output.status.success() {
        let first = first_error(&String::from_utf8_lossy(&output.stderr));
        return Err(RaiseError::new(
            "build",
            format!("{first}; the IR library at {lib} must be built"),
        ));
    }
    let run = Command::new(&binary)
        .output()
        .map_err(|e| RaiseError::new("run", format!("{}: {e}", binary.display())))?;
    print!("{}", String::from_utf8_lossy(&run.stdout));
    if !run.status.success() {
        let first = first_error(&String::from_utf8_lossy(&run.stderr));
        return Err(RaiseError::new(
            "run",
            format!("{first}; a kernel was refused by the trace arm"),
        ));
    }
    Ok(())
}

/// Removes the shadow tree and the generated build products after a green run.
fn cleanup(req: &Request) {
    for dir in ["shadow", "include"] {
        let _ = fs::remove_dir_all(req.out.join(dir));
    }
    for file in ["raise_main.cpp", "raise_bin"] {
        let _ = fs::remove_file(req.out.join(file));
    }
}
