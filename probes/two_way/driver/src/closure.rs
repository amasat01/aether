//! Step 1: the closure, the files the translation unit includes, asked from the compiler itself.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::request::Request;
use crate::{RaiseError, Res};

/// The files a translation unit includes.
#[derive(Debug)]
pub struct Closure {
    /// Every file, entry first.
    pub all: Vec<PathBuf>,
    /// The files under a root: the ones that are rewritten.
    pub project: Vec<PathBuf>,
}

/// The prerequisites of a make rule as the compiler's dependency listing writes it: lines joined
/// at a trailing backslash, the target before the first unescaped `:` dropped, `\ ` meaning a
/// space inside a name and `$$` a dollar.
#[must_use]
pub fn parse_make_rule(text: &str) -> Vec<String> {
    let joined = text.replace("\\\r\n", " ").replace("\\\n", " ");
    let mut names = Vec::new();
    let mut current = String::new();
    let mut in_target = true;
    let mut chars = joined.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if matches!(chars.peek(), Some(' ' | '#' | '\\')) => {
                current.push(chars.next().unwrap_or(' '));
            }
            '$' if chars.peek() == Some(&'$') => {
                chars.next();
                current.push('$');
            }
            ':' if in_target && matches!(chars.peek(), Some(' ' | '\n') | None) => {
                in_target = false;
                current.clear();
            }
            c if c.is_whitespace() => {
                if !current.is_empty() && !in_target {
                    names.push(std::mem::take(&mut current));
                }
                current.clear();
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() && !in_target {
        names.push(current);
    }
    names
}

/// Asks the compiler for the files the entry includes, under the flags of the request and the
/// trace arm (so that includes reached only there are in the closure), and keeps those under a
/// root as project files.
///
/// # Errors
/// The compiler cannot be run or fails; its first error line is given.
pub fn closure(req: &Request, flags: &[String]) -> Res<Closure> {
    let output = Command::new(&req.cxx)
        .args(flags)
        .arg("-MM")
        .arg(&req.entry)
        .output()
        .map_err(|e| {
            RaiseError::new(
                "closure",
                format!(
                    "cannot run {}: {e}; pass --cxx a compiler",
                    req.cxx.display()
                ),
            )
        })?;
    if !output.status.success() {
        let first = crate::pipeline::first_error(&String::from_utf8_lossy(&output.stderr));
        return Err(RaiseError::new(
            "closure",
            format!("{first}; fix the include path or definitions"),
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut all: Vec<PathBuf> = Vec::new();
    for name in parse_make_rule(&text) {
        let path = fs::canonicalize(&name).unwrap_or_else(|_| PathBuf::from(&name));
        if !all.contains(&path) {
            all.push(path);
        }
    }
    let project = all
        .iter()
        .filter(|f| req.roots.iter().any(|r| f.starts_with(r)))
        .cloned()
        .collect();
    Ok(Closure { all, project })
}
