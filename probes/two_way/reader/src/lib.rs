//! A type-free reader and rewriter for the control statements of C++ kernel sources.
//!
//! The lexer keeps every byte of a file in a token span; the reader finds function bodies and
//! reads their control statements as a tree whose conditions and bodies are text; the rewriter
//! turns the statements it knows exactly into recording calls and leaves every other byte alone.

pub mod lex;
pub mod read;
mod rewrite;

pub use rewrite::{analyse, Analysis, Report};

/// A function definition found in a file: what the driver needs to raise it by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// Qualified name, such as `physics::drag`.
    pub name: String,
    /// Parameter names in order; "" for an unnamed one.
    pub params: Vec<String>,
    /// 1-based line of the opening brace of the body.
    pub line: usize,
}

/// The named function definitions of `text`, in source order.
#[must_use]
pub fn signatures(text: &str) -> Vec<Signature> {
    let src = read::Source::new(text);
    let (functions, _) = read::read_functions(&src);
    functions
        .into_iter()
        .filter(|f| !f.name.is_empty())
        .map(|f| Signature {
            name: f.name,
            params: f.params,
            line: f.body.line,
        })
        .collect()
}
