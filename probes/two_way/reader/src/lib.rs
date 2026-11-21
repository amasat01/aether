//! A type-free reader and rewriter for the control statements of C++ kernel sources.
//!
//! The lexer keeps every byte of a file in a token span; the reader finds function bodies and
//! reads their control statements as a tree whose conditions and bodies are text; the rewriter
//! turns the statements it knows exactly into recording calls and leaves every other byte alone.

pub mod lex;
pub mod read;
mod rewrite;

pub use rewrite::{analyse, Analysis, Report};
