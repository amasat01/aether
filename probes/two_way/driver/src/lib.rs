//! The raise driver: reads one request, rewrites the project's kernel sources into a shadow tree,
//! builds the kernels against the trace arm and writes their IR. Every step is one function and
//! every failure is one line naming the step, what failed and the way out.

pub mod closure;
pub mod mirror;
pub mod pipeline;
pub mod request;

use std::fmt;

/// A failed step: which step, and one line saying what failed and what to do about it.
#[derive(Debug, PartialEq, Eq)]
pub struct RaiseError {
    /// The step that failed, such as `closure` or `syntax gate`.
    pub step: &'static str,
    /// What failed and the way out.
    pub message: String,
}

impl RaiseError {
    /// An error of `step`.
    pub fn new(step: &'static str, message: impl Into<String>) -> Self {
        RaiseError {
            step,
            message: message.into(),
        }
    }
}

impl fmt::Display for RaiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raise: {}: {}", self.step, self.message)
    }
}

impl std::error::Error for RaiseError {}

/// The result of a step.
pub type Res<T> = Result<T, RaiseError>;
