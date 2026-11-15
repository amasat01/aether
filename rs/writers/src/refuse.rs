//! The one refusal constructor: a construct the writers do not write is refused by name, never
//! written approximately.

/// The refusal of a construct this build does not write, named by `what`.
pub fn nib(what: impl std::fmt::Display) -> String {
    format!("not in this build: {what}")
}
