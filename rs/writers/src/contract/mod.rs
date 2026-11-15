//! The contract a unit arrives with: the JSON reader and the call description.
pub mod json;
#[cfg(feature = "old-contract")]
pub mod old;
pub mod types;
pub mod wire;

pub use json::*;
