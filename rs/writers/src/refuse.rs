//! The refusals of the writers: every one is a reason with a kind, so the layer above can tell a
//! construct this build does not write from a request that is malformed from a fault of the
//! writers themselves. A construct that is not written is refused by name, never written
//! approximately.
use std::fmt;

/// What a refusal says about the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// A well-formed request for a construct this build does not write.
    NotWritten,
    /// A request that is malformed: a key, a member or a version this build does not know, a part
    /// that is missing, a value of the wrong shape.
    BadRequest,
    /// A fault in the writers or in the structure they were given: an expectation of the walk
    /// that did not hold.
    Fault,
}

/// A refusal: its kind and the reason that names what was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    /// What kind of refusal it is.
    pub kind: RefusalKind,
    /// The reason, in plain words.
    pub reason: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(&self.reason)
    }
}

/// The refusal of a construct this build does not write, named by `what`.
pub fn nib(what: impl fmt::Display) -> Refusal {
    Refusal {
        kind: RefusalKind::NotWritten,
        reason: format!("not in this build: {what}"),
    }
}

/// The refusal of a construct this build does not write, with the reason as given.
pub fn not_written(reason: impl fmt::Display) -> Refusal {
    Refusal {
        kind: RefusalKind::NotWritten,
        reason: reason.to_string(),
    }
}

/// The refusal of a request this build does not know, named by `what` (the reason reads like
/// `nib`'s: the request names something that is not in this build).
pub fn unknown(what: impl fmt::Display) -> Refusal {
    Refusal {
        kind: RefusalKind::BadRequest,
        reason: format!("not in this build: {what}"),
    }
}

/// The refusal of a malformed request, with the reason as given.
pub fn bad(reason: impl fmt::Display) -> Refusal {
    Refusal {
        kind: RefusalKind::BadRequest,
        reason: reason.to_string(),
    }
}

/// The refusal for a fault of the writers or of the structure they were given.
pub fn fault(reason: impl fmt::Display) -> Refusal {
    Refusal {
        kind: RefusalKind::Fault,
        reason: reason.to_string(),
    }
}
