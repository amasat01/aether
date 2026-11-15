//! The reader of an operation node's kind: the `kind` text of the node, read into `OpKind`. A text
//! that names no kind is refused as a fault, naming the text.
use crate::ir::door;
use crate::kind::OpKind;
use crate::refuse::{fault, Refusal};

impl OpKind {
    /// The kind of an operation node, read from its `kind` text. A node whose text names no kind
    /// is a fault naming the text.
    pub fn of(ctx: impl Into<door::CtxPtr>, node: u32) -> Result<OpKind, Refusal> {
        let text = door::text_field(ctx, node, "kind")?;
        OpKind::parse(&text)
            .ok_or_else(|| fault(format!("node {node} has the unknown op kind {text}")))
    }

    /// The kind of a node that may be of another class: none for a node that is not an operation.
    pub fn of_op(ctx: impl Into<door::CtxPtr>, node: u32) -> Result<Option<OpKind>, Refusal> {
        let ctx = ctx.into();
        door::class_is(ctx, node, "Op")?
            .then(|| OpKind::of(ctx, node))
            .transpose()
    }
}
