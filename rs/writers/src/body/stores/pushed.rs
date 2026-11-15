//! The read-pushing rewrite of the walk: a runtime-index read of an element-wise vector value is
//! moved to the operands, `component_at(op(a, b), k)` becomes `op(component_at(a, k), b')` (a
//! rank-0 operand stays), so one element is computed and the vector is never built. The new nodes
//! are made through the door and the unit renders over the new canonical walk.
use crate::body::values::literal_of;
use crate::body::Class;
use crate::ir::door::Scalar;
use crate::ir::door::{self, rx_make_op_like, rx_rebuild, rx_substitute, RxCtx};
use crate::kind::OpKind;
use crate::refuse::{fault, Refusal};
use std::collections::HashMap;
use std::ffi::c_int;

/// The sink classes whose value operand the rewrite may replace.
const VALUE_SINKS: [&str; 4] = ["Finish", "SampleSum", "AccumWrite", "Assign"];

/// Whether an operation acts on each element alone: the binary and unary math and the ternary
/// forms.
fn is_elementwise(kind: OpKind) -> bool {
    use OpKind::*;
    matches!(
        kind,
        Add | Sub
            | Mul
            | Div
            | Pow
            | Min
            | Max
            | Atan2
            | Hypot
            | Copysign
            | Fmod
            | Remainder
            | Fdim
            | Neg
            | Abs
            | Sqrt
            | Rsqrt
            | Exp
            | Log
            | Sin
            | Cos
            | Tan
            | Tanh
            | Asin
            | Acos
            | Atan
            | Floor
            | Exp2
            | Expm1
            | Log2
            | Log10
            | Log1p
            | Cbrt
            | Sinh
            | Cosh
            | Asinh
            | Acosh
            | Atanh
            | Ceil
            | Trunc
            | Round
            | Rint
            | Sign
            | Erf
            | Erfc
            | Spacing
            | Fma
            | Clip
    )
}

/// The rewrite of reads over element-wise values: the nodes made so far (by value and index).
struct Pusher {
    /// The IR context the rewrite builds nodes in.
    ctx: *mut RxCtx,
    /// Nodes already rewritten, by (node, read) pair.
    memo: HashMap<(u32, u32), u32>,
}

/// Turns a door status code into the door's message when it is not OK.
fn status(ctx: *mut RxCtx, code: c_int) -> Result<(), Refusal> {
    if code == 0 {
        Ok(())
    } else {
        Err(door::failed(ctx, code))
    }
}

impl Pusher {
    /// The rank of a node's type.
    fn rank(&self, node: u32) -> Result<usize, Refusal> {
        Ok(crate::body::ty_of(self.ctx, node)?.shape.len())
    }

    /// An element-wise operation over a vector.
    fn is_elementwise_vector(&self, node: u32) -> Result<bool, Refusal> {
        let Some(kind) = OpKind::of_op(self.ctx, node)? else {
            return Ok(false);
        };
        Ok(self.rank(node)? == 1 && is_elementwise(kind))
    }

    /// A runtime-index read, without a literal, of an element-wise vector.
    fn is_pushable(&self, node: u32) -> Result<bool, Refusal> {
        if OpKind::of_op(self.ctx, node)? != Some(OpKind::ComponentAt) {
            return Ok(false);
        }
        let operands = door::operands(self.ctx, node)?;
        let plain = literal_of(self.ctx, node)? == Scalar::Absent;
        Ok(plain && operands.len() == 2 && self.is_elementwise_vector(operands[0])?)
    }

    /// A new operation node of the given kind over the operands, carrying the literal of the node
    /// `like` (none when `None`).
    fn make_op(&self, kind: OpKind, operands: &[u32], like: Option<u32>) -> Result<u32, Refusal> {
        let kind = std::ffi::CString::new(kind.name()).map_err(fault)?;
        let mut out = 0u32;
        let code = unsafe {
            rx_make_op_like(
                self.ctx,
                kind.as_ptr(),
                operands.as_ptr(),
                operands.len(),
                like.as_ref()
                    .map_or(std::ptr::null(), |node| node as *const u32),
                &mut out,
            )
        };
        status(self.ctx, code)?;
        Ok(out)
    }

    /// One element of a value: a rank-0 value is itself, an element-wise vector is the same
    /// operation over the elements of its operands, any other vector is read at the index.
    fn element(&mut self, value: u32, index: u32) -> Result<u32, Refusal> {
        if self.rank(value)? == 0 {
            return Ok(value);
        }
        if let Some(&made) = self.memo.get(&(value, index)) {
            return Ok(made);
        }
        let made = if self.is_elementwise_vector(value)? {
            let mut parts = vec![];
            for operand in door::operands(self.ctx, value)? {
                parts.push(self.element(operand, index)?);
            }
            let kind = OpKind::of(self.ctx, value)?;
            self.make_op(kind, &parts, Some(value))?
        } else {
            self.make_op(OpKind::ComponentAt, &[value, index], None)?
        };
        self.memo.insert((value, index), made);
        Ok(made)
    }

    /// Whether every sink commits a value the rewrite can replace.
    fn sinks_allow(&self, sinks: &[u32]) -> Result<bool, Refusal> {
        for &sink in sinks {
            let class = Class::of(self.ctx, sink)?;
            let known = VALUE_SINKS.contains(&class.as_str())
                || door::field_names(self.ctx, sink, "feathers")?
                    .unwrap_or_default()
                    .iter()
                    .any(|feather| feather == "elementwise_sink");
            let operands = door::operands(self.ctx, sink)?.len();
            if !known || !(operands == 1 || class == Class::AccumWrite) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The unit's sinks with every pushable read replaced, and the canonical walk over them (keeping
/// the unit's slots); the walk and sinks come back unchanged when nothing is pushable.
pub(crate) fn push_reads(
    ctx: *mut RxCtx,
    walk: u32,
    sinks: &[u32],
) -> Result<(u32, Vec<u32>), Refusal> {
    let mut pusher = Pusher {
        ctx,
        memo: HashMap::new(),
    };
    let mut reads = vec![];
    for node in door::walk_order(ctx, walk)? {
        if pusher.is_pushable(node)? {
            reads.push(node);
        }
    }
    if reads.is_empty() || !pusher.sinks_allow(sinks)? {
        return Ok((walk, sinks.to_vec()));
    }
    let mut targets = vec![];
    for &read in &reads {
        let operands = door::operands(ctx, read)?;
        targets.push(pusher.element(operands[0], operands[1])?);
    }
    let mut roots = vec![];
    for &sink in sinks {
        roots.extend(door::operands(ctx, sink)?);
    }
    let (mut done, mut count) = (vec![0u32; roots.len()], 0usize);
    let code = unsafe {
        rx_substitute(
            ctx,
            roots.as_ptr(),
            roots.len(),
            reads.as_ptr(),
            targets.as_ptr(),
            reads.len(),
            done.as_mut_ptr(),
            done.len(),
            &mut count,
        )
    };
    status(ctx, code)?;
    let (mut rebuilt, mut at) = (vec![], 0usize);
    for &sink in sinks {
        let width = door::operands(ctx, sink)?.len();
        let part = &done[at..at + width];
        at += width;
        let mut out = sink;
        if part != &roots[at - width..at] {
            status(ctx, unsafe {
                rx_rebuild(ctx, sink, part.as_ptr(), part.len(), &mut out)
            })?;
        }
        rebuilt.push(out);
    }
    Ok((door::canonical_over(ctx, &rebuilt, walk)?, rebuilt))
}
