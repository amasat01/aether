//! The lane split of a finishing kernel: the body cut at its one lowered loop with a tail
//! break. The text before the loop
//! (its head), one trip of the loop and the text after it (its tail) are rendered apart, so the
//! host can run the trips of
//! a whole tile of samples side by side.
use super::cut::{Carry, CutLoop};
pub use super::model::{LaneSplit, SampleFree};
use crate::body::frame::needs::Needs;
use crate::body::loops::{body_fanout, int, read_head, render_scope, LoopHead};
use crate::body::Buffer;
use crate::body::Class;
use crate::body::{Frame, LoopEmitter, Ty};
use crate::ir::analysis::{loops_flat, speculative_step};
use crate::ir::door::{self, RxCtx};
use crate::kind::OpKind;
use crate::refuse::Refusal;
use crate::spell::ident;
use std::collections::BTreeSet;

/// The one top-level loop the split admits: a tail break, counted from 0 by 1 with a static or
/// runtime bound, no inner
/// loop, tape or per-iteration commit, and no tape or counter anywhere in the kernel.
pub fn candidate(ctx: *mut RxCtx, walk: u32) -> Result<Option<u32>, Refusal> {
    let order = door::walk_order(ctx, walk)?;
    let mut loops = vec![];
    for &h in &order {
        if door::class_is(ctx, h, "Loop")? {
            loops.push(h)
        }
    }
    if loops.len() != 1 {
        return Ok(None);
    }
    let lp = loops[0];
    for h in loops_flat(ctx, &order)? {
        if door::class_is(ctx, h, "TapeRead")? || door::class_is(ctx, h, "LoopCount")? {
            return Ok(None);
        }
    }
    let (carries, nexts) = (
        door::field_node_list(ctx, lp, "carries")?.unwrap_or_default(),
        door::field_node_list(ctx, lp, "nexts")?.unwrap_or_default(),
    );
    let exit_values = door::field_node_list(ctx, lp, "exit_values")?.unwrap_or_default();
    let tail = !door::field_node_list(ctx, lp, "exit_cond")?
        .unwrap_or_default()
        .is_empty()
        && exit_values == nexts
        && exit_values != carries;
    if !tail
        || !door::field_node_list(ctx, lp, "body_sinks")?
            .unwrap_or_default()
            .is_empty()
    {
        return Ok(None);
    }
    let plain = door::field_scalar(ctx, lp, "start")? == door::Scalar::Int(0)
        && door::field_scalar(ctx, lp, "step")? == door::Scalar::Int(1)
        && !door::field_flag(ctx, lp, "count_tail")?.unwrap_or(false);
    if !plain {
        return Ok(None);
    }
    for h in door::body_nodes(ctx, lp)? {
        if door::class_is(ctx, h, "Loop")?
            || door::class_is(ctx, h, "TapeRead")?
            || door::class_is(ctx, h, "LoopCount")?
        {
            return Ok(None);
        }
    }
    Ok(Some(lp))
}

/// Opens the trip's scope for the split loop: the body's nodes, the index and the carries take slots beyond the flat table;
/// each carry reads through its usual name.
fn open_split_scope(
    f: &mut Frame,
    head: &LoopHead,
    index: &str,
) -> Result<crate::body::frame::scopes::LoopScope, Refusal> {
    let scope = f.open_loop(head.index_h, &head.carries, &head.body)?;
    let base = scope.base;
    f.set_text(base, index.to_string());
    for (j, c) in head.carries.iter().enumerate() {
        let slot = f.slot_of(*c);
        f.set_name_text(slot, head.names[j].clone())
    }
    Ok(scope)
}

/// Declares the trip's results: each carry's next value and the tail break's flag; returns the next values' names and the
/// flag's name.
fn declare_split_results(f: &mut Frame, head: &LoopHead) -> Result<(Vec<String>, String), Refusal> {
    let (pos, exit_cond) = (head.pos, head.exit_cond.unwrap());
    let next_names: Vec<String> = (0..head.names.len())
        .map(|j| f.names.next(pos, j).text)
        .collect();
    for j in 0..head.names.len() {
        let value = f.arg(head.nexts[j])?;
        f.declare_as(f.names.next(pos, j), head.elems[j].clone(), Some(&value));
    }
    let flag = f.names.exit_flag(pos).text;
    let value = f.arg(exit_cond)?;
    f.declare_as(f.names.exit_flag(pos), "bool".to_string(), Some(&value));
    Ok((next_names, flag))
}

/// Renders one trip of the split loop into its own text, the carried values read through their
/// usual names.
fn emit_split(f: &mut Frame, cuts: &mut Vec<CutLoop>, lp: u32) -> Result<(), Refusal> {
    let ctx = f.ctx();
    let head = read_head(f, lp)?;
    let pos = head.pos;
    f.record_carries(pos, head.names.clone());
    let init_texts = head
        .inits
        .iter()
        .map(|&i| f.peek_arg(i))
        .collect::<Result<Vec<_>, _>>()?;
    let index = f
        .names
        .index(
            pos,
            &ident(&door::text_field(f.ctx(), head.index_h, "name")?),
        )
        .text;
    let count = match door::field_node_list(ctx, lp, "count")?
        .unwrap_or_default()
        .first()
    {
        Some(&c) => Some(f.arg(c)?),
        None => None,
    };
    let trip = int(f, lp, "trip")?;
    let mark = f.buf.lines.len();
    let buffer = f.open_buffer();
    let scope = open_split_scope(f, &head, &index)?;
    let base = scope.base;
    let fan = body_fanout(f, &head, base, false)?;
    render_scope(f, &head, base, &fan, None, &[])?;
    let (next_names, flag) = declare_split_results(f, &head)?;
    f.close_loop(scope);
    let step = f.close_buffer(buffer);
    let mut carries = vec![];
    for j in 0..head.names.len() {
        carries.push(Carry {
            name: head.names[j].clone(),
            ty: head.types[j].clone(),
            init: init_texts[j].clone(),
            next: next_names[j].clone(),
            init_reads: f.reads_of(head.inits[j])?,
        })
    }
    cuts.push(CutLoop {
        mark,
        step,
        carries,
        index,
        start: 0,
        by: 1,
        trips: trip,
        exit: flag,
        speculative: false,
        bounds: None,
        bounds_reads: Default::default(),
        count,
    });
    Ok(())
}

/// Whether a value reads nothing per sample: parameters, sample counts, table lengths and
/// constants by non-random ops.
fn sample_free(f: &Frame, root: u32) -> Result<bool, Refusal> {
    let (mut stack, mut seen) = (vec![root], vec![]);
    while let Some(h) = stack.pop() {
        if seen.contains(&h) {
            continue;
        }
        seen.push(h);
        match Class::of(f.ctx(), h)? {
            Class::PlaneExtent | Class::Const | Class::RoleConst => {}
            Class::Leaf => {
                if !matches!(
                    door::text_field(f.ctx(), h, "role")?.as_str(),
                    "uniform" | "nsamples"
                ) {
                    return Ok(false);
                }
            }
            Class::Op => {
                if matches!(
                    OpKind::of(f.ctx(), h)?,
                    OpKind::RandomUniform | OpKind::RandomNormal | OpKind::RawDevice
                ) {
                    return Ok(false);
                }
                stack.extend(door::operands(f.ctx(), h)?);
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// `(j, rises)` when the exit condition compares the next value of exactly one scalar real
/// carry to a sample-free bound.
fn exit_order(f: &Frame, lp: u32) -> Result<Option<(usize, bool)>, Refusal> {
    let cond = door::field_node_list(f.ctx(), lp, "exit_cond")?.unwrap_or_default()[0];
    if !door::class_is(f.ctx(), cond, "Op")? {
        return Ok(None);
    }
    let kind = OpKind::of(f.ctx(), cond)?;
    let ops = door::operands(f.ctx(), cond)?;
    if ops.len() != 2 || !matches!(kind, OpKind::Ge | OpKind::Gt | OpKind::Le | OpKind::Lt) {
        return Ok(None);
    }
    let (nexts, carries) = (
        door::field_node_list(f.ctx(), lp, "nexts")?.unwrap_or_default(),
        door::field_node_list(f.ctx(), lp, "carries")?.unwrap_or_default(),
    );
    for first in 0..2 {
        let (value, bound) = (ops[first], ops[1 - first]);
        let kind = if first == 0 {
            kind
        } else {
            match kind {
                OpKind::Ge => OpKind::Le,
                OpKind::Gt => OpKind::Lt,
                OpKind::Le => OpKind::Ge,
                _ => OpKind::Gt,
            }
        };
        let hits: Vec<usize> = (0..nexts.len()).filter(|&j| nexts[j] == value).collect();
        if hits.len() != 1 || !sample_free(f, bound)? {
            continue;
        }
        let ty = crate::body::ty_of(f.ctx(), carries[hits[0]])?;
        if ty.shape.is_empty() && matches!(ty.dtype.as_str(), "f64" | "f32") {
            return Ok(Some((hits[0], matches!(kind, OpKind::Ge | OpKind::Gt))));
        }
    }
    Ok(None)
}

/// The values the head names and one trip reads, with their types; `None` when one trip reads a head value that is not on the
/// loop's boundary.
fn live_values(
    f: &Frame,
    lp: u32,
    cut: &CutLoop,
    head: &BTreeSet<String>,
) -> Result<Option<Vec<(String, Ty)>>, Refusal> {
    let carry_count = cut.carries.len();
    let mut live_ins: Vec<(String, Ty)> = vec![];
    for &free in door::operands(f.ctx(), lp)?.iter().skip(carry_count) {
        let name = f.arg(free)?;
        if head.contains(&name) && !live_ins.iter().any(|l| l.0 == name) {
            live_ins.push((name, crate::body::ty_of(f.ctx(), free)?));
        }
    }
    let step_mentions = cut.step.reads_of(head);
    let boundary: BTreeSet<&String> = live_ins.iter().map(|l| &l.0).collect();
    if step_mentions.iter().any(|w| !boundary.contains(w)) {
        return Ok(None);
    }
    live_ins.retain(|l| step_mentions.contains(&l.0));
    Ok(Some(live_ins))
}

/// The one-line declarations in the head of the live values that read nothing per sample and of the values they read.
fn sample_free_lines(
    f: &Frame,
    lp: u32,
    carry_count: usize,
    live_ins: &[(String, Ty)],
    pre: &Buffer,
) -> Result<SampleFree, Refusal> {
    let mut free_names: Vec<String> = vec![];
    for &free in door::operands(f.ctx(), lp)?.iter().skip(carry_count) {
        if sample_free(f, free)? {
            free_names.push(f.arg(free)?)
        }
    }
    let sample_free_names: Vec<String> = live_ins
        .iter()
        .map(|l| l.0.clone())
        .filter(|n| free_names.contains(n))
        .collect();
    let chain = pre.one_line_chain(&sample_free_names);
    Ok(SampleFree {
        at: chain.iter().map(|d| d.line).collect(),
        names: chain.iter().map(|d| d.name.text.clone()).collect(),
    })
}

/// The operations of the loop's own body nodes.
fn count_ops(f: &Frame, lp: u32) -> Result<usize, Refusal> {
    let mut op_count = 0;
    for h in door::body_nodes(f.ctx(), lp)? {
        if door::class_is(f.ctx(), h, "Op")? {
            op_count += 1
        }
    }
    Ok(op_count)
}

/// The loop emitter of the lane split: records the one loop's trip, and at the end the split of the render with its needs.
pub(crate) struct SplitEmitter {
    /// The loop the split is cut at.
    lp: u32,
    /// The cut loops written so far (at most the one).
    cuts: Vec<CutLoop>,
    /// The finished split and needs, set by `finish`; `None` when the body is not that shape.
    pub(crate) done: Option<(LaneSplit, Needs)>,
}

impl SplitEmitter {
    /// An emitter for the split at `lp`.
    pub(crate) fn new(lp: u32) -> Self {
        SplitEmitter {
            lp,
            cuts: vec![],
            done: None,
        }
    }
}

impl LoopEmitter for SplitEmitter {
    fn emit(&mut self, f: &mut Frame, lp: u32) -> Result<(), Refusal> {
        emit_split(f, &mut self.cuts, lp)
    }
    fn finish(&mut self, f: &Frame, guard: Option<String>) -> Result<(), Refusal> {
        let split = assemble(f, &self.cuts, guard, self.lp)?;
        self.done = split.map(|split| (split, f.recorded_needs()));
        Ok(())
    }
    fn one_trip(&self) -> bool {
        true
    }
}

/// The split of a finished render, or `None` when the tail reads a value the head named or one
/// trip reads a head value
/// that is not on the loop's boundary.
fn assemble(
    f: &Frame,
    cuts: &[CutLoop],
    guard: Option<String>,
    lp: u32,
) -> Result<Option<LaneSplit>, Refusal> {
    let Some(cut) = cuts.first() else {
        return Ok(None);
    };
    let (start, end) = if guard.is_some() {
        (1, f.buf.lines.len() - 1)
    } else {
        (0, f.buf.lines.len())
    };
    let (pre, post) = (f.buf.slice(start, cut.mark), f.buf.slice(cut.mark, end));
    let head = pre.declared();
    let post_mentions = post.reads_of(&head);
    if !post_mentions.is_empty() {
        return Ok(None);
    }
    let carry_count = cut.carries.len();
    let Some(live_ins) = live_values(f, lp, cut, &head)? else {
        return Ok(None);
    };
    let sample_free = sample_free_lines(f, lp, carry_count, &live_ins, &pre)?;
    let op_count = count_ops(f, lp)?;
    Ok(Some(LaneSplit {
        guard,
        pre,
        step: cut.step.clone(),
        post,
        carries: cut.carries.clone(),
        live_ins,
        count: cut.count.clone(),
        stop: int(f, lp, "stop")?,
        index: cut.index.clone(),
        exit: cut.exit.clone(),
        speculative: speculative_step(f.ctx(), lp)?,
        op_count,
        exit_order: exit_order(f, lp)?,
        sample_free,
    }))
}
