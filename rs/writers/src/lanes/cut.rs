//! Lane segments: the body of a kernel cut at the loops the host runs interchanged. A cut loop
//! renders one trip for one lane
//! into a text of its own; the text before it, between two cuts and after the last one are the
//! segments. The arm composes the
//! interchanged unit from them.
pub use super::model::{Carry, CutLoop, LaneSegments};
use crate::body::frame::buffer::Buffer;
use crate::body::frame::needs::Need;
use crate::body::frame::needs::Needs;
use crate::body::loops::{
    body_fanout, exit_part, for_header, int, loop_pos, read_head, render_scope, LoopHead,
};
use crate::body::{Frame, LoopEmitter, Ty};
use crate::ir::analysis::{is_resident, is_scratch, speculative_step};
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, Refusal};
use crate::spell::ident;

/// Refuses a loop the cut cannot write: one with a counter, a tape or sinks of its own.
fn check_cut_loop(f: &Frame, lp: u32, pos: usize) -> Result<(), Refusal> {
    let ctx = f.ctx();
    if f.count_names().iter().any(|c| c.0 == pos) || f.tape_names().iter().any(|t| (t.0).0 == pos) {
        return Err(nib("a cut loop with a counter or a tape"));
    }
    if !door::field_node_list(ctx, lp, "body_sinks")?
        .unwrap_or_default()
        .is_empty()
    {
        return Err(nib("a loop body with sinks"));
    }
    Ok(())
}

/// Which carried slots live in a plane (no local of their own: the carry is the plane's proxy, written in place); refuses a
/// planed carry under a head or middle break.
fn planed_carries(ctx: *mut RxCtx, head: &LoopHead) -> Result<Vec<bool>, Refusal> {
    // a carried slot that lives in a plane has no local of its own: the carry is the plane's proxy, written in place
    let mut planed = vec![];
    for &init in &head.inits {
        planed.push(is_scratch(ctx, init)? || is_resident(ctx, init)?)
    }
    // only the tail break is written: the end values are the next values, a planed one already the
    // proxy, and the trip is never speculative
    if planed.iter().any(|&p| p) && head.exit_cond.is_some() && head.exit_values != head.nexts {
        return Err(nib(
            "a carried slot in a plane of a loop with a head or middle break",
        ));
    }
    Ok(planed)
}

/// What the writing of a cut trip depends on beside the loop's head: the carried locals' names (a planed carry is its proxy),
/// which carries are planed, the names of the next values, the break flag and whether the trip may run for a stopped lane.
struct CutTrip {
    /// The carried locals' names.
    names: Vec<String>,
    /// Whether each carry lives in a plane.
    planed: Vec<bool>,
    /// The name of each carry's next value.
    nexts: Vec<String>,
    /// The name of the break flag.
    flag: String,
    /// Whether the trip may run for a stopped lane and be discarded by a select.
    speculative: bool,
}

/// The header of a counted loop as the cut records it: index type, begin, comparison, end; `None` for a static range.
fn cut_bounds(
    f: &mut Frame,
    head: &LoopHead,
    index: &str,
    trip: i64,
) -> Result<Option<Bounds>, Refusal> {
    let (ctx, lp, start, step) = (f.ctx(), head.lp, head.start, head.step);
    let index = index.to_string();
    let mut bounds = None;
    if let Some(&count) = door::field_node_list(ctx, lp, "count")?
        .unwrap_or_default()
        .first()
    {
        let header = for_header(
            &index,
            start,
            int(f, lp, "stop")?,
            step,
            trip,
            door::field_flag(f.ctx(), lp, "count_tail")?.unwrap_or(false),
            Some(&f.arg(count)?),
        );
        let recorded = (header.itype, header.begin, header.cmp, header.end);
        bounds = Some(recorded);
    }
    Ok(bounds)
}

/// The recorded header of a counted loop: index type, begin, comparison, end.
type Bounds = (String, String, String, String);

/// Opens the trip's scope: the body's nodes, the index and the carries take slots beyond the flat table; a planed carry is
/// written in place.
fn open_cut_scope(
    f: &mut Frame,
    head: &LoopHead,
    t: &CutTrip,
    index: &str,
) -> Result<crate::body::frame::scopes::LoopScope, Refusal> {
    let scope = f.open_loop(head.index_h, &head.carries, &head.body)?;
    let base = scope.base;
    f.set_text(base, index.to_string());
    for (j, c) in head.carries.iter().enumerate() {
        let slot = f.slot_of(*c);
        f.set_name_text(slot, t.names[j].clone());
        if t.planed[j] {
            f.mutable_carries.insert(*c);
        }
    }
    Ok(scope)
}

/// The trip of a loop with a break at its head or middle: the part before the break, the flag, the held values, then the rest
/// either speculatively (selected by the flag) or guarded by `if (!flag)`.
fn emit_break_trip(
    f: &mut Frame,
    head: &LoopHead,
    t: &CutTrip,
    base: usize,
    fan: &[usize],
) -> Result<(), Refusal> {
    let (pos, body, exits, edge) = (head.pos, &head.body, &head.exits, head);
    let (elems, names, flag, speculative) = (&head.elems, &t.names, &t.flag, t.speculative);
    let pre = exit_part(f, body, exits)?;
    render_scope(f, head, base, fan, Some(&pre), &[])?;
    let cond = f.arg(edge.exit_cond.unwrap())?;
    f.emit(format!("const bool {flag} = {cond};"));
    let held: Vec<String> = (0..names.len())
        .map(|j| f.names.exit_value(pos, j).text)
        .collect();
    for j in 0..names.len() {
        let line = format!(
            "const {} {} = {};",
            elems[j],
            held[j],
            f.arg(edge.exit_values[j])?
        );
        f.emit(line)
    }
    if speculative {
        emit_selected_rest(f, head, t, base, fan, &pre, &held)?
    } else {
        emit_guarded_rest(f, head, t, base, fan, &pre, &held)?
    }
    Ok(())
}

/// The rest of a speculative break trip: the whole body runs, and each next value is selected by the flag between the held
/// value and the computed one.
fn emit_selected_rest(
    f: &mut Frame,
    head: &LoopHead,
    t: &CutTrip,
    base: usize,
    fan: &[usize],
    pre: &[u32],
    held: &[String],
) -> Result<(), Refusal> {
    let (pos, edge) = (head.pos, head);
    let (elems, names, nexts, flag) = (&head.elems, &t.names, &t.nexts, &t.flag);
    render_scope(f, head, base, fan, None, pre)?;
    for j in 0..names.len() {
        let result = f.names.result(pos, j).text;
        let line = format!("const {} {result} = {};", elems[j], f.arg(edge.nexts[j])?);
        f.emit(line);
        let hold = if f.unit.build.blend && elems[j] == "Real" {
            f.need(Need::Blend);
            format!("hawk_abi::blend({flag}, {}, {result})", held[j])
        } else {
            format!("{flag} ? {} : {result}", held[j])
        };
        f.emit(format!("const {} {} = {hold};", elems[j], nexts[j]));
    }
    Ok(())
}

/// The rest of a guarded break trip: the carries take the held values, and the rest of the body runs under `if (!flag)`.
fn emit_guarded_rest(
    f: &mut Frame,
    head: &LoopHead,
    t: &CutTrip,
    base: usize,
    fan: &[usize],
    pre: &[u32],
    held: &[String],
) -> Result<(), Refusal> {
    let (pos, edge) = (head.pos, head);
    let (elems, names, nexts, flag) = (&head.elems, &t.names, &t.nexts, &t.flag);
    for j in 0..names.len() {
        f.emit(format!(
            "{} {} = {};",
            elems[j],
            f.names.write_back(pos, j).text,
            held[j]
        ))
    }
    f.emit(format!("if (!{flag}) {{"));
    f.buf.indent.push_str("    ");
    render_scope(f, head, base, fan, None, pre)?;
    for (j, &next) in edge.nexts[..names.len()].iter().enumerate() {
        let line = format!("{} = {};", f.names.write_back(pos, j).text, f.arg(next)?);
        f.emit(line)
    }
    f.buf.indent.truncate(f.buf.indent.len() - 4);
    f.emit("}".into());
    for j in 0..names.len() {
        f.emit(format!(
            "const {} {} = {};",
            elems[j],
            nexts[j],
            f.names.write_back(pos, j).text
        ))
    }
    Ok(())
}

/// The trip of a loop without a break at its head or middle: the body, the next values (a planed carry must already be stored
/// in place) and a tail break's flag.
fn emit_plain_trip(
    f: &mut Frame,
    head: &LoopHead,
    t: &CutTrip,
    base: usize,
    fan: &[usize],
) -> Result<(), Refusal> {
    let (edge, form) = (head, head.form);
    let (elems, names, nexts, flag, planed) = (&head.elems, &t.names, &t.nexts, &t.flag, &t.planed);
    render_scope(f, head, base, fan, None, &[])?;
    for j in 0..names.len() {
        if planed[j] {
            if f.arg(edge.nexts[j])? != names[j] {
                return Err(nib("a carried slot in a plane that is not stored in place"));
            }
            continue;
        }
        let line = format!(
            "const {} {} = {};",
            elems[j],
            nexts[j],
            f.arg(edge.nexts[j])?
        );
        f.emit(line)
    }
    if form == "tail" {
        let line = format!("const bool {flag} = {};", f.arg(edge.exit_cond.unwrap())?);
        f.emit(line)
    }
    Ok(())
}

/// What a cut loop records beside its head: where it starts in the output, its index name, the trip count, a counted header
/// and whether a trip may run for a stopped lane.
struct CutFacts {
    /// The number of lines emitted before the loop.
    mark: usize,
    /// The index variable's name.
    index: String,
    /// The trip count.
    trip: i64,
    /// The recorded header of a counted loop, if any.
    bounds: Option<Bounds>,
    /// Whether a trip may run for a stopped lane and be discarded by a select.
    speculative: bool,
}

/// Records the cut loop on the frame: its carried slots (a planed carry has no local), its index, range, break flag and bounds.
fn push_cut(
    f: &mut Frame,
    cuts: &mut Vec<CutLoop>,
    head: &LoopHead,
    t: &CutTrip,
    init_texts: Vec<String>,
    facts: CutFacts,
    step_text: Buffer,
) -> Result<(), Refusal> {
    let CutFacts {
        mark,
        index,
        trip,
        bounds,
        speculative,
    } = facts;
    let carries = (0..t.names.len())
        .filter(|&j| !t.planed[j])
        .map(|j| {
            Ok(Carry {
                name: t.names[j].clone(),
                ty: head.types[j].clone(),
                init: init_texts[j].clone(),
                next: t.nexts[j].clone(),
                init_reads: f.reads_of(head.inits[j])?,
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let exit = if head.exit_cond.is_none() {
        "false".to_string()
    } else {
        t.flag.clone()
    };
    let count_node = door::field_node_list(f.ctx(), head.lp, "count")?.unwrap_or_default();
    let bounds_reads = match (&bounds, count_node.first()) {
        (Some(_), Some(&count)) => f.reads_of(count)?,
        _ => Default::default(),
    };
    cuts.push(CutLoop {
        mark,
        step: step_text,
        carries,
        index,
        start: head.start,
        by: head.step,
        trips: trip,
        exit,
        speculative,
        bounds,
        bounds_reads,
        count: None,
    });
    Ok(())
}

/// Marks where the segment before the loop ends and renders one trip of it into its own text. A break at the head or the
/// middle of the body becomes a select: the trip's next values are the values at the break when the condition holds.
fn emit_cut(f: &mut Frame, cuts: &mut Vec<CutLoop>, lp: u32) -> Result<(), Refusal> {
    let ctx = f.ctx();
    let pos = loop_pos(f, lp)?;
    check_cut_loop(f, lp, pos)?;
    let head = read_head(f, lp)?;
    let planed = planed_carries(ctx, &head)?;
    let mut names = head.names.clone();
    for j in 0..names.len() {
        if planed[j] {
            names[j] = f.peek_arg(head.inits[j])?
        }
    }
    f.record_carries(pos, names.clone());
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
    let trip = int(f, lp, "trip")?;
    let bounds = cut_bounds(f, &head, &index, trip)?;
    // a trip that writes a plane is skipped by a lane that is not running
    let speculative = speculative_step(ctx, lp)? && !planed.iter().any(|&p| p);
    let mark = f.buf.lines.len();
    let buffer = f.open_buffer();
    let mut t = CutTrip {
        names,
        planed,
        nexts: vec![],
        flag: String::new(),
        speculative,
    };
    let scope = open_cut_scope(f, &head, &t, &index)?;
    let base = scope.base;
    let fan = body_fanout(f, &head, base, false)?;
    t.flag = f.names.exit_flag(pos).text;
    t.nexts = (0..t.names.len())
        .map(|j| f.names.next(pos, j).text)
        .collect();
    if head.form == "head" || head.form == "middle" {
        emit_break_trip(f, &head, &t, base, &fan)?
    } else {
        emit_plain_trip(f, &head, &t, base, &fan)?
    }
    f.close_loop(scope);
    let step_text = f.close_buffer(buffer);
    let facts = CutFacts {
        mark,
        index,
        trip,
        bounds,
        speculative,
    };
    push_cut(f, cuts, &head, &t, init_texts, facts, step_text)?;
    Ok(())
}

/// Whether a text is one identifier.
fn is_word(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// The loop emitter of the interchanged host text: records each cut loop, and at the end the segments of the render with the
/// needs its text recorded.
#[derive(Default)]
pub(crate) struct CutEmitter {
    /// The cut loops written so far, in order.
    cuts: Vec<CutLoop>,
    /// The finished segments and needs, set by `finish`.
    pub(crate) done: Option<(LaneSegments, Needs)>,
}

impl LoopEmitter for CutEmitter {
    fn emit(&mut self, f: &mut Frame, lp: u32) -> Result<(), Refusal> {
        emit_cut(f, &mut self.cuts, lp)
    }
    fn finish(&mut self, f: &Frame, guard: Option<String>) -> Result<(), Refusal> {
        let mut segs = segments_of(f, &self.cuts, guard)?;
        segs.loops = std::mem::take(&mut self.cuts);
        self.done = Some((segs, f.recorded_needs()));
        Ok(())
    }
}

/// The segments of a finished render.
fn segments_of(
    f: &Frame,
    cuts: &[CutLoop],
    guard: Option<String>,
) -> Result<LaneSegments, Refusal> {
    let (start, end) = if guard.is_some() {
        (1, f.buf.lines.len() - 1)
    } else {
        (0, f.buf.lines.len())
    };
    let marks: Vec<usize> = std::iter::once(start)
        .chain(cuts.iter().map(|c| c.mark))
        .chain(std::iter::once(end))
        .collect();
    let segments = marks.windows(2).map(|w| f.buf.slice(w[0], w[1])).collect();
    let mut types: Vec<(String, Ty)> = vec![];
    for (p, node) in f.nodes().enumerate() {
        let name = f.text(p);
        if is_word(name) && !types.iter().any(|t| &t.0 == name) {
            types.push((name.clone(), node.ty.clone()))
        }
    }
    for cut in cuts {
        for Carry { name, ty, .. } in &cut.carries {
            types.retain(|t| &t.0 != name);
            types.push((name.clone(), ty.clone()))
        }
    }
    Ok(LaneSegments {
        guard,
        segments,
        loops: vec![],
        types,
        proxies: f
            .proxies
            .iter()
            .filter(|x| x.width > 0)
            .map(|x| (x.name.clone(), x.plane.clone(), x.width))
            .collect(),
    })
}
