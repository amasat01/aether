//! Loops: a lowered loop is a statement (carried locals, a header, its own scope); its body is
//! read through the door's body
//! traversal. While a scope is open the body's nodes (and the loop's index and carries) are
//! appended to the frame's tables, so the
//! class writers work on them as on any node; closing the scope removes them again, as a name made
//! inside does not leak out.
use super::frame::disp::ScopeKind;
use super::{node_out, Frame, Name, NameKind, Out};
use crate::body::{Class, Ty};
use crate::ir::analysis::{is_resident, is_scratch};
use crate::ir::door;
use crate::refuse::{fault, nib, Refusal};
use crate::spell::{self, ident};

/// The integer a node's field holds.
pub(crate) fn int(f: &Frame, h: u32, name: &str) -> Result<i64, Refusal> {
    door::field_int(f.ctx(), h, name)?
        .ok_or_else(|| fault(format!("loop field {name} is not a number")))
}

/// A loop in the flat walk: its statement, then the rendering of the node is empty (its values are
/// read through LoopValue).
pub fn stmt(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    if f.table.loop_pos().is_empty() {
        index_loops(f)?
    }
    // the emitter is out of the frame while it writes the loop
    let emitter = if f.cut.contains(&h) {
        f.take_loop_emitter()
    } else {
        None
    };
    match emitter {
        Some(emit) => {
            let done = emit.emit(f, h);
            f.put_loop_emitter(emit);
            done?
        }
        None => emit_loop(f, h)?,
    }
    f.set_text(p, String::new());
    Ok(Out::Bound)
}

/// One carried slot's final value: the C++ local the loop left it in.
pub fn value(f: &Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let (lp, slot) = (f.node(p).ops[0], int(f, h, "literal")? as usize);
    let pos = loop_pos(f, lp)?;
    let names = f
        .carry_names()
        .iter()
        .find(|c| c.0 == pos)
        .ok_or_else(|| nib("a loop value read before its loop"))?;
    let name = names
        .1
        .get(slot)
        .cloned()
        .ok_or_else(|| fault("a loop value of a missing carry"))?;
    f.add_read(&name);
    Ok(Out::Value(name, None))
}

/// A forward loop's carried value at one iteration, out of its tape.
pub fn tape_read(f: &Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let (lp, slot) = (f.node(p).ops[0], int(f, h, "slot")? as usize);
    let pos = loop_pos(f, lp)?;
    let tape = &f
        .tape_names()
        .iter()
        .find(|t| t.0 == (pos, slot))
        .ok_or_else(|| fault("a tape read of a slot with no tape"))?
        .1;
    f.add_read(tape);
    Ok(Out::Value(
        format!(
            "{tape}[{}]",
            ordinal(
                f.text(f.opp(p)[1]),
                int(f, lp, "start")?,
                int(f, lp, "step")?
            )
        ),
        None,
    ))
}

/// The iteration ordinal of an index in a loop: the tape's subscript, folded away in the ordinary
/// range(N) case.
fn ordinal(index: &str, start: i64, step: i64) -> String {
    if (start, step) == (0, 1) {
        return index.to_string();
    }
    let shifted = if start == 0 {
        index.to_string()
    } else {
        format!("({index} - {start})")
    };
    if step == 1 {
        shifted
    } else {
        format!("({shifted} / {step})")
    }
}

/// The per-sample iteration counter of a loop whose count a derived node reads.
pub fn count(f: &Frame, p: usize) -> Result<Out, Refusal> {
    let pos = loop_pos(f, f.node(p).ops[0])?;
    Ok(Out::Value(
        f.count_names()
            .iter()
            .find(|c| c.0 == pos)
            .ok_or_else(|| fault("a loop count of a loop with no counter"))?
            .1
            .clone(),
        None,
    ))
}

/// The position of the loop node `lp` in the frame's table.
pub(crate) fn loop_pos(f: &Frame, lp: u32) -> Result<usize, Refusal> {
    f.table
        .loop_pos()
        .iter()
        .find(|x| x.0 == lp)
        .map(|x| x.1)
        .ok_or_else(|| fault("a loop that was not indexed"))
}

/// Every loop gets a stable suffix: a top-level loop its canonical position, a nested one (no
/// position of its own) the next number
/// above every position; then the loops whose count a derived node reads get a counter.
fn index_loops(f: &mut Frame) -> Result<(), Refusal> {
    let mut counter = f.count();
    let mut pending: std::collections::VecDeque<u32> = f.nodes().map(|n| n.h).collect();
    let mut all: Vec<u32> = vec![];
    while let Some(h) = pending.pop_front() {
        all.push(h);
        if door::is_a(f.ctx(), h, "loop")? {
            if !f.table.loop_pos().iter().any(|x| x.0 == h) {
                let held = f.at(h).ok().unwrap_or_else(|| {
                    counter += 1;
                    counter - 1
                });
                f.table.index_loop(h, held);
            }
            pending.extend(door::body_nodes(f.ctx(), h)?);
            pending.extend(door::field_refs(f.ctx(), h, "nexts")?.unwrap_or_default());
        }
    }
    for &h in &all {
        if door::is_a(f.ctx(), h, "tape_read")? {
            let pos = loop_pos(f, door::operands(f.ctx(), h)?[0])?;
            let slot = int(f, h, "slot")? as usize;
            if !f.tape_names().iter().any(|t| t.0 == (pos, slot)) {
                let name = f.names.tape(pos, slot).text;
                f.table.add_tape(pos, slot, name)
            }
        }
        if door::is_a(f.ctx(), h, "loop_count")? {
            let pos = loop_pos(f, door::operands(f.ctx(), h)?[0])?;
            if !f.count_names().iter().any(|c| c.0 == pos) {
                {
                    let name = f.names.counter(pos).text;
                    f.table.add_counter(pos, name)
                }
            }
        }
    }
    Ok(())
}

/// The parts of a `for` header: the index type, begin, comparison and end, and the text made from
/// them.
pub(crate) struct ForHeader {
    /// The index type.
    pub itype: String,
    /// The first value of the index.
    pub begin: String,
    /// The comparison against the end.
    pub cmp: String,
    /// The value the index is compared with.
    pub end: String,
    /// The index name.
    index: String,
    /// The step expression.
    incr: String,
}

impl ForHeader {
    /// The header line.
    pub fn text(&self) -> String {
        let ForHeader {
            itype,
            begin,
            cmp,
            end,
            index,
            incr,
        } = self;
        format!("for ({itype} {index} = {begin}; {index} {cmp} {end}; {incr}) {{")
    }
}

/// The plain `for` header: the direction follows the sign of the step; bounds are literals unless
/// a runtime count bounds the loop.
pub(crate) fn for_header(
    index: &str,
    start: i64,
    stop: i64,
    step: i64,
    trip: i64,
    count_tail: bool,
    count: Option<&str>,
) -> ForHeader {
    let cmp = if step > 0 { "<" } else { ">" };
    let incr = match step {
        1 => format!("++{index}"),
        -1 => format!("--{index}"),
        _ => format!("{index} += {step}"),
    };
    let Some(count) = count else {
        return ForHeader {
            itype: "Int".to_string(),
            begin: start.to_string(),
            cmp: cmp.to_string(),
            end: stop.to_string(),
            index: index.to_string(),
            incr,
        };
    };
    let (begin, end) = counted_range(start, stop, step, trip, count_tail, count);
    ForHeader {
        itype: "int".to_string(),
        begin,
        cmp: cmp.to_string(),
        end,
        index: index.to_string(),
        incr,
    }
}

/// The begin and end expressions of a counted loop: a counted tail starts at `trip - count` steps
/// before the stop, a head count
/// ends `count` steps after the start; a bound that is not a plain integer is cast to `int`.
fn counted_range(
    start: i64,
    stop: i64,
    step: i64,
    trip: i64,
    count_tail: bool,
    count: &str,
) -> (String, String) {
    let (begin, end) = if count_tail {
        (
            format!(
                "{start} {} ({trip} - {count}) * {}",
                if step > 0 { "+" } else { "-" },
                step.abs()
            ),
            stop.to_string(),
        )
    } else {
        (
            start.to_string(),
            if (start, step) == (0, 1) {
                format!("hawk_clamp_count({count})")
            } else {
                format!("{start} + {count} * {step}")
            },
        )
    };
    let digits = |t: &str| {
        let d = t.trim_start_matches('-');
        !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())
    };
    let begin = if digits(&begin) {
        begin
    } else {
        format!("static_cast<int>({begin})")
    };
    let end = if end.starts_with("hawk_clamp_count(") {
        end
    } else {
        format!("static_cast<int>({end})")
    };
    (begin, end)
}

/// The body nodes a break needs: its condition, the carried values at the break and everything
/// they read in the body.
pub(crate) fn exit_part(f: &Frame, body: &[u32], roots: &[u32]) -> Result<Vec<u32>, Refusal> {
    let (mut need, mut stack): (Vec<u32>, Vec<u32>) = (vec![], roots.to_vec());
    while let Some(h) = stack.pop() {
        if !body.contains(&h) || need.contains(&h) {
            continue;
        }
        need.push(h);
        stack.extend(door::operands(f.ctx(), h)?);
    }
    Ok(need)
}

/// The handles the loop's carries, exits and body sinks read: consumers that are no operation.
fn held_by_nothing_but_ops(f: &Frame, head: &LoopHead) -> Result<Vec<u32>, Refusal> {
    let mut held: Vec<u32> = head.nexts.iter().chain(&head.exits).copied().collect();
    for &sink in &head.body_sinks {
        held.extend(door::operands(f.ctx(), sink)?);
    }
    Ok(held)
}

/// Renders the body's nodes inside the open scope: a value read twice within the body, and a
/// vector, are named `tb<n>`.
pub(crate) fn render_scope(
    f: &mut Frame,
    head: &LoopHead,
    base: usize,
    sf: &[usize],
    only: Option<&[u32]>,
    skip: &[u32],
) -> Result<(), Refusal> {
    let body = &head.body;
    let exempt = held_by_nothing_but_ops(f, head)?;
    for &h in body {
        if skip.contains(&h) || only.is_some_and(|o| !o.contains(&h)) {
            continue;
        }
        let s = f.slot_of(h);
        if !f.is_outside(s) {
            continue;
        }
        if f.node(s).class == Class::Loop {
            emit_loop(f, h)?;
            f.set_text(s, String::new());
            continue;
        }
        let named = f.naming(s, ScopeKind::Loop, None, sf[s - base])?;
        match node_out(f, s)? {
            Out::Value(text, _) if named => {
                let name = f.names.body_temp();
                let ty = f.node(s).ty.clone();
                let held = sf[s - base] >= 2 && !f.leaf_valued(s)?;
                let raw = f.flows_raw(s, &exempt);
                f.declare(name.clone(), &ty.dtype, &ty.shape, &text, held, raw)?;
                f.set_text(s, name.text);
            }
            Out::Value(text, _) => f.set_text(s, text),
            Out::Bound => {}
        }
    }
    Ok(())
}

/// What a lowered loop is, read from the door before anything is written: its carries, their next
/// values, its sinks,
/// range, break and body, the names of its carried locals, and where its break sits. A later step
/// shares this with the cut and the split.
pub(crate) struct LoopHead {
    /// The loop node.
    pub lp: u32,
    /// The suffix the loop's identifiers carry.
    pub pos: usize,
    /// The carried placeholders.
    pub carries: Vec<u32>,
    /// The initial value of each carry.
    pub inits: Vec<u32>,
    /// The next value of each carry.
    pub nexts: Vec<u32>,
    /// The values the loop yields at a break.
    pub exit_values: Vec<u32>,
    /// The sinks the body commits each trip.
    pub body_sinks: Vec<u32>,
    /// The index node.
    pub index_h: u32,
    /// The break condition, if the loop has one.
    pub exit_cond: Option<u32>,
    /// The break condition (when there is one) followed by the break values.
    pub exits: Vec<u32>,
    /// The nodes of the body, in order.
    pub body: Vec<u32>,
    /// The first index value.
    pub start: i64,
    /// The index step.
    pub step: i64,
    /// The carried values' types.
    pub types: Vec<Ty>,
    /// The carried locals' C++ types.
    pub elems: Vec<String>,
    /// The carried locals' C++ names.
    pub names: Vec<String>,
    /// Where the break sits: "" (none), "head", "tail" or "middle".
    pub form: &'static str,
}

/// The locals a loop declares before its header: the tape arrays by carried slot and the iteration
/// counter's name.
struct LoopLocals {
    /// The tape array name of each carried slot that has one, by slot.
    tapes: Vec<(usize, String)>,
    /// The per-sample iteration counter, when the loop has one.
    counter: Option<String>,
    /// The carried slots that live in a plane (a placed render): the carry is the plane's proxy, written in place.
    planed: Vec<bool>,
}

/// Reads the loop's head from the door and names its carried locals; writes nothing.
pub(crate) fn read_head(f: &mut Frame, lp: u32) -> Result<LoopHead, Refusal> {
    let ctx = f.ctx();
    let pos = loop_pos(f, lp)?;
    let carries = door::field_node_list(ctx, lp, "carries")?.unwrap_or_default();
    let (nexts, exit_values) = (
        door::field_node_list(ctx, lp, "nexts")?.unwrap_or_default(),
        door::field_node_list(ctx, lp, "exit_values")?.unwrap_or_default(),
    );
    let body_sinks = door::field_node_list(ctx, lp, "body_sinks")?.unwrap_or_default();
    let inits: Vec<u32> = door::operands(ctx, lp)?
        .into_iter()
        .take(carries.len())
        .collect();
    let (mut elems, mut names, mut types) = (vec![], vec![], vec![]);
    for (j, &c) in carries.iter().enumerate() {
        let t = crate::body::ty_of(ctx, c)?;
        elems.push(
            if f.banded() && t.shape.is_empty() && matches!(t.dtype.as_str(), "f64" | "f32") {
                spell::BAND_TYPE.to_string()
            } else {
                spell::carry_type(&t.dtype, &t.shape)?
            },
        );
        types.push(t);
        names.push(
            f.names
                .carry(pos, j, &ident(&f.field_text(c, "name")?))
                .text,
        );
    }
    let index_h = door::field_refs(ctx, lp, "index")?.unwrap_or_default()[0];
    let exit_cond = door::field_refs(ctx, lp, "exit_cond")?
        .unwrap_or_default()
        .first()
        .copied();
    let body = door::body_nodes(ctx, lp)?;
    let exits: Vec<u32> = exit_cond
        .into_iter()
        .chain(exit_values.iter().copied())
        .collect();
    let form = match exit_cond {
        None => "",
        Some(_) if exit_values == carries => "head",
        Some(_) if exit_values == nexts => "tail",
        Some(_) => "middle",
    };
    Ok(LoopHead {
        lp,
        pos,
        carries,
        inits,
        nexts,
        exit_values,
        body_sinks,
        index_h,
        exit_cond,
        exits,
        body,
        start: int(f, lp, "start")?,
        step: int(f, lp, "step")?,
        types,
        elems,
        names,
        form,
    })
}

/// One lowered loop as statements: the carried locals, the header, the body in its own scope, the
/// results.
fn emit_loop(f: &mut Frame, lp: u32) -> Result<(), Refusal> {
    let mut head = read_head(f, lp)?;
    let mut planed = vec![];
    for (j, &init) in head.inits.iter().enumerate() {
        let in_plane = f.placed && (is_resident(f.ctx(), init)? || is_scratch(f.ctx(), init)?);
        if in_plane {
            head.names[j] = f.arg(init)?;
        }
        planed.push(in_plane);
    }
    let locals = declare_locals(f, &head, planed)?;
    let index = emit_header(f, &head)?;
    let scope = open_trip(f, &head, &locals, &index)?;
    let base = scope.base;
    let sf = body_fanout(f, &head, base)?;
    render_trip_body(f, &head, &locals, base, &sf)?;
    super::stores::storeplan::body_sinks(f, lp)?;
    emit_results(f, &head, &locals)?;
    f.close_loop(scope);
    f.emit("}".into());
    Ok(())
}

/// Declares the carried locals with their initial values, the tape arrays and the iteration
/// counter.
fn declare_locals(
    f: &mut Frame,
    head: &LoopHead,
    planed: Vec<bool>,
) -> Result<LoopLocals, Refusal> {
    let (pos, lp) = (head.pos, head.lp);
    f.record_carries(pos, head.names.clone());
    for j in (0..head.carries.len()).filter(|&j| !planed[j]) {
        let init = f.arg(head.inits[j])?;
        f.declare_var(
            Name::new(head.names[j].clone(), NameKind::Carry),
            head.elems[j].clone(),
            Some(&init),
        )
    }
    let tapes: Vec<(usize, String)> = {
        let mut t: Vec<(usize, String)> = f
            .tape_names()
            .iter()
            .filter(|t| (t.0).0 == pos)
            .map(|t| ((t.0).1, t.1.clone()))
            .collect();
        t.sort();
        t
    };
    let trip = int(f, lp, "trip")?;
    for (slot, name) in &tapes {
        f.emit(format!("{} {name}[{trip}];", head.elems[*slot]))
    }
    let counter = f
        .count_names()
        .iter()
        .find(|c| c.0 == pos)
        .map(|c| c.1.clone());
    if let Some(c) = &counter {
        f.declare_var(
            Name::new(c.clone(), NameKind::Counter),
            "Int".to_string(),
            Some("0"),
        )
    }
    Ok(LoopLocals {
        tapes,
        counter,
        planed,
    })
}

/// Writes the `for` header and returns the index variable's name.
fn emit_header(f: &mut Frame, head: &LoopHead) -> Result<String, Refusal> {
    let (ctx, lp) = (f.ctx(), head.lp);
    let index = f
        .names
        .index(head.pos, &ident(&f.field_text(head.index_h, "name")?))
        .text;
    let count = match door::field_refs(ctx, lp, "count")?
        .unwrap_or_default()
        .first()
    {
        Some(&c) => Some(f.arg(c)?),
        None => None,
    };
    let header = for_header(
        &index,
        int(f, lp, "start")?,
        int(f, lp, "stop")?,
        int(f, lp, "step")?,
        int(f, lp, "trip")?,
        door::field_flag(f.ctx(), lp, "count_tail")?.unwrap_or(false),
        count.as_deref(),
    );
    f.emit(header.text());
    Ok(index)
}

/// Opens the body's scope (the body's nodes, the index and the carries take slots beyond the flat
/// table), stores each tape's first element and binds the carries' texts to their locals.
fn open_trip(
    f: &mut Frame,
    head: &LoopHead,
    locals: &LoopLocals,
    index: &str,
) -> Result<super::frame::scopes::LoopScope, Refusal> {
    let scope = f.open_loop(head.index_h, &head.carries, &head.body)?;
    let base = scope.base;
    f.set_text(base, index.to_string());
    for (slot, name) in &locals.tapes {
        f.emit(format!(
            "{name}[{}] = {};",
            ordinal(index, head.start, head.step),
            head.names[*slot]
        ))
    }
    for (j, c) in head.carries.iter().enumerate() {
        let s = f.slot_of(*c);
        f.set_name_text(s, head.names[j].clone());
        if head.exit_cond.is_none() {
            f.mutable_carries.insert(*c);
        }
    }
    Ok(scope)
}

/// The fan-out of the body's slots, counting the carried results and the break values as readers too; the scope's outside
/// set is settled first.
pub(crate) fn body_fanout(
    f: &mut Frame,
    head: &LoopHead,
    base: usize,
) -> Result<Vec<usize>, Refusal> {
    let ctx = f.ctx();
    let mut roots: Vec<u32> = head
        .nexts
        .iter()
        .chain(head.exits.iter())
        .copied()
        .collect();
    for &sink in &head.body_sinks {
        match super::stores::storeplan::accumulate_links(f, sink)? {
            Some(links) => {
                for (key, element) in links {
                    roots.extend([key, element])
                }
            }
            None => roots.extend(door::operands(ctx, sink)?),
        }
    }
    super::dispatch::scope_outside(f, base, &roots)?;
    let mut sf = vec![0usize; f.count() - base];
    for &h in head
        .body
        .iter()
        .chain(head.nexts.iter())
        .chain(head.exits.iter())
    {
        for o in door::operands(ctx, h)? {
            let q = f.pos_of(o)?;
            if q >= base {
                sf[q - base] += 1
            }
        }
    }
    Ok(sf)
}

/// The body of one trip by the form of the break: a head or middle break renders the part before
/// it, tests, moves the carries and breaks; the rest renders whole.
fn render_trip_body(
    f: &mut Frame,
    head: &LoopHead,
    locals: &LoopLocals,
    base: usize,
    sf: &[usize],
) -> Result<(), Refusal> {
    let (pos, body, form) = (head.pos, &head.body, head.form);
    if form == "head" || form == "middle" {
        let pre = exit_part(f, body, &head.exits)?;
        render_scope(f, head, base, sf, Some(&pre), &[])?;
        let line = format!("if ({}) {{", f.arg(head.exit_cond.unwrap())?);
        f.emit(line);
        f.buf.indent.push_str("    ");
        let moved: Vec<usize> = (0..head.carries.len())
            .filter(|&j| head.exit_values[j] != head.carries[j])
            .collect();
        for &j in &moved {
            let value = f.arg(head.exit_values[j])?;
            f.declare_as(
                f.names.exit_value(pos, j),
                head.elems[j].clone(),
                Some(&value),
            )
        }
        for &j in &moved {
            f.emit(format!(
                "{} = {};",
                head.names[j],
                f.names.exit_value(pos, j).text
            ))
        }
        if let (Some(c), "middle") = (&locals.counter, form) {
            f.emit(format!("++{c};"))
        }
        f.emit("break;".into());
        f.buf.indent.truncate(f.buf.indent.len() - 4);
        f.emit("}".into());
        render_scope(f, head, base, sf, None, &pre)?;
    } else {
        render_scope(f, head, base, sf, None, &[])?
    }
    Ok(())
}

/// The results of a trip: each changed carry's next value, the tail break's flag, the carries'
/// updates, the counter and the tail break.
fn emit_results(f: &mut Frame, head: &LoopHead, locals: &LoopLocals) -> Result<(), Refusal> {
    let pos = head.pos;
    let mut results: Vec<Option<String>> = vec![];
    for j in 0..head.carries.len() {
        if head.nexts[j] == head.carries[j] {
            results.push(None);
            continue;
        }
        let next = f.names.next(pos, j);
        let r = next.text.clone();
        let value = f.arg(head.nexts[j])?;
        if locals.planed[j] {
            // the pass admits only chains stored in place: the next value is the proxy itself
            if value != head.names[j] {
                return Err(nib(
                    "a plane-carried slot whose next value is not stored in place",
                ));
            }
            results.push(None);
            continue;
        }
        f.declare_as(next, head.elems[j].clone(), Some(&value));
        results.push(Some(r));
    }
    if head.form == "tail" {
        let value = f.arg(head.exit_cond.unwrap())?;
        f.declare_as(f.names.exit_flag(pos), "bool".to_string(), Some(&value))
    }
    for (j, r) in results.iter().enumerate() {
        if let Some(r) = r {
            f.emit(format!("{} = {r};", head.names[j]))
        }
    }
    if let Some(c) = &locals.counter {
        f.emit(format!("++{c};"))
    }
    if head.form == "tail" {
        f.emit(format!("if ({}) break;", f.names.exit_flag(pos).text))
    }
    Ok(())
}
