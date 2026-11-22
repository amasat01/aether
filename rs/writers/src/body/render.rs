//! The render of one body as a pipeline of named steps: the unit's refusals, the sinks, the pushed
//! reads, the walk's
//! rows, the guard, the sink facts (companion planes, the block fold, the plane reads and the
//! early stores), the frame, the
//! plans (stores, placement, compensated target, commits), the zero lines, the walk of the
//! positions, the commit of the
//! sinks and the guard's close. The steps run in this order; the text depends on it.
use super::*;
use crate::refuse::{nib, Refusal};

/// The node table of a walk before a frame owns it: nodes in walk order, the operand positions of
/// each, how many readers each has, who they are and the clamped spans.
struct WalkRows {
    /// The nodes, in walk order.
    nodes: Vec<Node>,
    /// The position of each operand of each node.
    opp: Vec<Vec<usize>>,
    /// How many positions read each position (a primitive is not a reference).
    fanout: Vec<usize>,
    /// The readers of each position: (position, operand index).
    users: Vec<Vec<(usize, usize)>>,
    /// The clamped index span of each position, where it has one.
    spans: Vec<Option<Span>>,
}

/// The guard of the kernel: the lines that open it, the indent of what it holds and the condition
/// when there is one.
struct GuardPlan {
    /// The lines that open the body (the `if` of a guard).
    head: Vec<String>,
    /// The indent of the body's statements.
    indent: String,
    /// The condition under which a lane commits; `None` for an unguarded kernel.
    cond: Option<String>,
}

/// What the render plans beside the frame: the commit plan, the early stores and the correction
/// planes. The frame is built
/// before the plans (they read it), so they are passed next to it and not owned by it.
struct Plans {
    /// The commit plan: what is due, skipped, held and zeroed.
    commits: stores::commits::Plan,
    /// The own-column assigns stored right after their value renders: (position of the value, sink
    /// handle).
    early: Vec<(usize, u32)>,
    /// The correction planes of the compensated accumulates.
    companions: Vec<String>,
}

/// The position of node `h` in the walk.
fn at(ctx: *mut RxCtx, walk: u32, h: u32) -> Result<usize, Refusal> {
    walk_position(ctx, walk, h)
}

/// `render_frame` over the given sinks when there are any (and then with no refusal of a box or a
/// single step), else over the contract's own: the whole pipeline.
pub(super) fn render_frame_over<'e>(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
    cut: &[u32],
    emit: Option<&'e mut dyn LoopEmitter>,
    given_sinks: Option<&[u32]>,
) -> Result<(Frame<'e>, Option<String>), Refusal> {
    let one_trip = emit.as_ref().is_some_and(|e| e.one_trip());
    check_unit(walk, unit, cut, one_trip, given_sinks)?;
    let (guard_names, _) = guard_and_policy(unit);
    let sinks: Vec<u32> = given_sinks.map_or_else(|| unit.sinks.clone(), <[u32]>::to_vec);
    // runtime-index reads of element-wise vectors move to their operands before the walk is read
    let (walk, sinks) = stores::pushed::push_reads(ctx, walk, &sinks)?;
    let rows = load_rows(ctx, walk)?;
    let guard = plan_guard(ctx, walk, &guard_names, indent)?;
    let companions = companion_planes(ctx, walk, &rows.nodes, &sinks)?;
    let block_fold = block_fold_for(ctx, walk, unit, &rows.nodes, &sinks)?;
    // own-column assigns store right after their value renders, unless the body reads that plane
    let mut read = top_level_reads(ctx, &rows.nodes)?;
    let mut early = early_stores(ctx, walk, &rows, &sinks, &companions, &read)?;
    let count = rows.nodes.len();
    let outside = dispatch::outside(ctx, &rows.nodes, &rows.opp, &sinks, walk)?;
    let forced = forced_reads(ctx, &rows.nodes, &rows.opp)?;
    let mut f = new_frame(
        ctx,
        walk,
        unit,
        cut,
        emit,
        (rows, outside, forced),
        &guard,
        sinks,
        block_fold,
    );
    f.stores = stores::storeplan::Plan::build(&f)?;
    if f.placed {
        f.absorbed = placed::written(&f)?
    }
    add_loop_body_reads(&f, &mut read)?;
    plan_compensated(&mut f, unit, &mut early)?;
    let commits = commit_plan(&f, cut, &read, count)?;
    if !cut.is_empty() {
        early.clear()
    }
    let mut plans = Plans {
        commits,
        early,
        companions,
    };
    emit_zero_lines(&mut f, &mut plans);
    plans
        .early
        .retain(|e| !plans.commits.held.contains(&e.1) && !plans.commits.skip[e.0]);
    f.settle_disp(0, count, &plans.commits.skip);
    walk_positions(&mut f, &mut plans, count)?;
    commit_sinks(&mut f, &plans)?;
    if guard.cond.is_some() {
        let close = format!("{}}}", &f.buf.indent[4..]);
        f.buf.lines.push(close);
        f.buf.reads.push(Default::default())
    }
    Ok((f, guard.cond))
}

/// Refuses the units this render does not write: no walk, a fused
/// kernel without given sinks, and a boxed or two-path unit whose wrapper composes the body.
fn check_unit(
    walk: u32,
    unit: &Unit,
    cut: &[u32],
    split: bool,
    given_sinks: Option<&[u32]>,
) -> Result<(), Refusal> {
    if walk == u32::MAX {
        return Err(nib("a unit with no walk (team stage)"));
    }
    match &unit.form {
        Form::Fused(_) if given_sinks.is_none() => return Err(nib("kernel record FusedKernel")),
        _ => {}
    }
    // a boxed unit's body is the wrapper's: a boxed and a plain render side by side (the wrapper
    // owns it)
    if given_sinks.is_none() && !split && unit.thread_box.is_some() {
        return Err(nib("a boxed unit (the wrapper composes its body)"));
    }
    // a two-path unit's body is the wrapper's too: the fused step and the single step, each in its
    // own guarded frame
    if given_sinks.is_none() && !split && cut.is_empty() && unit.one_step.is_some() {
        return Err(nib("a two-path unit (the wrapper composes its body)"));
    }
    Ok(())
}

/// The walk's nodes with their operand positions, fan-out, readers and clamped spans.
fn load_rows(ctx: *mut RxCtx, walk: u32) -> Result<WalkRows, Refusal> {
    let order = door::walk_order(ctx, walk)?;
    let mut nodes = Vec::with_capacity(order.len());
    for &h in &order {
        let class = Class::of(ctx, h)?;
        nodes.push(Node {
            h,
            kind: crate::body::kind_of(ctx, h, &class)?,
            class,
            ty: crate::body::ty_of(ctx, h)?,
            ops: door::operands(ctx, h)?,
        })
    }
    let mut opp: Vec<Vec<usize>> = vec![];
    let mut fanout = vec![0usize; nodes.len()];
    let mut users: Vec<Vec<(usize, usize)>> = vec![vec![]; nodes.len()];
    for (u, n) in nodes.iter().enumerate() {
        let ps = n
            .ops
            .iter()
            .map(|&o| at(ctx, walk, o))
            .collect::<Result<Vec<_>, _>>()?;
        // a primitive is not a reference and not a referent: its forward expression reads what it
        // reads
        if n.class != Class::Primitive {
            for (k, &p) in ps.iter().enumerate() {
                fanout[p] += 1;
                users[p].push((u, k))
            }
        }
        opp.push(ps);
    }
    let spans = ops::clamp_spans(ctx, &nodes, &opp)?;
    Ok(WalkRows {
        nodes,
        opp,
        fanout,
        users,
        spans,
    })
}

/// The guard: each mask the default kind names counts only if the walk binds it in the
/// `terminated` role; the commit is
/// skipped when any is set. Opens an `if` and indents the body when there is one.
fn plan_guard(
    ctx: *mut RxCtx,
    walk: u32,
    guard_names: &[String],
    indent: &str,
) -> Result<GuardPlan, Refusal> {
    let mut indent = indent.to_string();
    let mut head: Vec<String> = vec![];
    let mut bound_names = vec![];
    for m in guard_names {
        if door::walk_binds(ctx, walk, "terminated", m)? {
            bound_names.push(m);
        }
    }
    let bound: Vec<String> = bound_names
        .into_iter()
        .map(|m| {
            format!(
                "{}[i].eval()",
                binding_name("terminated", 0, m).unwrap_or_default()
            )
        })
        .collect();
    let mut cond: Option<String> = None;
    if !bound.is_empty() {
        let text = if bound.len() == 1 {
            bound[0].clone()
        } else {
            format!("({})", bound.join(" || "))
        };
        head.push(format!("{indent}if (!{text}) {{"));
        indent.push_str("    ");
        cond = Some(text);
    }
    Ok(GuardPlan { head, indent, cond })
}

/// Whether the plain frame turns the block fold on: no sample sum among the sinks, and the unit's
/// own conditions (see
/// `block_fold_of`).
fn block_fold_for(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    nodes: &[Node],
    sinks: &[u32],
) -> Result<bool, Refusal> {
    let sums = sinks
        .iter()
        .any(|&q| at(ctx, walk, q).is_ok_and(|p| nodes[p].class == Class::SampleSum));
    block_fold_of(ctx, walk, unit, sums)
}

/// The correction planes of the compensated accumulates among the sinks.
fn companion_planes(
    ctx: *mut RxCtx,
    walk: u32,
    nodes: &[Node],
    sinks: &[u32],
) -> Result<Vec<String>, Refusal> {
    let mut companions: Vec<String> = vec![];
    for &q in sinks {
        let n = &nodes[at(ctx, walk, q)?];
        if n.class == Class::AccumUpdate && door::compensates(ctx, n.h)? {
            companions.push(format!("{}_c", field_text(ctx, n.h, "name")?))
        }
    }
    Ok(companions)
}

/// The (role, name) of every leaf among the top-level nodes: the planes the body reads outside
/// loop bodies.
fn top_level_reads(ctx: *mut RxCtx, nodes: &[Node]) -> Result<Vec<(String, String)>, Refusal> {
    nodes
        .iter()
        .filter(|n| n.class == Class::Leaf)
        .map(|n| Ok((field_text(ctx, n.h, "role")?, field_text(ctx, n.h, "name")?)))
        .collect()
}

/// The own-column assigns that store right after their value renders: not a companion plane, and
/// no leaf of the body (at
/// top level only: this is the first phase of the read list) reads the plane. Each is (position of
/// the value, sink).
fn early_stores(
    ctx: *mut RxCtx,
    walk: u32,
    rows: &WalkRows,
    sinks: &[u32],
    companions: &[String],
    read: &[(String, String)],
) -> Result<Vec<(usize, u32)>, Refusal> {
    let nodes = &rows.nodes;
    let mut early: Vec<(usize, u32)> = vec![];
    for &s in sinks {
        let p = at(ctx, walk, s)?;
        if nodes[p].class == Class::Assign
            && nodes[p].ops.len() == 1
            && !companions.contains(&field_text(ctx, nodes[p].h, "name")?)
            && !read.contains(&(
                field_text(ctx, nodes[p].h, "role")?,
                field_text(ctx, nodes[p].h, "name")?,
            ))
        {
            early.push((rows.opp[p][0], s))
        }
    }
    Ok(early)
}

/// The positions a loop reads from outside: each is named once, before the loop (a plane read
/// through At excepted).
fn forced_reads(ctx: *mut RxCtx, nodes: &[Node], opp: &[Vec<usize>]) -> Result<Vec<bool>, Refusal> {
    let mut forced = vec![false; nodes.len()];
    for (u, n) in nodes.iter().enumerate() {
        if n.class != Class::Loop {
            continue;
        }
        for &q in &opp[u] {
            let gather = nodes[q].class == Class::Leaf
                && matches!(
                    field_text(ctx, nodes[q].h, "role")?.as_str(),
                    "lookup" | "wide_in"
                );
            if !gather {
                forced[q] = true
            }
        }
    }
    Ok(forced)
}

/// The frame of the render: the rows (with what is outside and forced), the line buffer opened by
/// the guard and the unit's settings; no plan is in it yet.
#[allow(clippy::too_many_arguments)]
fn new_frame<'e>(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    cut: &[u32],
    emit: Option<&'e mut dyn LoopEmitter>,
    rows: (WalkRows, Vec<bool>, Vec<bool>),
    guard: &GuardPlan,
    sinks: Vec<u32>,
    block_fold: bool,
) -> Frame<'e> {
    let one_trip = emit.as_ref().is_some_and(|e| e.one_trip());
    let (
        WalkRows {
            nodes,
            opp,
            fanout,
            users,
            spans,
        },
        outside,
        forced,
    ) = rows;
    Frame {
        table: Table::build(
            ctx,
            walk,
            Positions::new(nodes, opp, fanout, users, spans, outside, forced),
            sinks,
        ),
        buf: Buffer::new(guard.head.clone(), guard.indent.clone()),
        block_fold,
        names: Namer::default(),
        needs: Default::default(),
        slot: Default::default(),
        cut: cut.to_vec(),
        loop_emitter: emit,
        sincos: vec![],
        compensated: None,
        placed: unit.target == crate::contract::types::Target::Device
            || (!cut.is_empty() && !one_trip),
        absorbed: Default::default(),
        proxies: vec![],
        stores: Default::default(),
        mutable_carries: Default::default(),
        unit: unit.clone(),
    }
}

/// The second phase of the read list: the plane reads of the whole body, loop bodies included (the
/// commit plan sees them; the early stores did not).
fn add_loop_body_reads(f: &Frame, read: &mut Vec<(String, String)>) -> Result<(), Refusal> {
    let ctx = f.ctx();
    for lp in (0..f.count())
        .filter(|&p| f.node(p).class == Class::Loop)
        .map(|p| f.node(p).h)
        .collect::<Vec<_>>()
    {
        let mut stack = vec![lp];
        while let Some(l) = stack.pop() {
            for h in door::body_nodes(ctx, l)? {
                if door::is_a(ctx, h, "leaf")? {
                    read.push((field_text(ctx, h, "role")?, field_text(ctx, h, "name")?))
                } else if door::is_a(ctx, h, "loop")? {
                    stack.push(h)
                }
            }
        }
    }
    Ok(())
}

/// Under the compensated sink policy: finds the correction plane's target and drops the early
/// stores.
fn plan_compensated(
    f: &mut Frame,
    unit: &Unit,
    early: &mut Vec<(usize, u32)>,
) -> Result<(), Refusal> {
    if let SinkPolicy::Compensated {
        plane: comp,
        output_slot: slot,
        output_names: names,
    } = &unit.kind.sink
    {
        let target = crate::ir::analysis::compensated_target(
            f.ctx(),
            comp,
            slot.as_deref(),
            names,
            f.sinks(),
        )?;
        f.compensated = Some(CompensatedPlan {
            plane: comp.clone(),
            target: target.map(|(name, own_column)| CompensatedTarget { name, own_column }),
        });
        early.clear();
    }
    Ok(())
}

/// The commit plan of the frame; a cut body commits nothing, so its plan is empty.
fn commit_plan(
    f: &Frame,
    cut: &[u32],
    read: &[(String, String)],
    count: usize,
) -> Result<stores::commits::Plan, Refusal> {
    Ok(if cut.is_empty() {
        stores::commits::plan(f, read)?
    } else {
        stores::commits::Plan {
            due: vec![],
            skip: vec![false; count],
            held: vec![],
            zero: vec![],
        }
    })
}

/// The zero lines: the slots an accumulate may reach before their own part is stored, then each
/// plane a loop accumulates into at runtime indices, zeroed whole once before the walk.
fn emit_zero_lines(f: &mut Frame, plans: &mut Plans) {
    for (target, at) in std::mem::take(&mut plans.commits.zero) {
        f.emit(format!(
            "{target}({at}, i.global()) = static_cast<Real>(0);"
        ))
    }
    for (target, width) in f.stores.accumulated.clone() {
        f.emit(format!(
            "for (Int hk = 0; hk < {width}; ++hk) {target}[i].at(hk) = static_cast<Real>(0);"
        ))
    }
}

/// The walk of the positions, each rendered in order with the early stores that follow it; refuses
/// a per-component commit that was planned for a value the walk never rendered.
fn walk_positions(f: &mut Frame, plans: &mut Plans, count: usize) -> Result<(), Refusal> {
    for p in 0..count {
        render_pos(f, &mut plans.commits, p)?;
        for &(_, s) in plans.early.iter().filter(|e| e.0 == p) {
            stores::sinks::store(f, s)?
        }
    }
    if !plans.commits.due.is_empty() {
        return Err(nib(
            "a per-component commit was planned for a value the walk never rendered",
        ));
    }
    Ok(())
}

/// The commit of every sink, in order: team and box stores, tape writes, sample sums, accumulates
/// and plain stores; the
/// finish comes last. A loop sink is emitted with its scope, and a compensated target or a
/// companion plane is skipped.
fn commit_sinks(f: &mut Frame, plans: &Plans) -> Result<(), Refusal> {
    let mut finish: Option<u32> = None;
    for s in f.sinks().to_vec() {
        let p = f.at(s)?;
        let (class, nops, h) = (f.node(p).class.clone(), f.node(p).ops.len(), f.node(p).h);
        if class == Class::Loop {
            continue;
        } // a lowered loop is a root of the walk: emitted with its scope, not a commit
        if matches!(
            class,
            Class::TeamStore | Class::TeamCopy | Class::ScatterLog | Class::LogStore
        ) {
            team::store(f, p)?;
            continue;
        }
        if class == Class::BoxStore {
            stores::sinks::box_store(f, p)?;
            continue;
        }
        if class == Class::WarpStore {
            let planned = plans.early.iter().any(|e| e.1 == s) || plans.commits.held.contains(&s);
            if !planned {
                stores::sinks::warp_store(f, p)?;
            }
            continue;
        }
        if class == Class::StoreUpdate {
            if f.unit.team.is_some() {
                return Err(nib("a wide write of a team stage"));
            }
            if nops >= 2 {
                stores::sinks::indexed_store(f, p)?;
                continue;
            }
        }
        if class == Class::Finish && nops == 1 {
            finish = Some(s);
            continue;
        }
        if class == Class::TapeWrite {
            stores::sinks::tape_write(f, p)?;
            continue;
        }
        if class == Class::SampleSum && nops == 1 {
            stores::sinks::sample_sum(f, s, p)?;
            continue;
        }
        if f.compensated
            .as_ref()
            .is_some_and(|c| c.plane == f.field_text(h, "name").unwrap_or_default())
        {
            continue;
        }
        if plans.companions.contains(&f.field_text(h, "name")?) {
            continue;
        }
        if class == Class::AccumUpdate && nops >= 2 {
            stores::sinks::accum(f, s, p)?;
            continue;
        }
        if !matches!(
            class,
            Class::Assign | Class::AccumUpdate | Class::StoreUpdate
        ) || nops != 1
        {
            return Err(nib(format!("sink {class} with {nops} operands")));
        }
        if !plans.early.iter().any(|e| e.1 == s) && !plans.commits.held.contains(&s) {
            stores::sinks::store(f, s)?
        }
    }
    if let Some(s) = finish {
        stores::sinks::finish(f, s)?
    }
    Ok(())
}
