//! Dispatch: a choice among branches by a clamped selector. The predicated form evaluates every
//! branch and chooses by a chain;
//! the switch form renders each branch inside its own case, with what every case needs computed
//! once above the switch.
use super::frame::disp::ScopeKind;
use super::stores::commits::Plan;
use super::{name_value, node_out, render_pos, Frame, Name, NameKind, Node, Out};
use crate::body::frame::needs::Need;
use crate::body::values::literal_of;
use crate::body::Class;
use crate::ir::analysis::switch_form;
use crate::ir::door::Scalar;
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, Refusal};
use crate::spell::{self};

/// The positions some path from a sink reaches without following the branches of a switch
/// dispatch (its selector is followed).
pub fn outside(
    ctx: *mut RxCtx,
    nodes: &[Node],
    opp: &[Vec<usize>],
    sinks: &[u32],
    walk: u32,
) -> Result<Vec<bool>, Refusal> {
    let mut form = vec![false; nodes.len()];
    for (p, n) in nodes.iter().enumerate() {
        if door::is_a(ctx, n.h, "dispatch")? {
            form[p] = switch_form(ctx, n.h)?
        }
    }
    if !form.contains(&true) {
        return Ok(vec![true; nodes.len()]);
    }
    let (mut seen, mut stack) = (
        vec![false; nodes.len()],
        sinks
            .iter()
            .map(|&s| door::position(ctx, walk, s))
            .collect::<Result<Vec<_>, _>>()?,
    );
    while let Some(p) = stack.pop() {
        if seen[p] {
            continue;
        }
        seen[p] = true;
        if form[p] {
            stack.extend(opp[p].first())
        } else {
            stack.extend(opp[p].iter())
        }
    }
    Ok(seen)
}

/// The body slots (from `base`) some path from the loop's roots reaches without following the
/// branches of a switch dispatch:
/// the rest are rendered inside the cases, and the scope skips them. Left alone when the body
/// holds no switch.
pub fn scope_outside(f: &mut Frame, base: usize, roots: &[u32]) -> Result<(), Refusal> {
    let mut form = vec![false; f.count()];
    for (s, is_switch) in form.iter_mut().enumerate().skip(base) {
        if door::is_a(f.ctx(), f.node(s).h, "dispatch")? {
            *is_switch = switch_form(f.ctx(), f.node(s).h)?
        }
    }
    if !form.contains(&true) {
        return Ok(());
    }
    let mut seen = vec![false; f.count()];
    let mut stack = roots
        .iter()
        .map(|&h| f.pos_of(h))
        .collect::<Result<Vec<_>, _>>()?;
    while let Some(s) = stack.pop() {
        if s < base || seen[s] {
            continue;
        }
        seen[s] = true;
        if form[s] {
            stack.extend(f.opp(s).first())
        } else {
            stack.extend(f.opp(s).iter())
        }
    }
    // the index and the carries are bound by the loop header, whatever reads them
    for (s, &reached) in seen.iter().enumerate().skip(base) {
        let bound = matches!(f.node(s).class, Class::LoopIndex | Class::LoopCarry);
        f.set_outside(s, reached || bound)
    }
    Ok(())
}

/// Whether the selector is an integer literal in `0..k`, or a select between such values.
fn in_range(f: &Frame, q: usize, k: usize) -> Result<bool, Refusal> {
    let n = &f.node(q);
    if n.class == Class::Const {
        return Ok(
            matches!(literal_of(f.ctx(), n.h)?, Scalar::Int(v) if v >= 0 && (v as usize) < k),
        );
    }
    if n.class == Class::Select || (n.class == Class::Op && f.field_text(n.h, "kind")? == "select")
    {
        for &o in &f.opp(q)[1..] {
            if !in_range(f, o, k)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    Ok(false)
}

/// The selector clamped to the branches, bound once; its name.
fn selector(f: &mut Frame, p: usize, tag: &str) -> Result<String, Refusal> {
    let (k, kname, sel) = (f.opp(p).len() - 1, f.names.selector(tag).text, f.opp(p)[0]);
    let text = f.text(sel).clone();
    let spelled = if in_range(f, sel, k)? {
        format!("static_cast<std::int32_t>({text})")
    } else {
        format!(
            "({text} < 0 ? 0 : ({text} > {} ? {} : static_cast<std::int32_t>({text})))",
            k - 1,
            k - 1
        )
    };
    f.declare_as(
        Name::new(kname.clone(), NameKind::Local),
        "std::int32_t".to_string(),
        Some(&spelled),
    );
    Ok(kname)
}

/// The suffix a dispatch's emitted names carry: its canonical position, or the next temporary
/// number for a dispatch inside a loop body (which has no position of its own).
fn tag_of(f: &mut Frame, p: usize) -> String {
    if f.is_slot(f.node(p).h) {
        f.names.body_tag()
    } else {
        p.to_string()
    }
}

/// A dispatch node's text: the switch form is written as a switch statement, the predicated form
/// as a chain.
pub fn stmt(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    if switch_form(f.ctx(), f.node(p).h)? {
        return Ok(Out::Value(switch(f, &[p])?.remove(0), None));
    }
    let tag = tag_of(f, p);
    let kname = selector(f, p, &tag)?;
    let k = f.opp(p).len() - 1;
    let texts: Vec<String> = f.opp(p)[1..].iter().map(|&q| f.text(q).clone()).collect();
    let mut expr = texts[k - 1].clone();
    for j in (0..k - 1).rev() {
        expr = if f.node(p).rank() == 0 {
            format!("({kname} == {j} ? {} : {expr})", texts[j])
        } else {
            let cond = f.hoist(format!(
                "aether::constant<aether::extents<>>({kname} == {j})"
            ));
            format!("aether::select({cond}, {}, {expr})", texts[j])
        };
    }
    Ok(Out::Value(expr, None))
}

/// The C++ element type of a dtype.
fn element(dtype: &str) -> Result<&'static str, Refusal> {
    match dtype {
        "f64" | "f32" => Ok("Real"),
        "i32" | "i64" => Ok("Int"),
        "bool" => Ok("bool"),
        d => Err(nib(format!("element type {d}"))),
    }
}

/// The literal zero (or a splat of one): a branch the switch need not render.
fn is_zero(f: &Frame, q: usize) -> Result<bool, Refusal> {
    let n = &f.node(q);
    if n.class == Class::Const {
        return Ok(match literal_of(f.ctx(), n.h)? {
            Scalar::Flag(held) => !held,
            Scalar::Int(v) => v == 0,
            Scalar::Real(v) => v == 0.0,
            _ => false,
        });
    }
    if n.class == Class::Op && f.field_text(n.h, "kind")? == "splat" && f.opp(q).len() == 1 {
        return is_zero(f, f.opp(q)[0]);
    }
    Ok(false)
}

/// The C++ type of a member's output local and the typed zero its elided branches write.
fn member_types(f: &Frame, p: usize) -> Result<(String, String), Refusal> {
    let (dtype, shape) = (&f.node(p).ty.dtype, &f.node(p).ty.shape);
    let elem = element(dtype)?;
    if shape.is_empty() {
        return Ok((elem.to_string(), format!("static_cast<{elem}>(0)")));
    }
    let extents = shape
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Ok((
        format!("aether::Item<{elem}, {extents}>"),
        format!("aether::constant<aether::extents<{extents}>>(static_cast<{elem}>(0))"),
    ))
}

/// Whether a dispatch node is one of the two kinds the switch form writes: a declared dispatch,
/// or an automatic one (its kind name is compared as the door states it).
fn declared_switch(f: &Frame, m: usize) -> Result<bool, Refusal> {
    let h = f.node(m).h;
    if door::is_a(f.ctx(), h, "auto_dispatch")? {
        return Ok(true);
    }
    Ok(door::is_a(f.ctx(), h, "dispatch")?
        && !door::is_a(f.ctx(), h, "predicated_dispatch")?
        && !door::is_a(f.ctx(), h, "segmented_dispatch")?)
}

/// One switch for a group of dispatches that share a selector and branch count (a lone dispatch
/// is a group of one).
/// Each case renders every member's branch in one scope; a member with an elided branch writes
/// its typed zero in
/// `default:`. Returns each member's output name.
fn switch(f: &mut Frame, members: &[usize]) -> Result<Vec<String>, Refusal> {
    for &m in members {
        if !declared_switch(f, m)? {
            return Err(nib("a dispatch of a strategy kind"));
        }
    }
    let k = f.opp(members[0]).len() - 1;
    let tags: Vec<String> = members.iter().map(|&m| tag_of(f, m)).collect();
    let kname = selector(f, members[0], &tags[0])?;
    let names: Vec<String> = tags.iter().map(|t| f.names.switch(t).text).collect();
    let mut elided: Vec<Vec<bool>> = vec![];
    for &m in members {
        elided.push(
            f.opp(m)[1..]
                .to_vec()
                .iter()
                .map(|&b| is_zero(f, b))
                .collect::<Result<_, _>>()?,
        )
    }
    let plain = elided.iter().any(|e| !e.contains(&true));
    let last = k - 1;
    let types = members
        .iter()
        .map(|&m| member_types(f, m))
        .collect::<Result<Vec<_>, _>>()?;
    let active: Vec<Vec<usize>> = (0..k)
        .map(|j| (0..members.len()).filter(|&i| !elided[i][j]).collect())
        .collect();
    let roots: Vec<Vec<usize>> = active
        .iter()
        .enumerate()
        .map(|(j, a)| a.iter().map(|&i| f.opp(members[i])[1 + j]).collect())
        .collect();
    for &m in members {
        if let Some(mix) = mix_plane(f, m)? {
            f.need(Need::MixCount);
            f.emit(format!("hawk_abi::mix_count({mix}, {k}, {kname});"))
        }
    }
    hoist_common(f, &roots)?;
    for (name, ty) in names.iter().zip(&types) {
        f.emit(format!("{} {name};", ty.0))
    }
    f.emit(format!("switch ({kname}) {{"));
    let outer = f.buf.indent.clone();
    f.buf.indent.push_str("    ");
    for (j, branch) in active.iter().enumerate().take(k) {
        if branch.is_empty() || (j == last && plain) {
            continue;
        }
        f.emit(format!("case {j}: {{"));
        case_body(f, &names, &types, members, &elided, j)?;
        f.emit("break;".into());
        f.emit("}".into());
    }
    if plain {
        f.emit("default: {".into());
        case_body(f, &names, &types, members, &elided, last)?;
        f.emit("}".into());
    } else {
        for (n, (name, ty)) in names.iter().zip(&types).enumerate() {
            f.emit(format!(
                "{}{name} = {};",
                if n == 0 { "default: " } else { "" },
                ty.1
            ))
        }
    }
    f.buf.indent = outer;
    f.emit("}".into());
    Ok(names)
}

/// One case's own scope: what it names does not leak to the next case.
fn case_body(
    f: &mut Frame,
    names: &[String],
    types: &[(String, String)],
    members: &[usize],
    elided: &[Vec<bool>],
    j: usize,
) -> Result<(), Refusal> {
    f.in_case(|f| {
        f.buf.indent.push_str("    ");
        let live: Vec<usize> = (0..members.len()).filter(|&i| !elided[i][j]).collect();
        let roots: Vec<usize> = live.iter().map(|&i| f.opp(members[i])[1 + j]).collect();
        render_branch(f, &roots)?;
        for (&i, &root) in live.iter().zip(&roots) {
            f.emit(format!("{} = {};", names[i], f.text(root)))
        }
        for i in (0..members.len()).filter(|&i| elided[i][j]) {
            f.emit(format!("{} = {};", names[i], types[i].1))
        }
        Ok(())
    })
}

/// Post-order over the roots' dependencies, leaving out what the enclosing scopes already bind.
/// A nested switch contributes its selector.
fn branch_order(f: &Frame, roots: &[usize]) -> Result<Vec<usize>, Refusal> {
    let (mut seen, mut order) = (vec![false; f.count()], vec![]);
    let mut stack: Vec<(usize, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    while let Some((n, expanded)) = stack.pop() {
        if seen[n] || f.is_scoped(n) || f.is_outside(n) {
            continue;
        }
        if !expanded {
            stack.push((n, true));
            let nested =
                door::is_a(f.ctx(), f.node(n).h, "dispatch")? && switch_form(f.ctx(), f.node(n).h)?;
            let children = if nested {
                &f.opp(n)[..1]
            } else {
                &f.opp(n)[..]
            };
            for &c in children.iter().rev() {
                stack.push((c, false))
            }
            continue;
        }
        seen[n] = true;
        order.push(n);
    }
    Ok(order)
}

/// Counts the reads of each operand by position: equal values are one value, bound once when two
/// readers use it.
fn tally_operands(f: &Frame, p: usize, tally: &mut [usize]) {
    for &q in f.opp(p) {
        tally[q] += 1
    }
}

/// Renders the nodes one branch of a switch case needs, naming those the case reads twice.
fn render_branch(f: &mut Frame, roots: &[usize]) -> Result<(), Refusal> {
    let order = branch_order(f, roots)?;
    let mut tally = vec![0usize; f.count()];
    for &p in &order {
        tally_operands(f, p, &mut tally)
    }
    for &r in roots {
        tally[r] += 1
    }
    render_nodes(f, &order, &tally)
}

/// A node that may run in a case that does not need it: it cannot fault and has no effect.
fn speculable(f: &Frame, p: usize) -> Result<bool, Refusal> {
    let n = &f.node(p);
    if n.class == Class::Loop {
        return Ok(false);
    }
    let kind = f.field_text(n.h, "kind").unwrap_or_default();
    if matches!(
        kind.as_str(),
        "floordiv" | "mod" | "set_component_at" | "add_at"
    ) {
        return Ok(false);
    }
    if n.class == Class::At {
        return Ok(n.ops.len() == 2
            && f.node(f.opp(p)[1]).class == Class::RoleConst
            && f.field_text(f.node(f.opp(p)[1]).h, "label")? == "param_cell");
    }
    if kind == "component_at" {
        let (v, i) = (f.opp(p)[0], f.opp(p)[1]);
        let idx = if f.node(i).class == Class::Const {
            match literal_of(f.ctx(), f.node(i).h)? {
                Scalar::Int(v) => Some(v),
                _ => None,
            }
        } else {
            None
        };
        return Ok(idx.is_some_and(|x| {
            x >= 0 && (x as usize) < f.node(v).ty.shape.first().copied().unwrap_or(0)
        }));
    }
    Ok(true)
}

/// Renders above the switch every node all cases reach, or with three or more cases all but
/// one, when it is safe to run anywhere.
/// A case with nothing to render makes any hoist speculative: then nothing is hoisted.
fn hoist_common(f: &mut Frame, case_roots: &[Vec<usize>]) -> Result<(), Refusal> {
    if case_roots.len() < 2 || case_roots.iter().any(|r| r.is_empty()) {
        return Ok(());
    }
    let orders = case_roots
        .iter()
        .map(|r| branch_order(f, r))
        .collect::<Result<Vec<_>, _>>()?;
    let n = orders.len();
    let mut count = vec![0usize; f.count()];
    for o in &orders {
        for &x in o {
            count[x] += 1
        }
    }
    let floor = if n >= 3 { n - 1 } else { n };
    let (mut hoisted, mut seen) = (vec![], vec![false; f.count()]);
    for o in &orders {
        for &x in o {
            if count[x] >= floor && !seen[x] {
                seen[x] = true;
                hoisted.push(x)
            }
        }
    }
    let (mut dropped, mut kept) = (vec![false; f.count()], vec![]);
    for &x in &hoisted {
        if count[x] < n && (!speculable(f, x)? || f.opp(x).iter().any(|&o| dropped[o])) {
            dropped[x] = true
        } else {
            kept.push(x)
        }
    }
    if kept.is_empty() {
        return Ok(());
    }
    let mut inside = vec![false; f.count()];
    for &x in &kept {
        inside[x] = true
    }
    let mut tally = vec![0usize; f.count()];
    for &x in &kept {
        tally_operands(f, x, &mut tally)
    }
    for (o, roots) in orders.iter().zip(case_roots) {
        for &x in o {
            if !inside[x] {
                tally_operands(f, x, &mut tally)
            }
        }
        for &r in roots {
            if inside[r] {
                tally[r] += 1
            }
        }
    }
    render_nodes(f, &kept, &tally)
}

/// Renders the nodes into the open scope, naming one the tally says is read twice.
fn render_nodes(f: &mut Frame, order: &[usize], tally: &[usize]) -> Result<(), Refusal> {
    for &p in order {
        let class = f.node(p).class.clone();
        if matches!(
            class,
            Class::Assign | Class::SampleSum | Class::AccumWrite | Class::WideWrite
        ) {
            return Err(nib(format!("a {class} inside a switch case")));
        }
        if let Out::Value(text, role) = node_out(f, p)? {
            let leaf_valued = f.leaf_valued(p)?;
            let named = f.naming(p, ScopeKind::Case, role.as_deref(), tally[p])?;
            if named {
                let name = f.names.body_temp();
                let ty = f.node(p).ty.clone();
                let held = tally[p] >= 2 && !leaf_valued;
                f.declare(name.clone(), &ty.dtype, &ty.shape, &text, held)?;
                f.set_text(p, name.text);
            } else {
                f.set_text(p, text)
            }
        }
        f.set_scoped(p, true);
    }
    Ok(())
}

/// The fused groups of the switch dispatches that share one selector position and operand count.
struct SelectorGroups {
    /// The selector's position and the number of operands (selector and branches).
    key: (usize, usize),
    /// The groups so far, each a list of dispatch positions in walk order.
    groups: Vec<Vec<usize>>,
}

/// The group of switch dispatches, in walk order, that position `p` renders with (the greedy
/// grouping of dispatches
/// sharing a selector and branch count, none reading a member), or none when it renders alone.
pub fn group_of(f: &Frame, p: usize) -> Result<Vec<usize>, Refusal> {
    if !door::is_a(f.ctx(), f.node(p).h, "dispatch")?
        || !f.is_outside(p)
        || f.is_slot(f.node(p).h)
        || !switch_form(f.ctx(), f.node(p).h)?
    {
        return Ok(vec![]);
    }
    let mut open: Vec<SelectorGroups> = vec![];
    for q in 0..f.count() {
        if !door::is_a(f.ctx(), f.node(q).h, "dispatch")?
            || !f.is_outside(q)
            || f.is_slot(f.node(q).h)
            || !switch_form(f.ctx(), f.node(q).h)?
        {
            continue;
        }
        // dispatches share a group when their selectors are the same value (one position)
        let key = (f.opp(q)[0], f.opp(q).len());
        let at = match open.iter().position(|g| g.key == key) {
            Some(i) => i,
            None => {
                open.push(SelectorGroups {
                    key,
                    groups: vec![],
                });
                open.len() - 1
            }
        };
        let deps = if open[at].groups.is_empty() {
            vec![]
        } else {
            closure(f, q)
        };
        match open[at]
            .groups
            .iter()
            .position(|g| !g.iter().any(|&m| deps[m]))
        {
            Some(i) => open[at].groups[i].push(q),
            None => open[at].groups.push(vec![q]),
        }
    }
    Ok(open
        .into_iter()
        .flat_map(|g| g.groups)
        .find(|g| g.len() > 1 && g.contains(&p))
        .unwrap_or_default())
}

/// Every position a node reads, through all operands (branches included), the node itself too.
fn closure(f: &Frame, root: usize) -> Vec<bool> {
    let (mut seen, mut stack) = (vec![false; f.count()], vec![root]);
    while let Some(n) = stack.pop() {
        if !seen[n] {
            seen[n] = true;
            stack.extend(f.opp(n).iter())
        }
    }
    seen
}

/// A fused group at its first member: what a later member's branches read that the walk has not
/// reached is rendered
/// first, then one switch for all, then each member is bound to its output.
pub fn emit_group(f: &mut Frame, plan: &mut Plan, group: &[usize]) -> Result<(), Refusal> {
    let (mut seen, mut stack, mut need) = (vec![false; f.count()], vec![], vec![]);
    for &m in group {
        stack.extend(f.opp(m)[1..].iter())
    }
    while let Some(n) = stack.pop() {
        if seen[n] || group.contains(&n) {
            continue;
        }
        seen[n] = true;
        if f.is_done(n) {
            continue;
        }
        if f.is_outside(n) {
            need.push(n)
        }
        stack.extend(f.opp(n).iter());
    }
    need.sort();
    for n in need {
        render_pos(f, plan, n)?
    }
    let names = switch(f, group)?;
    for (&m, name) in group.iter().zip(names) {
        f.set_done(m);
        name_value(f, m, name, None)?;
    }
    Ok(())
}

/// The plane an `auto` dispatch counts the branch it picks into (`lut_dispatch<n>_mix`, n the
/// dispatch's place among the walk's
/// dispatches, those inside loop bodies last), or none for a declared strategy.
fn mix_plane(f: &Frame, p: usize) -> Result<Option<String>, Refusal> {
    if !door::is_a(f.ctx(), f.node(p).h, "auto_dispatch")? {
        return Ok(None);
    }
    let Some(index) = door::dispatch_nodes(f.ctx(), f.walk())?
        .iter()
        .position(|&h| h == f.node(p).h)
    else {
        return Ok(None);
    };
    let name = crate::ir::runtime_names::mix_plane(&format!("dispatch{index}"));
    Ok(Some(spell::binding_name("lookup", &name)?))
}
