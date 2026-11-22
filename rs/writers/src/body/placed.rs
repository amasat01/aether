//! Placed walks: values a placement writes into a plane. The walk of an interchanged host unit
//! already holds the placement
//! (a `resident` op over the value and a plane leaf), so the text is the plane's lane proxy, the
//! statements that write the
//! value into it element by element, and no text at all for the values only those statements read.
use super::{Frame, Out, Proxy};
use crate::body::frame::needs::Need;
use crate::body::values::{literal_of, text_field};
use crate::body::Class;
use crate::contract::types::Target;
use crate::ir::analysis::{
    element_kind, gather, is_prior, is_resident, is_tiled, loops_flat, plane_widths, Gather,
};
use crate::ir::door;
use crate::kind::OpKind;
use crate::refuse::{fault, nib, Refusal};
use crate::spell::{binding_name, carry_type, element_at, scalar_op};
use std::collections::{HashMap, HashSet};

/// The key two equal values share: their canonical position, or the handle of a node inside a loop
/// body.
fn key(f: &Frame, h: u32) -> u64 {
    f.at(h).map_or((1u64 << 32) | u64::from(h), |p| p as u64)
}

/// The handles of the values a placement writes element by element (elementwise values and vectors
/// of parts, in the body or
/// inside a loop): the value of a placement and the values only those read, down to a leaf or
/// another placement.
///
/// The indices of gathered reads (the second loop): the element expression of a gathered run
/// spells its own index (`start + k * step`), so the index constants and index arithmetic of the
/// reads it replaces are never needed by name. An index value that is a node of the walk itself
/// is therefore always taken, whoever else reads it (a literal keeps its text for other readers:
/// `node_out`); an index value inside a loop body is taken only when every reader of it has been
/// taken. Asking "has every reader been taken" of walk nodes too would name each index of a
/// rolled read and unroll it: one recorded unit grows from 1,057 to 1,569 lines (512 named
/// indices), so the rule is the one that keeps a gathered read rolled.
pub(super) fn written(f: &Frame) -> Result<HashSet<u32>, Refusal> {
    let ctx = f.ctx();
    let mut scanned: Vec<u32> = f.nodes().map(|n| n.h).collect();
    let loops: Vec<u32> = f
        .nodes()
        .filter(|n| n.class == Class::Loop)
        .map(|n| n.h)
        .collect();
    for h in loops_flat(ctx, &loops)? {
        if !scanned.contains(&h) {
            scanned.push(h)
        }
    }
    let mut users: HashMap<u64, Vec<u32>> = HashMap::new();
    for &node in &scanned {
        if door::class_is(ctx, node, "Op")?
            || door::class_is(ctx, node, "Select")?
            || door::class_is(ctx, node, "Loop")?
        {
            for o in door::operands(ctx, node)? {
                users.entry(key(f, o)).or_default().push(node)
            }
        }
    }
    let (mut seen, mut keep): (HashSet<u32>, HashSet<u32>) = (HashSet::new(), HashSet::new());
    let mut reads: Vec<u32> = vec![];
    let mut stack: Vec<u32> = vec![];
    for &node in &scanned {
        if is_resident(ctx, node)? {
            stack.push(door::operands(ctx, node)?[0])
        }
    }
    while let Some(node) = stack.pop() {
        if seen.contains(&node)
            || !matches!(
                element_kind(ctx, node)?.as_str(),
                "parts" | "splat" | "elementwise"
            )
        {
            continue;
        }
        let mut only_readers = true;
        for &user in users.get(&key(f, node)).map_or(&[][..], |u| &u[..]) {
            let places =
                is_resident(ctx, user)? && key(f, door::operands(ctx, user)?[0]) == key(f, node);
            let chained = seen.contains(&user) && element_kind(ctx, user)? == "elementwise";
            if !(places || chained) {
                only_readers = false;
                break;
            }
        }
        if !only_readers {
            continue;
        }
        seen.insert(node);
        let operands = door::operands(ctx, node)?;
        if let Some(run) = gather(ctx, node)? {
            for &part in &operands {
                if users.get(&key(f, part)).map_or(0, |u| u.len()) == 1 {
                    reads.push(part)
                }
            }
            if let Some(base) = run.base {
                keep.insert(base);
            }
        }
        for &o in &operands {
            if crate::body::ty_of(ctx, o)?.shape.len() == 1 {
                stack.push(o)
            }
        }
    }
    seen.extend(reads.iter().copied());
    for &part in &reads {
        let mut constants = vec![door::operands(ctx, part)?[1]];
        while let Some(node) = constants.pop() {
            // a value with a canonical position is looked up by its handle, which keys no reader:
            // nothing stops it
            let readers_seen = f.at(node).is_ok()
                || users
                    .get(&key(f, node))
                    .is_none_or(|u| u.iter().all(|x| seen.contains(x)));
            if seen.contains(&node)
                || keep.contains(&node)
                || !(door::class_is(ctx, node, "Op")? || door::class_is(ctx, node, "Const")?)
                || !readers_seen
            {
                continue;
            }
            seen.insert(node);
            constants.extend(door::operands(ctx, node)?);
        }
    }
    Ok(seen)
}

/// Whether a value is written element by element and so never rendered whole.
pub(super) fn is_absorbed(f: &Frame, p: usize) -> bool {
    f.placed && f.absorbed.contains(&f.node(p).h)
}

/// The lane proxy of a plane leaf a placement writes: a row of the tile the arm declares, read by
/// the lane's own index.
pub(super) fn plane_leaf(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let (role, name) = (
        text_field(f.ctx(), h, "role")?,
        text_field(f.ctx(), h, "name")?,
    );
    let plane = binding_name(&role, f.node(p).rank(), &name)?;
    if !is_tiled(f.ctx(), h)? {
        // a mutable's own plane is its own proxy: the sample's view of it, with no row to declare
        // (width 0)
        let proxy = format!("{plane}[i]");
        if !f.proxies.iter().any(|x| x.name == proxy) {
            f.proxies.push(Proxy {
                name: proxy.clone(),
                plane,
                width: 0,
            })
        }
        f.set_text(p, proxy);
        return Ok(Out::Bound);
    }
    let proxy = format!("{plane}_i");
    let nodes: Vec<u32> = f.nodes().map(|n| n.h).collect();
    let wide = plane_widths(f.ctx(), &nodes)?
        .into_iter()
        .find(|w| w.0 == name)
        .ok_or_else(|| nib("a plane no placement records"))?
        .1;
    if f.unit.target == Target::Device {
        // a device lane has no tile around the body: the proxy is the sample's row of the plane, declared where it is read
        f.emit(format!(
            "const auto {proxy} = hawk_abi::tiled<Real, {wide}>({plane}, {});",
            crate::spell::SAMPLE_INDEX_IDENT
        ));
        f.need(Need::Tiled);
    }
    if !f.proxies.iter().any(|x| x.name == proxy) {
        f.proxies.push(Proxy {
            name: proxy.clone(),
            plane,
            width: wide,
        })
    }
    f.set_text(p, proxy);
    Ok(Out::Bound)
}

/// A value placed in its plane: the statements that write it there, then the proxy as the text of
/// the value.
pub(super) fn resident(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let (value, plane) = (f.node(p).ops[0], f.node(p).ops[1]);
    let proxy = f.text(f.opp(p)[1]).clone();
    if f.absorbed.contains(&value) {
        fill(f, &proxy, value)?;
    } else if !is_tiled(f.ctx(), plane)? && *f.peek(f.opp(p)[0]) != proxy {
        return Err(nib(
            "a value written whole into a mutable's own plane (no recorded unit)",
        ));
    } else if *f.peek(f.opp(p)[0]) != proxy {
        let held = match element_kind(f.ctx(), value)?.as_str() {
            "input" | "elementwise" | "parts" | "splat" => element_of(f, value, "hawk_lk", None)?,
            _ => format!("{}(hawk_lk)", f.text(f.opp(p)[0])),
        };
        f.emit_bare("AETHER_NO_UNROLL()".into());
        f.emit(format!(
            "for (Int hawk_lk = 0; hawk_lk < {}; ++hawk_lk) {proxy}.at(hawk_lk) = {held};",
            f.node(p).ty.shape[0]
        ));
    }
    f.set_text(p, proxy);
    Ok(Out::Bound)
}

/// Writes an elementwise value or a vector of parts into the proxy one element at a time: a rolled
/// loop over the elements when
/// each is the same expression of its index, else one statement per element.
fn fill(f: &mut Frame, proxy: &str, value: u32) -> Result<(), Refusal> {
    let ty = crate::body::ty_of(f.ctx(), value)?;
    let width = ty.shape[0];
    if rolled_elements(f, value)? {
        f.emit_bare("AETHER_NO_UNROLL()".into());
        let held = cast(&element_of(f, value, "hawk_lk", None)?, &ty.dtype)?;
        f.emit(format!(
            "for (Int hawk_lk = 0; hawk_lk < {width}; ++hawk_lk) {proxy}.at(hawk_lk) = {held};"
        ));
        return Ok(());
    }
    for k in 0..width {
        let held = cast(&element_of(f, value, &k.to_string(), Some(k))?, &ty.dtype)?;
        f.emit(format!("{proxy}.at({k}) = {held};"));
    }
    Ok(())
}

/// The text cast to an element type unless it already is.
pub(super) fn cast(text: &str, dtype: &str) -> Result<String, Refusal> {
    let elem = carry_type(dtype, &[])?;
    Ok(if text.starts_with(&format!("static_cast<{elem}>")) {
        text.to_string()
    } else {
        format!("static_cast<{elem}>({text})")
    })
}

/// Whether every element of the value is one expression of the element index: no vector of distinct
/// parts is under it.
fn rolled_elements(f: &Frame, value: u32) -> Result<bool, Refusal> {
    let mut stack = vec![value];
    let mut seen: Vec<u32> = vec![];
    while let Some(node) = stack.pop() {
        if seen.contains(&node) || crate::body::ty_of(f.ctx(), node)?.shape.len() != 1 {
            continue;
        }
        seen.push(node);
        match element_kind(f.ctx(), node)?.as_str() {
            "parts" => {
                let operands = door::operands(f.ctx(), node)?;
                if gather(f.ctx(), node)?.is_some() {
                    continue;
                }
                let literals: Vec<Option<door::Scalar>> = operands
                    .iter()
                    .map(|&o| match door::class_is(f.ctx(), o, "Const") {
                        Ok(true) => literal_of(f.ctx(), o).ok(),
                        _ => None,
                    })
                    .collect();
                let Some(first) = &literals[0] else {
                    return Ok(false);
                };
                if literals
                    .iter()
                    .any(|l| !l.as_ref().is_some_and(|l| l.same_value(first)))
                {
                    return Ok(false);
                }
            }
            "elementwise" => stack.extend(door::operands(f.ctx(), node)?),
            _ => {}
        }
    }
    Ok(true)
}

/// The text of element `k` of a value (`fixed` its number when it is a compile-time index): a plane
/// read at it, a part, or the
/// operation on the elements of its operands.
fn element_of(f: &Frame, value: u32, k: &str, fixed: Option<usize>) -> Result<String, Refusal> {
    let mut work = vec![(value, false)];
    let mut texts: Vec<String> = vec![];
    // the nodes whose texts the element is built from; an operation may fold some away
    let mut used: Vec<u32> = vec![];
    while let Some((node, expanded)) = work.pop() {
        let text = if expanded {
            None
        } else {
            element_text(f, node, k, fixed, &mut used)?
        };
        if let Some(text) = text {
            texts.push(text);
        } else if !expanded {
            work.push((node, true));
            work.extend(
                door::operands(f.ctx(), node)?
                    .into_iter()
                    .rev()
                    .map(|o| (o, false)),
            );
        } else {
            let count = door::operands(f.ctx(), node)?.len();
            let args = texts.split_off(texts.len() - count);
            texts.push(operation(f, node, &args)?);
        }
    }
    let element = texts.pop().ok_or_else(|| fault("an element of nothing"))?;
    for h in used {
        f.record_names_in(f.pos_of(h)?, &element);
    }
    Ok(element)
}

/// The text of element `k` of a value where it is not an operation on its operands' elements, else
/// `None`.
fn element_text(
    f: &Frame,
    node: u32,
    k: &str,
    fixed: Option<usize>,
    used: &mut Vec<u32>,
) -> Result<Option<String>, Refusal> {
    let ty = crate::body::ty_of(f.ctx(), node)?;
    if ty.shape.is_empty() {
        used.push(node);
        return Ok(Some(f.peek_arg(node)?));
    }
    if is_prior(f.ctx(), node)? {
        return Ok(Some(format!(
            "{}[i].at({k})",
            binding_name(
                &text_field(f.ctx(), node, "role")?,
                ty.shape.len(),
                &text_field(f.ctx(), node, "name")?
            )?
        )));
    }
    if !f.absorbed.contains(&node) {
        let text = f.peek_arg(node)?;
        if f.proxies.iter().any(|x| x.name == text) || f.is_lazy(f.pos_of(node)?) {
            f.record_read(f.pos_of(node)?);
            return Ok(Some(format!("{text}.at({k})")));
        }
    }
    let operands = door::operands(f.ctx(), node)?;
    Ok(match element_kind(f.ctx(), node)?.as_str() {
        "parts" => match gather(f.ctx(), node)? {
            Some(run) => Some(gathered(f, &run, k, fixed)?),
            None => {
                used.push(operands[fixed.unwrap_or(0)]);
                Some(f.peek_arg(operands[fixed.unwrap_or(0)])?)
            }
        },
        "splat" => {
            used.push(operands[0]);
            Some(f.peek_arg(operands[0])?)
        }
        "elementwise" => None,
        "input" => Some(format!(
            "{}[i].at({k})",
            binding_name(
                &text_field(f.ctx(), node, "role")?,
                ty.shape.len(),
                &text_field(f.ctx(), node, "name")?
            )?
        )),
        other => {
            return Err(nib(format!(
                "a value of kind {other} written into a plane element by element"
            )))
        }
    })
}

/// One element of an elementwise operation over the element texts.
fn operation(f: &Frame, node: u32, args: &[String]) -> Result<String, Refusal> {
    let kind = OpKind::of(f.ctx(), node)?;
    if kind == OpKind::Select {
        return Ok(format!("({} ? {} : {})", args[0], args[1], args[2]));
    }
    let dtype = crate::body::ty_of(f.ctx(), node)?.dtype;
    let mut used = vec![];
    let text = scalar_op(kind, args, &dtype, &mut used)?;
    f.need_spelled(&used);
    Ok(text)
}

/// Element `k` of a run of reads of one plane (`fixed` its number when it is a compile-time index).
fn gathered(f: &Frame, run: &Gather, k: &str, fixed: Option<usize>) -> Result<String, Refusal> {
    let at = match fixed {
        Some(n) => format!("static_cast<Int>({})", run.start + run.step * n as i64),
        None => format!(
            "(static_cast<Int>({}) + {k} * static_cast<Int>({}))",
            run.start, run.step
        ),
    };
    let index = match run.base {
        None => at,
        Some(base) => format!("({} + {at})", f.arg(base)?),
    };
    let plane_rank = crate::body::ty_of(f.ctx(), run.plane)?.shape.len();
    let plane = binding_name(
        &text_field(f.ctx(), run.plane, "role")?,
        plane_rank,
        &text_field(f.ctx(), run.plane, "name")?,
    )?;
    // a plane with axes is read through its view's call form, a plane without them through the sample index
    if door::axes_declared(f.ctx(), run.plane)? > 0 {
        let edge = if run.clamped { "clamp" } else { "unchecked" };
        let (element, _) = element_at(&plane, &[index], edge, true)?;
        if run.clamped {
            f.need(Need::Edge);
        }
        return Ok(element);
    }
    let index = if run.clamped {
        f.need(Need::Integers);
        format!("hawk_abi::int_clamp_index<Int>({index}, static_cast<Int>({plane}.samples()))")
    } else {
        index
    };
    Ok(format!(
        "{plane}[aether::SampleIndex::make(static_cast<std::size_t>({index}))].eval()"
    ))
}
