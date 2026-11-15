//! Values that are read, not computed: leaves, indexed reads, literals, plane extents, the sample
//! index and selects.
use super::{Frame, Out};
use crate::body::{kind_of, Class, OpKind};
use crate::ir::door;
use crate::refuse::{nib, Refusal};
use crate::spell::{self};

/// A leaf: a plane or scalar read from the kernel's arguments, bound whole or as its `[i]` proxy.
pub fn leaf(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let (role, name) = (
        text_field(f.ctx(), h, "role")?,
        text_field(f.ctx(), h, "name")?,
    );
    if crate::ir::analysis::is_scratch(f.ctx(), h)? {
        // the plane a placement writes: its lane proxy, or no text at all where the walk is not
        // placed
        if f.placed {
            return super::placed::plane_leaf(f, p);
        }
        f.set_text(p, String::new());
        return Ok(Out::Bound);
    }
    let (rank, numel, dtype) = (
        f.node(p).rank(),
        f.node(p).numel(),
        f.node(p).ty.dtype.clone(),
    );
    // a leaf only read element by element (static component reads) is bound as the `[i]` proxy, not
    // copied whole: a mutable
    // plane of rank 1 always, any other beyond the width threshold
    let uses = reader_kinds(f, p)?;
    let only_components = !uses.is_empty()
        && uses.iter().all(|(kind, slot, _)| {
            *slot == 0 && matches!(kind, Some(OpKind::Component | OpKind::ComponentAt))
        });
    // a read-only row read at a runtime index is served by the plane's own view, at any width
    let runtime = only_components
        && matches!(role.as_str(), "vec_in" | "per_sample")
        && rank == 1
        && uses
            .iter()
            .any(|(kind, _, _)| *kind == Some(OpKind::ComponentAt));
    // a plane a library call or a team store reads is bound as its proxy too
    let proxy = rank > 0
        && !matches!(role.as_str(), "mutable" | "uniform" | "nsamples")
        && super::linalg::leaf_is_proxy(f, p);
    let in_place = role != "mutable" || super::stores::rtstore::served_in_place(f, &name)?;
    let lazy = proxy
        || rank > 0
            && only_components
            && match role.as_str() {
                "uniform" | "nsamples" => false,
                "mutable" => rank == 1 && in_place,
                _ => numel > f.unit.build.form.lazy_leaf_width as usize || runtime,
            };
    f.set_lazy(p, lazy);
    let text = if lazy {
        format!("{}[i]", spell::binding_name(&role, &name)?)
    } else if rank > 0 {
        spell::row_leaf(&role, &name)?
    } else {
        spell::leaf(&role, &name, 0, &dtype)?
    };
    Ok(Out::Value(text, Some(role)))
}

/// An indexed read of a leaf at another lane or cell, spelled by the leaf's role.
pub fn at(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let n = &f.node(p);
    let (leaf_p, ops_n, rank) = (f.opp(p)[0], n.ops.len(), n.rank());
    let leaf_h = f.node(leaf_p).h;
    if ops_n == 2
        && f.node(f.opp(p)[1]).class == Class::RoleConst
        && text_field(f.ctx(), f.node(f.opp(p)[1]).h, "label")? == "param_cell"
    {
        let cell = spell::param_cell(
            &text_field(f.ctx(), leaf_h, "role")?,
            &text_field(f.ctx(), leaf_h, "name")?,
        )?;
        f.set_text(p, cell);
        return Ok(Out::Bound);
    }
    if f.node(leaf_p).class != Class::Leaf || ops_n != 2 {
        return Err(nib("an indexed read of a non-leaf"));
    }
    let role = text_field(f.ctx(), leaf_h, "role")?;
    let text = spell::at(
        &role,
        &text_field(f.ctx(), leaf_h, "name")?,
        f.text(f.opp(p)[1]),
        rank,
    )?;
    // a plane read at another lane that only library calls read is their per-sample operand as it
    // is, not a copy of the row
    if rank > 0
        && !matches!(role.as_str(), "lookup" | "wide_in")
        && super::linalg::leaf_is_proxy(f, p)
    {
        f.set_lazy(p, true);
        return Ok(Out::Value(
            text.trim_end_matches(".get()").to_string(),
            None,
        ));
    }
    Ok(Out::Value(text, None))
}

/// The extent of a plane.
pub fn plane_extent(f: &Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    Ok(Out::Value(
        spell::plane_extent(
            &text_field(f.ctx(), h, "role")?,
            &text_field(f.ctx(), h, "name")?,
        )?,
        None,
    ))
}

/// The launch position of the sample: `hawk_t` under an index map of lanes, else the lane's own
/// index.
pub fn sample_position(f: &Frame) -> Result<Out, Refusal> {
    Ok(Out::Value(
        format!(
            "static_cast<Int>({})",
            if f.unit.kind.lanes {
                "hawk_t"
            } else {
                "hawk_i"
            }
        ),
        None,
    ))
}

/// The index of the sample.
pub fn sample_index() -> Result<Out, Refusal> {
    Ok(Out::Value(spell::SAMPLE_INDEX.to_string(), None))
}

/// A constant, spelled for its type and rank.
pub fn constant(f: &Frame, p: usize) -> Result<Out, Refusal> {
    let n = &f.node(p);
    if f.banded() && matches!(n.ty.dtype.as_str(), "f64" | "f32") {
        return Err(nib("a banded literal"));
    }
    Ok(Out::Value(
        spell::constant(&n.ty.dtype, n.rank(), &held_of(&literal_of(f.ctx(), n.h)?))?,
        None,
    ))
}

/// A select between two values: a conditional expression when both arms may run freely, else the
/// library's select.
pub fn select(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let (o1, o2) = (f.node(p).ops[1], f.node(p).ops[2]);
    let host = f.unit.build.blend && evaluates_freely(f, o1, o2)?;
    let args: Vec<String> = f.opp(p).iter().map(|&q| f.text(q).clone()).collect();
    let (cond_rank, dtype, shape) = (
        f.node(f.opp(p)[0]).rank(),
        f.node(p).ty.dtype.clone(),
        f.node(p).ty.shape.clone(),
    );
    let mut used = vec![];
    let text = spell::select(
        &mut used,
        host,
        cond_rank,
        &dtype,
        &shape,
        &args,
        &mut |t| f.hoist(t),
    )?;
    f.hold_needs(p, &used);
    Ok(Out::Value(text, None))
}

/// Whether both arms may run whatever the condition says: no computed-index read, tape, loop or
/// run-time component index below them.
fn evaluates_freely(f: &Frame, a: u32, b: u32) -> Result<bool, Refusal> {
    let (mut stack, mut seen) = (vec![a, b], vec![]);
    while let Some(h) = stack.pop() {
        if seen.contains(&h) {
            continue;
        }
        seen.push(h);
        let class = Class::of(f.ctx(), h)?;
        if matches!(
            class,
            Class::At
                | Class::TeamRead
                | Class::TapeRead
                | Class::TapeWrite
                | Class::Loop
                | Class::Dispatch
        ) {
            return Ok(false);
        }
        if matches!(
            kind_of(f.ctx(), h, &class)?,
            Some(OpKind::RawDevice | OpKind::ComponentAt | OpKind::SetComponentAt | OpKind::AddAt)
        ) {
            return Ok(false);
        }
        stack.extend(door::operands(f.ctx(), h)?);
    }
    Ok(true)
}

/// How a leaf is read: (op kind, none for another class; operand index) of each reader. A loop reads its carried
/// initial values itself; what
/// its body takes from outside is read by the body's own nodes, nested loops included.
pub(super) fn reader_kinds(
    f: &Frame,
    p: usize,
) -> Result<Vec<(Option<OpKind>, usize, u32)>, Refusal> {
    let leaf = f.node(p).h;
    let mut uses = vec![];
    let mut loops: Vec<u32> = vec![];
    for &(user, slot) in f.users(p) {
        let node = &f.node(user);
        if node.class == Class::Loop {
            if slot
                < door::field_refs(f.ctx(), node.h, "carries")?
                    .unwrap_or_default()
                    .len()
            {
                uses.push((None, slot, node.h))
            }
            loops.push(node.h);
        } else {
            uses.push((node.kind, slot, node.h))
        }
    }
    while let Some(lp) = loops.pop() {
        for h in door::body_nodes(f.ctx(), lp)? {
            let class = Class::of(f.ctx(), h)?;
            let kind = kind_of(f.ctx(), h, &class)?;
            if class == Class::Loop {
                let carries = door::field_refs(f.ctx(), h, "carries")?
                    .unwrap_or_default()
                    .len();
                for (slot, o) in door::operands(f.ctx(), h)?.into_iter().enumerate() {
                    if o == leaf && slot < carries {
                        uses.push((kind, slot, h))
                    }
                }
                loops.push(h);
                continue;
            }
            for (slot, o) in door::operands(f.ctx(), h)?.into_iter().enumerate() {
                if o == leaf {
                    uses.push((kind, slot, h))
                }
            }
        }
    }
    Ok(uses)
}

/// The scalar a node's `literal` field holds (a number, a flag, a tuple of integers or nothing).
pub(crate) fn literal_of(ctx: *mut door::RxCtx, h: u32) -> Result<door::Scalar, Refusal> {
    door::field_scalar(ctx, h, "literal")
}

/// The value of a constant as the spelling takes it: a number, a flag, or the kind of anything
/// else.
fn held_of(scalar: &door::Scalar) -> spell::Held<'_> {
    match scalar {
        door::Scalar::Int(v) => spell::Held::Int(*v),
        door::Scalar::Real(v) => spell::Held::Real(*v),
        door::Scalar::Flag(v) => spell::Held::Flag(*v),
        other => spell::Held::Other(other.kind_name()),
    }
}

/// The static index a `component` node reads: the integers of its `literal` field. A literal that
/// is not integers is not in this build.
pub(crate) fn component_index(ctx: *mut door::RxCtx, h: u32) -> Result<Vec<i64>, Refusal> {
    match literal_of(ctx, h)? {
        door::Scalar::Ints(index) => Ok(index),
        door::Scalar::Int(v) => Ok(vec![v]),
        other => Err(nib(format!("component index of {}", other.kind_name()))),
    }
}

/// A node's text field (a kind, role, name, label or the like) without quotes, read through the
/// typed reader; a field that
/// holds nothing reads as the empty text.
pub(crate) fn text_field(ctx: *mut door::RxCtx, h: u32, name: &str) -> Result<String, Refusal> {
    Ok(door::field_text(ctx, h, name)?.unwrap_or_default())
}
