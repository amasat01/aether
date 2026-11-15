//! Reads and stores at a runtime index into a vector.
use crate::body::values::{literal_of, text_field};
use crate::body::{linalg, Frame, Out};
use crate::body::{Class, OpKind};
use crate::ir::door;
use crate::ir::door::Scalar;
use crate::refuse::{nib, Refusal};
use crate::spell::binding_name;

/// A component read at a runtime index: `plane[i].at(k)` straight from the view of a read-only
/// input row, else
/// `item(k)` of the value held once as an item.
pub(crate) fn component_at(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let (source, index) = (f.opp(p)[0], f.text(f.opp(p)[1]).clone());
    let warp = match literal_of(f.ctx(), f.node(p).h)? {
        Scalar::Absent => false,
        Scalar::Text(name) if name == "warp_read" => true,
        _ => return Err(nib("a runtime-index read with a literal")),
    };
    // a source already held in a placed plane is read from the plane, whatever made it
    if f.proxies.iter().any(|x| x.name == *f.peek(source)) {
        return Ok(Out::Value(format!("{}.at({index})", f.text(source)), None));
    }
    // an element-wise source is rewritten first: the read moves to its operands (a rewrite of the
    // walk, not written)
    if let (Some(kind), 1) = (f.node(source).kind, f.node(source).rank()) {
        if !matches!(
            kind,
            OpKind::Vec
                | OpKind::Mv
                | OpKind::Mm
                | OpKind::Cross
                | OpKind::AsVec3
                | OpKind::AsPure
                | OpKind::QuatMul
                | OpKind::QuatRotate
                | OpKind::QuatConj
                | OpKind::QuatRecip
                | OpKind::Splat
                | OpKind::Transpose
                | OpKind::AddAt
                | OpKind::SetComponentAt
        ) && !linalg::is_factor_kind(kind)
        {
            return Err(nib("a runtime-index read of an element-wise value (the read is pushed to its operands first)"));
        }
    }
    if f.node(source).class == Class::Leaf {
        let (role, name) = (
            text_field(f.ctx(), f.node(source).h, "role")?,
            text_field(f.ctx(), f.node(source).h, "name")?,
        );
        if role == "mutable" && !warp {
            return mutable_read(f, p, source, &name, &index);
        }
        // a read of one element of a plane the warp holds goes to the view at the run-time index
        let in_place =
            f.node(source).rank() == 1 && matches!(role.as_str(), "vec_in" | "per_sample");
        if warp || in_place {
            return Ok(Out::Value(
                format!("{}[i].at({index})", binding_name(&role, &name)?),
                None,
            ));
        }
    }
    let arg = f.text(source).clone();
    let held = linalg::held_item(f, source, &arg)?;
    Ok(Out::Value(format!("{held}({index})"), None))
}

/// Whether a mutable plane is stored by the kernel, when every store to it is a plain own-column
/// store (an error otherwise:
/// the plane is then not served in place).
pub(crate) fn plane_is_stored(f: &Frame, name: &str) -> Result<bool, Refusal> {
    let mut stored = false;
    for &s in f.sinks() {
        let sink = &f.node(f.at(s)?);
        if text_field(f.ctx(), sink.h, "role")? != "mutable"
            || text_field(f.ctx(), sink.h, "name")? != name
        {
            continue;
        }
        if !matches!(sink.class, Class::Assign | Class::WideWrite) || sink.ops.len() != 1 {
            return Err(nib("a mutable plane read in place but stored otherwise"));
        }
        stored = true;
    }
    Ok(stored)
}

/// Whether the plane's runtime-index reads may come straight from it: every store to it is a plain own-column store of the body,
/// and no loop stores into it.
pub(crate) fn served_in_place(f: &Frame, name: &str) -> Result<bool, Refusal> {
    Ok(plane_is_stored(f, name).is_ok() && !stored_inside_a_loop(f, name)?)
}

/// Names a read of a mutable plane served in place at its own position (`t<pos>`): it happens
/// before the stores, which the
/// final phase writes after every value.
pub(crate) fn named_before_store(f: &mut Frame, p: usize, text: &str) -> Out {
    let local = f.names.temp_at(p);
    f.declare_auto(local.clone(), text);
    f.set_text(p, local.text);
    Out::Bound
}

/// Whether a loop of the kernel (nested ones too) stores into the mutable plane of that name from
/// its body: such a plane is
/// not served in place.
fn stored_inside_a_loop(f: &Frame, name: &str) -> Result<bool, Refusal> {
    let loops: Vec<u32> = f
        .nodes()
        .filter(|n| n.class == Class::Loop)
        .map(|n| n.h)
        .collect();
    for h in crate::ir::analysis::loops_flat(f.ctx(), &loops)? {
        if !door::class_is(f.ctx(), h, "Loop")? {
            continue;
        }
        for sink in door::field_refs(f.ctx(), h, "body_sinks")?.unwrap_or_default() {
            if text_field(f.ctx(), sink, "role")? == "mutable"
                && text_field(f.ctx(), sink, "name")? == name
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// A runtime-index read of a mutable plane served by the plane's own view (`plane[i].at(k)`). With
/// a loop in the kernel only a
/// read inside a loop body is served so, when no loop stores into the plane: the read stays where
/// the body renders it.
fn mutable_read(
    f: &mut Frame,
    p: usize,
    source: usize,
    name: &str,
    index: &str,
) -> Result<Out, Refusal> {
    if f.node(source).rank() != 1 {
        return Err(nib("a runtime-index read of a mutable plane"));
    }
    let text = format!("{}[i].at({index})", binding_name("mutable", name)?);
    if f.nodes().any(|n| n.class == Class::Loop) {
        if !f.is_slot(f.node(p).h) || stored_inside_a_loop(f, name)? {
            return Err(nib("a runtime-index read of a mutable plane"));
        }
        plane_is_stored(f, name)?;
        return Ok(Out::Value(text, None));
    }
    if !plane_is_stored(f, name)? {
        return Ok(Out::Value(text, None));
    }
    Ok(named_before_store(f, p, &text))
}
