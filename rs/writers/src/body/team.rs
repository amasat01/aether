//! Team nodes: reads, copies and stores of a team stage.
use super::{linalg, Frame, Out};
use crate::body::frame::needs::Need;
use crate::body::values::text_field;
use crate::body::Class;
use crate::ir::door;
use crate::refuse::{nib, Refusal};
use crate::spell::binding_name;

/// A team node met in the walk: a read renders its value here; a store or copy is a sink, written
/// with the other sinks.
pub fn stmt(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    if f.node(p).class == Class::TeamRead {
        read(f, p)
    } else {
        Ok(Out::Bound)
    }
}

/// The offset of the lane's own slice of a scratch plane: `static_cast<std::size_t>(<index>) *
/// <size>`.
fn slice_base(index: &str, size: usize) -> String {
    format!("static_cast<std::size_t>({index}) * {size}")
}

/// A vector or matrix a team stage left in the lane's own slice of its scratch plane: one loop
/// ahead of the line that uses
/// it copies it row-major into a fresh item, whose name is the value.
fn read(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let (plane_p, index_p) = (f.opp(p)[0], f.opp(p)[1]);
    let plane = binding_name(
        &text_field(f.ctx(), f.node(plane_p).h, "role")?,
        &text_field(f.ctx(), f.node(plane_p).h, "name")?,
    )?;
    if f.node(p).rank() == 0 {
        return Err(nib("a team read of a scalar"));
    }
    let size = f.node(p).numel();
    let out = linalg::fresh_item(f, p)?;
    let base = slice_base(f.text(index_p), size);
    f.emit(format!("for (Int hawk_lk = 0; hawk_lk < {size}; ++hawk_lk) {out}.data()[hawk_lk] = {plane}.data()[{base} + hawk_lk];"));
    f.set_text(p, out);
    Ok(Out::Bound)
}

/// Element `hawk_lk` (row-major) of a per-sample operand: through `data()` for a fresh or held
/// item, else by index.
fn element(held: &str, shape: &[usize]) -> String {
    let item = |prefix: &str| {
        held.strip_prefix(prefix)
            .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
    };
    if item("hawk_la") || item("hawk_it") {
        format!("{held}.data()[hawk_lk]")
    } else if shape.len() == 1 {
        format!("{held}.at(hawk_lk)")
    } else {
        format!("{held}.at(hawk_lk / {}, hawk_lk % {})", shape[1], shape[1])
    }
}

/// A store of a value into the lane's own slice of a team stage's scratch plane. A scalar is one
/// assignment; a vector or
/// matrix is one loop over its elements, or (for a copy) a loop its team's threads share.
pub fn store(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    let (value_p, index_p) = (f.opp(p)[0], f.opp(p)[1]);
    let shape = f.node(value_p).ty.shape.clone();
    let (value, index) = (f.text(value_p).clone(), f.text(index_p).clone());
    let held = if shape.is_empty() {
        value
    } else {
        linalg::operand(f, value_p, &value)?
    };
    let copy = f.node(p).class == Class::TeamCopy;
    write(f, f.node(p).h, &held, &index, &shape, copy)
}

/// A log store inside a loop body, whose operands have no place of their own in the walk and are
/// rendered by handle: a scalar value into the slot its index names. A value of rank > 0 is refused
/// by name.
pub fn store_in_loop(f: &mut Frame, h: u32) -> Result<(), Refusal> {
    let operands = door::operands(f.ctx(), h)?;
    if !crate::body::ty_of(f.ctx(), operands[0])?.shape.is_empty() {
        return Err(nib("a log store of a rank > 0 value inside a loop body"));
    }
    let (value, index) = (f.arg(operands[0])?, f.arg(operands[1])?);
    write(f, h, &value, &index, &[], false)
}

/// The statement of a store into the lane's own slice of the scratch plane the node `h` names:
/// `held` the value (an element-readable item when `shape` is not empty), `index` its slice.
fn write(
    f: &mut Frame,
    h: u32,
    held: &str,
    index: &str,
    shape: &[usize],
    copy: bool,
) -> Result<(), Refusal> {
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let size: usize = shape.iter().product();
    let base = slice_base(index, size);
    if shape.is_empty() {
        f.emit(format!("{target}.data()[{base}] = {held};"));
        return Ok(());
    }
    let assign = format!(
        "{target}.data()[{base} + hawk_lk] = {};",
        element(held, shape)
    );
    if copy {
        f.need(Need::Linalg);
        f.need(Need::Team);
        f.emit(format!("{{ const aether::linalg::team::Team hawk_tm = aether::linalg::team::Team::here(); for (Int hawk_lk = hawk_tm.rank; hawk_lk < {size}; hawk_lk += hawk_tm.size) {assign} }}"));
    } else {
        f.emit(format!(
            "for (Int hawk_lk = 0; hawk_lk < {size}; ++hawk_lk) {assign}"
        ));
    }
    Ok(())
}
