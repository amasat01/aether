//! Scratch planes of a group inside a body: a read at the lane's own place or at a place moved from it by a whole number, and a
//! store at the lane's own place. The text of a phase is written twice, once for the interior lanes (every moved place is in
//! the plane, so the index is plain and the loop stays a straight run over unit-stride arrays) and once for the lanes at the
//! edges (the edge rule folds the index); the unit writer cuts the phase between the two.
use super::stores::sinks::{component_of, flat_indices};
use super::values::text_field;
use super::{Class, Frame, OpKind, Out};
use crate::body::values::literal_of;
use crate::ir::door::{self, RxCtx, Scalar};
use crate::refuse::{nib, Refusal};
use crate::spell::group::{scratch_cell, scratch_read, Fold, GROUP_EXTENT, GROUP_RANK, GROUP_SIZE};

/// Where a scratch element is: the lane's place moved by a whole number, or a place that does not depend on the lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// The lane's place in its group plus this many.
    Moved(i64),
    /// This place of the group, the same for every lane.
    Fixed(i64),
}

/// The text of the lane's place in its group inside a phase loop: the loop's own variable.
const RANK: &str = "hawk_rank";

/// The place the index node `index` states, or the refusal that a place of another shape is not written.
pub(crate) fn place_of(ctx: *mut RxCtx, index: u32, kernel: &str) -> Result<Place, Refusal> {
    let refuse = || {
        nib(format!(
            "{kernel:?}: a scratch place that is not the group rank moved by a whole number"
        ))
    };
    match Class::of(ctx, index)? {
        Class::GroupRank => Ok(Place::Moved(0)),
        Class::Const => match literal_of(ctx, index)? {
            Scalar::Int(k) => Ok(Place::Fixed(k)),
            _ => Err(refuse()),
        },
        Class::Op => {
            let kind = OpKind::of(ctx, index)?;
            let ops = door::operands(ctx, index)?;
            let [a, b] = ops.as_slice() else {
                return Err(refuse());
            };
            let (a, b) = (place_of(ctx, *a, kernel)?, place_of(ctx, *b, kernel)?);
            match (kind, a, b) {
                (OpKind::Add, Place::Moved(c), Place::Fixed(k))
                | (OpKind::Add, Place::Fixed(k), Place::Moved(c)) => Ok(Place::Moved(c + k)),
                (OpKind::Add, Place::Fixed(x), Place::Fixed(y)) => Ok(Place::Fixed(x + y)),
                (OpKind::Sub, Place::Moved(c), Place::Fixed(k)) => Ok(Place::Moved(c - k)),
                (OpKind::Sub, Place::Fixed(x), Place::Fixed(y)) => Ok(Place::Fixed(x - y)),
                _ => Err(refuse()),
            }
        }
        _ => Err(refuse()),
    }
}

/// The index text of a place: the loop variable moved by the offset, or the fixed place.
fn index_text(place: Place) -> String {
    match place {
        Place::Moved(0) => RANK.to_string(),
        Place::Moved(c) if c > 0 => format!("{RANK} + {c}"),
        Place::Moved(c) => format!("{RANK} - {}", -c),
        Place::Fixed(k) => k.to_string(),
    }
}

/// The text of the lane's place in its group.
pub(crate) fn rank() -> Out {
    Out::Value(GROUP_RANK.to_string(), None)
}

/// The text of the number of lanes of the lane's group.
pub(crate) fn size() -> Out {
    Out::Value(GROUP_SIZE.to_string(), None)
}

/// A read of a scratch plane (an `At` over a scratch leaf), written for the interior lanes of a phase or for its edge lanes
/// (`Frame::edge_lanes`).
pub(crate) fn read(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let kernel = f.unit.name.clone();
    let (plane, indices) = (f.opp(p)[0], f.opp(p)[1..].to_vec());
    let [index] = indices.as_slice() else {
        return Err(nib(format!(
            "{kernel:?}: a scratch read at several indices"
        )));
    };
    let name = text_field(f.ctx(), f.node(plane).h, "name")?;
    let place = place_of(f.ctx(), f.node(*index).h, &kernel)?;
    let edge = f.field_text(f.node(p).h, "edge")?;
    let ty = f.node(p).ty.clone();
    // a lane's own place is in the plane; a moved place is in it for an interior lane only; a fixed place is folded
    // by the rule in both forms, since it does not depend on the lane
    let fold = match place {
        Place::Moved(0) => Fold::Plain,
        Place::Moved(_) if !f.edge_lanes => Fold::Plain,
        _ => Fold::Edge(&edge),
    };
    let text = scratch_read(
        &name,
        (&ty.dtype, &ty.shape),
        &index_text(place),
        fold,
        GROUP_EXTENT,
    )?;
    if text.contains("aether::edge::") {
        f.need(super::frame::needs::Need::Edge);
    }
    Ok(Out::Value(text, None))
}

/// A store of a value into the lane's own place of a scratch plane (an `Update` of the role `scratch`).
pub(crate) fn store(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    let kernel = f.unit.name.clone();
    let (value, indices) = (f.opp(p)[0], f.opp(p)[1..].to_vec());
    let [index] = indices.as_slice() else {
        return Err(nib(format!(
            "{kernel:?}: a scratch store at several indices"
        )));
    };
    if place_of(f.ctx(), f.node(*index).h, &kernel)? != Place::Moved(0) {
        return Err(nib(format!(
            "{kernel:?}: a scratch store at another lane's place; a lane stores at its own rank"
        )));
    }
    let name = text_field(f.ctx(), f.node(p).h, "name")?;
    let shape = f.node(value).ty.shape.clone();
    let text = f.text(value).clone();
    if shape.is_empty() {
        f.emit(format!("{} = {text};", scratch_cell(&name, None, RANK)));
        return Ok(());
    }
    for (k, component) in flat_indices(&shape).into_iter().enumerate() {
        f.emit(format!(
            "{} = {};",
            scratch_cell(&name, Some(k), RANK),
            component_of(text.clone(), &component)
        ));
    }
    Ok(())
}
