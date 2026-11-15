//! The cut of a loop for the lane interchange and the lane split, and the arm text for each.
pub mod cut;
pub mod device;
pub mod device_text;
pub mod host_interchange;
pub mod host_split;
pub mod host_text;
pub mod model;
pub mod split;

use crate::body::frame::needs::Needs;
use crate::body::render_cut;
use crate::contract::types::Unit;
use crate::ir::door::RxCtx;
use crate::refuse::{fault, Refusal};

/// The body cut at the given top-level loops, which the host runs interchanged: `None` when a
/// value crossing a cut has no recorded type (the unit then keeps its plain body).
pub fn lane_segments(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
    cut: &[u32],
) -> Result<(cut::LaneSegments, Needs), Refusal> {
    let mut emitter = cut::CutEmitter::default();
    render_cut(ctx, walk, unit, indent, cut, &mut emitter)?;
    emitter
        .done
        .ok_or_else(|| fault("the cut render did not finish"))
}

/// The body cut at its one loop for the lane split of a finishing kernel: `None` when the body is
/// not that shape.
pub fn lane_split(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
) -> Result<Option<(split::LaneSplit, Needs)>, Refusal> {
    let Some(lp) = split::candidate(ctx, walk)? else {
        return Ok(None);
    };
    let mut emitter = split::SplitEmitter::new(lp);
    render_cut(ctx, walk, unit, indent, &[lp], &mut emitter)?;
    Ok(emitter.done)
}
