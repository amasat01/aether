//! The body of a two-path unit: the single step's body then the fused body, each over its own sinks
//! and walk, both led by
//! eight spaces. The entry joins them under a branch on the fused-steps word.
use crate::body::frame::needs::{Needs, Rendered};
use crate::body::render_body_over_with_needs;
use crate::contract::types::{Form, OneStep, SinkPolicy, Unit};
use crate::ir::door::{self, RxCtx};
use crate::refuse::{bad, nib, Refusal};

/// The single step of a unit written as two paths: not for a team stage or a fused group (each has a body of its own), nor
/// when a segment is given, nor for a compensated kind.
pub fn step_of(unit: &Unit) -> Option<&OneStep> {
    let compensated = matches!(unit.kind.sink, SinkPolicy::Compensated { .. });
    if matches!(unit.form, Form::TeamStage(_) | Form::Fused(_))
        || unit.segment.is_some()
        || compensated
    {
        return None;
    }
    unit.one_step.as_ref()
}

/// The single step's body, or `None` when the unit has no single step. Its walk is taken over the
/// step's sinks with the
/// slots the unit's own walk declares.
pub fn single_body(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<Option<Rendered>, Refusal> {
    let Some(step) = step_of(unit) else {
        return Ok(None);
    };
    let step_walk = door::canonical_over(ctx, &step.sinks, walk)?;
    render_body_over_with_needs(ctx, step_walk, unit, &" ".repeat(8), &step.sinks).map(Some)
}

/// The single step's walk handle and the walk record the unit holds for it.
pub fn step_walk(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<(u32, super::Walk), Refusal> {
    let step = step_of(unit).ok_or_else(|| nib("a two-path unit without a single step"))?;
    let record = step
        .walk
        .as_ref()
        .ok_or_else(|| bad("contract: one_step without a walk"))?;
    Ok((
        door::canonical_over(ctx, &step.sinks, walk)?,
        super::walk_from(record),
    ))
}

/// The two bodies of a two-path unit: the single step's and the fused one.
pub struct BodyParts {
    /// The single step's body.
    pub single: String,
    /// The fused body.
    pub fused: String,
    /// The needs of both bodies, the single step's first.
    pub needs: Needs,
}

impl BodyParts {
    /// The two bodies joined by a newline, the single step first.
    pub fn joined(&self) -> String {
        format!("{}\n{}", self.single, self.fused)
    }
}

/// The two bodies of a two-path unit, or `None` when the unit has no single step.
pub fn two_path_body(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
) -> Result<Option<BodyParts>, Refusal> {
    let Some(single) = single_body(ctx, walk, unit)? else {
        return Ok(None);
    };
    let fused = render_body_over_with_needs(ctx, walk, unit, &" ".repeat(8), &unit.sinks)?;
    let mut needs = single.needs.clone();
    needs.merge(&fused.needs);
    Ok(Some(BodyParts {
        single: single.text,
        fused: fused.text,
        needs,
    }))
}

/// The two bodies of a two-path unit that has a single step; a refusal when it has none.
pub fn parts(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<BodyParts, Refusal> {
    let parts = two_path_body(ctx, walk, unit)?
        .ok_or_else(|| nib("a two-path unit without a single step"))?;
    Ok(parts)
}
