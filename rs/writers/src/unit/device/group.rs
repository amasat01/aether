//! The device body of a unit whose walk is a group of lanes: a block serves one group, and each phase is a loop over the group's
//! lanes strided over the block's threads, with a barrier between phases.
//!
//! The call's range is a range of GROUPS: block `b` is the group `base + b`, and the launch asks for `scratch_bytes` of dynamic
//! shared memory, the sum of the scratch planes' bytes each rounded up to eight. A phase reads scratch at its own place or at a
//! place moved from it by a whole number: the lanes of a phase that moves a place fold the index by the edge rule, since the
//! interior/edge cut of the host's loops would only split work the threads already share.
use crate::body::frame::needs::{Needs, Rendered};
use crate::body::{render_phase_with_needs, PhaseForm};
use crate::contract::types::Unit;
use crate::ir::door::RxCtx;
use crate::refuse::Refusal;
use crate::spell::group::{lane_pointer_declarations, scratch_array, scratch_type};
use crate::unit::group::{
    axes_group_lines, lane_binding, lane_offset, reach, Lanes, Plan, ScratchPlane, GROUP_INDENT,
};

/// The indent of the statements inside a phase loop: the lane's binding and the phase's body.
const LOOP_INDENT: &str = "            ";

/// The lines that carve the scratch arrays out of the block's dynamic shared memory, one plane after another from offset zero, each
/// plane starting on an eight-byte boundary. Empty without scratch.
fn carve(scratch: &[ScratchPlane]) -> Result<Vec<String>, Refusal> {
    if scratch.is_empty() {
        return Ok(vec![]);
    }
    let mut lines = vec!["    extern __shared__ __align__(8) char hawk_scratch[];".to_string()];
    let mut offset = "0".to_string();
    for (k, plane) in scratch.iter().enumerate() {
        let element = scratch_type(&plane.dtype)?;
        let components: usize = plane.shape.iter().product();
        let bytes = format!("sizeof({element}) * {} * {components}", plane.bound);
        lines.push(format!(
            "    constexpr std::size_t hawk_scratch_at{k} = {offset};"
        ));
        for component in 0..components {
            let array = scratch_array(&plane.name, (!plane.shape.is_empty()).then_some(component));
            lines.push(format!(
                "    {element}* const {array} = reinterpret_cast<{element}*>(hawk_scratch + hawk_scratch_at{k}) + {component} * {};",
                plane.bound
            ));
        }
        offset = format!("hawk_scratch_at{k} + (({bytes}) + 7) / 8 * 8");
    }
    Ok(lines)
}

/// The statements that set `hawk_base` (the lane of rank 0) and `hawk_size` for the group `hawk_g`.
fn group_lines(lanes: &Lanes, size: &str) -> Vec<String> {
    match lanes {
        Lanes::Count(n) => vec![
            format!("{GROUP_INDENT}const std::int64_t hawk_base = hawk_g * {n};"),
            format!("{GROUP_INDENT}const std::int64_t hawk_size = nSamples - hawk_base < {n} ? nSamples - hawk_base : {n};"),
        ],
        Lanes::Axes(axes) => axes_group_lines(axes, size),
    }
}

/// The body of the unit: the scratch, the block's group, and one strided loop per run with a barrier between runs.
pub fn group_body(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    group: u32,
) -> Result<Rendered, Refusal> {
    let plan = Plan::read(ctx, walk, unit, group)?;
    let outputs = plan.pointer_outputs()?;
    let offset = lane_offset(&plan.lanes);
    let mut text = carve(&plan.scratch)?;
    for hoisted in &plan.hoisted {
        text.extend(
            lane_pointer_declarations(&hoisted.binding()?, hoisted.rank, hoisted.written)
                .into_iter()
                .map(|line| format!("    {line}")),
        );
    }
    text.extend([
        "    const std::int64_t hawk_g = base + static_cast<std::int64_t>(blockIdx.x);".to_string(),
        "    if (hawk_g >= base + count) return;".to_string(),
    ]);
    text.extend(group_lines(&plan.lanes, &plan.size));
    let mut needs = Needs::default();
    for (k, run) in plan.runs.iter().enumerate() {
        let (before, after) = reach(ctx, &run.sinks, &unit.name)?;
        let phase = render_phase_with_needs(
            ctx,
            walk,
            unit,
            LOOP_INDENT,
            &run.sinks,
            PhaseForm {
                edge_lanes: before + after != 0,
                pointer_outputs: &outputs,
            },
        )?;
        needs.merge(&phase.needs);
        text.push(format!("{GROUP_INDENT}// {}", run.label));
        text.push(format!(
            "{GROUP_INDENT}for (std::int64_t hawk_rank = threadIdx.x; hawk_rank < hawk_size; hawk_rank += blockDim.x) {{"
        ));
        text.push(lane_binding(&offset));
        text.push(phase.text);
        text.push(format!("{GROUP_INDENT}}}"));
        if k + 1 < plan.runs.len() {
            text.push(format!("{GROUP_INDENT}__syncthreads();"));
        }
    }
    Ok(Rendered {
        text: text.join("\n"),
        needs,
    })
}
