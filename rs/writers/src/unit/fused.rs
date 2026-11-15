//! A fused lane group: one kernel record holding several lanes, each with its own sinks and walk.
//! The body is the lanes' bodies
//! in order, each in a scope of its own led by a `// lane <name>` comment, rendered by the same
//! renderer as a solo kernel.
use crate::body::frame::needs::{Needs, Rendered};
use crate::body::render_body_over_with_needs;
use crate::contract::types::{Form, Unit};
use crate::ir::door::{self, RxCtx};
use crate::refuse::nib;

/// The body of a fused group: per lane, the open comment, the lane's prologue (if any) one level
/// in, the lane's body rendered
/// over its own sinks and walk, and the close comment. A lane name given twice is refused, as the
/// old composer does.
pub fn fused_body(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
) -> Result<Rendered, String> {
    let Form::Fused(lanes) = &unit.form else {
        return Err(nib("a fused group without lanes"));
    };
    if lanes.is_empty() {
        return Err(nib("a fused group with no lanes"));
    }
    let mut seen: Vec<&str> = vec![];
    let mut chunks: Vec<String> = vec![];
    let mut needs = Needs::default();
    for lane in lanes {
        let name = lane.name.as_str();
        if seen.contains(&name) {
            return Err(nib("a fused group with two lanes of one name"));
        }
        seen.push(name);
        let sinks = &lane.sinks;
        let lane_walk = door::canonical_over(ctx, sinks, walk)?;
        let inner = format!("{indent}    ");
        chunks.push(format!("{indent}{{  // lane {name}"));
        for line in lane.prologue.lines() {
            chunks.push(format!("{inner}{line}"))
        }
        let lane_body = render_body_over_with_needs(ctx, lane_walk, unit, &inner, sinks)?;
        needs.merge(&lane_body.needs);
        chunks.push(lane_body.text);
        chunks.push(format!("{indent}}}  // lane {name}"));
    }
    Ok(Rendered {
        text: chunks.join("\n"),
        needs,
    })
}
