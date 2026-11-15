//! The body of a thread-box unit: the plain body for a launch of fewer rows than the box holds, and
//! the boxed body (the
//! sinks copied once per lane of the box), joined in one frame that picks between them at run time.
use crate::body::frame::needs::Rendered;
use crate::body::render_body_over_with_needs;
use crate::contract::types::Unit;
use crate::ir::analysis as an;
use crate::ir::door::{self, RxCtx};
use crate::refuse::nib;

/// The box of a unit: (lanes per thread, stride, lane axis).
fn box_record(unit: &Unit) -> Option<(usize, i64, usize)> {
    let record = unit.thread_box.as_ref()?;
    Some((
        record.t as usize,
        record.stride,
        record.axis.unwrap_or(0) as usize,
    ))
}

/// The two bodies of a boxed unit joined by the `nSamples` test, or `None` when the unit has no
/// box. A box whose sinks
/// the analysis cannot copy is refused by name.
pub fn boxed_body(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<Option<Rendered>, String> {
    let Some((lanes, stride, axis)) = box_record(unit) else {
        return Ok(None);
    };
    let sinks = unit.sinks.clone();
    let copies = an::boxed_sinks(ctx, &sinks, axis, lanes, stride)
        .map_err(|why| nib(format!("a boxed unit ({why})")))?;
    let plain = render_body_over_with_needs(ctx, walk, unit, &" ".repeat(16), &sinks)?;
    let boxed_walk = door::canonical_over(ctx, &copies, walk)?;
    let boxed = render_body_over_with_needs(ctx, boxed_walk, unit, &" ".repeat(12), &copies)?;
    let mut needs = plain.needs.clone();
    needs.merge(&boxed.needs);
    Ok(Some(Rendered {
        text: frame(lanes as i64, stride, &plain.text, &boxed.text),
        needs,
    }))
}

/// The frame that runs the boxed body over a thread's box, or, for fewer rows than the box holds,
/// the plain body over each
/// lane the thread owns in turn.
fn frame(lanes: i64, stride: i64, plain: &str, boxed: &str) -> String {
    let bind = |pad: &str| {
        format!("const aether::SampleIndex i =\n{pad}    aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n{pad}(void)hawk_i;\n{pad}(void)nSamples;")
    };
    let (pad8, pad12, pad16) = (" ".repeat(8), " ".repeat(12), " ".repeat(16));
    let own = "hawk_box_own";
    [
        format!("{pad8}if (nSamples < {}) {{", lanes * stride),
        format!("{pad12}for (long long hawk_k = 0; hawk_k < {lanes}; ++hawk_k) {{"),
        format!("{pad16}const long long hawk_i = {own} + hawk_k * {stride};"),
        format!("{pad16}if (hawk_i >= nSamples) break;"),
        format!("{pad16}{}", bind(&pad16)),
        plain.to_string(),
        format!("{pad12}}}"),
        format!("{pad8}}} else {{"),
        format!(
            "{pad12}const long long hawk_i = {own} + {} < nSamples",
            (lanes - 1) * stride
        ),
        format!(
            "{pad12}    ? {own} : nSamples - {} + {own} % {stride};",
            lanes * stride
        ),
        format!("{pad12}{}", bind(&pad12)),
        boxed.to_string(),
        format!("{pad8}}}"),
    ]
    .join("\n")
}
