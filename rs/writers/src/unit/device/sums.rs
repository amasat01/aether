//! The device's frame around the body of a kernel with sample-sum sinks: a warp per row sum, with
//! no early return, so a
//! lane past the launch's count still takes part in its warp's sum with zero terms.
use super::prologue::{index_prologue, Geometry};
use crate::contract::types::{Active, Build, Segment};
use crate::refuse::{bad, fault, Refusal};
use crate::unit::sums::{identity_fill, op_code, store_lambda, SumSink};

/// The name of the lane's terms array, which the body writes its sums into.
const TERMS: &str = "hawk_ss";

/// The rows one warp folds in one batched sum, for sinks of `width` columns: the largest power of
/// two up to the wing's limit whose columns keep the warp under the residency line; the build's
/// forced value wins when set.
fn fold_rows(build: &Build, width: i64) -> Result<i64, Refusal> {
    if let Some(forced) = build.fold_k {
        return Ok(forced);
    }
    let fold = build
        .fold_limits
        .as_ref()
        .ok_or_else(|| bad("the device wing has no fold k_max"))?;
    let mut rows = fold.k_max;
    while rows > 1 && fold.regs_base + fold.regs_column * (rows * width - 1) > fold.residency_regs {
        rows /= 2;
    }
    Ok(rows)
}

/// The line that says whether the lane takes part: inside the launch's count, and inside the active
/// set's live count.
fn active_line(live_count: Option<&str>) -> String {
    match live_count {
        Some(count) => {
            let live = format!("static_cast<long long>({count})");
            format!("    const bool hawk_active = hawk_flat < count && base + hawk_flat < {live};")
        }
        None => "    const bool hawk_active = hawk_flat < count;".to_string(),
    }
}

/// The frame for one row per warp: the prologue's early return becomes a skip of the body.
fn single_row_frame(
    sinks: &[SumSink],
    store: &str,
    live_count: Option<&str>,
) -> Result<(String, String), Refusal> {
    let (width, first) = (sinks.len() as i64, &sinks[0]);
    let exit_line = "    if (hawk_flat >= count) return;\n    {";
    let plain = index_prologue(&Geometry::default());
    let fill = identity_fill(&first.op, width, TERMS, &first.element)?;
    let replacement = [
        "    if ((hawk_flat & ~31LL) >= count) return;".to_string(),
        format!("    {} {TERMS}[{width}] = {{}};", first.element),
        format!("{fill}{}", active_line(live_count)),
        "    if (hawk_active) {".to_string(),
    ];
    let opened = plain.replace(exit_line, &replacement.join("\n"));
    let code = op_code(&first.op)?;
    let closed = [
        "    }".to_string(),
        "    const long long hawk_row = (base + hawk_flat) >> 5;".to_string(),
        store.to_string(),
        format!("    hawk_abi::sampleSumWarp_<{width}, {code}>({TERMS}, hawk_ss_store);"),
    ];
    Ok((opened, closed.join("\n")))
}

/// The frame of a warp that serves `rows` consecutive rows: the folds ride one warp sum of `rows *
/// width` columns.
fn folded_rows_frame(
    sinks: &[SumSink],
    rows: i64,
    store: &str,
    live_count: Option<&str>,
) -> Result<(String, String), Refusal> {
    let (width, first) = (sinks.len() as i64, &sinks[0]);
    let (columns, element) = (rows * width, &first.element);
    let plain = index_prologue(&Geometry::default());
    // the prologue's own flat-index line: the thread
    let lane_line = plain.split("    if (hawk_flat").next().unwrap_or("");
    let fill = identity_fill(&first.op, columns, "hawk_terms", element)?;
    let opened = [
        lane_line.trim_end_matches('\n').to_string(),
        "    const long long hawk_gid = hawk_flat;".to_string(),
        format!("    const long long hawk_wbase = (hawk_gid & ~31LL) * {rows};"),
        "    if (hawk_wbase >= count) return;".to_string(),
        format!("    {element} hawk_terms[{columns}] = {{}};"),
        format!("{fill}#pragma unroll"),
        format!("    for (int hawk_r = 0; hawk_r < {rows}; ++hawk_r) {{"),
        format!("    {element}* const {TERMS} = hawk_terms + hawk_r * {width};"),
        "    const long long hawk_flat = hawk_wbase + 32 * hawk_r + (hawk_gid & 31);".to_string(),
        "    const long long hawk_i = base + hawk_flat;".to_string(),
        "    const aether::SampleIndex i =".to_string(),
        "        aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));".to_string(),
        "    (void)hawk_i;".to_string(),
        "    (void)nSamples;".to_string(),
        active_line(live_count),
        "    if (hawk_active) {".to_string(),
    ];
    let code = op_code(&first.op)?;
    let closed = [
        "    }\n    }".to_string(),
        "    const long long hawk_row0 = (base + hawk_wbase + (hawk_gid & 31)) >> 5;".to_string(),
        store.to_string(),
        format!("    hawk_abi::sampleSumWarp_<{columns}, {code}>(hawk_terms, hawk_ss_store);"),
    ];
    Ok((opened.join("\n"), closed.join("\n")))
}

/// The refusal of sinks of different folds, for every device path (plain and banded): the warp sum takes one fold for all its
/// columns, and the band form is chosen once for all of them.
pub(super) fn refuse_mixed_folds(name: &str, sinks: &[SumSink]) -> Result<(), Refusal> {
    if sinks.iter().any(|sink| sink.op != sinks[0].op) {
        return Err(fault(format!(
            "{name}: sample reductions of different folds in one kernel are not written on the device yet"
        )));
    }
    Ok(())
}

/// The refusal of a sample sum whose unit does not run its samples by launch position: a mapped lane set or a segment. The
/// warp rows here are written for a plain or a prefix launch only; a prefix lane set runs lane `t` as sample `t` and stays.
pub(super) fn refuse_unwritten_sum_geometry(
    name: &str,
    active: Option<&Active>,
    segment: Option<&Segment>,
) -> Result<(), Refusal> {
    if active.is_some_and(|set| set.map_binding.is_some()) {
        return Err(fault(format!(
            "{name}: a sum over samples with a mapped lane set is not written on the device yet"
        )));
    }
    if segment.is_some() {
        return Err(fault(format!(
            "{name}: a sum over samples in a segmented unit is not written on the device yet"
        )));
    }
    Ok(())
}

/// The frame the body of a kernel with the sample-sum `sinks` sits in, as its opening and its
/// close.
pub fn frame(
    build: &Build,
    sinks: &[SumSink],
    cells: bool,
    live_count: Option<&str>,
) -> Result<(String, String), Refusal> {
    let rows = fold_rows(build, sinks.len() as i64)?;
    let store = store_lambda(sinks, rows, build.cells_row, cells)?;
    if rows > 1 {
        folded_rows_frame(sinks, rows, &store, live_count)
    } else {
        single_row_frame(sinks, &store, live_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sink(op: &str) -> SumSink {
        SumSink {
            op: op.into(),
            role: "out".into(),
            name: op.into(),
            element: "double".into(),
        }
    }

    #[test]
    fn sinks_of_different_folds_are_refused_by_the_kernel_name() {
        let refusal = refuse_mixed_folds("k", &[sink("min"), sink("max")]);
        let message = format!("{:?}", refusal.expect_err("a refusal"));
        assert!(message.contains(
            "k: sample reductions of different folds in one kernel are not written on the device yet"
        ));
        assert!(refuse_mixed_folds("k", &[sink("min"), sink("min")]).is_ok());
    }

    #[test]
    fn a_sum_and_a_maximum_are_refused_in_a_banded_mode_too() {
        let banded = |op: &str, element: &str| SumSink {
            element: element.into(),
            ..sink(op)
        };
        let sinks = [
            banded("sum", "aether::banded::BandAccum"),
            banded("max", "aether::banded::BandAccum"),
        ];
        assert!(refuse_mixed_folds("k", &sinks).is_err());
    }

    #[test]
    fn a_sum_with_a_mapped_lane_set_or_a_segment_is_refused_by_name() {
        let set = |map: Option<&str>| Active {
            map_binding: map.map(str::to_string),
            count_binding: "count".into(),
            group: 1,
            map_bits: crate::contract::types::MapBits::B32,
            lane_set: true,
        };
        let mapped = refuse_unwritten_sum_geometry("k", Some(&set(Some("map"))), None);
        assert!(format!("{:?}", mapped.expect_err("a refusal")).contains(
            "k: a sum over samples with a mapped lane set is not written on the device yet"
        ));
        let run = Segment {
            offsets_binding: "offsets".into(),
            j: 0,
        };
        let segmented = refuse_unwritten_sum_geometry("k", None, Some(&run));
        assert!(format!("{:?}", segmented.expect_err("a refusal")).contains(
            "k: a sum over samples in a segmented unit is not written on the device yet"
        ));
        assert!(refuse_unwritten_sum_geometry("k", Some(&set(None)), None).is_ok());
        assert!(refuse_unwritten_sum_geometry("k", None, None).is_ok());
    }
}
