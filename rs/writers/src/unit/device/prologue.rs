//! The device entry's geometry: the one place a unit names a launch-geometry builtin. It owns the
//! index prologue in its
//! forms (flat lane, team stage, active-set map, run segment, thread box) and the entry signature.
use crate::contract::types::{Active, MapBits};
use crate::ir::analysis as an;
use crate::refuse::Refusal;
use crate::spell::binding_name;
use crate::unit::{active_reads, view_type, Slot, Walk};

/// The qualifier of every helper function of a device unit: force-inlined, so no call frame is
/// added.
pub const QUALIFIER: &str = "__device__ __forceinline__";

/// The lane of a flat launch: one thread per sample, leaving when it is past the launch's count.
const FLAT_LANE: &str = "    const long long hawk_flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (hawk_flat >= count) return;
    {
        const long long hawk_i = base + hawk_flat;
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;";

/// The lane of a team stage: the lane is the block.
const TEAM_LANE: &str = "    const long long hawk_flat =
        static_cast<long long>(blockIdx.x);
    if (hawk_flat >= count) return;
    {
        const long long hawk_i = base + hawk_flat;
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;";

/// The line that binds the lane's own sample in the flat prologues; the other forms replace it.
const OWN_SAMPLE: &str = "        const long long hawk_i = base + hawk_flat;";

/// The lane's place in its warp, bound for a warp-per-lane team stage.
const WARP_LANE: &str = "
        const Int hawk_wl = static_cast<Int>(threadIdx.x & 31);
        (void)hawk_wl;";

/// How the unit's lanes map onto the launch.
#[derive(Default)]
pub struct Geometry {
    /// The team stage's `(threads per team, warps per block)`.
    pub team: Option<(i64, i64)>,
    /// The active-set form: the count and map bindings.
    pub active: Option<Active>,
    /// The offsets plane's binding and the run's index.
    pub segment: Option<(String, i64)>,
    /// `(threads per box, stride)`.
    pub boxed: Option<(i64, i64)>,
}

/// The lane of a team stage: the block, or a block of warps, with a warp-per-lane stage's lane
/// place.
fn team_prologue(threads: i64, warps: i64) -> String {
    let mut plain = TEAM_LANE.to_string();
    let warps = if threads == 32 { warps } else { 1 };
    if warps > 1 {
        // k warps a block: the lane is the block's warp, and the lane count is nSamples
        plain = plain
            .replace(
                "static_cast<long long>(blockIdx.x);",
                &format!("static_cast<long long>(blockIdx.x) * {warps} + (threadIdx.x >> 5);"),
            )
            .replace("hawk_flat >= count", "hawk_flat >= nSamples");
    }
    if threads == 32 {
        plain.push_str(WARP_LANE);
    }
    plain
}

/// The prologue with the own-sample line replaced by the active-set map read behind a live-count
/// exit.
fn active_prologue(plain: &str, active: &Active) -> String {
    let (count_expr, sample_expr) = active_reads(active, "hawk_t");
    // a 32-bit lane set's map word is a 32-bit sample, so `hawk_i` and every index chain off it is
    // one register
    let narrow = active.lane_set && active.map_bits == MapBits::B32 && active.group == 1;
    let sample = if narrow {
        format!("        const int hawk_i = {sample_expr};")
    } else {
        format!("        const long long hawk_i = static_cast<long long>({sample_expr});")
    };
    let past = if active.group > 1 {
        "\n        if (hawk_i >= nSamples) return;"
    } else {
        ""
    };
    let own = [
        "        const long long hawk_t = base + hawk_flat;".to_string(),
        format!("        if (hawk_t >= static_cast<long long>({count_expr})) return;"),
        format!("{sample}{past}"),
    ];
    plain.replace(OWN_SAMPLE, &own.join("\n"))
}

/// The prologue with the own-sample line replaced by the run's bounds read off the offsets plane.
fn segment_prologue(plain: &str, offsets: &str, run: i64) -> String {
    let index =
        |position: i64| format!("aether::SampleIndex::make(static_cast<std::size_t>({position}))");
    let (begin, end) = (
        format!("{offsets}[{}].eval()", index(run)),
        format!("{offsets}[{}].eval()", index(run + 1)),
    );
    let own = [
        "        (void)base;".to_string(),
        format!("        const long long hawk_seg_begin = static_cast<long long>({begin});"),
        format!("        const long long hawk_seg_end = static_cast<long long>({end});"),
        "        const long long hawk_i = hawk_seg_begin + hawk_flat;".to_string(),
        "        if (hawk_i >= hawk_seg_end) return;".to_string(),
    ];
    plain.replace(OWN_SAMPLE, &own.join("\n"))
}

/// The prologue of a thread box: a thread, ending at its first owned lane.
fn box_prologue(threads: i64, stride: i64) -> String {
    let before_own = &FLAT_LANE[..FLAT_LANE.find(OWN_SAMPLE).unwrap_or(0)];
    let own = [
        "        const long long hawk_t = base + hawk_flat;".to_string(),
        format!(
            "        const long long hawk_box_own = hawk_t + hawk_t / {stride} * {};",
            (threads - 1) * stride
        ),
        "        if (hawk_box_own >= nSamples) return;".to_string(),
    ];
    format!("{before_own}{}", own.join("\n"))
}

/// The index prologue of the unit: the one geometry seam. A thread box ignores the other forms, as
/// the writer does.
pub fn index_prologue(geometry: &Geometry) -> String {
    if let Some((threads, stride)) = geometry.boxed {
        return box_prologue(threads, stride);
    }
    let plain = match geometry.team {
        Some((threads, warps)) => team_prologue(threads, warps),
        None => FLAT_LANE.to_string(),
    };
    match (&geometry.active, &geometry.segment) {
        (Some(active), _) => active_prologue(&plain, active),
        (None, Some((offsets, run))) => segment_prologue(&plain, offsets, *run),
        (None, None) => plain,
    }
}

/// What closes the prologue's block.
pub fn index_epilogue() -> &'static str {
    "    }"
}

/// The by-value ABI shape one slot crosses the launch boundary as.
fn param_type(slot: &Slot) -> Result<String, Refusal> {
    let shape = an::mirror_of(&slot.role, &slot.dtype, slot.shape.len())?;
    if shape == "value" {
        view_type(slot)
    } else {
        Ok(format!("eagle::plugin::{shape}"))
    }
}

/// The identifier the entry takes a slot's mirror under.
pub fn param_name(slot: &Slot) -> Result<String, Refusal> {
    Ok(format!(
        "p_{}",
        binding_name(&slot.role, slot.shape.len(), &slot.name)?
    ))
}

/// The `__global__` entry: the slots by value, then the partition triple.
pub fn entry_signature(
    walk: &Walk,
    name: &str,
    launch_bounds: Option<(i64, i64)>,
) -> Result<String, Refusal> {
    let mut params = vec![];
    for slot in &walk.slots {
        params.push(format!("    {} {}", param_type(slot)?, param_name(slot)?));
    }
    params.extend(
        [
            "    long long base",
            "    long long count",
            "    long long nSamples",
        ]
        .map(String::from),
    );
    let bound = launch_bounds.map_or(String::new(), |(threads, blocks)| {
        format!(" __launch_bounds__({threads}, {blocks})")
    });
    Ok(format!(
        "extern \"C\" __global__ void{bound} {name}(\n{})",
        params.join(",\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_lane_is_the_default_prologue() {
        assert!(
            index_prologue(&Geometry::default()).contains("blockIdx.x) * blockDim.x + threadIdx.x")
        );
    }

    #[test]
    fn team_of_a_warp_binds_the_lane_place() {
        let geometry = Geometry {
            team: Some((32, 2)),
            ..Geometry::default()
        };
        let text = index_prologue(&geometry);
        assert!(
            text.contains("(threadIdx.x >> 5)")
                && text.contains("hawk_flat >= nSamples")
                && text.contains("hawk_wl")
        );
    }

    #[test]
    fn segment_reads_the_run_bounds() {
        let geometry = Geometry {
            segment: Some(("lut_offsets".into(), 3)),
            ..Geometry::default()
        };
        let text = index_prologue(&geometry);
        assert!(text.contains(
            "lut_offsets[aether::SampleIndex::make(static_cast<std::size_t>(4))].eval()"
        ));
    }
}
