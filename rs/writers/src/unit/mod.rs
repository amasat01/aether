//! What both unit writers share: the scalar mode, the walk as the contract carries it, the
//! spellings of its slots, the
//! prelude with the helpers a body asks for, the export block, the run-commit frame and the
//! unpacking of the entry's
//! arguments. The host and device arms differ only in the arguments they pass here.
use crate::body::frame::needs::{Need, Needs as BodyNeeds};
use crate::contract::types::Role;
use crate::contract::types::{Active, Axis, Exports, Scalar, Target, ThreadBox, Walk as TypedWalk};
use crate::ir::door::RxCtx;
use crate::refuse::{fault, not_written, Refusal};
use crate::spell::{binding_name, written_role, KEPT_PLANE, ZEROED_PLANE};
use text::*;

pub mod boxed;
pub mod device;
pub mod entry;
pub mod fused;
pub mod host;
pub mod sums;
mod team_banded;
pub mod team_stage;
pub mod text;
pub mod two_path;
use crate::ir::analysis as an;

/// The target's spellings of the helpers that have one arm per target.
pub struct Arms {
    /// Spelling of the atomic add of a term into a slot.
    pub accum_add: &'static str,
    /// Spelling of the compensated atomic add.
    pub accum_add_compensated: &'static str,
    /// Spelling of the atomic or.
    pub accum_or: &'static str,
    /// Spelling of the finished-count commit.
    pub finish_count: &'static str,
    /// Spelling of the mix-count commit.
    pub mix_count: &'static str,
    /// Spelling of the spacing helper.
    pub spacing: &'static str,
    /// The device-only helpers a body may name; empty for the host.
    pub tiled: &'static str,
    /// Spelling of the warp-sum helper; empty for the host.
    pub warp_sum: &'static str,
    /// The device's warp form of the row sum, appended to the shared one; empty for the host.
    pub sample_sum_warp: &'static str,
    /// The headers a banded unit with a row sum, a team or a warp sum includes; empty for the host.
    pub banded_include: &'static str,
    /// The banded forms of the row sum: the pair combine, and the helpers that replace the pinning
    /// one.
    pub banded_sum: &'static str,
    /// The banded forms of the row sum's output helpers; empty for the host.
    pub banded_ss_helpers: &'static str,
}

/// One binding per `arg_spec` slot, in order, each rebuilt from `source_of(slot)`; `binding_of` may
/// replace a slot's own
/// two lines (the host binds a mask through its own type). A finishing kernel's writable twin and a
/// promoted lookup
/// plane's cell follow the slot.
pub fn unpack_lines(
    ctx: *mut RxCtx,
    walk: u32,
    w: &Walk,
    source_of: &dyn Fn(&Slot) -> Result<String, Refusal>,
    binding_of: &dyn Fn(&Slot) -> Result<Option<String>, Refusal>,
) -> Result<String, Refusal> {
    let mut lines: Vec<String> = vec![];
    for s in &w.slots {
        let id = binding_name(&s.role, s.shape.len(), &s.name)?;
        let q = if WRITABLE.contains(&s.role.as_str()) {
            ""
        } else {
            "const "
        };
        let src = reconstruct(s, &source_of(s)?, &runtime_extents(w, s, source_of)?)?;
        match binding_of(s)? {
            Some(own) => lines.push(own),
            None => {
                lines.push(format!("    {q}{} {id} =", view_type(s)?));
                lines.push(format!("        {src};"));
            }
        }
        lines.push(format!("    (void){id};"));
        if s.role == "terminated" && w.finish.as_deref() == Some(s.name.as_str()) {
            // the finish epilogue's writable twin: same mirror, non-const view
            lines.push(format!("    {} {id}_w =\n        {src};", view_type(s)?));
            lines.push(format!("    (void){id}_w;"));
        }
        if s.role == "lookup" && an::param_plane(ctx, walk, &s.name)? {
            // a promoted uniform: ONE load, ahead of the sample index
            lines.push(format!("    const auto {id}_cell = {id}[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval();"));
            lines.push(format!("    (void){id}_cell;"));
        }
    }
    Ok(lines.join("\n"))
}

/// The slot roles an entry may write through.
pub const WRITABLE: [&str; 4] = ["mutable", "out", KEPT_PLANE, ZEROED_PLANE];

/// The element types an index plane may hold.
pub const INT_PLANE: [&str; 2] = ["int32", "uint32"];

/// The tag of the ABI the text is written against.
pub const ABI_TAG: &str = "aether-abi/2";

/// The sizes the entry exports when the contract states none.
pub const DERIVED_SIZES: [&str; 5] = [
    "sizeof(eagle::plugin::GRefMirror)",
    "sizeof(eagle::plugin::ScalarHandle)",
    "sizeof(eagle::plugin::IntHandle)",
    "sizeof(EAGLE_ABI_INDEX_T)",
    "sizeof(eagle::plugin::PartitionTriple)",
];

/// The template with its qualifier and arm markers filled.
pub fn fill(t: &str, qual: &str, arm: &str) -> String {
    t.replace("@QUAL@", qual).replace("@ARM@", arm)
}

/// The scalar mode by value: its id and `Real` spelling and headers.
pub struct Mode {
    /// The scalar mode's id.
    pub id: &'static str,
    /// The spelling of `Real`.
    pub real: &'static str,
    /// The headers the mode adds.
    pub includes: &'static str,
    /// Whether the mode is the banded one.
    pub cells: bool,
}

/// The facts of a rendered unit that its caller reads back: a typed record, written as one JSON object by `json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    /// The target the unit was written for.
    pub target: Target,
    /// The scalar mode's id.
    pub mode: &'static str,
    /// Whether the lane loop runs its lanes side by side; the device text never does.
    pub vectorised: bool,
    /// The thread box, when each thread serves several samples and the box has an axis.
    pub thread_box: Option<ThreadBox>,
}

impl Meta {
    /// The facts as JSON text: target, mode, vectorised and the thread box (null when none).
    pub fn json(&self) -> String {
        let boxed = match &self.thread_box {
            Some(ThreadBox {
                axis: Some(axis),
                t,
                stride,
            }) => format!("{{\"axis\":{axis},\"T\":{t},\"stride\":{stride}}}"),
            _ => "null".to_string(),
        };
        let target = match self.target {
            Target::Host => "host",
            Target::Device => "cuda",
        };
        format!(
            "{{\"target\":\"{target}\",\"mode\":\"{}\",\"vectorised\":{},\"box\":{boxed}}}",
            self.mode, self.vectorised
        )
    }
}

/// A companion plane of a compensated sum: its name, the plane it compensates and the symbol of the
/// host entry or device kernel that folds it into that plane.
#[derive(Debug, Clone, PartialEq)]
pub struct CompanionFact {
    /// The companion plane's name.
    pub plane: String,
    /// The name of the plane it compensates.
    pub target: String,
    /// The symbol of the host entry or device kernel that folds it.
    pub fold_entry: String,
}

/// The lane split of a finishing unit's step, as the caller prices it.
#[derive(Debug, Clone, PartialEq)]
pub struct StepFacts {
    /// The operations of one step.
    pub ops: i64,
    /// Whether the fast entries count the samples finished on entry (the body is guarded by its mask).
    pub entry_counts: bool,
}

/// What the writer decided about a rendered unit that its caller states about it: the symbol the
/// text exports for its entry, the row planes of its sample sums, its companion planes and the
/// step of a finishing unit that splits, and the branch-mix planes of its auto dispatch groups.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Facts {
    /// The symbol the text exports for its entry.
    pub symbol: String,
    /// The row plane of each sample-sum sink.
    pub row_planes: Vec<sums::RowPlane>,
    /// The companion planes.
    pub companions: Vec<CompanionFact>,
    /// The step of a finishing unit whose body is a lane split; none for any other unit.
    pub step: Option<StepFacts>,
    /// The branch-mix plane of each auto dispatch group, as the group's name and the plane's name.
    pub mix_planes: Vec<(String, String)>,
    /// The persistent entry the device text holds; none when the unit has no fast entries.
    pub persist_function: Option<String>,
    /// The contiguous-range entry the device text holds (under the build's fast-entries macro); none when the unit has no
    /// fast entries.
    pub range_function: Option<String>,
}

/// One argument slot of the entry: role, name, element type and shape.
pub struct Slot {
    /// The slot's role.
    pub role: String,
    /// The slot's name.
    pub name: String,
    /// The slot's element type.
    pub dtype: String,
    /// The slot's extents.
    pub shape: Vec<i64>,
    /// The slot's position in the argument list.
    pub index: i64,
    /// The axes of the plane, in storage order; empty for a plane without axes (one runtime axis).
    pub axes: Vec<Axis>,
}

/// The name of the slot that carries the runtime extent of one axis of a plane.
pub fn extent_name(plane: &str, label: &str) -> String {
    format!("{plane}_{label}")
}

/// The kernel's walk as the contract carries it.
pub struct Walk {
    /// The access class of the walk.
    pub cls: String,
    /// The name of the finishing slot, when the walk finishes.
    pub finish: Option<String>,
    /// The walk's slots in argument order.
    pub slots: Vec<Slot>,
}

/// The C++ element spelling of an element type of a plane.
pub fn element(dtype: &str) -> Result<String, Refusal> {
    Ok(match dtype {
        "f64" | "f32" => "Real".into(),
        "i32" | "i64" => "Int".into(),
        d if INT_PLANE.contains(&d) => format!("std::{d}_t"),
        "bool" => "bool".into(),
        d => {
            return Err(not_written(format!(
                "no C++ element spelling for dtype {d:?}"
            )))
        }
    })
}

/// The C++ type an entry binds a slot to: a plain value or a view over its extents. A plane with axes is a view of its value
/// shape followed by its axes (a static extent as a number, a runtime one as `dyn`); a plane without axes has one runtime axis.
pub fn view_type(s: &Slot) -> Result<String, Refusal> {
    if an::mirror_of(&s.role, &s.dtype, s.shape.len())? == "value" {
        if s.role == "nsamples" {
            return Ok("EAGLE_ABI_INDEX_T".into());
        }
        if s.role == "extent" {
            return Ok("std::int64_t".into());
        }
        return if INT_PLANE.contains(&s.dtype.as_str()) {
            Ok("Int".into())
        } else {
            element(&s.dtype)
        };
    }
    Ok(format!(
        "aether::View<{}, {}, aether::layout_right>",
        element(&s.dtype)?,
        extents_type(s)
    ))
}

/// The extents of a plane's view: its value shape, then its axes (a static extent as a number, a runtime one as `dyn`); a plane
/// without axes has one runtime axis.
fn extents_type(s: &Slot) -> String {
    let axes: Vec<String> = if s.axes.is_empty() {
        vec!["aether::dyn".into()]
    } else {
        s.axes
            .iter()
            .map(|a| a.extent.map_or("aether::dyn".into(), |e| e.to_string()))
            .collect()
    };
    let all: Vec<String> = s.shape.iter().map(|e| e.to_string()).chain(axes).collect();
    format!("aether::extents<{}>", all.join(", "))
}

/// The expression that rebuilds a slot's view from the entry's source text for it. A plane with axes, or one holding a value of
/// rank three or more, is built from the handle's data pointer and device and its extents: `runtime_extents` are the texts of
/// its runtime axes, in axis order. A plane without axes holding a value of rank zero to two is read through the unit's helpers
/// when it crosses as a mirror; one that crosses as a scalar handle (a lookup) is built from its extents as soon as its value has
/// a rank, since the helpers read a mirror.
pub fn reconstruct(s: &Slot, src: &str, runtime_extents: &[String]) -> Result<String, Refusal> {
    if an::mirror_of(&s.role, &s.dtype, s.shape.len())? == "value" {
        return Ok(src.into());
    }
    let e = element(&s.dtype)?;
    let mirror = an::mirror_of(&s.role, &s.dtype, s.shape.len())? == "GRefMirror";
    // the helpers every unit carries spell a mirror plane of rank zero to two without axes; any other plane is built from its
    // extents
    if !s.axes.is_empty() || s.shape.len() > 2 || (!mirror && !s.shape.is_empty()) {
        let (data, kind, id) = if mirror {
            ("data_", "deviceType_", "deviceId_")
        } else {
            ("data", "deviceType", "deviceId")
        };
        let ext = extents_type(s);
        return Ok(format!(
            "aether::View<{e}, {ext}, aether::layout_right>(\n            reinterpret_cast<{e}*>(({src}).{data}),\n            aether::layout_right::mapping<{ext}>({ext}({})),\n            aether::Device(static_cast<DLDeviceType>(({src}).{kind}), ({src}).{id}))",
            // a plane without axes has one runtime axis: its samples. A mirror's pitch counts them; a handle states the elements of
            // the whole plane (value components included), so the samples are those over the value's components.
            if s.axes.is_empty() {
                if mirror {
                    format!("static_cast<std::size_t>(({src}).compStride_)")
                } else {
                    let per_sample: i64 = s.shape.iter().product();
                    format!("static_cast<std::size_t>(({src}).samples / {per_sample})")
                }
            } else {
                runtime_extents.join(", ")
            }
        ));
    }
    Ok(match s.shape.len() {
        0 => format!("hawk_abi::scalar_view<{e}>({src})"),
        1 => format!("hawk_abi::vec_view<{e}, {}>({src})", s.shape[0]),
        _ => format!(
            "hawk_abi::mat_view<{e}, {}, {}>({src})",
            s.shape[0], s.shape[1]
        ),
    })
}

/// The texts of a plane's runtime extents, read from the slots that carry them, in axis order.
fn runtime_extents(
    w: &Walk,
    s: &Slot,
    source_of: &dyn Fn(&Slot) -> Result<String, Refusal>,
) -> Result<Vec<String>, Refusal> {
    let mut texts = vec![];
    for axis in s.axes.iter().filter(|a| a.extent.is_none()) {
        let wanted = extent_name(&s.name, &axis.label);
        let slot = w
            .slots
            .iter()
            .find(|x| x.role == "extent" && x.name == wanted)
            .ok_or_else(|| {
                fault(format!(
                    "plane {:?} has a runtime axis {:?} and the walk carries no extent slot for it",
                    s.name, axis.label
                ))
            })?;
        texts.push(source_of(slot)?);
    }
    Ok(texts)
}

/// The row-sum text of a banded mode: value-initialised zeros (an accumulator has no `E(0)`), the
/// warp shuffle through
/// the accumulator's own, and the generic output helpers in place of the pinning one.
fn banded_sample_sum(text: &str, helpers: &str) -> String {
    text.replace("E(0)", "E{}")
        .replace("-static_cast<E>(__builtin_inf())", "ssInf_<E>(-1.0f)")
        .replace("static_cast<E>(__builtin_inf())", "ssInf_<E>(1.0f)")
        .replace("__shfl_xor_sync(0xffffffffu, sh, off)", "ssShfl_(sh, off)")
        .replace("__shfl_xor_sync(0xffffffffu, sc, off)", "ssShfl_(sc, off)")
        .replacen(
            "template <class E>\n__device__ __forceinline__ void ssPin_",
            helpers,
            1,
        )
}

/// The flags the body text sets in the prelude.
pub struct Needs {
    /// Whether the body compensates a scattered store.
    pub store_compensated: bool,
    /// Whether the body names the tile helper.
    pub tiled: bool,
    /// Whether the body names the warp sum.
    pub warp_sum: bool,
    /// Whether the body draws random numbers.
    pub random: bool,
    /// Whether the body names the spacing helper.
    pub spacing: bool,
    /// Whether the body names the mix count.
    pub mix: bool,
    /// Whether the body names the integer helpers.
    pub integers: bool,
    /// Whether the body names the linear algebra library.
    pub linalg: bool,
    /// Whether the body names the runtime expression library.
    pub rt: bool,
    /// Whether the body names the geometry library.
    pub rays: bool,
    /// Whether the body names the team library.
    pub team: bool,
    /// Whether the body commits an or into a slot.
    pub scatter_or: bool,
    /// Whether the body commits an or at an element of a plane with axes.
    pub scatter_or_at: bool,
    /// Whether the body reads a value of rank above zero out of a plane with axes.
    pub view_item: bool,
    /// Whether the body adds a term at an element of a plane with axes.
    pub scatter_at: bool,
    /// Whether the body adds a compensated term at an element of a plane with axes.
    pub compensated_at: bool,
    /// Whether the body names an edge-rule helper of an indexed access.
    pub edges: bool,
    /// Whether the body adds compensated terms.
    pub compensated: bool,
    /// Whether the unit folds a companion into its target.
    pub fold: bool,
    /// Whether a folded plane has axes, which gives its fold the helper that walks the data in storage order.
    pub fold_axes: bool,
    /// Whether the body opens run commits.
    pub runs: bool,
    /// Whether the body updates an element through the library's named atomic helpers.
    pub element_atomic: bool,
    /// Whether the unit finishes lanes.
    pub finish: bool,
    /// Whether lane indices are kept in 32 bits.
    pub narrow_int: bool,
    /// Whether the unit writes sample sums.
    pub sample_sum: bool,
    /// The row width of the sample sums.
    pub row: i64,
}

/// The includes, the scalar aliases and the ABI helpers of the unit; `arms` are the target's
/// spellings of the helpers.
pub fn prelude(m: &Mode, q: &str, arms: &Arms, n: &Needs) -> String {
    let includes = include_block(m, arms, n);
    let helpers = helper_block(m, q, arms, n);
    format!(
        "{includes}\nusing Real = {};\nusing Int = {};\n\n{}{}",
        m.real,
        if n.narrow_int { "int" } else { "long long" },
        fill(CLAMP_COUNT, q, ""),
        fill(ABI_HELPERS, q, arms.accum_add).replace("@HELPERS@", &helpers)
    )
}

/// The include block: the common includes, then those the mode and the needs add after the marker.
fn include_block(m: &Mode, arms: &Arms, n: &Needs) -> String {
    let mut includes = UNIT_INCLUDES.to_string();
    let extra = [
        (n.linalg || n.rays || n.team, FACTS_INCLUDE),
        (!m.includes.is_empty(), m.includes),
        (n.random, RANDOM_INCLUDE),
        (n.edges, EDGE_INCLUDE),
        (n.runs, RUN_INCLUDE),
        (n.element_atomic, ELEMENT_ATOMIC_INCLUDE),
        (
            n.compensated || n.compensated_at || n.store_compensated || n.fold || n.sample_sum,
            TWO_SUM_INCLUDE,
        ),
        (n.linalg, LINALG_INCLUDE),
        (n.rt, RT_INCLUDE),
        (
            m.cells && (n.sample_sum || n.team || n.warp_sum),
            arms.banded_include,
        ),
        (n.rays, RAY_INCLUDE),
        (n.team, TEAM_INCLUDE),
    ]
    .iter()
    .filter(|x| x.0)
    .map(|x| x.1)
    .collect::<String>();
    if !extra.is_empty() {
        includes = includes.replacen(
            RANDOM_INCLUDE_MARKER,
            &format!("{RANDOM_INCLUDE_MARKER}{extra}"),
            1,
        )
    }
    includes
}

/// The helper definitions the needs ask for, in the order the text lists them.
fn helper_block(m: &Mode, q: &str, arms: &Arms, n: &Needs) -> String {
    let mut helpers = String::new();
    if n.compensated || n.store_compensated || n.fold || n.sample_sum {
        helpers += &fill(TWO_SUM_ERR, q, "")
    }
    if n.fold {
        helpers += &fill(
            if n.fold_axes {
                FOLD_COMPANION_AXES
            } else {
                FOLD_COMPANION
            },
            q,
            "",
        )
    }
    if n.sample_sum {
        let halves: Vec<String> = (1..(64 - (n.row as u64).leading_zeros()))
            .map(|k| (n.row >> k).to_string())
            .collect();
        if m.cells {
            helpers += &fill(arms.banded_sum, q, "")
        }
        let sums = format!("{SAMPLE_SUM}{}", arms.sample_sum_warp);
        let sums = if m.cells {
            banded_sample_sum(&sums, arms.banded_ss_helpers)
        } else {
            sums
        };
        helpers += &fill(&sums, q, "")
            .replace("@ROW@", &n.row.to_string())
            .replace("@HALF@", &(n.row / 2).to_string())
            .replace("@OFFSETS@", &halves.join(", "));
    }
    if n.compensated {
        helpers += &fill(ACCUM_ADD_COMPENSATED, q, arms.accum_add_compensated)
    }
    if n.compensated_at {
        helpers += &fill(ACCUM_ADD_COMPENSATED_AT, q, arms.accum_add_compensated)
    }
    if n.store_compensated {
        helpers += &fill(STORE_COMPENSATED, q, "")
    }
    if n.finish {
        helpers += &fill(FINISH_COUNT, q, arms.finish_count)
    }
    if n.spacing {
        helpers += &fill(SPACING, q, arms.spacing)
    }
    if n.mix {
        helpers += &fill(MIX_COUNT, q, arms.mix_count)
    }
    if n.integers {
        helpers += &fill(INT_HELPERS, q, "")
    }
    if n.scatter_or {
        helpers += &fill(ACCUM_OR, q, arms.accum_or)
    }
    if n.view_item {
        helpers += &fill(VIEW_ITEM, q, "")
    }
    if n.scatter_or_at {
        helpers += &fill(ACCUM_OR_AT, q, arms.accum_or)
    }
    if n.scatter_at {
        helpers += &fill(ACCUM_ADD_AT, q, arms.accum_add)
    }
    if n.tiled {
        helpers += &fill(arms.tiled, q, "")
    }
    if n.warp_sum {
        helpers += &fill(arms.warp_sum, q, "")
    }
    helpers
}

/// The loop between the declarations of the run states and their closes.
pub fn with_runs(loop_: Vec<String>, runs: &[String]) -> Vec<String> {
    if runs.is_empty() {
        return loop_;
    }
    let open = runs
        .iter()
        .map(|t| format!("    auto hawk_run_{t} = aether::accum::run::open({t});"))
        .collect::<Vec<_>>()
        .join("\n");
    let close = runs
        .iter()
        .map(|t| format!("    aether::accum::run::close(hawk_run_{t});"))
        .collect::<Vec<_>>()
        .join("\n");
    std::iter::once(open)
        .chain(loop_)
        .chain(std::iter::once(close))
        .collect()
}

/// The prelude flags the body's recorded needs set; `random`, `fold`, `finish`, `narrow_int`,
/// `sample_sum` and
/// `row` come from
/// the unit rather than the text.
pub fn needs_of_body(
    body: &BodyNeeds,
    random: bool,
    fold: bool,
    finish: bool,
    narrow_int: bool,
    sample_sum: bool,
    row: i64,
) -> Needs {
    Needs {
        store_compensated: body.has(Need::StoreCompensated),
        tiled: body.has(Need::Tiled),
        warp_sum: body.has(Need::WarpSum),
        random,
        spacing: body.has(Need::Spacing),
        mix: body.has(Need::MixCount),
        integers: body.has(Need::Integers),
        linalg: body.has(Need::Linalg),
        rt: body.has(Need::Rt),
        rays: body.has(Need::Geometry),
        team: body.has(Need::Team),
        scatter_or: body.has(Need::AccumOr),
        view_item: body.has(Need::ViewItem),
        scatter_or_at: body.has(Need::AccumOrAt),
        scatter_at: body.has(Need::AccumAt),
        edges: body.has(Need::Edge),
        compensated: body.has(Need::AccumCompensated),
        compensated_at: body.has(Need::AccumCompensatedAt),
        fold,
        fold_axes: false,
        runs: body.has(Need::AccumRun),
        element_atomic: body.has(Need::ElementAtomic),
        finish,
        narrow_int,
        sample_sum,
        row,
    }
}

/// Whether a companion's target plane has axes: the walk's slot of that name states them.
pub fn folds_axes(w: &Walk, companions: &[(String, String)]) -> bool {
    companions.iter().any(|(_, target)| {
        w.slots
            .iter()
            .any(|s| &s.name == target && !s.axes.is_empty())
    })
}

/// The scalar mode of a unit by value.
pub fn mode_of(scalar: Scalar) -> Mode {
    match scalar {
        Scalar::Float64 => Mode {
            id: "float64",
            real: "double",
            includes: "",
            cells: false,
        },
        Scalar::Float32 => Mode {
            id: "float32",
            real: "float",
            includes: "",
            cells: false,
        },
        Scalar::Banded => Mode {
            id: "banded",
            real: "aether::banded::BandedReal",
            includes:
                "#include <aether/banded/banded.h>\n#include <aether/banded/BandedRealOps.h>\n",
            cells: true,
        },
    }
}

/// The walk the writers spell slots from, taken from the unit's typed arguments.
pub fn walk_from(walk: &TypedWalk) -> Walk {
    Walk {
        cls: walk.access.name().to_string(),
        finish: walk.finish.clone(),
        slots: walk
            .slots
            .iter()
            .map(|slot| Slot {
                role: match slot.role {
                    Role::PlaneOut => written_role(slot.zeroed).to_string(),
                    other => other.name().to_string(),
                },
                name: slot.name.clone(),
                dtype: slot.dtype.name().to_string(),
                shape: slot.shape.clone(),
                index: slot.index,
                axes: slot.axes.clone().unwrap_or_default(),
            })
            .collect(),
    }
}

/// The export block of the unit: the layout self-check, a host object or a device object
/// (`device`), or the finished text the unit carries.
pub fn exports(exports: &Exports, device: bool) -> Result<String, Refusal> {
    let values: Vec<String> = match exports {
        Exports::Text(text) => return Ok(text.clone()),
        Exports::Derived => DERIVED_SIZES.iter().map(|s| s.to_string()).collect(),
        Exports::Sizes(sizes) => sizes.iter().map(|n| format!("{n}ull")).collect(),
    };
    let head = if device {
        format!("extern \"C\" __device__ const char eagle_abi_tag[] = \"{ABI_TAG}\";\nextern \"C\" __device__ unsigned long long eagle_layout_sizes[5] = {{")
    } else {
        format!("extern \"C\" const char eagle_abi_tag[] = \"{ABI_TAG}\";\nextern \"C\" const std::uint64_t eagle_layout_sizes[5] = {{")
    };
    Ok(format!(
        "\n// The self-check exports, DERIVED from this build.\n{head}\n    {},\n}};\n",
        values.join(",\n    ")
    ))
}

/// `(count expression, sample expression at launch position `position`)` of an active set.
pub fn active_reads(active: &Active, position: &str) -> (String, String) {
    let word = |binding: &str, at: &str, bits: i64| {
        format!("reinterpret_cast<const std::int{bits}_t*>({binding}.data())[{at}]")
    };
    let bits = match active.map_bits {
        crate::contract::types::MapBits::B32 => 32,
        crate::contract::types::MapBits::B64 => 64,
    };
    let count = word(&active.count_binding, "0", 32);
    match &active.map_binding {
        None => (count, format!("({position})")),
        Some(map) if active.group > 1 => {
            let group = active.group;
            let index = word(map, &format!("({position}) / {group}"), bits);
            (
                count,
                format!("static_cast<long long>({index}) * {group} + ({position}) % {group}"),
            )
        }
        Some(map) => (count, word(map, position, bits)),
    }
}

/// Whether the unit's `Int` is 32 bits: a lane set of 32-bit width that holds no integer plane or
/// by-value integer.
pub fn narrows_int(active: &Active, walk: &Walk) -> bool {
    let wide = |slot: &Slot| {
        slot.dtype == "i32"
            || slot.dtype == "i64"
            || (slot.role == "uniform" && INT_PLANE.contains(&slot.dtype.as_str()))
    };
    let narrow = active.map_bits == crate::contract::types::MapBits::B32;
    active.lane_set && narrow && !walk.slots.iter().any(wide)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exports_derived_and_overridden() {
        let derived = exports(&Exports::Derived, false).unwrap();
        assert!(derived.starts_with("\n// The self-check exports, DERIVED from this build.\nextern \"C\" const char eagle_abi_tag[] = \"aether-abi/2\";"));
        assert!(derived.ends_with("    sizeof(eagle::plugin::PartitionTriple),\n};\n"));
        assert!(exports(&Exports::Sizes([1, 2, 3, 4, 5]), false)
            .unwrap()
            .contains("    1ull,\n    2ull,"));
        assert!(exports(&Exports::Derived, true)
            .unwrap()
            .contains("extern \"C\" __device__ unsigned long long eagle_layout_sizes[5] = {"));
    }
    /// Arms with no text: the tests below look at the helper text, not at an arm's.
    fn empty_arms() -> Arms {
        Arms {
            accum_add: "",
            accum_add_compensated: "",
            accum_or: "",
            finish_count: "",
            mix_count: "",
            spacing: "",
            tiled: "",
            warp_sum: "",
            sample_sum_warp: "",
            banded_include: "",
            banded_sum: "",
            banded_ss_helpers: "",
        }
    }
    #[test]
    fn a_fold_of_a_plane_without_axes_keeps_the_flat_helper() {
        let m = mode_of(Scalar::Float32);
        let mut n = needs_of_body(&BodyNeeds::default(), false, true, false, false, false, 0);
        let plain = helper_block(&m, "static inline", &empty_arms(), &n);
        assert!(plain.contains("fold_companion") && !plain.contains("flat_element"));
        n.fold_axes = true;
        let walked = helper_block(&m, "static inline", &empty_arms(), &n);
        assert!(walked.contains("flat_element"));
    }
    #[test]
    fn the_compensated_add_at_an_element_is_a_helper_of_its_own() {
        let m = mode_of(Scalar::Float32);
        let mut n = needs_of_body(&BodyNeeds::default(), false, false, false, false, false, 0);
        n.compensated = true;
        let flat = helper_block(&m, "static inline", &empty_arms(), &n);
        assert!(flat.contains("ViewT& target, ViewT& comp") && !flat.contains("compensated_at"));
        n.compensated = false;
        n.compensated_at = true;
        let at = helper_block(&m, "static inline", &empty_arms(), &n);
        assert!(at.contains("accum_add_compensated_at") && !at.contains("ViewT& target"));
    }
    #[test]
    fn prelude_is_filled() {
        let arms = Arms {
            accum_add: "",
            accum_add_compensated: "",
            accum_or: "",
            finish_count: "",
            mix_count: "",
            spacing: "",
            tiled: "",
            warp_sum: "",
            sample_sum_warp: "",
            banded_include: "",
            banded_sum: "",
            banded_ss_helpers: "",
        };
        let m = mode_of(Scalar::Float32);
        let n = Needs {
            store_compensated: false,
            tiled: false,
            warp_sum: false,
            random: false,
            spacing: false,
            mix: false,
            integers: false,
            linalg: false,
            rt: false,
            rays: false,
            team: false,
            scatter_or: false,
            view_item: false,
            scatter_or_at: false,
            scatter_at: false,
            edges: false,
            compensated: false,
            compensated_at: false,
            fold: false,
            fold_axes: false,
            runs: false,
            element_atomic: false,
            finish: false,
            narrow_int: false,
            sample_sum: false,
            row: 0,
        };
        let p = prelude(&m, "static inline", &arms, &n);
        assert!(p.contains("using Real = float;\nusing Int = long long;") && !p.contains('@'));
    }

    /// Oracle: the extents written by hand, the value shape first, then each axis as a number or `dyn`.
    #[test]
    fn a_plane_with_axes_is_a_view_of_its_axes_rebuilt_from_the_handle_and_the_runtime_extents() {
        let plane = Slot {
            role: "lookup".into(),
            name: "p".into(),
            dtype: "f64".into(),
            shape: vec![],
            index: 0,
            axes: vec![
                Axis {
                    label: "i".into(),
                    extent: None,
                },
                Axis {
                    label: "k".into(),
                    extent: Some(3),
                },
            ],
        };
        assert_eq!(
            view_type(&plane).unwrap(),
            "aether::View<Real, aether::extents<aether::dyn, 3>, aether::layout_right>"
        );
        let text = reconstruct(&plane, "h", &["n".into()]).unwrap();
        assert!(
            text.contains("aether::extents<aether::dyn, 3>(n)"),
            "{text}"
        );
        assert!(text.contains("(h).data"), "{text}");
        assert_eq!(extent_name("p", "i"), "p_i");
    }
}
