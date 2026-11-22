//! The device translation unit: prelude, `__global__` entry and one lane per thread around a
//! rendered body, written from
//! the walk, the build contract and the body string. A construct this build does not write yet is
//! refused as
//! `not in this build: <construct>`.
use super::prologue::{self, Geometry, QUALIFIER};
use super::text::*;
use crate::body::frame::needs::Need;
use crate::contract::types::{SinkPolicy, Target, Unit};
use crate::ir::analysis as an;
use crate::ir::door::RxCtx;
use crate::lanes::device as persist;
use crate::refuse::{nib, Refusal};
use crate::spell::binding_name;
use crate::unit::entry::{HostSource, Render};
use crate::unit::sums;
use crate::unit::two_path;
use crate::unit::{self, Arms, Meta, Walk};

/// The device's spellings of the helpers that have one arm per target.
const DEVICE_ARMS: Arms = Arms {
    accum_add: ARM_ACCUM_ADD,
    accum_add_compensated: ARM_ACCUM_ADD_COMPENSATED,
    accum_or: ARM_ACCUM_OR,
    finish_count: ARM_FINISH_COUNT,
    mix_count: ARM_MIX_COUNT,
    spacing: ARM_SPACING,
    tiled: TILED,
    warp_sum: WARP_SUM,
    sample_sum_warp: SAMPLE_SUM_WARP,
    banded_include: BANDED_SUM_INCLUDE,
    banded_sum: BANDED_SUM,
    banded_ss_helpers: BANDED_SS_HELPERS,
};

/// The device's spellings for a scalar mode: the banded mode's warp sum is an accumulator fold.
fn arms_of(banded: bool) -> Arms {
    if banded {
        Arms {
            warp_sum: BANDED_WARP_SUM,
            ..DEVICE_ARMS
        }
    } else {
        DEVICE_ARMS
    }
}

/// Refuses the forms of the call the device arm does not write yet, by name.
fn refuse_unwritten(unit: &Unit) -> Result<(), Refusal> {
    // a unit with a single step the entry does not write as two paths (a caller's body, a segment,
    // a compensated kind)
    let two_path = unit.one_step.is_some() && two_path::step_of(unit).is_none();
    let refused = [(two_path, "two-path entry")];
    match refused.iter().find(|(set, _)| *set) {
        Some((_, construct)) => Err(nib(construct)),
        None => Ok(()),
    }
}

/// The lines around the body of one path of a two-path unit: the staged-rows frame over the
/// path's own sinks when the unit serves several rows per block, else the plain lane loop.
fn path_frame(
    ctx: *mut RxCtx,
    input: &Render,
    geometry: &Geometry,
    sinks: &[u32],
) -> Result<(String, String), Refusal> {
    let unit = input.unit;
    if unit.rows > 1 {
        return super::staged::frame(ctx, sinks, unit.rows, &unit.build);
    }
    Ok((
        prologue::index_prologue(geometry),
        prologue::index_epilogue().to_string(),
    ))
}

/// One path of a two-path unit: its frame (opened, closed) and its body.
struct Path<'a> {
    /// The lines that open and close the path's lane loop.
    frame: &'a (String, String),
    /// The path's body.
    body: &'a str,
}

/// The entry lines of a two-path unit: a branch on the fused-steps word between the single step's
/// lane loop and the fused
/// one, each opened by its own marker statement so the compiler keeps the two sides apart.
fn two_path_lines(single: &Path, fused: &Path) -> Result<Vec<String>, Refusal> {
    let word = format!(
        "{}[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()",
        binding_name("lookup", 0, "fused_steps")?
    );
    let path = |label: &str, p: &Path| {
        vec![
            format!("    asm volatile(\"// hawk: {label}\");"),
            p.frame.0.clone(),
            p.body.to_string(),
            p.frame.1.clone(),
        ]
    };
    let mut lines = vec![format!("    if ({word} == 1) {{")];
    lines.extend(path("one step", single));
    lines.push("    } else {".to_string());
    lines.extend(path("fused steps", fused));
    lines.push("    }".to_string());
    Ok(lines)
}

/// The lane split a two-path unit's fast entries are cut from: only the finishing, sample-local kernel with the fused-steps
/// word whose fused body cuts into head, one trip with a runtime count, and tail has them. `None` for any other two-path unit,
/// which has no fast entries.
fn lane_split_of(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    w: &Walk,
) -> Result<Option<crate::lanes::split::LaneSplit>, Refusal> {
    let has_word = w.slots.iter().any(|slot| slot.name == "fused_steps");
    if w.finish.is_none() || w.cls != "sample_local" || !has_word {
        return Ok(None);
    }
    Ok(
        match crate::lanes::lane_split(ctx, walk, unit, "        ")? {
            Some((split, _)) if split.count.is_some() => Some(split),
            _ => None,
        },
    )
}

/// The kernel that folds the companion of `target` into it, for the kernel named `name`: the one place
/// that names it.
pub(crate) fn fold_entry(name: &str, target: &str) -> String {
    format!("{name}__fold_{target}")
}

/// The lines of one fold entry per companion: the kernel's own signature and arguments, one lane
/// per slot of the target.
fn fold_entries(
    ctx: *mut RxCtx,
    walk: u32,
    w: &Walk,
    name: &str,
    bounds: Option<(i64, i64)>,
) -> Result<Vec<String>, Refusal> {
    let mut lines = vec![];
    for (companion, target) in an::companions(ctx, walk)? {
        lines.push(prologue::entry_signature(
            w,
            &fold_entry(name, &target),
            bounds,
        )?);
        lines.extend([
            "{".to_string(),
            unpack(ctx, walk, w)?,
            prologue::index_prologue(&Geometry::default()),
        ]);
        let (into, from) = (
            binding_name("accum_out", 0, &target)?,
            binding_name("accum_out", 0, &companion)?,
        );
        lines.push(format!(
            "        hawk_abi::fold_companion({into}, {from}, i);"
        ));
        lines.extend([
            prologue::index_epilogue().to_string(),
            "}".to_string(),
            String::new(),
        ]);
    }
    Ok(lines)
}

/// The entry's argument bindings: each slot is a by-value parameter of the entry.
fn unpack(ctx: *mut RxCtx, walk: u32, w: &Walk) -> Result<String, Refusal> {
    let source = |slot: &unit::Slot| prologue::param_name(slot);
    unit::unpack_lines(ctx, walk, w, &source, &|_| Ok(None))
}

/// The sample-sum sinks of the unit, with the element type the mode gives their plane: a banded
/// mode sums in accumulators,
/// not in the plane's element type.
fn sample_sum_sinks(
    ctx: *mut RxCtx,
    unit: &Unit,
    w: &Walk,
    mode: &unit::Mode,
) -> Result<Vec<sums::SumSink>, Refusal> {
    let sum_nodes = sums::sample_sum_nodes(ctx, &unit.sinks)?;
    let mut sum_sinks = sums::sum_sinks(ctx, &sum_nodes, w)?;
    if mode.cells {
        let accumulator = if sum_sinks.first().is_some_and(|sink| sink.op == "sum") {
            "BandAccum"
        } else {
            "Band"
        };
        for sink in &mut sum_sinks {
            sink.element = format!("aether::banded::{accumulator}");
        }
    }
    Ok(sum_sinks)
}

/// The prelude flags of the unit's text: the body's recorded needs plus what the unit itself adds
/// (fold companions, finish,
/// narrow lane index, sample sums, a compensated kind's run helper).
fn needs_of_unit(
    input: &Render,
    w: &Walk,
    has_companions: bool,
    folds_axes: bool,
    has_sums: bool,
) -> unit::Needs {
    let unit = input.unit;
    let scanned = &input.needs;
    let narrow_int = unit
        .active
        .as_ref()
        .is_some_and(|active| unit::narrows_int(active, w));
    let row = if has_sums { unit.build.cells_row } else { 0 };
    let mut needs = unit::needs_of_body(
        scanned,
        input.needs_random,
        has_companions,
        w.finish.is_some(),
        narrow_int,
        has_sums,
        row,
    );
    needs.fold_axes = folds_axes;
    // a compensated kind's scattered target commits through a run of the kind's correction plane:
    // the helper is needed there too
    let compensated_kind = matches!(unit.kind.sink, SinkPolicy::Compensated { .. });
    if compensated_kind && scanned.has(Need::CompensatedRun) {
        needs.compensated = true
    }
    needs
}

/// The launch geometry the unit's entry indexes its lanes by: team, active set, segment and thread
/// box.
fn geometry_of(unit: &Unit) -> Geometry {
    Geometry {
        team: unit.team.as_ref().map(|team| (team.threads, team.warps)),
        active: unit.active.clone(),
        segment: unit
            .segment
            .as_ref()
            .map(|segment| (segment.offsets_binding.clone(), segment.j)),
        boxed: unit.thread_box.as_ref().map(|b| (b.t, b.stride)),
    }
}

/// The lines around the body of a one-path entry: the staged-rows frame, the sample-sum frame, or
/// the plain lane loop.
fn framed_lines(
    ctx: *mut RxCtx,
    input: &Render,
    geometry: &Geometry,
    sum_sinks: &[sums::SumSink],
    mode: &unit::Mode,
) -> Result<Vec<String>, Refusal> {
    let unit = input.unit;
    let body = input.body.clone();
    if unit.rows > 1 {
        let (opened, closed) = super::staged::frame(ctx, &unit.sinks, unit.rows, &unit.build)?;
        Ok(vec![opened, body, closed])
    } else if sum_sinks.is_empty() {
        Ok(vec![
            prologue::index_prologue(geometry),
            body,
            prologue::index_epilogue().to_string(),
        ])
    } else {
        let live_count = geometry
            .active
            .as_ref()
            .map(|active| unit::active_reads(active, "0").0);
        let (opened, closed) =
            super::sums::frame(&unit.build, sum_sinks, mode.cells, live_count.as_deref())?;
        Ok(vec![opened, body, closed])
    }
}

/// The entry lines of a two-path unit, and the lines of its fast entries cut from the single step.
fn two_path_entry(
    ctx: *mut RxCtx,
    walk: u32,
    input: &Render,
    w: &Walk,
    geometry: &Geometry,
) -> Result<(Vec<String>, Vec<String>), Refusal> {
    let unit = input.unit;
    let two_path::BodyParts { single, fused, .. } = two_path::parts(ctx, walk, unit)?;
    let step_sinks = two_path::step_of(unit).map_or(&[][..], |step| &step.sinks[..]);
    let single_frame = path_frame(ctx, input, geometry, step_sinks)?;
    let fused_frame = path_frame(ctx, input, geometry, &unit.sinks)?;
    let lines = two_path_lines(
        &Path {
            frame: &single_frame,
            body: &single,
        },
        &Path {
            frame: &fused_frame,
            body: &fused,
        },
    )?;
    let Some(split) = lane_split_of(ctx, walk, unit, w)? else {
        return Ok((lines, vec![]));
    };
    let signature = prologue::entry_signature(w, &unit.name, unit.build.launch_bounds)?;
    let counts_finished = split.guard.is_some();
    let cut = persist::Cut {
        name: &unit.name,
        signature,
        unpack: unpack(ctx, walk, w)?,
        split: &split,
        counts_finished,
        plain_prologue: prologue::index_prologue(&Geometry::default()),
    };
    let fast = persist::fast_entries(&cut, &single, geometry.active.is_some())?;
    Ok((lines, fast))
}

/// The unit's text and facts, from the unit, the body written for it and the contract the frame
/// reads.
pub fn render(ctx: *mut RxCtx, walk: u32, input: &Render) -> Result<HostSource, Refusal> {
    let unit = input.unit;
    let name = unit.name.clone();
    let mode = unit::mode_of(unit.scalar);
    let typed_walk = unit.walk.as_ref();
    let w = unit::walk_from(typed_walk.ok_or_else(|| nib("a unit without a kernel walk"))?);
    refuse_unwritten(unit)?;
    let sum_sinks = sample_sum_sinks(ctx, unit, &w, &mode)?;
    let companions = an::companions(ctx, walk)?;
    let needs = needs_of_unit(
        input,
        &w,
        !companions.is_empty(),
        unit::folds_axes(&w, &companions),
        !sum_sinks.is_empty(),
    );
    let geometry = geometry_of(unit);
    let runs = &input.needs.runs;
    let step = two_path::step_of(unit).is_some();
    if step && (!runs.is_empty() || !sum_sinks.is_empty()) {
        return Err(nib("a two-path unit with run commits or sample sums"));
    }
    let (looped, fast) = if step {
        two_path_entry(ctx, walk, input, &w, &geometry)?
    } else {
        let framed = framed_lines(ctx, input, &geometry, &sum_sinks, &mode)?;
        (unit::with_runs(framed, runs), vec![])
    };
    let exports = unit::exports(&unit.exports, true)?;
    let mut parts = vec![
        format!(
            "// HAWK-emitted cuda translation unit (scalar mode: {}).\n{}",
            mode.id, "// Body text is the ONE renderer's; this file is the entry wrapper.\n"
        ),
        unit::prelude(&mode, QUALIFIER, &arms_of(mode.cells), &needs),
        exports,
        String::new(),
        prologue::entry_signature(&w, &name, unit.build.launch_bounds)?,
        "{".to_string(),
        unpack(ctx, walk, &w)?,
    ];
    parts.extend(looped);
    parts.extend(["}".to_string(), String::new()]);
    parts.extend(fold_entries(
        ctx,
        walk,
        &w,
        &name,
        unit.build.launch_bounds,
    )?);
    let fast_names = (!fast.is_empty()).then(|| {
        (
            persist::persist_function(&name),
            persist::range_function(&name),
        )
    });
    parts.extend(fast);
    let mut text = parts.join("\n");
    if unit.literal_table && mode.id == "float64" {
        text = super::literals::literal_table(&text);
    }
    let (persist_function, range_function) = fast_names.unzip();
    Ok(HostSource {
        text,
        body: input.body.clone(),
        entry: name,
        meta: Meta {
            target: Target::Device,
            mode: mode.id,
            vectorised: false,
            thread_box: unit.thread_box.clone(),
        },
        facts: crate::unit::Facts {
            persist_function,
            range_function,
            ..crate::unit::Facts::default()
        },
        #[cfg(feature = "shadow")]
        needs_diffs: None,
    })
}
