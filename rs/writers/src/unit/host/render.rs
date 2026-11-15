//! The host translation unit: declarations, entry and loop around a rendered body, written from the
//! walk, the build
//! contract and the body string. A construct this build does not write yet is refused as `not in
//! this build: <construct>`.
use super::text::*;
use crate::body::frame::needs::Need;
use crate::body::frame::needs::Needs as BodyNeeds;
use crate::body::Class;
use crate::contract::types::{Active, Machine, SinkPolicy, Unit};
use crate::ir::analysis as an;
use crate::ir::door::{self, RxCtx};
use crate::lanes::host_interchange as interchange;
use crate::lanes::host_split as split;
use crate::lanes::host_text::{HOST_MASK_HELPERS, INDEPENDENT_ITERATIONS};
use crate::refuse::nib;
use crate::spell::binding_name;
use crate::unit::entry::{HostSource, Render};
use crate::unit::sums::*;
use crate::unit::two_path;
use crate::unit::*;

/// The host's spellings of the helpers that have one arm per target.
const HOST_ARMS: Arms = Arms {
    accum_add: ARM_ACCUM_ADD,
    accum_add_compensated: ARM_ACCUM_ADD_COMPENSATED,
    accum_or: ARM_ACCUM_OR,
    finish_count: ARM_FINISH_COUNT,
    mix_count: ARM_MIX_COUNT,
    spacing: ARM_SPACING,
    tiled: "",
    warp_sum: "",
    sample_sum_warp: "",
    banded_include: "",
    banded_sum: "",
    banded_ss_helpers: "",
};

/// The `params[k]` cast one slot's by-value mirror is read through.
fn deref(s: &Slot) -> Result<String, String> {
    let shape = an::mirror_of(&s.role, &s.dtype, s.shape.len())?;
    Ok(if shape == "value" {
        format!(
            "*static_cast<const {}*>(params[{}])",
            view_type(s)?,
            s.index
        )
    } else {
        format!(
            "*static_cast<const eagle::plugin::{shape}*>(params[{}])",
            s.index
        )
    })
}

/// The entry's argument bindings: each slot read from `params[k]`, a rank-0 mask bound through the
/// host's mask type.
fn unpack(ctx: *mut RxCtx, walk: u32, w: &Walk) -> Result<String, String> {
    let mask = |s: &Slot| -> Result<Option<String>, String> {
        if !is_mask(w, s) {
            return Ok(None);
        }
        let id = binding_name(&s.role, &s.name)?;
        Ok(Some(format!(
            "    const hawk_abi::HostMask {id} =\n        hawk_abi::host_mask({});",
            deref(s)?
        )))
    };
    unpack_lines(ctx, walk, w, &deref, &mask)
}

/// Whether the slot is a rank-0 terminated mask of a sample-local walk, which the host binds
/// through its mask type.
fn is_mask(w: &Walk, s: &Slot) -> bool {
    w.cls == "sample_local" && s.role == "terminated" && s.shape.is_empty() && s.dtype == "bool"
}

/// The text that opens the loop over a contiguous run of samples: the plain loop, or the blocked
/// loop that fills the lane guards first under the lane-guard macro.
fn contiguous_loop(guards: &[String], finish: Option<&FinishLanes>) -> String {
    let plain = format!("{INDEPENDENT_ITERATIONS}\n    for (aether::offset_t hawk_j = hawk_lo; hawk_j < hawk_hi; ++hawk_j) {{");
    if guards.is_empty() {
        return plain;
    }
    let mut l: Vec<String> = vec!["#if HAWK_HOST_LANE_GUARDS".into()];
    for k in 0..guards.len() {
        l.push(format!(
            "    alignas(64) HAWK_LANE_T hawk_lanes{k}[hawk_abi::kHostLaneBlock];"
        ))
    }
    l.push("    for (aether::offset_t hawk_blo = hawk_lo; hawk_blo < hawk_hi;".into());
    l.push("         hawk_blo += hawk_abi::kHostLaneBlock) {".into());
    l.push(
        "    const aether::offset_t hawk_bhi = hawk_abi::lane_block_end(hawk_blo, hawk_hi);".into(),
    );
    for (k, g) in guards.iter().enumerate() {
        l.push(format!("    const hawk_abi::HostLaneMask hawk_lane{k} =\n        {g}.lanes(hawk_lanes{k}, hawk_blo, hawk_bhi);"))
    }
    if let Some(FinishLanes { mask, counter, .. }) = finish {
        l.push(format!(
            "    const hawk_abi::HostMask hawk_finish_mask = {mask};"
        ));
        l.push(format!("    const auto& hawk_finish_count = {counter};"));
        l.push("    HAWK_LANE_T hawk_finished = 0.0;".into());
    }
    for (k, g) in guards.iter().enumerate() {
        l.push(format!(
            "    const hawk_abi::HostLaneMask {g} = hawk_lane{k};"
        ))
    }
    if let Some(FinishLanes {
        mask,
        twin,
        counter,
    }) = finish
    {
        let k = guards.iter().position(|g| g == mask).unwrap_or(0);
        l.push(format!(
            "    const hawk_abi::HostLaneFinish {twin}{{hawk_lanes{k}, hawk_blo}};"
        ));
        l.push(format!(
            "    const hawk_abi::HostLaneTally {counter}{{&hawk_finished}};"
        ));
        l.push(format!("    (void){twin};"));
        l.push(format!("    (void){counter};"));
    }
    l.extend([
        INDEPENDENT_ITERATIONS.to_string(),
        "    for (aether::offset_t hawk_j = hawk_blo; hawk_j < hawk_bhi; ++hawk_j) {".into(),
        "#else".into(),
        plain,
        "#endif".into(),
    ]);
    l.join("\n")
}

/// The text that closes the loop `contiguous_loop` opened, with the finish commit when the lanes
/// finish.
fn contiguous_close(guards: &[String], finish: Option<&FinishLanes>) -> String {
    if guards.is_empty() {
        return "    }".into();
    }
    let commit = finish.map_or(String::new(), |FinishLanes { mask, .. }| format!("\n    hawk_abi::finish_lanes(hawk_finish_mask, hawk_lanes{}, hawk_blo, hawk_bhi, hawk_finished, hawk_finish_count);", guards.iter().position(|g| g == mask).unwrap_or(0)));
    format!("    }}\n#if HAWK_HOST_LANE_GUARDS{commit}\n    }}\n#endif")
}

/// Whether a loop of the walk leaves the lanes vectorised: an interchanged loop must be a
/// speculative step with no library
/// call in the loops nested in it; any other loop needs a compile-time trip count the compiler
/// fully unrolls (or one nested
/// in an interchanged loop).
fn loop_keeps_lanes(
    ctx: *mut RxCtx,
    node: u32,
    lanes: &[u32],
    inner_of_lanes: &[u32],
) -> Result<bool, String> {
    if lanes.contains(&node) {
        if !an::speculative_step(ctx, node)? {
            return Ok(false);
        }
        // a library call inside a loop nested in an interchanged one stays scalar
        let mut inside = door::body_nodes(ctx, node)?;
        inside.extend(door::field_refs(ctx, node, "nexts")?.unwrap_or_default());
        inside.extend(door::field_refs(ctx, node, "exit_cond")?.unwrap_or_default());
        for inner in an::loops_flat(ctx, &inside)? {
            if !door::class_is(ctx, inner, "Loop")? {
                continue;
            }
            for h in an::loops_flat(ctx, &door::body_nodes(ctx, inner)?)? {
                if an::is_library_call(ctx, h)? {
                    return Ok(false);
                }
            }
        }
        return Ok(true);
    }
    // a loop with a run-time trip count, or more trips than the compiler fully unrolls, stays its
    // lane's own `for`
    let dynamic = door::field_text(ctx, node, "dynamic")?.is_some();
    let trip = door::field_int(ctx, node, "trip")?
        .ok_or_else(|| "loop trip is not a number".to_string())?;
    Ok(!(dynamic
        || (trip > crate::ir::runtime_names::UNROLL_LIMIT && !inner_of_lanes.contains(&node))))
}

/// Whether the host loop over samples vectorises: the verdict of the host's rules, over the IR
/// answers it asks for.
fn vectorises(ctx: *mut RxCtx, walk: u32, w: &Walk, machine: &Machine) -> Result<bool, String> {
    let lanes = an::interchanged_loops(ctx, walk)?;
    if w.cls != "sample_local" && !(w.cls == "cross_sample_read" && !lanes.is_empty()) {
        return Ok(false);
    }
    // a fixed-count loop nested in an interchanged one is vectorised across the lanes whatever its
    // length
    let mut inner_of_lanes: Vec<u32> = vec![];
    for &lp in &lanes {
        let inside = an::loops_flat(ctx, &door::body_nodes(ctx, lp)?)?;
        for h in inside {
            if door::class_is(ctx, h, "Loop")? {
                inner_of_lanes.push(h)
            }
        }
    }
    let (exact, routed) = (machine.exact_libm, machine.routed_pow);
    let mut branching = door::constant_list(ctx, "LINALG_FACTORISATIONS")?;
    branching.extend(door::constant_list(ctx, "LINALG_CONSUMERS")?);
    for node in an::kernel_nodes(ctx, walk)? {
        if exact
            && an::is_library_call(ctx, node)?
            && !(routed && door::field_text(ctx, node, "kind")?.as_deref() == Some("pow"))
        {
            return Ok(false);
        }
        // a factorisation or a consumer branches per lane (a pivot search or a convergence skip):
        // the kinds are the writer's rule
        if an::is_linalg(ctx, node)?
            && branching.contains(&door::field_text(ctx, node, "kind")?.unwrap_or_default())
        {
            return Ok(false);
        }
        match Class::of(ctx, node)? {
            // a strategy kind (auto, predicated) is a dispatch too
            class if class.as_str().ends_with("Dispatch") => {
                if an::switch_form(ctx, node)? {
                    return Ok(false);
                }
            }
            Class::Loop if !loop_keeps_lanes(ctx, node, &lanes, &inner_of_lanes)? => {
                return Ok(false)
            }
            _ => {}
        }
    }
    Ok(true)
}

/// The names of a finish epilogue that moved onto a guarded mask.
pub struct FinishLanes {
    /// The lane guard of the terminated mask the finish reads.
    pub mask: String,
    /// The lane finish object that stands for the mask's twin.
    pub twin: String,
    /// The finished-count binding the lanes tally into.
    pub counter: String,
}

/// The lane form of a walk: how the host loop runs its lanes.
pub struct LaneShape {
    /// Whether the loop runs its lanes side by side.
    pub simd: bool,
    /// The lane-guard names of the walk's masks.
    pub guards: Vec<String>,
    /// The finish epilogue, when it moves onto a guarded mask.
    pub finish: Option<FinishLanes>,
}

/// The unit's lane shape and optional loop forms: what the entry's loop is built around.
struct EntryShape {
    /// The kernel's walk, as the entry reads its slots.
    walk: Walk,
    /// Whether the loop runs its lanes side by side, with its lane guards and finish epilogue.
    lane: LaneShape,
    /// Whether a team stage runs the plain lane loop, a team of one.
    team: bool,
    /// The active-set form of the unit, when it has one.
    active: Option<Active>,
    /// The segment form: the offsets binding and the segment index.
    segment: Option<(String, i64)>,
    /// The thread box form: the box's period and stride.
    boxed: Option<(i64, i64)>,
    /// Whether the unit writes its entry as a single-step path and a fused one.
    two_path: bool,
}

/// The interchanged-loop form of a unit, when its walk has one: the loops the host runs
/// interchanged are a lane split for
/// a finishing kernel, else the body cut at them, run as trips outside and the tile's lanes inside.
#[derive(Default)]
struct Interchange {
    /// The lane text, when the unit has the interchanged form.
    text: Option<String>,
    /// The needs of the lane text, recorded while it was written.
    needs: crate::body::frame::needs::Needs,
    /// The prelude and export of a lane split.
    split: Option<SplitExtra>,
}

/// The helper text and the export a lane split adds to the unit.
struct SplitExtra {
    /// The helpers the split's text needs.
    prelude: String,
    /// The split's exported function.
    export: String,
}

/// Refuses the unit shapes the host text does not write: a two-path entry the unit does not
/// support, staged rows and the banded mode.
fn refuse_unwritten(unit: &Unit, m: &Mode, two_path: bool) -> Result<(), String> {
    for (set, what) in [
        (unit.one_step.is_some() && !two_path, "two-path entry"),
        (
            two_path && (unit.thread_box.is_some() || (unit.source_call && unit.team.is_some())),
            "two-path entry with a box or a team",
        ),
        (unit.rows != 1, "staged rows"),
        (m.cells, "banded mode"),
    ] {
        if set {
            return Err(nib(what));
        }
    }
    Ok(())
}

/// The helpers the text adds after the common prelude: the lane split's own, the interchanged
/// loops' (with the tile helper when
/// their text names it), or the blend and mask helpers the body asks for.
fn extra_prelude(interchange: &Interchange, masks: bool, blend: bool) -> String {
    let mask_helpers = if masks { HOST_MASK_HELPERS } else { "" };
    if let Some(split) = &interchange.split {
        format!("{BLEND_HELPERS}\n{}", split.prelude)
    } else if interchange.text.is_some() {
        format!(
            "{BLEND_HELPERS}\n{}",
            interchange::prelude(mask_helpers, interchange.needs.has(Need::TileLane))
        )
    } else {
        let blend_helpers = if blend {
            format!("{BLEND_HELPERS}\n")
        } else {
            String::new()
        };
        format!("{blend_helpers}{mask_helpers}")
    }
}

/// Whether the unit's frame is the plain one, so the interchanged forms may apply: the unit's own
/// body and loop, with no
/// active set, box, segment, run commits or sample sums.
fn is_plain_frame(unit: &Unit, scanned: &BodyNeeds, sums: &[u32]) -> bool {
    !unit.prerendered_body
        && unit.active.is_none()
        && unit.thread_box.is_none()
        && unit.segment.is_none()
        && scanned.runs.is_empty()
        && sums.is_empty()
}

/// The lane split of a finishing kernel, when the unit has one: its text, needs, prelude and
/// export.
fn lane_split_form(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    shape: &EntryShape,
    fin: &FinishLanes,
    m: &Mode,
) -> Result<Option<Interchange>, String> {
    let real_bytes = if m.real == "float" { 4 } else { 8 };
    let slots: Vec<(String, String)> = shape
        .walk
        .slots
        .iter()
        .map(|s| (s.role.clone(), s.name.clone()))
        .collect();
    let finish = (fin.mask.clone(), fin.twin.clone(), fin.counter.clone());
    let split = split::split_unit(
        ctx,
        walk,
        unit,
        &split::SplitRequest {
            name: &unit.name,
            guards: &shape.lane.guards,
            finish: &finish,
            real_bytes,
            slots: &slots,
            finish_name: shape.walk.finish.as_deref().unwrap_or(""),
        },
    )?;
    Ok(split.map(|split| Interchange {
        text: Some(split.text),
        needs: split.needs,
        split: Some(SplitExtra {
            prelude: split.prelude,
            export: split.export,
        }),
    }))
}

/// The interchanged-loop form of the unit: a finishing kernel as a lane split, else the body cut at
/// the loops the host runs
/// interchanged. Empty when the frame is not the plain one or the walk has neither.
fn interchange_of(
    ctx: *mut RxCtx,
    walk: u32,
    input: &Render,
    shape: &EntryShape,
    sums: &[u32],
) -> Result<Interchange, String> {
    let unit = input.unit;
    let scanned = &input.needs;
    let eligible = is_plain_frame(unit, scanned, sums)
        && !scanned.has(Need::AccumCompensated)
        && !scanned.has(Need::StoreCompensated)
        && an::lane_form_access(ctx, walk)?;
    if !eligible {
        return Ok(Interchange::default());
    }
    if let Some(fin) = &shape.lane.finish {
        let m = mode_of(unit.scalar);
        if let Some(found) = lane_split_form(ctx, walk, unit, shape, fin, &m)? {
            return Ok(found);
        }
    }
    let cut = an::interchanged_loops(ctx, walk)?;
    if cut.is_empty() {
        return Ok(Interchange::default());
    }
    let (segments, mut needs) = crate::lanes::lane_segments(ctx, walk, unit, "        ", &cut)?;
    let text = interchange::loop_interchange_text(&segments, &mut needs)?;
    Ok(Interchange {
        text: Some(text),
        needs,
        split: None,
    })
}

/// The prelude flags of the unit's text: the body's recorded needs, what the unit itself adds, and
/// what the interchanged
/// loop names (it may name a library the body does not).
fn needs_of_unit(
    input: &Render,
    shape: &EntryShape,
    has_companions: bool,
    has_sums: bool,
    interchange: &Interchange,
) -> Needs {
    let unit = input.unit;
    let scanned = &input.needs;
    // a lane set of 32-bit width keeps its lanes, and so every index of them, in 32 bits
    let narrow_int = shape
        .active
        .as_ref()
        .is_some_and(|a| narrows_int(a, &shape.walk));
    let mut needs = needs_of_body(
        scanned,
        input.needs_random,
        has_companions,
        shape.walk.finish.is_some(),
        narrow_int,
        has_sums,
        unit.build.cells_row,
    );
    // a compensated kind's scattered target commits through a run of the kind's correction plane:
    // the helper is needed there too
    let compensated_kind = matches!(unit.kind.sink, SinkPolicy::Compensated { .. });
    if compensated_kind && scanned.has(Need::CompensatedRun) {
        needs.compensated = true
    }
    if interchange.text.is_some() {
        needs.linalg |= interchange.needs.has(Need::Linalg);
        needs.rt |= interchange.needs.has(Need::Rt)
    }
    needs
}

/// The lines of the entry's body: the loop around the body (with its run commits), the sample-sum
/// frame, or the branch
/// between the single-step and the fused loops of a two-path unit; wrapped in the interchanged form
/// when the unit has one.
fn entry_lines(
    ctx: *mut RxCtx,
    walk: u32,
    input: &Render,
    shape: &EntryShape,
    frame: Option<(String, String)>,
    tiled: &Option<String>,
    sums: &[u32],
) -> Result<Vec<String>, String> {
    let runs = &input.needs.runs;
    let (prologue, epilogue) = loop_frame(&shape.lane, &shape.active, &shape.segment, shape.boxed)?;
    if shape.two_path {
        if !runs.is_empty() || !sums.is_empty() {
            return Err(nib("a two-path unit with run commits or sample sums"));
        }
        let unit = input.unit;
        let two_path::BodyParts { single, fused, .. } = two_path::parts(ctx, walk, unit)?;
        let (step_handle, step_walk) = two_path::step_walk(ctx, walk, unit)?;
        let step_lane = lane_shape(ctx, step_handle, &step_walk, shape.team, &shape.active)?;
        let (one_open, one_close) =
            loop_frame(&step_lane, &shape.active, &shape.segment, shape.boxed)?;
        let one = PathLines {
            open: one_open,
            body: single,
            close: one_close,
        };
        let many = PathLines {
            open: prologue,
            body: fused,
            close: epilogue,
        };
        return two_path_lines(one, many, tiled);
    }
    let framed = match frame {
        Some((open, close)) => vec![open, input.body.clone(), close],
        None => vec![prologue, input.body.clone(), epilogue],
    };
    Ok(tiled_or(tiled, with_runs(framed, runs)))
}

/// The export lines of the unit: its exported sizes, and the lane split's export or the blank line
/// the interchanged text
/// follows.
fn export_lines(unit: &Unit, interchange: &Interchange) -> Result<String, String> {
    let base = exports(&unit.exports, false)?;
    Ok(match &interchange.split {
        Some(split) => format!("{base}\n{}", split.export),
        None if interchange.text.is_some() => format!("{base}\n"),
        None => base,
    })
}

/// The lines of one fold entry per companion: the kernel's own signature and arguments, one fold of
/// the companion into its
/// target over the call's range.
fn companion_folds(
    ctx: *mut RxCtx,
    walk: u32,
    w: &Walk,
    name: &str,
    companions: &[(String, String)],
) -> Result<Vec<String>, String> {
    let mut lines = vec![];
    for (comp, target) in companions {
        lines.extend([signature(&format!("{name}__fold_{target}")), "{".into(), unpack(ctx, walk, w)?,
            "    const std::int64_t hawk_end = base + count;\n    for (std::int64_t hawk_i = base; hawk_i < hawk_end; ++hawk_i) {\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n        (void)hawk_i;\n        (void)nSamples;".into(),
            format!("        hawk_abi::fold_companion({}, {}, i);", binding_name("accum_out", target)?, binding_name("accum_out", comp)?), "    }".into(), "}".into(), String::new()]);
    }
    Ok(lines)
}

/// The unit's facts as JSON: the target, the scalar mode and whether the lane loop vectorises.
fn meta_of(mode: &Mode, vectorised: bool) -> String {
    format!(
        "{{\"target\":\"host\",\"mode\":\"{}\",\"vectorised\":{vectorised},\"box\":null}}",
        mode.id
    )
}

/// The text of the unit and its facts, from the unit, the body written for it and the contract the
/// frame reads.
pub fn render(ctx: *mut RxCtx, walk: u32, input: &Render) -> Result<HostSource, String> {
    let unit = input.unit;
    let name = unit.name.clone();
    let m = mode_of(unit.scalar);
    let typed_walk = unit.walk.as_ref();
    let w = walk_from(typed_walk.ok_or_else(|| nib("a unit without a kernel walk"))?);
    let two_path = two_path::step_of(unit).is_some();
    refuse_unwritten(unit, &m, two_path)?;
    let companions = an::companions(ctx, walk)?;
    let sums = sample_sum_nodes(ctx, &unit.sinks)?;
    let team = unit.team.is_some();
    let active = unit.active.clone();
    let lane = lane_shape(ctx, walk, &w, team, &active)?;
    let frame = if sums.is_empty() {
        None
    } else {
        let live = active.as_ref().map(|a| active_reads(a, "0").0);
        Some(sample_sum_frame(
            ctx,
            &sums,
            &w,
            unit.build.cells_row,
            live,
        )?)
    };
    let shape = EntryShape {
        walk: w,
        lane,
        team,
        active,
        segment: unit
            .segment
            .as_ref()
            .map(|s| (s.offsets_binding.clone(), s.j)),
        boxed: unit.thread_box.as_ref().map(|b| (b.t, b.stride)),
        two_path,
    };
    let interchange = interchange_of(ctx, walk, input, &shape, &sums)?;
    let needs = needs_of_unit(
        input,
        &shape,
        !companions.is_empty(),
        !sums.is_empty(),
        &interchange,
    );
    let masks =
        shape.walk.cls == "sample_local" && shape.walk.slots.iter().any(|s| s.role == "terminated");
    let blend = input.needs.has(Need::Blend) || input.needs.has(Need::SelectItem);
    let extra = extra_prelude(&interchange, masks, blend);
    let entry = entry_lines(ctx, walk, input, &shape, frame, &interchange.text, &sums)?;
    let mut parts = vec![
        format!("// HAWK-emitted host translation unit (scalar mode: {}).\n// Body text is the ONE renderer's; this file is the entry wrapper.\n", m.id),
        prelude(&m, "static inline", &HOST_ARMS, &needs),
        extra,
        export_lines(unit, &interchange)?,
        String::new(),
        signature(&name),
        "{".into(),
        unpack(ctx, walk, &shape.walk)?,
    ];
    parts.extend(entry);
    parts.extend(["}".into(), String::new()]);
    parts.extend(companion_folds(ctx, walk, &shape.walk, &name, &companions)?);
    let vectorised =
        shape.boxed.is_none() && !two_path && vectorises(ctx, walk, &shape.walk, &unit.machine)?;
    #[cfg(feature = "shadow")]
    let needs_diffs = interchange.text.as_deref().map(|text| {
        let shown = BodyNeeds::scan(text);
        [Need::Linalg, Need::Rt, Need::TileLane]
            .into_iter()
            .filter(|n| shown.has(*n) != interchange.needs.has(*n))
            .map(|n| format!("{n:?} (lane text)"))
            .collect()
    });
    Ok(HostSource {
        text: parts.join("\n").replace("HAWK_LANE_T", m.real),
        body: input.body.clone(),
        entry: name,
        #[cfg(feature = "shadow")]
        needs_diffs,
        meta: meta_of(&m, vectorised),
    })
}

/// The lane form of a walk: whether the host loop runs its lanes side by side, the lane-guard names
/// of its masks, and the
/// finish epilogue's names when it moves onto a guarded mask.
fn lane_shape(
    ctx: *mut RxCtx,
    walk: u32,
    w: &Walk,
    team: bool,
    active: &Option<Active>,
) -> Result<LaneShape, String> {
    let reads_position = active.is_some() && reads_sample_position(ctx, walk)?;
    let simd = w.cls == "sample_local"
        && !team
        && !active
            .as_ref()
            .is_some_and(|a| a.group > 1 || reads_position);
    let guards: Vec<String> = if simd {
        w.slots
            .iter()
            .filter(|s| is_mask(w, s))
            .map(|s| binding_name(&s.role, &s.name))
            .collect::<Result<_, _>>()?
    } else {
        vec![]
    };
    let finish = match &w.finish {
        Some(f) if guards.contains(&binding_name("terminated", f)?) => Some(FinishLanes {
            mask: binding_name("terminated", f)?,
            twin: format!("{}_w", binding_name("terminated", f)?),
            counter: binding_name("lookup", "finished_count")?,
        }),
        _ => None,
    };
    Ok(LaneShape {
        simd,
        guards,
        finish,
    })
}

/// Whether any node of the walk reads the sample's position.
fn reads_sample_position(ctx: *mut RxCtx, walk: u32) -> Result<bool, String> {
    for node in door::walk_order(ctx, walk)? {
        if door::class_is(ctx, node, "SamplePosition")? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The loop around a body, over the lanes of a walk whose `lane_shape` is given: the index prologue
/// and epilogue of the entry.
/// `active`, `segment` and `boxed` are the unit's own forms.
fn loop_frame(
    shape: &LaneShape,
    active: &Option<Active>,
    segment: &Option<(String, i64)>,
    boxed: Option<(i64, i64)>,
) -> Result<(String, String), String> {
    let (simd, guards, finish) = (shape.simd, &shape.guards, &shape.finish);
    let loop_head = "        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));\n        (void)hawk_i;\n        (void)nSamples;";
    let index = |p: i64| format!("aether::SampleIndex::make(static_cast<std::size_t>({p}))");
    let fin_args: Vec<String> = finish
        .as_ref()
        .map_or(vec![], |f| vec![f.twin.clone(), f.counter.clone()]);
    let prologue = match (&segment, simd) {
        _ if active.is_some() => {
            let a = active.as_ref().unwrap();
            let (count_expr, sample_expr) = active_reads(a, "hawk_t");
            let bound = format!("    const std::int64_t hawk_live = static_cast<std::int64_t>({count_expr});\n    const std::int64_t hawk_end = base + count;\n    const std::int64_t hawk_stop = hawk_end < hawk_live ? hawk_end : hawk_live;\n");
            if !simd {
                let past = if a.group > 1 { "        if (hawk_i >= nSamples) continue;\n" } else { "" };
                format!("{bound}    for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t) {{\n        const std::int64_t hawk_i = static_cast<std::int64_t>({sample_expr});\n{past}        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n        (void)hawk_i;\n        (void)nSamples;")
            } else {
                let mut params: String = guards.iter().map(|g| format!(", const auto& {g}")).collect();
                if let [t, k] = fin_args.as_slice() { params += &format!(", auto&& {t}, const auto& {k}") }
                format!("{bound}    auto hawk_sample = [&](const std::int64_t hawk_i{params})\n        __attribute__((always_inline)) {{\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n        (void)hawk_i;\n        (void)nSamples;")
            }
        }
        _ if boxed.is_some() => {
            // the body is the box frame the renderer composed over the entry's lane variable
            let (t, st) = boxed.unwrap_or((1, 1));
            format!("    const std::int64_t hawk_end = base + count;\n    for (std::int64_t hawk_t = base; hawk_t < hawk_end; ++hawk_t) {{\n        const long long hawk_box_own = hawk_t + hawk_t / {st} * {};\n        if (hawk_box_own >= nSamples) break;", (t - 1) * st)
        }
        (None, true) => format!("    const aether::offset_t hawk_lo =\n        static_cast<aether::offset_t>(base);\n    const aether::offset_t hawk_hi =\n        static_cast<aether::offset_t>(base + count);\n{}\n{loop_head}", contiguous_loop(guards, finish.as_ref())),
        (None, false) => "    const std::int64_t hawk_end = base + count;\n    for (std::int64_t hawk_i = base; hawk_i < hawk_end; ++hawk_i) {\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n        (void)hawk_i;\n        (void)nSamples;".to_string(),
        (Some((off, j)), simd) => {
            let (b, e) = (format!("{off}[{}].eval()", index(*j)), format!("{off}[{}].eval()", index(j + 1)));
            if simd {
                format!("    const std::int64_t hawk_seg_end = static_cast<std::int64_t>({e});\n    const std::int64_t hawk_seg_begin = static_cast<std::int64_t>({b}) + base;\n    const std::int64_t hawk_seg_stop = hawk_seg_begin + count < hawk_seg_end\n        ? hawk_seg_begin + count : hawk_seg_end;\n    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(hawk_seg_begin);\n    const aether::offset_t hawk_hi = hawk_seg_stop > hawk_seg_begin\n        ? static_cast<aether::offset_t>(hawk_seg_stop) : hawk_lo;\n{}\n{loop_head}", contiguous_loop(guards, finish.as_ref()))
            } else {
                format!("    const std::int64_t hawk_seg_begin = static_cast<std::int64_t>({b});\n    const std::int64_t hawk_seg_end = static_cast<std::int64_t>({e});\n    for (std::int64_t hawk_tid = base; hawk_tid < base + count; ++hawk_tid) {{\n        const std::int64_t hawk_i = hawk_seg_begin + hawk_tid;\n        if (hawk_i >= hawk_seg_end) break;\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n        (void)hawk_i;\n        (void)nSamples;")
            }
        }
    };
    let epilogue = if let (Some(a), true) = (&active, simd) {
        let at = |pos: &str| format!("static_cast<std::int64_t>({})", active_reads(a, pos).1);
        let args: String = guards
            .iter()
            .chain(fin_args.iter())
            .map(|g| format!(", {g}"))
            .collect();
        format!("    }};\n    bool hawk_identity = base >= hawk_stop\n        || ({} == base && {} == hawk_stop - 1);\n    if (hawk_identity) {{\n        std::int64_t hawk_strays = 0;\n        for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t)\n            hawk_strays |= {} ^ hawk_t;\n        hawk_identity = hawk_strays == 0;\n    }}\n    if (hawk_identity) {{\n    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);\n    const aether::offset_t hawk_hi = hawk_stop > base\n        ? static_cast<aether::offset_t>(hawk_stop) : hawk_lo;\n{}\n        hawk_sample(static_cast<std::int64_t>(hawk_j){args});\n{}\n    }} else {{\n{INDEPENDENT_ITERATIONS}\n    for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t) {{\n        hawk_sample({}{args});\n    }}\n    }}", at("base"), at("hawk_stop - 1"), at("hawk_t"), contiguous_loop(guards, finish.as_ref()), contiguous_close(guards, finish.as_ref()), at("hawk_t"))
    } else if simd {
        contiguous_close(guards, finish.as_ref())
    } else {
        "    }".to_string()
    };
    Ok((prologue, epilogue))
}

/// The three texts of one path of a two-path entry: the loop's opening, the body inside it and its
/// close.
struct PathLines {
    /// The index prologue that opens the loop.
    open: String,
    /// The body of the path.
    body: String,
    /// The epilogue that closes the loop.
    close: String,
}

/// The entry lines of a two-path unit: a branch on the fused-steps word between the single step's
/// loop and the fused one.
fn two_path_lines(
    single: PathLines,
    fused: PathLines,
    tiled: &Option<String>,
) -> Result<Vec<String>, String> {
    let word = format!(
        "{}[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()",
        binding_name("lookup", "fused_steps")?
    );
    let mut lines = vec![
        format!("    if ({word} == 1) {{"),
        single.open,
        single.body,
        single.close,
        "    } else {".to_string(),
    ];
    lines.extend(tiled_or(tiled, vec![fused.open, fused.body, fused.close]));
    lines.push("    }".to_string());
    Ok(lines)
}

/// The plain loop, or the interchanged loop under its macro with the plain one kept as the
/// macro-off build.
fn tiled_or(tiled: &Option<String>, plain: Vec<String>) -> Vec<String> {
    match tiled {
        None => plain,
        Some(text) => std::iter::once("#if HAWK_HOST_INTERCHANGE".to_string())
            .chain([text.clone(), "#else".to_string()])
            .chain(plain)
            .chain(["#endif".to_string()])
            .collect(),
    }
}

/// The serial loop over the rows that start in the call's range, and its close, for the sample-sum
/// sinks.
fn sample_sum_frame(
    ctx: *mut RxCtx,
    sums: &[u32],
    w: &Walk,
    row: i64,
    live: Option<String>,
) -> Result<(String, String), String> {
    if row < 1 || row & (row - 1) != 0 {
        return Err(format!(
            "the host wing's cells.row must be a power of two; it is {row}"
        ));
    }
    let sinks = sum_sinks(ctx, sums, w)?;
    let width = sums.len() as i64;
    let store = store_lambda(&sinks, 1, row, false)?;
    let ops: Vec<&str> = sinks.iter().map(|sink| sink.op.as_str()).collect();
    let op = ops[0];
    let code = op_code(op)?;
    let elem = sinks[0].element.clone();
    let fill = identity_fill(op, row * width, "hawk_ss_rows", &elem)?;
    let opened = format!("    const std::int64_t hawk_lo = (base + {r1}) / {row} * {row};\n    std::int64_t hawk_hi = (base + count + {r1}) / {row} * {row};\n    if (hawk_hi > nSamples) hawk_hi = nSamples;\n    {elem} hawk_ss_rows[{row} * {width}] = {{}};\n{fill}    std::int64_t hawk_row = 0;\n{}\n    for (std::int64_t hawk_i = hawk_lo; hawk_i < hawk_hi; ++hawk_i) {{\n        {elem}* const hawk_ss = hawk_ss_rows + (hawk_i & {r1}) * {width};\n        const aether::SampleIndex i =\n            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));", store, r1 = row - 1);
    // an active-set run: a sample at or past the live count adds the identity (its row is still
    // stored)
    let opened = match &live {
        Some(count) => {
            format!("{opened}\n        if (hawk_i < static_cast<std::int64_t>({count})) {{")
        }
        None => opened,
    };
    let identity = if op == "sum" {
        "0".to_string()
    } else {
        format!("hawk_abi::ssIdentity_<{code}, {elem}>()")
    };
    let skip = if live.is_some() {
        format!("\n        }} else {{\n            for (int hawk_k = 0; hawk_k < {width}; ++hawk_k) hawk_ss[hawk_k] = {identity};\n        }}\n")
    } else {
        String::new()
    };
    let closed = format!("{skip}        if ((hawk_i & 31) == 31 || hawk_i + 1 == hawk_hi) {{\n            hawk_row = hawk_i >> 5;\n            hawk_abi::sampleSumRow_<{width}, {code}>(hawk_ss_rows, hawk_ss_store);\n        }}\n    }}");
    Ok((opened, closed))
}

/// The signature of the host entry named `name`.
fn signature(name: &str) -> String {
    format!("extern \"C\" void {name}_host(void* const* params,\n    std::int64_t base,\n    std::int64_t count,\n    std::int64_t nSamples)")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_loop_forms() {
        assert!(contiguous_loop(&[], None).ends_with("hawk_j < hawk_hi; ++hawk_j) {"));
        assert!(contiguous_loop(&["trm_x".into()], None)
            .contains("trm_x.lanes(hawk_lanes0, hawk_blo, hawk_bhi)"));
        assert_eq!(contiguous_close(&[], None), "    }");
    }
}
