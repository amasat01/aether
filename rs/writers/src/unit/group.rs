//! The host body of a unit whose walk is a group of lanes: groups run one after another inside one call, each wholly on the
//! calling thread, and each phase of a group is one loop over its lanes.
//!
//! The call's range is a range of GROUPS: `base` is the first group and `count` the number of groups, `nSamples` the lanes of
//! the whole launch. The lanes of a group are found from the IR's own facts of the group: `count` N takes the consecutive
//! lanes `[g * N, g * N + size)` (the last group of a launch may be partial, and its `size` is its real one); `axes` takes the
//! lanes along the spanned axes of the launch's lane grid, the other axes enumerating the groups in lane order.
//!
//! A phase reads scratch at its own place or at a place moved from it by a whole number. When a phase moves a place, its lanes
//! are cut where the edge rules act: the interior lanes (every moved place is in the plane) are one loop with plain indices,
//! and the few lanes at the two ends are written separately with the edge rule folding the index. No index in the interior
//! loop carries a remainder or a branch, which keeps the compiler's vectoriser on it.
use crate::body::frame::needs::{Needs, Rendered};
use crate::body::scratch::{place_of, Place};
use crate::body::{render_phase_with_needs, PhaseForm};
use crate::contract::types::{Target, Unit};
use crate::ir::door::{self, RxCtx};
use crate::refuse::{fault, not_written, Refusal};
use crate::spell::binding_name;
use crate::spell::group::{
    group_calls, group_function_open, lane_pointer_declarations, scratch_declare,
};

/// The indent of the statements of the group loop.
const GROUP_INDENT: &str = "        ";
/// The indent of the statements inside a phase loop: the lane's binding and the phase's body.
const LOOP_INDENT: &str = "            ";

/// The `Group` node among the unit's sinks, if there is one. A unit with no walk (`walk` is `u32::MAX`) has none: its sinks are
/// not nodes of this context, and the render refuses it by name.
pub fn of(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<Option<u32>, Refusal> {
    if walk == u32::MAX {
        return Ok(None);
    }
    for &sink in &unit.sinks {
        if door::class_is(ctx, sink, "Group")? {
            return Ok(Some(sink));
        }
    }
    Ok(None)
}

/// One lane axis of the launch: its extent, its stride (lanes between two neighbours along it) and whether it spans a group.
struct Axis {
    /// The text of the extent.
    extent: String,
    /// The text of the stride.
    stride: String,
    /// Whether the axis is one of those that span a group.
    spanned: bool,
}

/// How the lanes of a group are found.
enum Lanes {
    /// `N` consecutive lanes from `group * N`.
    Count(i64),
    /// The lanes along the spanned axes of the grid.
    Axes(Vec<Axis>),
}

/// Why a kernel's group is not written on the host.
fn unwritten(kernel: &str, what: &str) -> Refusal {
    not_written(format!("{kernel:?}: {what} is not written on the host yet"))
}

/// The product of the texts of `factors`, `1` for none.
fn product(factors: &[&str]) -> String {
    if factors.is_empty() {
        "1".to_string()
    } else {
        factors.join(" * ")
    }
}

/// The lane axes of an `axes` group, from the group's grid: the extents (a run-time extent is the launch's lanes over the others)
/// and strides in lane order, and which axes span a group.
fn grid_axes(ctx: *mut RxCtx, group: u32, kernel: &str) -> Result<(Vec<Axis>, String), Refusal> {
    let grid = door::field_axes(ctx, group, "grid")?;
    let named = door::field_names(ctx, group, "axes")?.unwrap_or_default();
    let fixed: i64 = grid.iter().filter_map(|(_, e)| *e).product();
    if grid.iter().filter(|(_, e)| e.is_none()).count() > 1 {
        return Err(unwritten(
            kernel,
            "a group over a launch with two run-time lane axes",
        ));
    }
    let extents: Vec<String> = grid
        .iter()
        .map(|(_, e)| e.map_or_else(|| format!("(nSamples / {fixed})"), |e| e.to_string()))
        .collect();
    let axes: Vec<Axis> = grid
        .iter()
        .enumerate()
        .map(|(k, (label, _))| {
            let later: Vec<&str> = extents[k + 1..].iter().map(String::as_str).collect();
            Axis {
                extent: extents[k].clone(),
                stride: product(&later),
                spanned: named.contains(label),
            }
        })
        .collect();
    let size = {
        let spanned: Vec<&str> = axes
            .iter()
            .filter(|a| a.spanned)
            .map(|a| a.extent.as_str())
            .collect();
        product(&spanned)
    };
    Ok((axes, size))
}

/// The statements that set `hawk_base` (the lane of rank 0) and `hawk_size` for the group `hawk_g` of an `axes` group.
fn axes_group_lines(axes: &[Axis], size: &str) -> Vec<String> {
    let others: Vec<&Axis> = axes.iter().filter(|a| !a.spanned).collect();
    let mut terms = vec![];
    for (j, axis) in others.iter().enumerate() {
        let inner: Vec<&str> = others[j + 1..].iter().map(|a| a.extent.as_str()).collect();
        let divided = if inner.is_empty() {
            "hawk_g".to_string()
        } else {
            format!("hawk_g / ({})", product(&inner))
        };
        let coordinate = if j == 0 {
            divided
        } else {
            format!("({divided}) % {}", axis.extent)
        };
        terms.push(format!("({coordinate}) * {}", axis.stride));
    }
    let base = if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    };
    vec![
        format!("{GROUP_INDENT}const std::int64_t hawk_base = {base};"),
        format!("{GROUP_INDENT}const std::int64_t hawk_size = {size};"),
    ]
}

/// The offset of the lane of place `hawk_rank` from the group's first lane: the rank itself for consecutive lanes, else the
/// rank split over the runs of spanned axes that are neighbours in the grid.
fn lane_offset(lanes: &Lanes) -> String {
    let Lanes::Axes(axes) = lanes else {
        return "hawk_rank".to_string();
    };
    let spanned: Vec<(usize, &Axis)> = axes.iter().enumerate().filter(|(_, a)| a.spanned).collect();
    // runs of spanned axes that are neighbours in the grid: (extents of the run, stride of its innermost axis)
    let mut runs: Vec<(Vec<&str>, &str)> = vec![];
    let mut previous: Option<usize> = None;
    for (k, axis) in &spanned {
        match runs.last_mut() {
            Some(run) if previous.is_some_and(|j| j + 1 == *k) => {
                run.0.push(&axis.extent);
                run.1 = &axis.stride;
            }
            _ => runs.push((vec![&axis.extent], &axis.stride)),
        }
        previous = Some(*k);
    }
    let sizes: Vec<String> = runs
        .iter()
        .map(|(extents, _)| format!("({})", product(extents)))
        .collect();
    let terms: Vec<String> = runs
        .iter()
        .enumerate()
        .map(|(r, (_, stride))| {
            let inner: Vec<&str> = sizes[r + 1..].iter().map(String::as_str).collect();
            let divided = if inner.is_empty() {
                "hawk_rank".to_string()
            } else {
                format!("hawk_rank / ({})", product(&inner))
            };
            let coordinate = if r == 0 {
                divided
            } else {
                format!("({divided}) % {}", sizes[r])
            };
            if *stride == "1" {
                coordinate
            } else {
                format!("({coordinate}) * {stride}")
            }
        })
        .collect();
    terms.join(" + ")
}

/// Every node the roots reach through their operands, the roots included.
fn reachable(ctx: *mut RxCtx, roots: &[u32]) -> Result<Vec<u32>, Refusal> {
    let (mut seen, mut stack) = (vec![], roots.to_vec());
    while let Some(node) = stack.pop() {
        if !seen.contains(&node) {
            seen.push(node);
            stack.extend(door::operands(ctx, node)?);
        }
    }
    Ok(seen)
}

/// The bindings and ranks of the input planes the sinks read at the lane's own element: the planes a phase loop reads through a
/// pointer (and, for a vector or matrix plane, a component pitch) hoisted before the group loop (`body::values::is_lane_pointer_plane`
/// states the same rule for the read). Only an input plane qualifies: a pointer is `__restrict`, a promise that nothing else in
/// the loop stores through the plane's memory, which holds for a plane the unit never writes. A plane the body also writes, a
/// plane with named axes (read through an index) and a uniform (loop-invariant already) keep their own reads.
fn input_planes(ctx: *mut RxCtx, sinks: &[u32]) -> Result<Vec<(String, usize)>, Refusal> {
    let mut planes: Vec<(String, usize)> = vec![];
    for node in reachable(ctx, sinks)? {
        if door::class_is(ctx, node, "Leaf")?
            && door::text_field(ctx, node, "role")? == "plane_in"
            && door::axes_declared(ctx, node)? == 0
        {
            let rank = crate::body::ty_of(ctx, node)?.shape.len();
            let name = door::text_field(ctx, node, "name")?;
            let binding = binding_name("plane_in", rank, &name)?;
            if !planes.iter().any(|(known, _)| *known == binding) {
                planes.push((binding, rank));
            }
        }
    }
    Ok(planes)
}

/// The bindings of the scalar output planes a phase loop writes through a pointer hoisted before the group loop (the store form
/// `body::stores::sinks::store` spells for them). A written plane gets the `__restrict` pointer only when the unit's own facts
/// keep the promise that nothing else reaches its memory inside the loops: (1) no node of the unit reads the plane, in the body or
/// beside the group, (2) no sink but a plain own-element store (`Assign`) names it: a keyed add, an append or any other write
/// is at another lane or through a key, and (3) the value stored is a scalar. Any other written plane keeps its view store.
fn output_planes(ctx: *mut RxCtx, sinks: &[u32]) -> Result<Vec<String>, Refusal> {
    let read = touched(ctx, sinks)?.read;
    let (mut own, mut elsewhere): (Vec<(String, String)>, Vec<String>) = (vec![], vec![]);
    for &sink in sinks {
        // a barrier is no sink and names no plane; every sink class carries the plane's name
        if !door::is_a(ctx, sink, "sink")? {
            continue;
        }
        let name = door::text_field(ctx, sink, "name")?;
        let scalar_value = || -> Result<bool, Refusal> {
            let value = door::operands(ctx, sink)?[0];
            Ok(crate::body::ty_of(ctx, value)?.shape.is_empty())
        };
        if door::class_is(ctx, sink, "Assign")? && scalar_value()? {
            let binding = binding_name(&door::text_field(ctx, sink, "role")?, 0, &name)?;
            own.push((name, binding));
        } else {
            elsewhere.push(name);
        }
    }
    let mut planes: Vec<String> = vec![];
    for (name, binding) in own {
        if !read.contains(&name) && !elsewhere.contains(&name) && !planes.contains(&binding) {
            planes.push(binding);
        }
    }
    Ok(planes)
}

/// What one run of sinks does to the planes: the planes it writes (and whether each write accumulates), the planes its values read,
/// and those it reads through an index (at another lane of the launch).
struct Touch {
    /// The planes the sinks write, each with whether the write is an accumulate.
    written: Vec<(String, bool)>,
    /// The planes the sinks' values read.
    read: Vec<String>,
    /// The planes the sinks' values read through an index.
    read_elsewhere: Vec<String>,
}

/// What `sinks` do to the planes.
fn touched(ctx: *mut RxCtx, sinks: &[u32]) -> Result<Touch, Refusal> {
    let mut written = vec![];
    for &sink in sinks {
        let assigns = door::class_is(ctx, sink, "Assign")?;
        if (assigns || door::class_is(ctx, sink, "Update")?)
            && door::text_field(ctx, sink, "role")? != "scratch"
        {
            let accumulates = !assigns && door::text_field(ctx, sink, "op")? != "store";
            written.push((door::text_field(ctx, sink, "name")?, accumulates));
        }
    }
    let (mut read, mut read_elsewhere) = (vec![], vec![]);
    for node in reachable(ctx, sinks)? {
        if door::class_is(ctx, node, "Leaf")? {
            read.push(door::text_field(ctx, node, "name")?);
        }
        if !door::class_is(ctx, node, "At")? {
            continue;
        }
        let ops = door::operands(ctx, node)?;
        // a read of a uniform's cell (an `At` with a role constant) is no read at another lane
        if ops.len() >= 2
            && door::class_is(ctx, ops[0], "Leaf")?
            && !door::class_is(ctx, ops[1], "RoleConst")?
        {
            read_elsewhere.push(door::text_field(ctx, ops[0], "name")?);
        }
    }
    Ok(Touch {
        written,
        read,
        read_elsewhere,
    })
}

/// Whether the sinks beside the group run after the group's phases (else before them), refusing a pair that no order serves.
///
/// Every read of a plane sees its launch-start value, so a read must run before the write of the same lane's cell: the sinks
/// beside the group run BEFORE the group's first phase when they read a plane the body writes, and AFTER its last phase when
/// they write a plane the body reads (before, when neither). Both at once is a cycle, and is refused. Across the lanes of
/// different groups the call has no order to give: a plane one run writes may be read by the other only at the lane's own
/// element, and written by the other only by accumulating (adds commute).
fn outside_runs_after(
    ctx: *mut RxCtx,
    kernel: &str,
    body: &[u32],
    outside: &[u32],
) -> Result<bool, Refusal> {
    let (inside, beside) = (touched(ctx, body)?, touched(ctx, outside)?);
    for (writer, reader, who) in [
        (&inside, &beside, "the sinks beside the group"),
        (&beside, &inside, "the group body"),
    ] {
        for (plane, accumulates) in &writer.written {
            let rewritten = reader
                .written
                .iter()
                .any(|(other, adds)| other == plane && !(*adds && *accumulates));
            if rewritten || reader.read_elsewhere.contains(plane) {
                return Err(unwritten(
                    kernel,
                    &format!("the plane '{plane}' written on one side of the group and rewritten or read at another lane by {who}"),
                ));
            }
        }
    }
    let reads_body_writes = inside.written.iter().any(|(p, _)| beside.read.contains(p));
    let writes_body_reads = beside.written.iter().any(|(p, _)| inside.read.contains(p));
    if reads_body_writes && writes_body_reads {
        return Err(unwritten(
            kernel,
            "sinks beside a group that read a plane the body writes and write a plane the body reads",
        ));
    }
    Ok(writes_body_reads)
}

/// The scratch reads among `nodes`: each read's place node and the plane's name.
fn scratch_reads(ctx: *mut RxCtx, nodes: &[u32]) -> Result<Vec<(u32, String)>, Refusal> {
    let mut reads = vec![];
    for &node in nodes {
        if !door::class_is(ctx, node, "At")? {
            continue;
        }
        let ops = door::operands(ctx, node)?;
        if let [plane, index] = ops.as_slice() {
            if door::class_is(ctx, *plane, "Scratch")? {
                reads.push((*index, door::text_field(ctx, *plane, "name")?));
            }
        }
    }
    Ok(reads)
}

/// The scratch plane a sink stores, if it stores one.
fn scratch_stored(ctx: *mut RxCtx, sink: u32) -> Result<Option<String>, Refusal> {
    let is_store =
        door::class_is(ctx, sink, "Update")? && door::text_field(ctx, sink, "role")? == "scratch";
    Ok(if is_store {
        Some(door::text_field(ctx, sink, "name")?)
    } else {
        None
    })
}

/// One loop's worth of sinks: sinks of one phase none of which reads a scratch plane an earlier one of the chunk stores (a value
/// read before such a store would be bound ahead of it, so a read that must see the store starts a new loop).
fn chunks(ctx: *mut RxCtx, sinks: &[u32]) -> Result<Vec<Vec<u32>>, Refusal> {
    let mut out: Vec<Vec<u32>> = vec![];
    let mut stored: Vec<String> = vec![];
    for &sink in sinks {
        let reads = scratch_reads(ctx, &reachable(ctx, &[sink])?)?;
        if reads.iter().any(|(_, name)| stored.contains(name)) {
            stored.clear();
            out.push(vec![]);
        } else if out.is_empty() {
            out.push(vec![]);
        }
        if let Some(chunk) = out.last_mut() {
            chunk.push(sink);
        }
        stored.extend(scratch_stored(ctx, sink)?);
    }
    Ok(out)
}

/// How far a chunk reads before and after the lane's own place: the largest negative and positive move of any scratch read.
fn reach(ctx: *mut RxCtx, sinks: &[u32], kernel: &str) -> Result<(i64, i64), Refusal> {
    let (mut before, mut after) = (0i64, 0i64);
    for (index, _) in scratch_reads(ctx, &reachable(ctx, sinks)?)? {
        if let Place::Moved(c) = place_of(ctx, index, kernel)? {
            before = before.max(-c);
            after = after.max(c);
        }
    }
    Ok((before, after))
}

/// The lines that bind the lane of the loop's `hawk_rank`.
fn lane_binding(offset: &str) -> String {
    format!(
        "{LOOP_INDENT}const std::int64_t hawk_i = hawk_base + {offset};\n{LOOP_INDENT}const aether::SampleIndex i =\n{LOOP_INDENT}    aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n{LOOP_INDENT}(void)hawk_i;\n{LOOP_INDENT}(void)nSamples;"
    )
}

/// How the loops of a group reach the lanes of the planes.
struct LoopLanes<'a> {
    /// The offset of the lane of place `hawk_rank` from the group's first lane.
    offset: &'a str,
    /// The bindings of the output planes the loops write through a hoisted pointer (`output_planes`).
    outputs: &'a [String],
}

/// The text of one chunk of a phase: its loop over the interior lanes, and, when the chunk moves a scratch place, the loop over
/// the edge lanes beside it.
fn chunk_text(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    sinks: &[u32],
    lanes: &LoopLanes,
    (label, phase): (&str, usize),
    needs: &mut Needs,
) -> Result<String, Refusal> {
    let (before, after) = reach(ctx, sinks, &unit.name)?;
    let interior = render_phase_with_needs(
        ctx,
        walk,
        unit,
        LOOP_INDENT,
        sinks,
        PhaseForm {
            edge_lanes: false,
            pointer_outputs: lanes.outputs,
        },
    )?;
    needs.merge(&interior.needs);
    let binding = lane_binding(lanes.offset);
    if before + after == 0 {
        return Ok(format!(
            "{GROUP_INDENT}// {label}\n{GROUP_INDENT}for (std::int64_t hawk_rank = 0; hawk_rank < hawk_size; ++hawk_rank) {{\n{binding}\n{}\n{GROUP_INDENT}}}",
            interior.text
        ));
    }
    let edge = render_phase_with_needs(
        ctx,
        walk,
        unit,
        LOOP_INDENT,
        sinks,
        PhaseForm {
            edge_lanes: true,
            pointer_outputs: lanes.outputs,
        },
    )?;
    needs.merge(&edge.needs);
    let (a, b) = (format!("hawk_lo{phase}"), format!("hawk_hi{phase}"));
    Ok([
        format!("{GROUP_INDENT}// {label}: the interior lanes, then the lanes at the edges"),
        format!("{GROUP_INDENT}const std::int64_t {a} = hawk_size < {before} ? hawk_size : {before};"),
        format!("{GROUP_INDENT}const std::int64_t {b} = hawk_size - {after} > {a} ? hawk_size - {after} : {a};"),
        format!("{GROUP_INDENT}for (std::int64_t hawk_rank = {a}; hawk_rank < {b}; ++hawk_rank) {{"),
        binding.clone(),
        interior.text,
        format!("{GROUP_INDENT}}}"),
        format!("{GROUP_INDENT}for (std::int64_t hawk_edge = 0; hawk_edge < {a} + (hawk_size - {b}); ++hawk_edge) {{"),
        format!("{LOOP_INDENT}const std::int64_t hawk_rank = hawk_edge < {a} ? hawk_edge : hawk_edge - {a} + {b};"),
        binding,
        edge.text,
        format!("{GROUP_INDENT}}}"),
    ]
    .join("\n"))
}

/// The group's body: the scratch arrays, then the loop over the groups of the call with one loop per phase inside.
pub fn group_body(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    group: u32,
) -> Result<Rendered, Refusal> {
    let kernel = unit.name.as_str();
    if unit.target != Target::Host {
        return Err(unwritten(kernel, "a group body for the device target"));
    }
    let membership = door::text_field(ctx, group, "membership")?;
    for node in door::walk_order(ctx, walk)? {
        for (class, what) in [
            ("GroupReduce", "a group reduce"),
            ("GroupScan", "a group scan"),
        ] {
            if door::class_is(ctx, node, class)? {
                return Err(unwritten(kernel, what));
            }
        }
    }
    let (lanes, size, bound) = match membership.as_str() {
        "count" => {
            let n = door::field_int(ctx, group, "count")?
                .ok_or_else(|| fault("a count group without its lanes"))?;
            (Lanes::Count(n), n.to_string(), Some(n))
        }
        "axes" => {
            let (axes, size) = grid_axes(ctx, group, kernel)?;
            let fixed: Option<i64> = axes
                .iter()
                .filter(|a| a.spanned)
                .map(|a| a.extent.parse::<i64>().ok())
                .product();
            (Lanes::Axes(axes), size, fixed)
        }
        "key" => return Err(unwritten(kernel, "a group by key")),
        other => return Err(fault(format!("a group of membership {other}"))),
    };
    let body = door::field_refs(ctx, group, "body_sinks")?.unwrap_or_default();
    let phases = door::field_ints(ctx, group, "phases")?.unwrap_or_default();
    let mut scratch = vec![];
    for plane in door::field_refs(ctx, group, "scratch")?.unwrap_or_default() {
        let ty = crate::body::ty_of(ctx, plane)?;
        let bound = bound.ok_or_else(|| fault("a scratch plane in a group with no static size"))?;
        let name = door::text_field(ctx, plane, "name")?;
        scratch.extend(scratch_declare(&name, &ty.dtype, &ty.shape, bound)?);
    }
    let offset = lane_offset(&lanes);
    let everything: Vec<u32> = body
        .iter()
        .chain(unit.sinks.iter().filter(|&&s| s != group))
        .copied()
        .collect();
    let outputs = output_planes(ctx, &everything)?;
    let mut text: Vec<String> = vec![];
    let hoisted = input_planes(ctx, &everything)?
        .into_iter()
        .map(|(binding, rank)| (binding, rank, false))
        .chain(outputs.iter().map(|binding| (binding.clone(), 0, true)));
    for (binding, rank, written) in hoisted {
        text.extend(
            lane_pointer_declarations(&binding, rank, written)
                .into_iter()
                .map(|line| format!("    {line}")),
        );
    }
    let loop_lanes = LoopLanes {
        offset: &offset,
        outputs: &outputs,
    };
    let outside: Vec<u32> = unit.sinks.iter().copied().filter(|&s| s != group).collect();
    let after = outside_runs_after(ctx, kernel, &body, &outside)?;
    let mut needs = Needs::default();
    let mut per_group: Vec<String> = vec![];
    let phase_count = phases.iter().max().map_or(0, |m| *m as usize + 1);
    if !outside.is_empty() && !after {
        let label = ("sinks beside the group, before its phases", phase_count);
        per_group.push(chunk_text(
            ctx,
            walk,
            unit,
            &outside,
            &loop_lanes,
            label,
            &mut needs,
        )?);
    }
    for phase in 0..phase_count {
        let mut sinks = vec![];
        for (&sink, &of) in body.iter().zip(&phases) {
            if of as usize == phase && !door::class_is(ctx, sink, "Barrier")? {
                sinks.push(sink);
            }
        }
        for chunk in chunks(ctx, &sinks)? {
            let label = (format!("phase {phase}"), phase);
            per_group.push(chunk_text(
                ctx,
                walk,
                unit,
                &chunk,
                &loop_lanes,
                (&label.0, label.1),
                &mut needs,
            )?);
        }
    }
    if !outside.is_empty() && after {
        let label = ("sinks beside the group, after its phases", phase_count);
        per_group.push(chunk_text(
            ctx,
            walk,
            unit,
            &outside,
            &loop_lanes,
            label,
            &mut needs,
        )?);
    }
    match &lanes {
        Lanes::Count(n) => {
            text.push(group_function_open());
            text.extend(scratch.iter().map(|line| format!("{GROUP_INDENT}{line}")));
            text.extend(per_group);
            text.push("    };".to_string());
            text.extend(group_calls(*n));
        }
        Lanes::Axes(axes) => {
            text.extend(scratch.iter().map(|line| format!("    {line}")));
            text.push(
                "    for (std::int64_t hawk_g = base; hawk_g < base + count; ++hawk_g) {"
                    .to_string(),
            );
            text.extend(axes_group_lines(axes, &size));
            text.extend(per_group);
            text.push("    }".to_string());
        }
    }
    Ok(Rendered {
        text: text.join("\n"),
        needs,
    })
}
