//! Questions about the IR that the host arm asks. Each is a C function of the wings library,
//! declared here and resolved
//! when the final binary is linked; the arm ports none of them. Handles are those of the walk's
//! context.
use crate::ir::door::{CtxPtr, RxCtx, RX_OK, RX_TRUNC};
use crate::refuse::{fault, Refusal, RefusalKind};
use std::ffi::{c_char, c_int, CString};

pub use super::runtime_names;

/// One placed plane, as the wings answer it: the leaf that names it, its layout and width.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PlaneRow {
    /// The plane the target writes.
    plane: u32,
    /// The layout class of the target.
    layout: c_int,
    /// The width of the target in values.
    width: i64,
}

/// The key run of a sink, as the wings answer it (`found` 0: none).
#[repr(C)]
#[derive(Default)]
pub struct KeyRun {
    /// Whether a target was found.
    found: c_int,
    /// The form of the run.
    form: c_int,
    /// The width of the run.
    width: i64,
    /// Whether the run is blocked.
    block: c_int,
}

/// The name a node carries, as plain text.
fn name_of(ctx: *mut RxCtx, node: u32) -> Result<String, Refusal> {
    let name = crate::ir::door::field_text(ctx, node, "name")?;
    name.ok_or_else(|| fault(format!("node {node} has no name")))
}

extern "C" {
    /// Whether the lanes of a tile may run side by side over the walk (`*out` is 0 or 1).
    pub fn rw_cpu_lane_form_access(ctx: *mut RxCtx, walk: u32, out: *mut c_int) -> c_int;
    /// The top-level loops the host runs interchanged, in walk order.
    pub fn rw_cpu_interchanged_loops(
        ctx: *mut RxCtx,
        walk: u32,
        unroll_limit: i64,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// Whether the node is a call of a library function (`*out` is 0 or 1).
    pub fn rw_cost_is_library_call(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether the node is a linear-algebra node of any kind (`*out` is 0 or 1); which kinds branch
    /// per lane is the writer's.
    pub fn rw_linalg_is_linalg(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether a dispatch node renders as a switch (`*out` is 0 or 1).
    pub fn rw_dispatch_switch_form(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether the walk promotes the lookup plane of that name to a parameter cell (`*out` is 0 or
    /// 1).
    pub fn rw_ir_param_plane(
        ctx: *mut RxCtx,
        walk: u32,
        name: *const c_char,
        out: *mut c_int,
    ) -> c_int;
    /// The walk's companions as lines `companion=target`, in the walk's order, by the caller-buffer
    /// protocol.
    /// The key run of one keyed sink of the kernel with the given sinks: "none" or "<form> <width>
    /// <block 0|1>".
    fn rw_lanes_key_run_in_kernel(
        ctx: *mut RxCtx,
        sinks: *const u32,
        nsinks: usize,
        sink: u32,
        run_max: i64,
        block_run: i64,
        out: *mut KeyRun,
    ) -> c_int;
    /// Whether one trip of the loop is safe to run and discard for a stopped lane (`*out` is 0 or
    /// 1).
    pub fn rw_loops_speculative_step(ctx: *mut RxCtx, lp: u32, out: *mut c_int) -> c_int;
    /// Whether the node places a value in a plane (`*out` is 0 or 1).
    pub fn rw_wide_is_resident(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether the node names a scratch plane the runtime made (`*out` is 0 or 1).
    pub fn rw_wide_is_tiled(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether the node names a plane a placement writes (`*out` is 0 or 1).
    pub fn rw_wide_is_scratch(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// Whether the node is the launch-start value of a vector mutable (`*out` is 0 or 1).
    pub fn rw_wide_is_prior(ctx: *mut RxCtx, node: u32, out: *mut c_int) -> c_int;
    /// How element k of a rank-1 node is read: "proxy", "input", "parts", "splat", "elementwise" or
    /// "none".
    pub fn rw_wide_element_kind(ctx: *mut RxCtx, node: u32, kind: *mut c_int) -> c_int;
    /// Whether a vector is reads of one plane at base + start + k * step (`*found` is 0 or 1).
    pub fn rw_wide_gather(
        ctx: *mut RxCtx,
        node: u32,
        found: *mut c_int,
        plane: *mut u32,
        base: *mut u32,
        start: *mut i64,
        step: *mut i64,
        clamped: *mut c_int,
    ) -> c_int;
    /// The placements of the given nodes (loop bodies included) as lines `plane resident layout
    /// width`, by plane name.
    pub fn rw_wide_record(
        ctx: *mut RxCtx,
        order: *const u32,
        n: usize,
        rows: *mut PlaneRow,
        cap: usize,
        count: *mut usize,
    ) -> c_int;
    /// Whether each node is proven equal over every aligned run of `warp` lanes (`flags` holds one
    /// int per node).
    pub fn rw_lanes_warp_uniform(
        ctx: *mut RxCtx,
        sinks: *const u32,
        nsinks: usize,
        warp: *const i64,
        nodes: *const u32,
        nnodes: usize,
        flags: *mut c_int,
    ) -> c_int;
    /// Whether the node is spelled as its runtime twin; `form` is "auto", "rolled" or "static"
    /// (`*out` is 0 or 1).
    pub fn rw_fold_rolled(
        ctx: *mut RxCtx,
        node: u32,
        fold_min: *const i64,
        form: *const c_char,
        out: *mut c_int,
    ) -> c_int;
    /// The nodes and everything inside their loop bodies, in order.
    pub fn rw_loops_flat(
        ctx: *mut RxCtx,
        nodes: *const u32,
        n: usize,
        buf: *mut u32,
        cap: usize,
        count: *mut usize,
    ) -> c_int;
    /// The target of a compensated sink: a struct (found, own column) and its name by the caller-
    /// buffer protocol.
    fn rw_split_compensated_target(
        ctx: *mut RxCtx,
        sink_id: *const c_char,
        compensation: *const c_char,
        returned_slot: *const c_char,
        output_names: *const c_char,
        sinks: *const u32,
        nsinks: usize,
        out: *mut CompensatedTarget,
        name_buf: *mut c_char,
        cap: usize,
        len: *mut usize,
    ) -> c_int;
    /// The branch-mix plane of a dispatch, "<dispatch_name>_mix", by the caller-buffer protocol.

    /// The sinks of a boxed entry: each sink once per lane of the box, as copies made in the
    /// context (which must declare
    /// the box_store kind). `buf` holds `nsinks * lanes` nodes.
    fn rw_box_boxed_sinks(
        ctx: *mut RxCtx,
        sinks: *const u32,
        nsinks: usize,
        axis: usize,
        lanes: usize,
        stride: i64,
        buf: *mut u32,
        cap: usize,
        count: *mut usize,
    ) -> c_int;
    /// The sinks with every least-squares node lowered to a factorisation and its solve (`buf`
    /// holds `nsinks` nodes;
    /// `*changed` is 0 when the sinks come back as given). `max_dim` is null when the target states
    /// none.
    fn rw_lstsq_lowered(
        ctx: *mut RxCtx,
        sinks: *const u32,
        nsinks: usize,
        max_dim: *const i64,
        buf: *mut u32,
        cap: usize,
        count: *mut usize,
        changed: *mut c_int,
    ) -> c_int;
    /// The companion planes of the walk, as the door reports them.
    pub fn rw_ir_companions(
        ctx: *mut RxCtx,
        walk: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
}

/// The branch-mix plane of the dispatch of that name.
pub fn mix_name(dispatch_name: &str) -> Result<String, Refusal> {
    Ok(runtime_names::mix_plane(dispatch_name))
}

/// The refusal a non-OK status stands for: the reason is the text the context holds.
fn refusal(ctx: *mut RxCtx, what: &str, st: c_int) -> Refusal {
    let mut refused = crate::ir::door::failed(ctx, st);
    refused.reason = format!("analysis {what} refused ({st}): {}", refused.reason);
    refused
}

/// How a slot of this role crosses the launch boundary: "GRefMirror", "ScalarHandle" or "value".
pub fn mirror_of(role: &str, _dtype: &str, rank: usize) -> Result<String, Refusal> {
    runtime_names::mirror_of(role, rank).map(String::from)
}

/// The layout of `rw_compensated_target` in raptor_wings.h.
#[repr(C)]
struct CompensatedTarget {
    /// Whether a compensated target was found.
    found: c_int,
    /// Whether the target is the kind's own column.
    own_column: c_int,
}

/// The text a two-call text export answers (the length first, then the text), or the refusal.
fn text(
    ctx: *mut RxCtx,
    what: &str,
    call: impl Fn(*mut c_char, usize, *mut usize) -> c_int,
) -> Result<String, Refusal> {
    let mut len = 0usize;
    let st = call(std::ptr::null_mut(), 0, &mut len);
    if st == RX_OK {
        return Ok(String::new());
    }
    if st != RX_TRUNC {
        return Err(refusal(ctx, what, st));
    }
    let mut buf = vec![0u8; len + 1];
    let st = call(buf.as_mut_ptr() as *mut c_char, len + 1, &mut len);
    if st != RX_OK {
        return Err(refusal(ctx, what, st));
    }
    buf.truncate(len);
    String::from_utf8(buf).map_err(fault)
}

/// A yes-or-no answer of the analyses, or the refusal.
fn flag(ctx: *mut RxCtx, what: &str, f: impl Fn(*mut c_int) -> c_int) -> Result<bool, Refusal> {
    let mut out: c_int = 0;
    let st = f(&mut out);
    if st == RX_OK {
        Ok(out != 0)
    } else {
        Err(refusal(ctx, what, st))
    }
}

/// A list answer of the analyses, or the refusal.
fn list<T: Copy + Default>(
    ctx: *mut RxCtx,
    what: &str,
    f: impl Fn(*mut T, usize, *mut usize) -> c_int,
) -> Result<Vec<T>, Refusal> {
    let mut n = 0usize;
    let st = f(std::ptr::null_mut(), 0, &mut n);
    if st == RX_OK {
        return Ok(vec![]);
    }
    if st != RX_TRUNC {
        return Err(refusal(ctx, what, st));
    }
    let mut buf = vec![T::default(); n];
    let st = f(buf.as_mut_ptr(), n, &mut n);
    if st != RX_OK {
        return Err(refusal(ctx, what, st));
    }
    buf.truncate(n);
    Ok(buf)
}

/// Whether the walk is written in lane form.
pub fn lane_form_access(ctx: impl Into<CtxPtr>, walk: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "lane_form_access", |o| unsafe {
        rw_cpu_lane_form_access(ctx, walk, o)
    })
}
/// The top-level loops the host runs interchanged.
pub fn interchanged_loops(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    list(ctx, "interchanged_loops", |b, c, n| unsafe {
        rw_cpu_interchanged_loops(ctx, walk, runtime_names::UNROLL_LIMIT, b, c, n)
    })
}
/// The whole kernel's nodes and everything inside their loop bodies, in order.
pub fn kernel_nodes(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    loops_flat(ctx, &crate::ir::door::walk_order(ctx, walk)?)
}
/// Whether the node is a call into the math library.
pub fn is_library_call(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_library_call", |o| unsafe {
        rw_cost_is_library_call(ctx, node, o)
    })
}
/// Whether the loop's step is speculative.
pub fn speculative_step(ctx: impl Into<CtxPtr>, lp: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "speculative_step", |o| unsafe {
        rw_loops_speculative_step(ctx, lp, o)
    })
}
/// Whether the node's value stays resident.
pub fn is_resident(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_resident", |o| unsafe {
        rw_wide_is_resident(ctx, node, o)
    })
}
/// Whether the node is tiled.
pub fn is_tiled(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_tiled", |o| unsafe {
        rw_wide_is_tiled(ctx, node, o)
    })
}
/// Whether the node is a scratch plane.
pub fn is_scratch(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_scratch", |o| unsafe {
        rw_wide_is_scratch(ctx, node, o)
    })
}
/// Whether the node is a prior read.
pub fn is_prior(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_prior", |o| unsafe {
        rw_wide_is_prior(ctx, node, o)
    })
}
/// How element k of a rank-1 node is read: "proxy", "input", "parts", "splat", "elementwise" or
/// "none".
pub fn element_kind(ctx: impl Into<CtxPtr>, node: u32) -> Result<String, Refusal> {
    let ctx = ctx.into().raw();
    let mut kind: c_int = 0;
    let st = unsafe { rw_wide_element_kind(ctx, node, &mut kind) };
    if st != RX_OK {
        return Err(refusal(ctx, "element_kind", st));
    }
    let words = ["none", "proxy", "input", "parts", "splat", "elementwise"];
    Ok(words.get(kind as usize).unwrap_or(&"none").to_string())
}
/// The width of each plane the placements of the given nodes (loop bodies included) write, by plane
/// name.
pub fn plane_widths(
    ctx: impl Into<CtxPtr>,
    nodes: &[u32],
) -> Result<Vec<(String, usize)>, Refusal> {
    let ctx = ctx.into().raw();
    let rows: Vec<PlaneRow> = list(ctx, "plane_widths", |b, c, n| unsafe {
        rw_wide_record(ctx, nodes.as_ptr(), nodes.len(), b, c, n)
    })?;
    rows.iter()
        .map(|row| Ok((name_of(ctx, row.plane)?, row.width as usize)))
        .collect()
}

/// Whether each of the nodes is proven equal over every aligned run of `warp` lanes of the kernel
/// with these sinks.
pub fn warp_uniform(
    ctx: impl Into<CtxPtr>,
    sinks: &[u32],
    warp: Option<i64>,
    nodes: &[u32],
) -> Result<Vec<bool>, Refusal> {
    let ctx = ctx.into().raw();
    let mut flags = vec![0 as c_int; nodes.len()];
    let run = warp.unwrap_or(0);
    let st = unsafe {
        rw_lanes_warp_uniform(
            ctx,
            sinks.as_ptr(),
            sinks.len(),
            if warp.is_some() {
                &run
            } else {
                std::ptr::null()
            },
            nodes.as_ptr(),
            nodes.len(),
            flags.as_mut_ptr(),
        )
    };
    if st != RX_OK {
        return Err(refusal(ctx, "warp_uniform", st));
    }
    Ok(flags.into_iter().map(|x| x != 0).collect())
}

/// Whether a fold or product is spelled as its runtime twin under the form setting and the fold
/// threshold.
pub fn fold_rolled(
    ctx: impl Into<CtxPtr>,
    node: u32,
    fold_min: i64,
    form: &str,
) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let form = CString::new(form).map_err(fault)?;
    flag(ctx, "fold_rolled", |o| unsafe {
        rw_fold_rolled(ctx, node, &fold_min, form.as_ptr(), o)
    })
}

/// A vector that is the reads of one plane at `base + start + k * step`.
pub struct Gather {
    /// The plane the run is over.
    pub plane: u32,
    /// The base index the run starts from, when it has one.
    pub base: Option<u32>,
    /// The first index of the run.
    pub start: i64,
    /// The distance between indexes of the run.
    pub step: i64,
    /// Whether the run is clamped to the plane.
    pub clamped: bool,
}
/// The run of reads a vector is, when it is one.
pub fn gather(ctx: impl Into<CtxPtr>, node: u32) -> Result<Option<Gather>, Refusal> {
    let ctx = ctx.into().raw();
    let (mut found, mut clamped, mut plane, mut base, mut start, mut step) =
        (0 as c_int, 0 as c_int, 0u32, 0u32, 0i64, 0i64);
    let st = unsafe {
        rw_wide_gather(
            ctx,
            node,
            &mut found,
            &mut plane,
            &mut base,
            &mut start,
            &mut step,
            &mut clamped,
        )
    };
    if st != RX_OK {
        return Err(refusal(ctx, "gather", st));
    }
    Ok((found != 0).then(|| Gather {
        plane,
        base: (base != u32::MAX).then_some(base),
        start,
        step,
        clamped: clamped != 0,
    }))
}
/// The nodes and everything inside their loop bodies, in order.
pub fn loops_flat(ctx: impl Into<CtxPtr>, nodes: &[u32]) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    list(ctx, "loops_flat", |b, c, n| unsafe {
        rw_loops_flat(ctx, nodes.as_ptr(), nodes.len(), b, c, n)
    })
}
/// Whether the node is a linear-algebra call.
pub fn is_linalg(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "is_linalg", |o| unsafe {
        rw_linalg_is_linalg(ctx, node, o)
    })
}
/// Whether the node's switch is written as a switch.
pub fn switch_form(ctx: impl Into<CtxPtr>, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    flag(ctx, "switch_form", |o| unsafe {
        rw_dispatch_switch_form(ctx, node, o)
    })
}
/// Whether the walk has a parameter plane of that name.
pub fn param_plane(ctx: impl Into<CtxPtr>, walk: u32, name: &str) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let n = CString::new(name).map_err(fault)?;
    flag(ctx, "param_plane", |o| unsafe {
        rw_ir_param_plane(ctx, walk, n.as_ptr(), o)
    })
}
/// The companion pairs of the walk: (plane, its correction plane).
pub fn companions(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<(String, String)>, Refusal> {
    let ctx = ctx.into().raw();
    let sinks: Vec<u32> = list(ctx, "companions", |b, c, n| unsafe {
        rw_ir_companions(ctx, walk, b, c, n)
    })?;
    sinks
        .iter()
        .map(|&sink| {
            let name = name_of(ctx, sink)?;
            Ok((runtime_names::companion_of(&name), name))
        })
        .collect()
}

/// The key run of a sink: its form, or none when keys are not proven to run.
pub fn key_run(
    ctx: impl Into<CtxPtr>,
    sinks: &[u32],
    sink: u32,
    run_max: i64,
    block_run: i64,
) -> Result<Option<(String, u32, bool)>, Refusal> {
    let ctx = ctx.into().raw();
    let mut run = KeyRun::default();
    let st = unsafe {
        rw_lanes_key_run_in_kernel(
            ctx,
            sinks.as_ptr(),
            sinks.len(),
            sink,
            run_max,
            block_run,
            &mut run,
        )
    };
    if st != RX_OK {
        return Err(crate::refuse::nib("needs analysis lanes.key_run"));
    }
    let form = ["none", "proven", "sorted", "tested"]
        .get(run.form as usize)
        .copied();
    Ok((run.found != 0).then(|| {
        (
            form.unwrap_or("none").to_string(),
            run.width as u32,
            run.block != 0,
        )
    }))
}

/// The compensated target of a kernel's sinks: its name and whether it is the lane's own column;
/// `None` when there is
/// none. Refuses (the door's message) when the outputs do not name exactly one target.
pub fn compensated_target(
    ctx: impl Into<CtxPtr>,
    compensation: &str,
    slot: Option<&str>,
    names: &[String],
    sinks: &[u32],
) -> Result<Option<(String, bool)>, Refusal> {
    let ctx = ctx.into().raw();
    let (id, comp) = (
        CString::new("compensated").map_err(fault)?,
        CString::new(compensation).map_err(fault)?,
    );
    let slot_c = slot.map(CString::new).transpose().map_err(fault)?;
    let names_c = CString::new(names.join("\n")).map_err(fault)?;
    let raw = std::cell::Cell::new(CompensatedTarget {
        found: 0,
        own_column: 0,
    });
    let name = text(ctx, "compensated_target", |b, c, n| unsafe {
        rw_split_compensated_target(
            ctx,
            id.as_ptr(),
            comp.as_ptr(),
            slot_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            names_c.as_ptr(),
            sinks.as_ptr(),
            sinks.len(),
            raw.as_ptr(),
            b,
            c,
            n,
        )
    })?;
    let target = raw.into_inner();
    Ok((target.found != 0).then_some((name, target.own_column != 0)))
}

/// The sinks of a boxed entry for a box of `lanes` lanes along lane axis `axis`, `stride` flat
/// lanes apart.
pub fn boxed_sinks(
    ctx: impl Into<CtxPtr>,
    sinks: &[u32],
    axis: usize,
    lanes: usize,
    stride: i64,
) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    let mut buf = vec![0u32; sinks.len() * lanes];
    let mut count = 0usize;
    let st = unsafe {
        rw_box_boxed_sinks(
            ctx,
            sinks.as_ptr(),
            sinks.len(),
            axis,
            lanes,
            stride,
            buf.as_mut_ptr(),
            buf.len(),
            &mut count,
        )
    };
    if st != RX_OK {
        return Err(crate::ir::door::failed(ctx, st));
    }
    buf.truncate(count);
    Ok(buf)
}

/// The kernel's sinks with the least-squares nodes lowered, or `None` when there is nothing to
/// lower. `max_dim` is the
/// target's largest dimension one lane factors alone.
pub fn lowered_sinks(
    ctx: impl Into<CtxPtr>,
    sinks: &[u32],
    max_dim: Option<i64>,
) -> Result<Option<Vec<u32>>, Refusal> {
    let ctx = ctx.into().raw();
    let mut buf = vec![0u32; sinks.len()];
    let (mut count, mut changed) = (0usize, 0 as c_int);
    let limit = max_dim.unwrap_or(0);
    let st = unsafe {
        rw_lstsq_lowered(
            ctx,
            sinks.as_ptr(),
            sinks.len(),
            if max_dim.is_some() {
                &limit
            } else {
                std::ptr::null()
            },
            buf.as_mut_ptr(),
            buf.len(),
            &mut count,
            &mut changed,
        )
    };
    if st != RX_OK {
        let reason = format!("least squares ({})", crate::ir::door::last_error(ctx));
        return Err(match crate::ir::door::failed(ctx, st).kind {
            RefusalKind::NotWritten => crate::refuse::nib(reason),
            kind => Refusal { kind, reason },
        });
    }
    buf.truncate(count);
    Ok((changed != 0).then_some(buf))
}
