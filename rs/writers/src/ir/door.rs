//! The IR core's C door as seen from this crate: the functions are declared here as `extern "C"`
//! (their meaning is that of
//! raptor_ir.h) and are resolved when the final binary is linked. The helpers return owned values.
use crate::refuse::{bad, fault, nib, Refusal};
use std::ffi::{c_char, c_int, CString};

#[repr(C)]
/// The IR core's context, as an opaque type; it is only ever used through a pointer.
pub struct RxCtx {
    /// Keeps the type from being built.
    _private: [u8; 0],
}

/// A context pointer as the door's wrappers take it: any `*mut RxCtx` made by the IR core converts
/// to it. Holding one
/// states that the pointer is a live context (the contract of every wrapper); the wrappers read it
/// back with `raw`.
#[derive(Clone, Copy)]
pub struct CtxPtr(*mut RxCtx);

impl From<*mut RxCtx> for CtxPtr {
    fn from(ctx: *mut RxCtx) -> Self {
        CtxPtr(ctx)
    }
}

impl CtxPtr {
    /// The raw pointer, for the extern functions.
    pub fn raw(self) -> *mut RxCtx {
        self.0
    }
}

/// Success.
pub const RX_OK: c_int = 0;
/// The call was refused; the context holds the reason.
pub const RX_REFUSED: c_int = 1;
/// An argument was missing or malformed.
pub const RX_BADARG: c_int = 2;
/// The buffer was too small; the length is filled in.
pub const RX_TRUNC: c_int = 3;
/// The asked-for value is absent.
pub const RX_ABSENT: c_int = 5;

/// The layout of `rx_type` in raptor_ir_query.h.
#[repr(C)]
pub struct RxType {
    /// The number of dimensions.
    rank: u32,
    /// The extents.
    dims: [i64; 8],
    /// The element type code.
    dtype: c_int,
    /// The tag, as text.
    tag: [c_char; 64],
}

/// The dtype names in the order of the header's dtype codes.
const DTYPE_NAMES: [&str; 7] = ["f64", "f32", "i32", "i64", "bool", "int32", "uint32"];

/// A node's type: extents, dtype name and tag (`None` for no tag).
#[derive(Debug, PartialEq, Eq)]
pub struct Ttype {
    /// The extents of the type.
    pub shape: Vec<i64>,
    /// The element type, by name.
    pub dtype: String,
    /// The tag of the type, when it has one.
    pub tag: Option<String>,
}

extern "C" {
    /// The reason of the last refusal.
    pub fn rx_last_error(ctx: *const RxCtx) -> *const c_char;
    /// The class of a node, as text.
    pub fn rx_class(
        ctx: *mut RxCtx,
        node: u32,
        buf: *mut c_char,
        cap: usize,
        len: *mut usize,
    ) -> c_int;
    /// The operands of a node.
    pub fn rx_operands(
        ctx: *mut RxCtx,
        node: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The nodes of a walk, in order.
    pub fn rx_walk_order(
        ctx: *mut RxCtx,
        walk: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// Whether a walk binds the slot (role, name).
    pub fn rx_walk_binds(
        ctx: *mut RxCtx,
        walk: u32,
        role: *const c_char,
        name: *const c_char,
        out: *mut c_int,
    ) -> c_int;
    /// The slots a walk binds, as (role, name) texts back to back.
    pub fn rx_walk_args(
        ctx: *mut RxCtx,
        walk: u32,
        buf: *mut c_char,
        cap: usize,
        len: *mut usize,
        count: *mut usize,
    ) -> c_int;
    /// The canonical walk over sinks that keeps the slots of another walk.
    pub fn rx_canonical_like(
        ctx: *mut RxCtx,
        sinks: *const u32,
        n: usize,
        like: u32,
        out: *mut u32,
    ) -> c_int;
    /// The position of a node in a walk.
    pub fn rx_walk_position(ctx: *mut RxCtx, walk: u32, node: u32, pos: *mut u32) -> c_int;
    /// The nodes in the body of a loop.
    pub fn rx_body_nodes(
        ctx: *mut RxCtx,
        lp: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The roots of the body of a loop.
    pub fn rx_body_roots(
        ctx: *mut RxCtx,
        lp: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The nodes a dispatch holds.
    pub fn rx_dispatch_nodes(
        ctx: *mut RxCtx,
        walk: u32,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The kind of value a field holds (an `RX_KIND_*` code).
    pub fn rx_field_kind(ctx: *mut RxCtx, node: u32, name: *const c_char, out: *mut c_int)
        -> c_int;
    /// A field that holds a list of integers.
    pub fn rx_field_ints(
        ctx: *mut RxCtx,
        node: u32,
        name: *const c_char,
        buf: *mut i64,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The node references a field holds, positionally, repeats kept.
    pub fn rx_field_node_list(
        ctx: *mut RxCtx,
        node: u32,
        name: *const c_char,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// The type of a node, typed.
    pub fn rx_ttype_of(ctx: *mut RxCtx, node: u32, out: *mut RxType) -> c_int;
    /// A field of a node, as text.
    pub fn rx_field_text(
        ctx: *mut RxCtx,
        node: u32,
        name: *const c_char,
        buf: *mut c_char,
        cap: usize,
        len: *mut usize,
    ) -> c_int;
    /// A field of a node, as an integer.
    pub fn rx_field_int(ctx: *mut RxCtx, node: u32, name: *const c_char, out: *mut i64) -> c_int;
    /// A field of a node, as a real number.
    pub fn rx_field_real(ctx: *mut RxCtx, node: u32, name: *const c_char, out: *mut f64) -> c_int;
    /// A field of a node, as a flag.
    pub fn rx_field_flag(ctx: *mut RxCtx, node: u32, name: *const c_char, out: *mut c_int)
        -> c_int;
    /// A field of a node, as a list of nodes.
    pub fn rx_field_nodes(
        ctx: *mut RxCtx,
        node: u32,
        name: *const c_char,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// A field of a node, as a list of names.
    pub fn rx_field_names(
        ctx: *mut RxCtx,
        node: u32,
        name: *const c_char,
        buf: *mut c_char,
        cap: usize,
        len: *mut usize,
        count: *mut usize,
    ) -> c_int;
    /// Whether the node is inside the body of a loop.
    pub fn rx_in_loop_body(ctx: *mut RxCtx, walk: u32, node: u32, out: *mut c_int) -> c_int;
    /// The texts of a list constant of the IR core, back to back.
    pub fn rx_const_names(
        ctx: *mut RxCtx,
        name: *const c_char,
        buf: *mut c_char,
        cap: usize,
        len: *mut usize,
        count: *mut usize,
    ) -> c_int;
    /// Whether the node is of a class (or a subclass).
    pub fn rx_is_a(ctx: *mut RxCtx, node: u32, name: *const c_char, out: *mut c_int) -> c_int;
    /// The digest of a walk.
    pub fn rx_digest(ctx: *mut RxCtx, walk: u32, out65: *mut c_char) -> c_int;
    /// Makes a node of an operation that carries the literal of another node (NULL: none).
    pub fn rx_make_op_like(
        ctx: *mut RxCtx,
        kind: *const c_char,
        ops: *const u32,
        nops: usize,
        like: *const u32,
        out: *mut u32,
    ) -> c_int;
    /// Substitutes one node for another.
    pub fn rx_substitute(
        ctx: *mut RxCtx,
        roots: *const u32,
        nroots: usize,
        from: *const u32,
        to: *const u32,
        nmap: usize,
        buf: *mut u32,
        cap: usize,
        n: *mut usize,
    ) -> c_int;
    /// Rebuilds a node over new operands.
    pub fn rx_rebuild(
        ctx: *mut RxCtx,
        node: u32,
        ops: *const u32,
        nops: usize,
        out: *mut u32,
    ) -> c_int;
    /// The node a value passes through.
    pub fn rx_through(ctx: *mut RxCtx, node: u32, out: *mut u32) -> c_int;
    /// Whether two index nodes are the same index.
    pub fn rx_same_index(ctx: *mut RxCtx, a: u32, b: u32, out: *mut c_int) -> c_int;
}

/// The node a value passes through (the node itself when it has none).
pub fn through(ctx: impl Into<CtxPtr>, h: u32) -> Result<u32, Refusal> {
    let ctx = ctx.into().raw();
    let mut out = h;
    let st = unsafe { rx_through(ctx, h, &mut out) };
    if st != RX_OK {
        return Err(failed(ctx, st));
    }
    Ok(out)
}

/// Whether two indexed accesses are at the same index.
pub fn same_index(ctx: impl Into<CtxPtr>, a: u32, b: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let mut out: c_int = 0;
    let st = unsafe { rx_same_index(ctx, a, b, &mut out) };
    if st != RX_OK {
        return Err(failed(ctx, st));
    }
    Ok(out != 0)
}

/// The text of the last refusal of the context.
pub fn last_error(ctx: impl Into<CtxPtr>) -> String {
    let ctx = ctx.into().raw();
    unsafe { std::ffi::CStr::from_ptr(rx_last_error(ctx)) }
        .to_string_lossy()
        .into_owned()
}

/// The refusal a failed door status stands for, with the context's reason: a construct the core
/// refuses is not written, a bad argument is a bad request, anything else is a fault.
pub fn failed(ctx: impl Into<CtxPtr>, status: c_int) -> Refusal {
    let reason = last_error(ctx);
    match status {
        RX_REFUSED => nib(reason),
        RX_BADARG => bad(reason),
        _ => fault(reason),
    }
}

/// Turns a status into a result, with the context's reason for a refusal.
fn check(ctx: *mut RxCtx, st: c_int) -> Result<(), Refusal> {
    if st == RX_OK {
        Ok(())
    } else {
        Err(failed(ctx, st))
    }
}

/// A text output of the door: asks for the length, then for the text (the caller-buffer protocol).
fn text(
    ctx: *mut RxCtx,
    f: impl Fn(*mut c_char, usize, *mut usize) -> c_int,
) -> Result<String, Refusal> {
    let mut len = 0usize;
    let st = f(std::ptr::null_mut(), 0, &mut len);
    if st != RX_TRUNC {
        check(ctx, st)?
    }
    let mut buf = vec![0u8; len + 1];
    check(ctx, f(buf.as_mut_ptr() as *mut c_char, len + 1, &mut len))?;
    buf.truncate(len);
    String::from_utf8(buf).map_err(fault)
}

/// A handle list output of the door (`n` is set even when the buffer is too small).
fn handles(
    ctx: *mut RxCtx,
    f: impl Fn(*mut u32, usize, *mut usize) -> c_int,
) -> Result<Vec<u32>, Refusal> {
    let mut n = 0usize;
    let st = f(std::ptr::null_mut(), 0, &mut n);
    if st != RX_TRUNC {
        check(ctx, st)?;
        return Ok(vec![]);
    }
    let mut buf = vec![0u32; n];
    check(ctx, f(buf.as_mut_ptr(), n, &mut n))?;
    buf.truncate(n);
    Ok(buf)
}

/// A C string of the text, or a refusal for an interior NUL.
fn cs(s: &str) -> Result<CString, Refusal> {
    CString::new(s).map_err(|_| bad("interior NUL"))
}

/// The class name of a node.
pub(crate) fn class_name(ctx: impl Into<CtxPtr>, node: u32) -> Result<String, Refusal> {
    let ctx = ctx.into().raw();
    text(ctx, |b, c, l| unsafe { rx_class(ctx, node, b, c, l) })
}
/// The operands of a node.
pub fn operands(ctx: impl Into<CtxPtr>, node: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    handles(ctx, |b, c, n| unsafe { rx_operands(ctx, node, b, c, n) })
}
/// The nodes of a walk, in order.
pub fn walk_order(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    handles(ctx, |b, c, n| unsafe { rx_walk_order(ctx, walk, b, c, n) })
}
/// Whether the node's class is exactly `name` (a refinement of it does not count, unlike `is_a`).
pub fn class_is(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<bool, Refusal> {
    Ok(class_name(ctx, node)? == name)
}

/// The plain text of a field that always holds text (no quotes around it); a field that holds
/// nothing is refused.
pub fn text_field(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<String, Refusal> {
    let held = field_text(ctx, node, name)?;
    held.ok_or_else(|| fault(format!("node {node} has no {name}")))
}

/// Whether the walk binds the slot (`role`, `name`).
pub fn walk_binds(
    ctx: impl Into<CtxPtr>,
    walk: u32,
    role: &str,
    name: &str,
) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let (role, name) = (cs(role)?, cs(name)?);
    let mut out: c_int = 0;
    check(ctx, unsafe {
        rx_walk_binds(ctx, walk, role.as_ptr(), name.as_ptr(), &mut out)
    })?;
    Ok(out != 0)
}

/// The slots the walk binds, as (role, name) pairs in argument order.
pub fn walk_args(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<(String, String)>, Refusal> {
    let ctx = ctx.into().raw();
    let (mut len, mut count) = (0usize, 0usize);
    let status = unsafe { rx_walk_args(ctx, walk, std::ptr::null_mut(), 0, &mut len, &mut count) };
    if status != RX_TRUNC {
        check(ctx, status)?;
        return Ok(vec![]);
    }
    let mut buf = vec![0u8; len + 1];
    let status = unsafe {
        rx_walk_args(
            ctx,
            walk,
            buf.as_mut_ptr() as *mut c_char,
            len + 1,
            &mut len,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(len);
    let text = String::from_utf8(buf).map_err(fault)?;
    let texts: Vec<&str> = text.split_terminator('\0').collect();
    Ok(texts
        .chunks(2)
        .filter(|pair| pair.len() == 2)
        .map(|pair| (pair[0].to_string(), pair[1].to_string()))
        .collect())
}

/// The digest of a walk.
pub fn digest(ctx: impl Into<CtxPtr>, walk: u32) -> Result<String, Refusal> {
    let ctx = ctx.into().raw();
    let mut out = [0 as c_char; 65];
    check(ctx, unsafe { rx_digest(ctx, walk, out.as_mut_ptr()) })?;
    Ok(unsafe { std::ffi::CStr::from_ptr(out.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// The canonical position of any node the walk reached (a duplicate answers its representative's
/// position).
pub fn position(ctx: impl Into<CtxPtr>, walk: u32, node: u32) -> Result<usize, Refusal> {
    let ctx = ctx.into().raw();
    let mut p = 0u32;
    check(ctx, unsafe { rx_walk_position(ctx, walk, node, &mut p) })?;
    Ok(p as usize)
}
/// The nodes in the body of a loop.
pub fn body_nodes(ctx: impl Into<CtxPtr>, lp: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    handles(ctx, |b, c, n| unsafe { rx_body_nodes(ctx, lp, b, c, n) })
}
/// The roots of the body of a loop.
pub fn body_roots(ctx: impl Into<CtxPtr>, lp: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    handles(ctx, |b, c, n| unsafe { rx_body_roots(ctx, lp, b, c, n) })
}

/// A constant raptor owns, by name, as the list of texts it holds; refuses a name that is no list.
pub fn constant_list(ctx: impl Into<CtxPtr>, name: &str) -> Result<Vec<String>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let (mut len, mut count) = (0usize, 0usize);
    let status = unsafe {
        rx_const_names(
            ctx,
            name.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut len,
            &mut count,
        )
    };
    if status != RX_TRUNC {
        check(ctx, status)?;
        return Ok(vec![]);
    }
    let mut buf = vec![0u8; len + 1];
    let status = unsafe {
        rx_const_names(
            ctx,
            name.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            len + 1,
            &mut len,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(len);
    let text = String::from_utf8(buf).map_err(fault)?;
    Ok(text.split_terminator('\0').map(String::from).collect())
}

/// The dispatch nodes of a walk, top-level first, then those inside loop bodies: the order that
/// names its dispatch groups.
pub fn dispatch_nodes(ctx: impl Into<CtxPtr>, walk: u32) -> Result<Vec<u32>, Refusal> {
    let ctx = ctx.into().raw();
    handles(ctx, |b, c, n| unsafe {
        rx_dispatch_nodes(ctx, walk, b, c, n)
    })
}

/// The canonical walk over the given sinks, keeping the slots the unit's own walk declares (its
/// leaves), so a sub-render
/// of a unit (a boxed body, a single step) binds the same arguments as the entry. No plane is
/// exempt.
pub fn canonical_over(
    ctx: impl Into<CtxPtr>,
    sinks: &[u32],
    unit_walk: u32,
) -> Result<u32, Refusal> {
    let ctx = ctx.into().raw();
    let mut walk = u32::MAX;
    let status =
        unsafe { rx_canonical_like(ctx, sinks.as_ptr(), sinks.len(), unit_walk, &mut walk) };
    check(ctx, status)?;
    Ok(walk)
}

/// The contract name whose accumulate commits through a companion plane when the walk is keyed
/// automatically.
const KEYED_AUTO_CONTRACT: &str = "keyed_auto";

/// Whether an accumulate under the contract commits through a companion plane: the contracts
/// `compensated` and
/// the automatic keyed one.
pub fn compensates(contract: &str) -> Result<bool, Refusal> {
    Ok(contract == "compensated" || contract == KEYED_AUTO_CONTRACT)
}

/// Whether the node is of the named node kind or a kind derived from it.
pub fn is_a(ctx: impl Into<CtxPtr>, node: u32, kind: &str) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(kind)?;
    let mut out: c_int = 0;
    check(ctx, unsafe { rx_is_a(ctx, node, name.as_ptr(), &mut out) })?;
    Ok(out != 0)
}

/// The type of a node. Refuses (the door's message) a rank over 8 or an unknown node.
pub fn ttype_of(ctx: impl Into<CtxPtr>, node: u32) -> Result<Ttype, Refusal> {
    let ctx = ctx.into().raw();
    let mut raw = RxType {
        rank: 0,
        dims: [0; 8],
        dtype: 0,
        tag: [0; 64],
    };
    check(ctx, unsafe { rx_ttype_of(ctx, node, &mut raw) })?;
    let dtype = DTYPE_NAMES
        .get(raw.dtype as usize)
        .ok_or_else(|| fault(format!("dtype code {}", raw.dtype)))?;
    let tag = unsafe { std::ffi::CStr::from_ptr(raw.tag.as_ptr()) }.to_string_lossy();
    Ok(Ttype {
        shape: raw.dims[..raw.rank as usize].to_vec(),
        dtype: dtype.to_string(),
        tag: (!tag.is_empty()).then(|| tag.into_owned()),
    })
}

/// Reads a status of a typed reader: `Ok(true)` for a value, `Ok(false)` for absence, the door's
/// message otherwise.
fn present(ctx: *mut RxCtx, status: c_int) -> Result<bool, Refusal> {
    match status {
        RX_ABSENT => Ok(false),
        other => check(ctx, other).map(|()| true),
    }
}

/// A field that holds text, without quotes; `None` when the field holds nothing; an error when it
/// holds another kind.
pub fn field_text(
    ctx: impl Into<CtxPtr>,
    node: u32,
    name: &str,
) -> Result<Option<String>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut len = 0usize;
    let mut status =
        unsafe { rx_field_text(ctx, node, name.as_ptr(), std::ptr::null_mut(), 0, &mut len) };
    if status != RX_TRUNC {
        // an empty text needs no second call
        return Ok(present(ctx, status)?.then(String::new));
    }
    let mut buf = vec![0u8; len + 1];
    status = unsafe {
        rx_field_text(
            ctx,
            node,
            name.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            len + 1,
            &mut len,
        )
    };
    check(ctx, status)?;
    buf.truncate(len);
    String::from_utf8(buf).map(Some).map_err(fault)
}

/// A field that holds an integer; `None` when the field holds nothing.
pub fn field_int(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<Option<i64>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut out = 0i64;
    let status = unsafe { rx_field_int(ctx, node, name.as_ptr(), &mut out) };
    Ok(present(ctx, status)?.then_some(out))
}

/// A field that holds a real number (an integer reads as its value); `None` when the field holds
/// nothing.
pub fn field_real(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<Option<f64>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut out = 0f64;
    let status = unsafe { rx_field_real(ctx, node, name.as_ptr(), &mut out) };
    Ok(present(ctx, status)?.then_some(out))
}

/// A field that holds a flag; `None` when the field holds nothing.
pub fn field_flag(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<Option<bool>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut out: c_int = 0;
    let status = unsafe { rx_field_flag(ctx, node, name.as_ptr(), &mut out) };
    Ok(present(ctx, status)?.then_some(out != 0))
}

/// The node references a field holds, each distinct node once in order of first appearance; `None`
/// when the field
/// holds nothing (a field holding no reference answers an empty list).
pub fn field_refs(
    ctx: impl Into<CtxPtr>,
    node: u32,
    name: &str,
) -> Result<Option<Vec<u32>>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut count = 0usize;
    let status = unsafe {
        rx_field_nodes(
            ctx,
            node,
            name.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if status != RX_TRUNC {
        return Ok(present(ctx, status)?.then(Vec::new));
    }
    let mut buf = vec![0u32; count];
    let status = unsafe {
        rx_field_nodes(
            ctx,
            node,
            name.as_ptr(),
            buf.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(count);
    Ok(Some(buf))
}

/// The names a field holds as a list of texts; `None` when the field holds nothing.
pub fn field_names(
    ctx: impl Into<CtxPtr>,
    node: u32,
    name: &str,
) -> Result<Option<Vec<String>>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let (mut len, mut count) = (0usize, 0usize);
    let status = unsafe {
        rx_field_names(
            ctx,
            node,
            name.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut len,
            &mut count,
        )
    };
    if status != RX_TRUNC {
        return Ok(present(ctx, status)?.then(Vec::new));
    }
    let mut buf = vec![0u8; len + 1];
    let status = unsafe {
        rx_field_names(
            ctx,
            node,
            name.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            len + 1,
            &mut len,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(len);
    let text = String::from_utf8(buf).map_err(fault)?;
    Ok(Some(
        text.split_terminator('\0').map(String::from).collect(),
    ))
}

/// Whether the node lies inside a loop body of the walk (the walk's flat traversal gives it no
/// position).
pub fn in_loop_body(ctx: impl Into<CtxPtr>, walk: u32, node: u32) -> Result<bool, Refusal> {
    let ctx = ctx.into().raw();
    let mut out: c_int = 0;
    check(ctx, unsafe { rx_in_loop_body(ctx, walk, node, &mut out) })?;
    Ok(out != 0)
}

/// The node references a field holds in the order the field names them, a node listed as often as it
/// is named (a positional list, such as the carried values of a loop); `None` when the field holds
/// nothing, an empty list when it holds no reference.
pub fn field_node_list(
    ctx: impl Into<CtxPtr>,
    node: u32,
    name: &str,
) -> Result<Option<Vec<u32>>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut count = 0usize;
    let status = unsafe {
        rx_field_node_list(
            ctx,
            node,
            name.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if status != RX_TRUNC {
        return Ok(present(ctx, status)?.then(Vec::new));
    }
    let mut buf = vec![0u32; count];
    let status = unsafe {
        rx_field_node_list(
            ctx,
            node,
            name.as_ptr(),
            buf.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(count);
    Ok(Some(buf))
}

/// A field that holds a list of integers; `None` when the field holds nothing.
pub fn field_ints(
    ctx: impl Into<CtxPtr>,
    node: u32,
    name: &str,
) -> Result<Option<Vec<i64>>, Refusal> {
    let ctx = ctx.into().raw();
    let name = cs(name)?;
    let mut count = 0usize;
    let status = unsafe {
        rx_field_ints(
            ctx,
            node,
            name.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if status != RX_TRUNC {
        return Ok(present(ctx, status)?.then(Vec::new));
    }
    let mut buf = vec![0i64; count];
    let status = unsafe {
        rx_field_ints(
            ctx,
            node,
            name.as_ptr(),
            buf.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    check(ctx, status)?;
    buf.truncate(count);
    Ok(Some(buf))
}

/// What a field that holds a scalar, a list of integers or nothing holds (a node literal).
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    /// The field holds nothing.
    Absent,
    /// A text.
    Text(String),
    /// An integer.
    Int(i64),
    /// A real number.
    Real(f64),
    /// A flag.
    Flag(bool),
    /// A tuple of integers (the index of a component read).
    Ints(Vec<i64>),
}

/// The kind code of `rx_field_kind` for a field that holds nothing.
const KIND_ABSENT: c_int = 0;
/// The kind code for a field that holds text.
const KIND_TEXT: c_int = 1;
/// The kind code for a field that holds an integer.
const KIND_INTEGER: c_int = 2;
/// The kind code for a field that holds a real number.
const KIND_REAL: c_int = 3;
/// The kind code for a field that holds a flag.
const KIND_FLAG: c_int = 4;
/// The kind code for a field that holds a tuple of integers.
const KIND_INTS: c_int = 7;

/// The scalar a field holds, read by the typed reader its kind picks; refuses a field that holds
/// anything else.
pub fn field_scalar(ctx: impl Into<CtxPtr>, node: u32, name: &str) -> Result<Scalar, Refusal> {
    let ctx = ctx.into().raw();
    let cname = cs(name)?;
    let mut kind: c_int = 0;
    check(ctx, unsafe {
        rx_field_kind(ctx, node, cname.as_ptr(), &mut kind)
    })?;
    let missing = || fault(format!("node {node} lost its {name}"));
    Ok(match kind {
        KIND_ABSENT => Scalar::Absent,
        KIND_TEXT => Scalar::Text(field_text(ctx, node, name)?.ok_or_else(missing)?),
        KIND_INTEGER => Scalar::Int(field_int(ctx, node, name)?.ok_or_else(missing)?),
        KIND_REAL => Scalar::Real(field_real(ctx, node, name)?.ok_or_else(missing)?),
        KIND_FLAG => Scalar::Flag(field_flag(ctx, node, name)?.ok_or_else(missing)?),
        KIND_INTS => Scalar::Ints(field_ints(ctx, node, name)?.ok_or_else(missing)?),
        _ => return Err(fault(format!("the field {name} holds more than a scalar"))),
    })
}

impl Scalar {
    /// Whether two scalars are the same value: of one kind, and a real bit for bit (so `0.0` and
    /// `-0.0` differ and a not-a-number equals itself).
    pub fn same_value(&self, other: &Scalar) -> bool {
        match (self, other) {
            (Scalar::Real(a), Scalar::Real(b)) => a.to_bits() == b.to_bits(),
            _ => self == other,
        }
    }

    /// The kind of the scalar in words, for a message.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Scalar::Absent => "nothing",
            Scalar::Text(_) => "text",
            Scalar::Int(_) => "an integer",
            Scalar::Real(_) => "a real",
            Scalar::Flag(_) => "a flag",
            Scalar::Ints(_) => "a tuple of integers",
        }
    }
}
