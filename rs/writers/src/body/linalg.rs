//! Linear algebra: factorisations, solves and field reads through aether::linalg.
use super::{Frame, NameKind, Out};
use crate::body::frame::needs::Need;
use crate::body::values::{literal_of, text_field};
use crate::body::{Class, OpKind};
use crate::ir::door;
use crate::ir::door::Scalar;
use crate::refuse::{nib, Refusal};
use crate::spell::binding_name;

/// The op kinds that are written through the library.
pub fn is_factor_kind(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Lu
            | OpKind::Chol
            | OpKind::Eigh
            | OpKind::CholFrom
            | OpKind::Qr
            | OpKind::LuSolve
            | OpKind::LuSolveT
            | OpKind::ChoSolve
            | OpKind::TriLower
            | OpKind::TriUpperT
            | OpKind::CholFactor
            | OpKind::EighW
            | OpKind::EighV
            | OpKind::LuDet
    ) || is_info_kind(kind)
}

/// The kinds that read the `info` field of a factor.
fn is_info_kind(kind: OpKind) -> bool {
    matches!(kind, OpKind::LuInfo | OpKind::CholInfo | OpKind::EighInfo)
}

/// The library call a solve kind is spelled with.
fn solve_name(kind: OpKind) -> Option<&'static str> {
    Some(match kind {
        OpKind::LuSolve => "luSolve",
        OpKind::LuSolveT => "luSolveT",
        OpKind::ChoSolve => "choSolve",
        _ => return None,
    })
}

/// The library call of a solve against the factor's lower triangle alone.
fn triangular_name(kind: OpKind) -> Option<&'static str> {
    Some(match kind {
        OpKind::TriLower => "triLower",
        OpKind::TriUpperT => "triUpperT",
        _ => return None,
    })
}

/// The element type the library is instantiated with.
fn element(dtype: &str) -> Result<&'static str, Refusal> {
    match dtype {
        "f64" | "f32" => Ok("Real"),
        d => Err(nib(format!("linear algebra over dtype {d}"))),
    }
}

/// The extents of a shape as a comma-separated list, as a template argument list spells them.
fn extents(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The node classes that store a value into a team or log plane.
const STORE_CLASSES: [&str; 4] = ["TeamStore", "TeamCopy", "ScatterLog", "LogStore"];

/// Whether a leaf is a library operand bound as the plane's own `[i]` proxy: every one of its
/// readers is a library call
/// taking it as the matrix (factorisation) or the right-hand side (solve), a team store, or a
/// component read, and at least one is a call.
pub fn leaf_is_proxy(f: &Frame, p: usize) -> bool {
    let (mut proxied, mut all_read) = (false, true);
    let readers = super::values::reader_kinds(f, p).unwrap_or_default();
    all_read &= !readers.is_empty();
    for (kind, slot, reader) in readers {
        let stored = slot == 0
            && STORE_CLASSES
                .iter()
                .any(|k| door::class_is(f.ctx(), reader, k).unwrap_or(false));
        let library_use = stored || (matches!(kind, Some(OpKind::Lu | OpKind::Chol | OpKind::Eigh | OpKind::CholFrom)) && slot == 0)
            || (kind.is_some_and(|k| solve_name(k).or(triangular_name(k)).is_some()) && slot == 1)
            // a rolled product reads its operands as proxies, and so does a value no statement
            // renders
            // a placed value's store loop reads its operand by element, and so does a value only
            // that loop reads
            || (f.placed && (f.absorbed.contains(&reader) || kind == Some(OpKind::Resident)))
            || super::ops::is_rolled_handle(f, reader) || crate::ir::door::position(f.ctx(), f.walk(), reader).ok().is_some_and(|u| super::ops::is_absorbed(f, u).unwrap_or(false));
        proxied |= library_use;
        all_read &= library_use || (kind == Some(OpKind::Component) && slot == 0);
    }
    proxied && all_read
}

/// The plane target a library result is written into directly: the own-column store of the only
/// reader, when the kernel has
/// no loop and never reads that plane's prior value. None when the result goes to a fresh local.
fn direct_target(f: &Frame, p: usize) -> Result<Option<String>, Refusal> {
    if f.fanout(p) != 1 || f.nodes().any(|n| n.class == Class::Loop) {
        return Ok(None);
    }
    let mut read_planes = vec![];
    for n in f.nodes().filter(|n| n.class == Class::Leaf) {
        if text_field(f.ctx(), n.h, "role")? == "mutable" {
            read_planes.push(text_field(f.ctx(), n.h, "name")?)
        }
    }
    let mut companions = vec![];
    for &s in f.sinks() {
        let sink = &f.node(f.at(s)?);
        if sink.class == Class::AccumWrite {
            companions.push(format!("{}_c", text_field(f.ctx(), sink.h, "name")?))
        }
    }
    for &s in f.sinks() {
        let sp = f.at(s)?;
        let sink = &f.node(sp);
        if sink.class != Class::Assign || sink.ops.len() != 1 || f.opp(sp)[0] != p {
            continue;
        }
        let name = text_field(f.ctx(), sink.h, "name")?;
        if read_planes.contains(&name) || companions.contains(&name) {
            return Ok(None);
        }
        return Ok(Some(format!(
            "{}[i]",
            binding_name(&text_field(f.ctx(), sink.h, "role")?, sink.rank(), &name)?
        )));
    }
    Ok(None)
}

/// Declares an unset `aether::Item` for library calls to fill, named by the shared counter; returns
/// its name.
pub(super) fn fresh_item(f: &mut Frame, p: usize) -> Result<String, Refusal> {
    let name = f.names.fresh_item();
    let node = &f.node(p);
    let cty = format!(
        "aether::Item<{}, {}>",
        element(&node.ty.dtype)?,
        extents(&node.ty.shape)
    );
    let text = name.text.clone();
    f.declare_as(name, cty, None);
    Ok(text)
}

/// Whether a text names a fresh item of a library call (those are filled element by element through
/// `data()`).
pub(super) fn is_fresh(f: &Frame, name: &str) -> bool {
    let found = f
        .buf
        .decls
        .iter()
        .find(|d| d.name.text == name && d.name.kind == NameKind::FreshItem);
    found.is_some()
}

/// Whether an earlier declaration named `name` a materialised row or item (so a library can read it
/// in place).
fn declared_as_item(f: &Frame, name: &str) -> bool {
    // the names a loop body gives its values are not recorded as items
    let found = f.buf.decls.iter().find(|d| {
        d.name.text == name
            && d.constant
            && d.name.kind != NameKind::BodyTemp
            && d.value.as_ref().is_some_and(|value| {
                d.cty == "auto" && value.ends_with("[i].get()")
                    || d.cty.starts_with("aether::Item<")
            })
    });
    found.is_some()
}

/// The name an earlier declaration gave to the held item of `text`, if one did.
fn held_item_of(f: &Frame, text: &str) -> Option<String> {
    let found = f
        .buf
        .decls
        .iter()
        .find(|d| d.name.kind == NameKind::HeldItem && d.value.as_deref() == Some(text));
    found.map(|d| d.name.text.clone())
}

/// A value as the per-sample operand a library call reads by element. A plane leaf is its proxy
/// (the leaf's own local
/// when bound as one, else `plane[i]`), a named item itself, any other value an item held once and
/// declared ahead.
pub(super) fn operand(f: &mut Frame, source: usize, arg: &str) -> Result<String, Refusal> {
    let (class, rank) = (f.node(source).class.clone(), f.node(source).rank());
    // a value that lives in a plane: its proxy
    if f.proxies.iter().any(|x| x.name == arg) {
        f.record_read(source);
        return Ok(arg.to_string());
    }
    if class == Class::Leaf && rank > 0 {
        let (role, name) = (
            text_field(f.ctx(), f.node(source).h, "role")?,
            text_field(f.ctx(), f.node(source).h, "name")?,
        );
        if matches!(
            role.as_str(),
            "mutable" | "uniform" | "nsamples" | "lookup" | "wide_in"
        ) {
            return Err(nib(format!("a library operand from a {role} leaf")));
        }
        return if leaf_is_proxy(f, source) {
            f.record_read(source);
            Ok(arg.to_string())
        } else {
            Ok(format!(
                "{}[i]",
                binding_name(&role, f.node(source).rank(), &name)?
            ))
        };
    }
    if class == Class::At {
        // a plane read at another lane bound as its proxy is the operand as it is
        return if f.is_lazy(source) {
            f.record_read(source);
            Ok(arg.to_string())
        } else {
            Err(nib("a library operand read at another lane"))
        };
    }
    f.record_read(source);
    let assembled = f.node(source).kind == Some(OpKind::Vec);
    if is_fresh(f, arg) || assembled {
        return Ok(arg.to_string());
    }
    held_item(f, source, arg)
}

/// The name of a value materialised once as a const item, declared ahead of the line on first use
/// (`arg` is its expression).
pub(super) fn held_item(f: &mut Frame, source: usize, arg: &str) -> Result<String, Refusal> {
    if declared_as_item(f, arg) {
        return Ok(arg.to_string());
    }
    if let Some(held) = held_item_of(f, arg) {
        return Ok(held);
    }
    let held = f.names.held_item();
    let cty = format!(
        "aether::Item<{}, {}>",
        element(&f.node(source).ty.dtype)?,
        extents(&f.node(source).ty.shape)
    );
    let text = held.text.clone();
    f.declare_as(held, cty, Some(arg));
    Ok(text)
}

/// `aether::linalg::<name><targs>(args)`.
fn call(name: &str, targs: &[usize], args: &[&str]) -> String {
    let targs = if targs.is_empty() {
        String::new()
    } else {
        format!(
            "<{}>",
            targs
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!("aether::linalg::{name}{targs}({})", args.join(", "))
}

/// A factorisation: declares the struct (`hawk_lf<k>`) and fills it in place from the matrix
/// operand; the value is its name.
/// Eigenvalues alone use the cheaper struct when no `eigh_v` reads this factor.
fn factorisation(f: &mut Frame, p: usize, kind: OpKind) -> Result<Out, Refusal> {
    let n = f.node(f.opp(p)[0]).ty.shape[0];
    let factor = f.names.factor();
    let name = factor.text.clone();
    let vectors = f
        .nodes()
        .enumerate()
        .any(|(q, m)| m.kind == Some(OpKind::EighV) && f.opp(q).first() == Some(&p));
    let (function, structure) = match (kind, vectors) {
        (OpKind::Lu, _) => ("lu", "Lu"),
        (OpKind::Chol, _) => ("cholesky", "Cholesky"),
        (OpKind::CholFrom, _) => ("choleskyFrom", "Cholesky"),
        (OpKind::Eigh, false) => ("eighValues", "EighValues"),
        (OpKind::Eigh, true) => ("eigh", "Eigh"),
        _ => return Err(nib(format!("op {kind} of rank > 0"))),
    };
    let elem = element(&f.node(p).ty.dtype)?;
    f.need(Need::Linalg);
    f.declare_var(
        factor,
        format!("aether::linalg::{structure}<{elem}, {n}>"),
        None,
    );
    let matrix = f.opp(p)[0];
    let held = operand(f, matrix, &f.text(matrix).clone())?;
    f.emit(format!("{};", call(function, &[n], &[&name, &held])));
    f.set_text(p, name);
    Ok(Out::Bound)
}

/// Whether a value is the literal identity matrix (an assembled square matrix of ones on the
/// diagonal and zeros elsewhere).
pub(super) fn is_identity(f: &Frame, q: usize) -> Result<bool, Refusal> {
    let node = &f.node(q);
    if node.kind != Some(OpKind::Vec) || node.rank() != 2 || node.ty.shape[0] != node.ty.shape[1] {
        return Ok(false);
    }
    let n = node.ty.shape[0];
    for (k, &entry) in f.opp(q).iter().enumerate() {
        if f.node(entry).class != Class::Const {
            return Ok(false);
        }
        let want = if k / n == k % n { 1.0 } else { 0.0 };
        let held = match literal_of(f.ctx(), f.node(entry).h)? {
            Scalar::Int(v) => Some(v as f64),
            Scalar::Real(v) => Some(v),
            _ => None,
        };
        if held != Some(want) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A solve (`luSolve`, `choSolve`, `triLower`, ...) or a field read (`choleskyFactor`, the eigen
/// fields): written by
/// statements ahead of the line into the plane directly or into a fresh item; the value is that
/// target.
fn consumer(f: &mut Frame, p: usize, kind: OpKind) -> Result<Out, Refusal> {
    let n = f.node(f.opp(p)[0]).ty.shape[0];
    let out = match direct_target(f, p)? {
        Some(target) => target,
        None => fresh_item(f, p)?,
    };
    let factor = f.text(f.opp(p)[0]).clone();
    if solve_name(kind).is_some() || triangular_name(kind).is_some() {
        let rhs = f.opp(p)[1];
        let columns = if f.node(rhs).rank() == 2 {
            f.node(rhs).ty.shape[1]
        } else {
            1
        };
        if kind == OpKind::LuSolve && is_identity(f, rhs)? {
            f.emit(format!("{};", call("luInverse", &[], &[&factor, &out])));
            f.set_text(p, out);
            return Ok(Out::Bound);
        }
        let held = operand(f, rhs, &f.text(rhs).clone())?;
        let text = match triangular_name(kind) {
            Some(name) => call(name, &[n, columns], &[&format!("{factor}.l"), &held, &out]),
            None => call(
                solve_name(kind).unwrap_or_default(),
                &[columns],
                &[&factor, &held, &out],
            ),
        };
        f.emit(format!("{text};"));
    } else if kind == OpKind::CholFactor {
        f.emit(format!(
            "{};",
            call("choleskyFactor", &[], &[&factor, &out])
        ));
    } else {
        let field = if kind == OpKind::EighW { "w" } else { "v" };
        let shape = f.node(p).ty.shape.clone();
        let count: usize = shape.iter().product();
        let slot = if is_fresh(f, &out) {
            format!("{out}.data()[hawk_lk]")
        } else if shape.len() == 1 {
            format!("{out}.at(hawk_lk)")
        } else {
            format!("{out}.at(hawk_lk / {}, hawk_lk % {})", shape[1], shape[1])
        };
        f.emit(format!("for (Int hawk_lk = 0; hawk_lk < {count}; ++hawk_lk) {slot} = {factor}.{field}[hawk_lk];"));
    }
    f.set_text(p, out);
    Ok(Out::Bound)
}

/// The text of a factor kind: a factorisation, a solve or field read, `<factor>.info`, or the
/// determinant call.
pub fn op(f: &mut Frame, p: usize, kind: OpKind) -> Result<Out, Refusal> {
    match kind {
        OpKind::Lu | OpKind::Chol | OpKind::CholFrom | OpKind::Eigh => factorisation(f, p, kind),
        _ if is_info_kind(kind) => Ok(Out::Value(format!("{}.info", f.text(f.opp(p)[0])), None)),
        OpKind::LuDet => Ok(Out::Value(call("luDet", &[], &[f.text(f.opp(p)[0])]), None)),
        OpKind::LuSolve
        | OpKind::LuSolveT
        | OpKind::ChoSolve
        | OpKind::TriLower
        | OpKind::TriUpperT
        | OpKind::CholFactor
        | OpKind::EighW
        | OpKind::EighV => consumer(f, p, kind),
        _ => Err(nib(format!("op {kind} of rank > 0"))),
    }
}
