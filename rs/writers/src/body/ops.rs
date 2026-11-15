//! Operations: scalar and tensor spellings, casts, integer calls and index clamps.
use super::{linalg, Frame, Name, NameKind, Node, Out, Span};
use crate::body::frame::needs::Need;
use crate::body::values::{component_index, literal_of, text_field};
use crate::body::Class;
use crate::ir::door::Scalar;
use crate::ir::door::{self, RxCtx};
use crate::refuse::nib;
use crate::spell::{self};

/// The text of the operation at position `p`: a statement bound ahead of the line (`Out::Bound`) or
/// the expression that
/// stands for it. The operation kinds are told apart by family: rolled folds, whole-value forms (a
/// value placed in its plane,
/// a raw block, a store), scalar spellings, and tensor spellings (library calls, rays, expression
/// methods).
pub fn op(f: &mut Frame, p: usize) -> Result<Out, String> {
    let h = f.node(p).h;
    let kind = text_field(f.ctx(), h, "kind")?;
    if is_absorbed(f, p)? {
        f.set_text(p, String::new());
        return Ok(Out::Bound);
    }
    if let Some(out) = rolled_fold(f, p, &kind)? {
        return Ok(out);
    }
    if kind == "vec"
        && f.node(p).rank() == 2
        && f.node(p).numel() > f.unit.build.form.rolled_fold_min as usize
    {
        return assembled(f, p);
    }
    // a runtime-index read spells its own form of the plane it reads, so it records only what it embeds
    let args: Vec<String> = if matches!(
        kind.as_str(),
        "component_at" | "resident" | "ray_tri" | "ray_aabb" | "ray_sphere"
    ) {
        f.opp(p).iter().map(|&q| f.peek(q).clone()).collect()
    } else {
        f.opp(p).iter().map(|&q| f.text(q).clone()).collect()
    };
    if let Some(out) = whole_value(f, p, &kind, &args)? {
        return Ok(out);
    }
    let ranks: Vec<usize> = f.opp(p).iter().map(|&q| f.node(q).rank()).collect();
    match scalar_text(f, p, &kind, &args, &ranks)? {
        Some(text) => Ok(Out::Value(text, None)),
        None => tensor_value(f, p, &kind, &args, &ranks),
    }
}

/// A product or a vector reduction past the fold threshold, spelled as its `aether::rt` twin; none
/// for any other operation.
fn rolled_fold(f: &mut Frame, p: usize, kind: &str) -> Result<Option<Out>, String> {
    let h = f.node(p).h;
    let shape = |slot: usize| f.node(f.opp(p)[slot]).ty.shape.clone();
    let (fold, size) = match kind {
        "mv" | "mm" => (shape(0).get(1).copied().unwrap_or(0), f.node(p).numel()),
        "dot" | "sum" | "norm" if shape(0).len() == 1 => (shape(0)[0], 1),
        _ => (0, 0),
    };
    let rolled = fold + size > 0
        && crate::ir::analysis::fold_rolled(
            f.ctx(),
            h,
            f.unit.build.form.rolled_fold_min,
            f.unit.build.fold_form.name(),
        )?;
    if !rolled {
        return Ok(None);
    }
    if matches!(kind, "dot" | "sum" | "norm") {
        return rolled_reduction(f, p, kind).map(Some);
    }
    rolled_product(f, p, kind).map(Some)
}

/// The operations that are written as a whole, whatever their rank: a value placed in its plane,
/// the full sum of a matrix, a
/// raw block, a refused least-squares solve, a store at a runtime index. None for any other kind.
fn whole_value(
    f: &mut Frame,
    p: usize,
    kind: &str,
    args: &[String],
) -> Result<Option<Out>, String> {
    // a value placed in its plane: the host text has no planes, so the value is its own text and is
    // never named
    if kind == "resident" && f.placed {
        return super::placed::resident(f, p).map(Some);
    }
    if kind == "resident" {
        return if f.unit.build.blend {
            f.record_read(f.opp(p)[0]);
            f.set_text(p, args[0].clone());
            Ok(Some(Out::Bound))
        } else {
            Err(nib("a value placed in its plane (device text)"))
        };
    }
    if kind == "sum" && f.node(f.opp(p)[0]).rank() == 2 {
        return matrix_sum(f, p).map(Some);
    }
    if kind == "raw_device" {
        return raw_block(f, p, args).map(Some);
    }
    if kind.starts_with("lstsq") {
        return Err(nib(format!(
            "op {kind}: least squares is lowered by raptor.wings.base.lstsq before the seam"
        )));
    }
    if matches!(kind, "set_component_at" | "add_at") {
        return super::stores::storeplan::store(f, p).map(Some);
    }
    Ok(None)
}

/// The text of a scalar operation (every operand of rank 0, or the index clamp, sign and cast of
/// one scalar); none when the
/// operation is a tensor one.
fn scalar_text(
    f: &mut Frame,
    p: usize,
    kind: &str,
    args: &[String],
    ranks: &[usize],
) -> Result<Option<String>, String> {
    let rank = f.node(p).rank();
    let dtype = f.node(p).ty.dtype.clone();
    if kind == "clamp_index" && rank == 0 {
        return Ok(Some(clamp_index_text(f, p, args)));
    }
    if kind == "sign" && rank == 0 && ranks == [0] {
        return Ok(Some(sign_text(f, &args[0])));
    }
    if kind == "cast" && rank == 0 && ranks == [0] {
        let source = f.node(f.opp(p)[0]).ty.dtype.clone();
        let mut used = vec![];
        let text = spell::cast_op(&args[0], &dtype, &source, &mut used)?;
        f.hold_needs(p, &used);
        return Ok(Some(text));
    }
    if rank == 0 && ranks.iter().all(|&r| r == 0) {
        let mut used = vec![];
        let text = spell::scalar_op(kind, args, &dtype, &mut used)?;
        f.hold_needs(p, &used);
        return Ok(Some(text));
    }
    Ok(None)
}

/// An index clamp of a scalar: the library call, or, where one range test of the group's span
/// proves every read of the group
/// in range, the plain read guarded by that test.
fn clamp_index_text(f: &mut Frame, p: usize, args: &[String]) -> String {
    f.need(Need::Integers);
    let held = format!("hawk_abi::int_clamp_index<Int>({})", args.join(", "));
    match f.span(p) {
        None => held,
        Some(Span { base, lo, hi }) => {
            let b = format!("static_cast<Int>({})", f.text(base));
            format!("(__builtin_expect(({b} >= static_cast<Int>({})) & ({b} < {} - static_cast<Int>({hi})), 1) ? {} : {held})", -lo, args[1], args[0])
        }
    }
}

/// The sign of a scalar: it maps NaN to itself (the argument is read twice, so it is held in a name
/// unless it is one).
fn sign_text(f: &mut Frame, arg: &str) -> String {
    let is_name = arg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !arg.starts_with(|c: char| c.is_ascii_digit());
    let x = if is_name {
        arg.to_string()
    } else {
        f.hoist(arg.to_string())
    };
    format!("(aether::math::isnan({x}) ? {x} : aether::math::sign({x}))")
}

/// Whether an operand text is the lane proxy of a placed value.
fn is_proxy(f: &Frame, text: &str) -> bool {
    f.proxies.iter().any(|x| x.name == text)
}

/// The text of an operation over tensors: a library call, a ray primitive, a read at a runtime
/// index, an element read, an
/// expression method, or the tensor spelling.
fn tensor_value(
    f: &mut Frame,
    p: usize,
    kind: &str,
    args: &[String],
    ranks: &[usize],
) -> Result<Out, String> {
    let h = f.node(p).h;
    if linalg::is_factor_kind(kind) {
        return linalg::op(f, p, kind);
    }
    if matches!(kind, "ray_tri" | "ray_aabb" | "ray_sphere") {
        return Ok(Out::Value(ray_call(f, p, kind, args, ranks)?, None));
    }
    if kind == "component_at" {
        return super::stores::rtstore::component_at(f, p);
    }
    // a placed value is an aether expression to the whole-value ops; a static element read reads it
    // by element
    if kind == "component" && args.first().is_some_and(|a| is_proxy(f, a)) {
        let index = component_index(f.ctx(), h)?;
        return Ok(Out::Value(
            format!("{}.at({})", args[0], join_index(&index)),
            None,
        ));
    }
    let args: Vec<String> = args
        .iter()
        .map(|a| {
            if is_proxy(f, a) && kind != "component" {
                format!("{a}.expr()")
            } else {
                a.clone()
            }
        })
        .collect();
    if let Some(text) = expression_method(kind, &args, ranks) {
        return Ok(Out::Value(text, None));
    }
    spell_tensor(f, p, kind, &args, ranks)
}

/// The integers of a static index spelled as C++ arguments: `2` or `1, 2`.
fn join_index(index: &[i64]) -> String {
    index
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The tensor spelling of an operation. A static read of a mutable plane served in place is named
/// where it stands, ahead of
/// the stores.
fn spell_tensor(
    f: &mut Frame,
    p: usize,
    kind: &str,
    args: &[String],
    ranks: &[usize],
) -> Result<Out, String> {
    let h = f.node(p).h;
    let (dtype, shape) = (f.node(p).ty.dtype.clone(), f.node(p).ty.shape.clone());
    let component = if kind == "component" {
        component_index(f.ctx(), h)?
    } else {
        vec![]
    };
    let source = f.opp(p).first().copied();
    let lazy_source = if kind == "component" {
        source
            .filter(|&q| f.is_lazy(q))
            .map(|q| text_field(f.ctx(), f.node(q).h, "role").is_ok_and(|role| role == "mutable"))
    } else {
        None
    };
    let op = spell::TensorOp {
        kind,
        args,
        ranks,
        dtype: &dtype,
        shape: &shape,
        component: &component,
        lazy_source,
    };
    let text = spell::tensor_op(&op, &mut |t| f.hoist(t))?;
    if lazy_source == Some(true) && !f.nodes().any(|n| n.class == Class::Loop) {
        let name = text_field(f.ctx(), f.node(f.opp(p)[0]).h, "name")?;
        if super::stores::rtstore::plane_is_stored(f, &name)? {
            return Ok(super::stores::rtstore::named_before_store(f, p, &text));
        }
    }
    Ok(Out::Value(text, None))
}

/// One index clamp read at `base + offset`, by its own position.
struct ClampRead {
    /// The position of the clamp.
    at: usize,
    /// The position of the runtime base of the index.
    base: usize,
    /// The static offset added to the base.
    offset: i64,
}

/// The clamp reads that share one base position and one bound position.
struct ClampGroup {
    /// The position of the base the group shares.
    base: usize,
    /// The position of the bound the index is clamped to.
    bound: usize,
    /// The reads of the group, in walk order.
    reads: Vec<ClampRead>,
}

/// The `clamp_index` reads at `base + c` (c a static integer) that share `base` and the bound, two
/// or more per group: each maps to
/// (base position, min c, max c) of its group, by its own position.
pub fn clamp_spans(
    ctx: *mut RxCtx,
    nodes: &[Node],
    opp: &[Vec<usize>],
) -> Result<Vec<Option<Span>>, String> {
    let int = |q: usize| -> Result<Option<i64>, String> {
        if nodes[q].class != Class::Const {
            return Ok(None);
        }
        Ok(match literal_of(ctx, nodes[q].h)? {
            Scalar::Int(v) => Some(v),
            _ => None,
        })
    };
    let mut groups: Vec<ClampGroup> = vec![];
    for (p, n) in nodes.iter().enumerate() {
        if n.class != Class::Op || text_field(ctx, n.h, "kind")? != "clamp_index" {
            continue;
        }
        let (index, bound) = (opp[p][0], opp[p][1]);
        let (mut base, mut offset) = (index, 0i64);
        if nodes[index].class == Class::Op
            && matches!(
                text_field(ctx, nodes[index].h, "kind")?.as_str(),
                "add" | "sub"
            )
            && opp[index].len() == 2
        {
            let (a, b) = (opp[index][0], opp[index][1]);
            let add = text_field(ctx, nodes[index].h, "kind")? == "add";
            if let Some(c) = int(b)? {
                (base, offset) = (a, if add { c } else { -c })
            } else if add {
                if let Some(c) = int(a)? {
                    (base, offset) = (b, c)
                }
            }
        }
        let read = ClampRead {
            at: p,
            base,
            offset,
        };
        match groups
            .iter_mut()
            .find(|g| (g.base, g.bound) == (base, bound))
        {
            Some(group) => group.reads.push(read),
            None => groups.push(ClampGroup {
                base,
                bound,
                reads: vec![read],
            }),
        }
    }
    let mut spans = vec![None; nodes.len()];
    for group in groups.iter().filter(|g| g.reads.len() >= 2) {
        let offsets = group.reads.iter().map(|r| r.offset);
        let (lo, hi) = (offsets.clone().min().unwrap(), offsets.max().unwrap());
        for read in &group.reads {
            spans[read.at] = Some(Span {
                base: read.base,
                lo,
                hi,
            })
        }
    }
    Ok(spans)
}

/// The whole-value operations of aether's expressions that are one method or free function over the
/// operand texts:
/// the quaternion calls, the vector views and the component-wise quotient of two tensors. None for
/// any other kind.
fn expression_method(kind: &str, args: &[String], ranks: &[usize]) -> Option<String> {
    let method0 = match kind {
        "quat_conj" => "quatConj",
        "quat_recip" => "quatReciprocal",
        "as_pure" => "asPureQuaternion",
        "as_vec3" => "asBack3DVector",
        _ => "",
    };
    let method1 = match kind {
        "quat_mul" => "quatMul",
        "quat_rotate" => "quatRotate",
        _ => "",
    };
    match (args.len(), kind) {
        (1, _) if !method0.is_empty() => Some(format!("{}.{method0}()", args[0])),
        (2, _) if !method1.is_empty() => Some(format!("{}.{method1}({})", args[0], args[1])),
        (2, "div") if ranks[0] > 0 && ranks[1] > 0 => {
            Some(format!("aether::cwiseDiv({}, {})", args[0], args[1]))
        }
        _ => None,
    }
}

/// A product past the fold threshold, as its `aether::rt` twin written into a fresh item ahead of
/// the line that uses it.
/// A transposed operand is read through its own operand, so the transposed value is never built.
fn rolled_product(f: &mut Frame, p: usize, kind: &str) -> Result<Out, String> {
    let out = linalg::fresh_item(f, p)?;
    let (left, right) = (
        product_operand(f, f.opp(p)[0])?,
        product_operand(f, f.opp(p)[1])?,
    );
    let twin = if kind == "mv" {
        "matVecItems"
    } else {
        "matMat"
    };
    f.need(Need::Rt);
    f.emit(format!("aether::rt::{twin}({out}, {left}, {right});"));
    f.set_text(p, out);
    Ok(Out::Bound)
}

/// A fold over a vector past the threshold, as its `aether::rt` twin reading its operands per
/// sample.
fn rolled_reduction(f: &mut Frame, p: usize, kind: &str) -> Result<Out, String> {
    let mut operands = vec![];
    for k in 0..f.opp(p).len() {
        let q = f.opp(p)[k];
        let arg = f.text(q).clone();
        operands.push(linalg::operand(f, q, &arg)?)
    }
    f.need(Need::Rt);
    Ok(Out::Value(
        format!("aether::rt::{kind}({})", operands.join(", ")),
        None,
    ))
}

/// The full reduction of a matrix: the Frobenius inner product with a matrix of ones, which is held
/// once ahead of the line.
fn matrix_sum(f: &mut Frame, p: usize) -> Result<Out, String> {
    let q = f.opp(p)[0];
    let ty = f.node(q).ty.clone();
    let extents = ty
        .shape
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let one = super::placed::cast("1", &ty.dtype)?;
    let ones = f.hoist(format!(
        "aether::constant<aether::extents<{extents}>>({one})"
    ));
    Ok(Out::Value(format!("{}.matDot({ones})", f.text(q)), None))
}

/// An operand of a rolled product as a per-sample operand: a transpose is `aether::rt::transposed`
/// of its operand.
fn product_operand(f: &mut Frame, q: usize) -> Result<String, String> {
    if f.node(q).class == Class::Op && text_field(f.ctx(), f.node(q).h, "kind")? == "transpose" {
        let inner = product_operand(f, f.opp(q)[0])?;
        f.need(Need::Rt);
        return Ok(format!("aether::rt::transposed({inner})"));
    }
    let arg = f.text(q).clone();
    linalg::operand(f, q, &arg)
}

/// A matrix built from more entries than the fold threshold: a fresh item its entries are stored
/// into, one statement each.
fn assembled(f: &mut Frame, p: usize) -> Result<Out, String> {
    let out = linalg::fresh_item(f, p)?;
    let columns = f.node(p).ty.shape[1];
    for (k, &entry) in f.opp(p).clone().iter().enumerate() {
        let line = format!(
            "{out}({}, {}) = {};",
            k / columns,
            k % columns,
            f.text(entry)
        );
        f.emit(line);
    }
    f.set_text(p, out);
    Ok(Out::Bound)
}

/// Whether an operation (by handle) is a product or a fold the threshold and the form spell as its
/// `aether::rt` twin.
pub(super) fn is_rolled_handle(f: &Frame, h: u32) -> bool {
    if !door::class_is(f.ctx(), h, "Op").unwrap_or(false) {
        return false;
    }
    if !matches!(
        text_field(f.ctx(), h, "kind").unwrap_or_default().as_str(),
        "mv" | "mm" | "dot" | "sum" | "norm"
    ) {
        return false;
    }
    crate::ir::analysis::fold_rolled(
        f.ctx(),
        h,
        f.unit.build.form.rolled_fold_min,
        f.unit.build.fold_form.name(),
    )
    .unwrap_or(false)
}

/// Whether the operation at a position is a rolled product (which reads its transposed operands
/// through them).
pub(super) fn is_rolled_product(f: &Frame, p: usize) -> bool {
    matches!(
        text_field(f.ctx(), f.node(p).h, "kind")
            .unwrap_or_default()
            .as_str(),
        "mv" | "mm"
    ) && is_rolled_handle(f, f.node(p).h)
}

/// Whether a value is read only through the operand it hides: a transpose every reader of which is
/// a rolled product (which
/// reads through it), or the identity matrix a solve replaces by an inverse. No statement renders
/// such a value.
pub(super) fn is_absorbed(f: &Frame, p: usize) -> Result<bool, String> {
    if f.node(p).class != Class::Op || f.users(p).is_empty() {
        return Ok(false);
    }
    let kind = text_field(f.ctx(), f.node(p).h, "kind")?;
    if kind == "transpose" {
        return Ok(f.users(p).len() == f.fanout(p)
            && f.users(p).iter().all(|&(u, _)| is_rolled_product(f, u)));
    }
    if kind == "vec" && linalg::is_identity(f, p)? {
        return Ok(f.users(p).len() == f.fanout(p)
            && f.users(p).iter().all(|&(u, slot)| {
                slot == 1 && text_field(f.ctx(), f.node(u).h, "kind").is_ok_and(|k| k == "lu_solve")
            }));
    }
    Ok(false)
}

/// A ray primitive: the library returns its four values by value, held once, and the node is the
/// item of them. A vector operand
/// goes in as a per-sample operand the library reads by element; a rank-0 one as it is.
fn ray_call(
    f: &mut Frame,
    p: usize,
    kind: &str,
    args: &[String],
    ranks: &[usize],
) -> Result<String, String> {
    let function = match kind {
        "ray_tri" => "rayTriangle",
        "ray_aabb" => "rayAabb",
        _ => "raySphere",
    };
    let element = match f.node(p).ty.dtype.as_str() {
        "f64" | "f32" => "Real",
        d => return Err(nib(format!("a ray over dtype {d}"))),
    };
    let mut spelled = vec![];
    for (k, arg) in args.iter().enumerate() {
        spelled.push(if ranks[k] > 0 {
            linalg::operand(f, f.opp(p)[k], arg)?
        } else {
            f.record_read(f.opp(p)[k]);
            arg.clone()
        })
    }
    f.need(Need::Geometry);
    let held = f.hoist(format!(
        "aether::geometry::{function}<{element}>({})",
        spelled.join(", ")
    ));
    Ok(format!(
        "aether::Item<{element}, 4>{{{}}}",
        (0..4)
            .map(|k| format!("{held}.v[{k}]"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// A raw block of the aether language, spliced in parenthesised. Its operands are declared as
/// `hawk_raw_arg<k>` locals: the
/// first block that has operands declares them in the kernel's own scope, a later one inside its
/// own lambda so two blocks never
/// redefine the names.
fn raw_block(f: &mut Frame, p: usize, args: &[String]) -> Result<Out, String> {
    let h = f.node(p).h;
    let language = text_field(f.ctx(), h, "language")?;
    if language != "aether" {
        return Err(nib(format!("a raw block written in {language}")));
    }
    let text = text_field(f.ctx(), h, "text")?;
    let decls: Vec<(Name, &String)> = args
        .iter()
        .enumerate()
        .map(|(k, arg)| (Name::new(format!("hawk_raw_arg{k}"), NameKind::RawArg), arg))
        .collect();
    if decls.is_empty() {
        return Ok(Out::Value(format!("({text})"), None));
    }
    if !f.buf.decls.iter().any(|d| d.name.text == "hawk_raw_arg0") {
        for (name, arg) in decls {
            f.declare_auto(name, arg)
        }
        return Ok(Out::Value(format!("({text})"), None));
    }
    Ok(Out::Value(
        format!(
            "[&] {{ {} return ({text}); }}()",
            decls
                .iter()
                .map(|(name, arg)| format!("const auto {} = {arg};", name.text))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        None,
    ))
}
