//! Operations: scalar and tensor spellings, casts, integer calls and index clamps.
use super::{linalg, Frame, Name, NameKind, Node, Out, Span};
use crate::body::frame::needs::Need;
use crate::body::values::{component_index, literal_of, real_value, text_field};
use crate::body::{Class, OpKind};
use crate::ir::door::Scalar;
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, Refusal};
use crate::spell::{self};

/// The text of the operation at position `p`: a statement bound ahead of the line (`Out::Bound`) or
/// the expression that
/// stands for it. The operation kinds are told apart by family: rolled folds, whole-value forms (a
/// value placed in its plane,
/// a raw block, a store), scalar spellings, and tensor spellings (library calls, rays, expression
/// methods).
pub fn op(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let kind = f.op_kind(p)?;
    if is_absorbed(f, p)? {
        f.set_text(p, String::new());
        return Ok(Out::Bound);
    }
    if kind == OpKind::WarpSum {
        f.need(crate::body::frame::needs::Need::WarpSum);
        let value = f.text(f.opp(p)[0]).clone();
        let partial = if f.node(f.opp(p)[0]).rank() > 0 {
            format!("{value}.sum()")
        } else {
            value
        };
        return Ok(Out::Value(
            format!("hawk_abi::warp_sum<Real>(static_cast<Real>({partial}))"),
            None,
        ));
    }
    if let Some(out) = rolled_fold(f, p, kind)? {
        return Ok(out);
    }
    if kind == OpKind::Vec
        && f.node(p).rank() == 2
        && f.node(p).numel() > f.unit.build.form.rolled_fold_min as usize
    {
        return assembled(f, p);
    }
    // a runtime-index read spells its own form of the plane it reads, so it records only what it embeds
    let args: Vec<String> = if matches!(
        kind,
        OpKind::ComponentAt
            | OpKind::Resident
            | OpKind::RayTri
            | OpKind::RayAabb
            | OpKind::RaySphere
    ) {
        f.opp(p).iter().map(|&q| f.peek(q).clone()).collect()
    } else {
        f.opp(p).iter().map(|&q| f.text(q).clone()).collect()
    };
    if let Some(out) = whole_value(f, p, kind, &args)? {
        return Ok(out);
    }
    let ranks: Vec<usize> = f.opp(p).iter().map(|&q| f.node(q).rank()).collect();
    match scalar_text(f, p, kind, &args, &ranks)? {
        Some(text) => Ok(Out::Value(text, None)),
        None => tensor_value(f, p, kind, &args, &ranks),
    }
}

/// A product or a vector reduction past the fold threshold, spelled as its `aether::rt` twin; none
/// for any other operation.
fn rolled_fold(f: &mut Frame, p: usize, kind: OpKind) -> Result<Option<Out>, Refusal> {
    let h = f.node(p).h;
    let shape = |slot: usize| f.node(f.opp(p)[slot]).ty.shape.clone();
    let (fold, size) = match kind {
        OpKind::Mv | OpKind::Mm => (shape(0).get(1).copied().unwrap_or(0), f.node(p).numel()),
        OpKind::Dot | OpKind::Sum | OpKind::Norm if shape(0).len() == 1 => (shape(0)[0], 1),
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
    if matches!(kind, OpKind::Dot | OpKind::Sum | OpKind::Norm) {
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
    kind: OpKind,
    args: &[String],
) -> Result<Option<Out>, Refusal> {
    // a value placed in its plane: the host text has no planes, so the value is its own text and is
    // never named
    if kind == OpKind::Resident && f.placed {
        return super::placed::resident(f, p).map(Some);
    }
    if kind == OpKind::Resident {
        // not placed (a device text, or a host text without lane planes): the value is its own text
        f.record_read(f.opp(p)[0]);
        f.set_text(p, args[0].clone());
        return Ok(Some(Out::Bound));
    }
    if kind == OpKind::Sum && f.node(f.opp(p)[0]).rank() == 2 {
        return matrix_sum(f, p).map(Some);
    }
    if kind == OpKind::RawDevice {
        return raw_block(f, p, args).map(Some);
    }
    if matches!(
        kind,
        OpKind::Lstsq | OpKind::LstsqGramSolve | OpKind::LstsqInfo
    ) {
        return Err(nib(format!(
            "op {kind}: least squares is lowered by raptor.wings.base.lstsq before the seam"
        )));
    }
    if matches!(kind, OpKind::SetComponentAt | OpKind::AddAt) {
        return super::stores::storeplan::store(f, p).map(Some);
    }
    Ok(None)
}

/// Writes, in a banded unit, the fused add of a product, and `exp`, `sin` and `cos`: the raw twin when every reader takes a raw band, the
/// shared `sincos` of a sine and cosine of one operand at the top scope, else none (the plain math call).
fn banded_special(f: &mut Frame, p: usize, kind: OpKind, args: &[String]) -> Option<String> {
    if f.banded() && args.len() == 2 {
        if let Some(text) = literal_op(f, p, kind, args) {
            return Some(text);
        }
        if kind == OpKind::Add {
            return fused_add(f, p, args);
        }
    }
    if !f.banded() || args.len() != 1 {
        return None;
    }
    if matches!(kind, OpKind::Sin | OpKind::Cos) {
        if let Some((sin, cos)) = banded_sincos(f, p, kind, &args[0]) {
            return Some(if kind == OpKind::Sin { sin } else { cos });
        }
    }
    let float = matches!(f.node(p).ty.dtype.as_str(), "f64" | "f32");
    let users = f.users(p);
    let twin = float
        && !users.is_empty()
        && users
            .iter()
            .all(|&(u, _)| f.node(u).kind.is_some_and(spell::raw_reader));
    if twin {
        spell::banded_raw_twin(kind, &args[0])
    } else {
        None
    }
}

/// `x <op> c` with exactly one real literal operand, by its exact class (see `spell::literal_op`).
fn literal_op(f: &Frame, p: usize, kind: OpKind, args: &[String]) -> Option<String> {
    if !matches!(kind, OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div) {
        return None;
    }
    let ops = f.opp(p);
    match (real_value(f, ops[0]), real_value(f, ops[1])) {
        (Some(c), None) => spell::literal_op(kind, &args[1], c, true),
        (None, Some(c)) => spell::literal_op(kind, &args[0], c, false),
        _ => None,
    }
}

/// `a*b + c` (either order) as the fused intake `fma(a, b, c)`: a float add whose product has this add alone as
/// reader, is not named, and has no literal factor, and whose addend is a plane read or a normalised result (a door); else none
/// (the add is left to the raw chain, the plain operator).
fn fused_add(f: &mut Frame, p: usize, args: &[String]) -> Option<String> {
    if !matches!(f.node(p).ty.dtype.as_str(), "f64" | "f32") {
        return None;
    }
    let float =
        |q: usize| matches!(f.node(q).ty.dtype.as_str(), "f64" | "f32") && f.node(q).rank() == 0;
    for j in 0..2 {
        let (prod, addend) = (f.opp(p)[j], f.opp(p)[1 - j]);
        let operands = f.opp(prod).clone();
        let n = f.node(addend);
        let door = n.class == Class::Op && n.kind.is_some_and(spell::banded_door);
        if f.node(prod).kind != Some(OpKind::Mul)
            || operands.len() != 2
            || f.fanout(prod) != 1
            || f.is_declared(f.peek(prod))
            || operands.iter().any(|&o| f.node(o).class == Class::Const)
            || !(matches!(n.class, Class::Leaf | Class::At) || door)
            || !float(addend)
        {
            continue;
        }
        let (a, b) = (f.text(operands[0]).clone(), f.text(operands[1]).clone());
        return Some(spell::banded_fma(&a, &b, &args[1 - j]));
    }
    None
}

/// The names of the shared sine and cosine of the operand of the sine or cosine at `p`: bound the first time one of the
/// pair is reached; none when the node is not one of a pair at the top scope.
fn banded_sincos(f: &mut Frame, p: usize, kind: OpKind, arg: &str) -> Option<(String, String)> {
    let top = f.walk_count();
    let operand = f.opp(p)[0];
    let partner = if kind == OpKind::Sin {
        OpKind::Cos
    } else {
        OpKind::Sin
    };
    let paired = (0..top)
        .any(|q| f.node(q).kind == Some(partner) && f.opp(q)[0] == operand && f.is_outside(q));
    if p >= top || !f.is_outside(p) || !paired {
        return None;
    }
    if let Some((_, s, c)) = f.sincos.iter().find(|e| e.0 == operand) {
        return Some((s.clone(), c.clone()));
    }
    let k = f.sincos.len();
    let (sin, cos) = (format!("hawk_sin{k}"), format!("hawk_cos{k}"));
    let [bind, call] = spell::banded_sincos(&sin, &cos, arg);
    f.emit_bare(bind);
    f.emit(call);
    f.sincos.push((operand, sin.clone(), cos.clone()));
    Some((sin, cos))
}

/// Refuses, in a banded unit, the operations whose banded spelling is not written: an explicit fma.
fn refuse_banded_forms(f: &Frame, kind: OpKind) -> Result<(), Refusal> {
    if !f.banded() {
        return Ok(());
    }
    if kind == OpKind::Fma {
        return Err(nib("a banded fma (the fused form)"));
    }
    Ok(())
}

/// The text of a scalar operation (every operand of rank 0, or the index clamp, sign and cast of
/// one scalar); none when the
/// operation is a tensor one.
fn scalar_text(
    f: &mut Frame,
    p: usize,
    kind: OpKind,
    args: &[String],
    ranks: &[usize],
) -> Result<Option<String>, Refusal> {
    let rank = f.node(p).rank();
    let dtype = f.node(p).ty.dtype.clone();
    if kind == OpKind::ClampIndex && rank == 0 {
        return Ok(Some(clamp_index_text(f, p, args)));
    }
    if kind == OpKind::Sign && rank == 0 && ranks == [0] {
        return Ok(Some(sign_text(f, &args[0])));
    }
    if kind == OpKind::Cast && rank == 0 && ranks == [0] {
        let source = f.node(f.opp(p)[0]).ty.dtype.clone();
        let mut used = vec![];
        let text = spell::cast_op(&args[0], &dtype, &source, &mut used)?;
        f.hold_needs(p, &used);
        return Ok(Some(text));
    }
    refuse_banded_forms(f, kind)?;
    if rank == 0 && ranks.iter().all(|&r| r == 0) {
        if let Some(text) = banded_special(f, p, kind, args) {
            return Ok(Some(text));
        }
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
    kind: OpKind,
    args: &[String],
    ranks: &[usize],
) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    if linalg::is_factor_kind(kind) {
        return linalg::op(f, p, kind);
    }
    if matches!(kind, OpKind::RayTri | OpKind::RayAabb | OpKind::RaySphere) {
        return Ok(Out::Value(ray_call(f, p, kind, args, ranks)?, None));
    }
    if kind == OpKind::ComponentAt {
        return super::stores::rtstore::component_at(f, p);
    }
    // a placed value is an aether expression to the whole-value ops; a static element read reads it
    // by element
    if kind == OpKind::Component && args.first().is_some_and(|a| is_proxy(f, a)) {
        let index = component_index(f.ctx(), h)?;
        return Ok(Out::Value(
            format!("{}.at({})", args[0], join_index(&index)),
            None,
        ));
    }
    let args: Vec<String> = args
        .iter()
        .map(|a| {
            if is_proxy(f, a) && kind != OpKind::Component {
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
    kind: OpKind,
    args: &[String],
    ranks: &[usize],
) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let (dtype, shape) = (f.node(p).ty.dtype.clone(), f.node(p).ty.shape.clone());
    let component = if kind == OpKind::Component {
        component_index(f.ctx(), h)?
    } else {
        vec![]
    };
    let source = f.opp(p).first().copied();
    let lazy_source = if kind == OpKind::Component {
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
) -> Result<Vec<Option<Span>>, Refusal> {
    let int = |q: usize| -> Result<Option<i64>, Refusal> {
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
        if n.kind != Some(OpKind::ClampIndex) {
            continue;
        }
        let (index, bound) = (opp[p][0], opp[p][1]);
        let (mut base, mut offset) = (index, 0i64);
        if matches!(nodes[index].kind, Some(OpKind::Add | OpKind::Sub)) && opp[index].len() == 2 {
            let (a, b) = (opp[index][0], opp[index][1]);
            let add = nodes[index].kind == Some(OpKind::Add);
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
fn expression_method(kind: OpKind, args: &[String], ranks: &[usize]) -> Option<String> {
    let method0 = match kind {
        OpKind::QuatConj => "quatConj",
        OpKind::QuatRecip => "quatReciprocal",
        OpKind::AsPure => "asPureQuaternion",
        OpKind::AsVec3 => "asBack3DVector",
        _ => "",
    };
    let method1 = match kind {
        OpKind::QuatMul => "quatMul",
        OpKind::QuatRotate => "quatRotate",
        _ => "",
    };
    match (args.len(), kind) {
        (1, _) if !method0.is_empty() => Some(format!("{}.{method0}()", args[0])),
        (2, _) if !method1.is_empty() => Some(format!("{}.{method1}({})", args[0], args[1])),
        (2, OpKind::Div) if ranks[0] > 0 && ranks[1] > 0 => {
            Some(format!("aether::cwiseDiv({}, {})", args[0], args[1]))
        }
        _ => None,
    }
}

/// A product past the fold threshold, as its `aether::rt` twin written into a fresh item ahead of
/// the line that uses it.
/// A transposed operand is read through its own operand, so the transposed value is never built.
fn rolled_product(f: &mut Frame, p: usize, kind: OpKind) -> Result<Out, Refusal> {
    let out = linalg::fresh_item(f, p)?;
    let (left, right) = (
        product_operand(f, f.opp(p)[0])?,
        product_operand(f, f.opp(p)[1])?,
    );
    let twin = if kind == OpKind::Mv {
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
fn rolled_reduction(f: &mut Frame, p: usize, kind: OpKind) -> Result<Out, Refusal> {
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
fn matrix_sum(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
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
fn product_operand(f: &mut Frame, q: usize) -> Result<String, Refusal> {
    if f.node(q).kind == Some(OpKind::Transpose) {
        let inner = product_operand(f, f.opp(q)[0])?;
        f.need(Need::Rt);
        return Ok(format!("aether::rt::transposed({inner})"));
    }
    let arg = f.text(q).clone();
    linalg::operand(f, q, &arg)
}

/// A matrix built from more entries than the fold threshold: a fresh item its entries are stored
/// into, one statement each.
fn assembled(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
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
    if !OpKind::of(f.ctx(), h).is_ok_and(|kind| {
        matches!(
            kind,
            OpKind::Mv | OpKind::Mm | OpKind::Dot | OpKind::Sum | OpKind::Norm
        )
    }) {
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
    matches!(f.node(p).kind, Some(OpKind::Mv | OpKind::Mm)) && is_rolled_handle(f, f.node(p).h)
}

/// Whether a value is read only through the operand it hides: a transpose every reader of which is
/// a rolled product (which
/// reads through it), or the identity matrix a solve replaces by an inverse. No statement renders
/// such a value.
pub(super) fn is_absorbed(f: &Frame, p: usize) -> Result<bool, Refusal> {
    if f.node(p).class != Class::Op || f.users(p).is_empty() {
        return Ok(false);
    }
    let kind = f.op_kind(p)?;
    if kind == OpKind::Transpose {
        return Ok(f.users(p).len() == f.fanout(p)
            && f.users(p).iter().all(|&(u, _)| is_rolled_product(f, u)));
    }
    if kind == OpKind::Vec && linalg::is_identity(f, p)? {
        return Ok(f.users(p).len() == f.fanout(p)
            && f.users(p)
                .iter()
                .all(|&(u, slot)| slot == 1 && f.node(u).kind == Some(OpKind::LuSolve)));
    }
    Ok(false)
}

/// A ray primitive: the library returns its four values by value, held once, and the node is the
/// item of them. A vector operand
/// goes in as a per-sample operand the library reads by element; a rank-0 one as it is.
fn ray_call(
    f: &mut Frame,
    p: usize,
    kind: OpKind,
    args: &[String],
    ranks: &[usize],
) -> Result<String, Refusal> {
    let function = match kind {
        OpKind::RayTri => "rayTriangle",
        OpKind::RayAabb => "rayAabb",
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
fn raw_block(f: &mut Frame, p: usize, args: &[String]) -> Result<Out, Refusal> {
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
