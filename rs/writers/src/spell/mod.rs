//! The C++ expression spelling of the aether writer (rank-0 subset): literals, leaves, scalar
//! operations, stores. A construct
//! it does not spell yet is a refusal "not in this build: <construct>".
mod band;
use crate::kind::OpKind;
use crate::refuse::{nib, Refusal};
pub use band::{band_literal, literal_op};

/// The spelling role of a written plane the runtime leaves as it is: the IR role `plane_out` for a plane a store writes.
pub const KEPT_PLANE: &str = "plane_out";

/// The spelling role of a written plane the runtime zeroes before the run: an accumulate target, which names its prefix
/// apart from the planes a store writes.
pub const ZEROED_PLANE: &str = "plane_out+zeroed";

/// The spelling role of a written plane by whether the runtime zeroes it.
pub fn written_role(zeroed: bool) -> &'static str {
    if zeroed {
        ZEROED_PLANE
    } else {
        KEPT_PLANE
    }
}

/// The prefix of the names a role's slots are bound under, none for a role that has no binding. A plane read per sample is
/// named by the rank of its value: scalar, vector, matrix, or a general plane.
pub fn ident_prefix(role: &str, rank: usize) -> Option<&'static str> {
    Some(match role {
        "mutable" => "mut",
        "out" => "out",
        KEPT_PLANE => "wout",
        ZEROED_PLANE => "aout",
        "plane_in" => match rank {
            0 => "psc",
            1 => "vin",
            2 => "mtx",
            _ => "pin",
        },
        "wide_in" => "win",
        "lookup" => "lut",
        "terminated" => "trm",
        "uniform" => "uni",
        "nsamples" => "nsm",
        "extent" => "ext",
        _ => return None,
    })
}

/// The C++ name a slot of `role`, value rank `rank` and `name` is bound under.
pub fn binding_name(role: &str, rank: usize, name: &str) -> Result<String, Refusal> {
    ident_prefix(role, rank)
        .map(|p| format!("{p}_{name}"))
        .ok_or_else(|| nib(format!("slot role {role}")))
}

/// The C++ text of a real number: the shortest digits that read back as the value, positional for
/// decimal exponents in [-4, 16), else scientific with a signed two-digit exponent (`1e-05`,
/// `1e+16`); plus `.0` when no point or exponent shows (`100.0`, `-0.0`). Not a number and the
/// infinities are the macros `NAN`, `INFINITY` and `-INFINITY`. This is the one place a real is
/// printed.
pub fn real_literal(x: f64) -> String {
    if x.is_nan() {
        return "NAN".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-INFINITY" } else { "INFINITY" }.into();
    }
    let e = format!("{:e}", x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let text = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int = exp as usize + 1;
            let padded = format!("{digits:0<int$}");
            let (a, b) = padded.split_at(int);
            format!("{sign}{a}.{}", if b.is_empty() { "0" } else { b })
        } else {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    };
    if text.contains(['.', 'e', 'E']) {
        text
    } else {
        text + ".0"
    }
}

/// A rank-0 leaf read at the body's own sample.
pub fn leaf(role: &str, name: &str, rank: usize, dtype: &str) -> Result<String, Refusal> {
    let b = binding_name(role, rank, name)?;
    if rank > 0 {
        return Err(nib(format!("leaf of rank {rank} ({role})")));
    }
    if role == "uniform" && int_plane(dtype) {
        return Ok(format!("static_cast<std::{dtype}_t>({b})"));
    }
    Ok(match role {
        "uniform" => b,
        "nsamples" => format!("static_cast<Int>({b})"),
        "mutable" => format!("{b}[i].get()()"),
        _ => format!("{b}[i].eval()"),
    })
}

/// The index text of one axis under an edge rule, and the condition under which the access exists: `clamp` and `wrap` fold the
/// index into the axis, `zero` keeps it and states the range test (a read outside yields zero, a write outside is dropped), and
/// `unchecked` passes it on the caller's promise. `extent` is the text of the axis's extent.
fn edged(edge: &str, index: &str, extent: &str) -> Result<(String, Option<String>), Refusal> {
    Ok(match edge {
        "clamp" | "wrap" | "unchecked" => {
            (format!("aether::edge::{edge}({index}, {extent})"), None)
        }
        "zero" => (
            format!("static_cast<std::size_t>({index})"),
            Some(format!("aether::edge::inside({index}, {extent})")),
        ),
        other => return Err(nib(format!("the edge rule {other:?}"))),
    })
}

/// The element of a plane at one index per axis, under an edge rule, and the range test a `zero` edge adds. A plane with axes is
/// addressed through its view's call form, a plane without axes (one runtime axis) through the sample index.
pub fn element_at(
    binding: &str,
    indices: &[String],
    edge: &str,
    axes: bool,
) -> Result<(String, Option<String>), Refusal> {
    if !axes {
        let [index] = indices else {
            return Err(nib("an access to a plane without axes at several indices"));
        };
        let (position, test) = match edge {
            "unchecked" => (format!("static_cast<std::size_t>({index})"), None),
            _ => edged(edge, index, &format!("{binding}.samples()"))?,
        };
        return Ok((
            format!("{binding}[aether::SampleIndex::make({position})]"),
            test,
        ));
    }
    let (args, test) = axis_args(binding, 0, indices, edge)?;
    Ok((format!("{binding}({})", args.join(", ")), test))
}

/// The index text of every axis of a plane with axes under an edge rule, and the joined range test of a `zero` edge. The view
/// stores the components of a value of rank `value_rank` ahead of the axes, so axis `n` is the view's dimension `value_rank + n`.
fn axis_args(
    binding: &str,
    value_rank: usize,
    indices: &[String],
    edge: &str,
) -> Result<(Vec<String>, Option<String>), Refusal> {
    let mut args = vec![];
    let mut tests = vec![];
    for (axis, index) in indices.iter().enumerate() {
        let (arg, test) = edged(
            edge,
            index,
            &format!("{binding}.extent({})", value_rank + axis),
        )?;
        args.push(arg);
        tests.extend(test);
    }
    Ok((args, (!tests.is_empty()).then(|| tests.join(" && "))))
}

/// A read guarded by a `zero` edge's range test: the read inside, a zero outside; the read itself otherwise. A value of rank
/// zero has the element type's default as its zero; a value of rank above zero a constant of its extents, held as an item so
/// that both arms have one type.
fn guarded(
    read: String,
    test: Option<String>,
    dtype: &str,
    shape: &[usize],
) -> Result<String, Refusal> {
    let Some(test) = test else {
        return Ok(read);
    };
    if shape.is_empty() {
        return Ok(format!(
            "(({test}) ? {read} : std::decay_t<decltype({read})>{{}})"
        ));
    }
    let zero = format!(
        "{}(aether::constant<aether::extents<{}>>({}))",
        carry_type(dtype, shape)?,
        ext(shape),
        cast("0", dtype)?
    );
    Ok(format!("(({test}) ? {read} : {zero})"))
}

/// A plane read at the indices an `At` names, one per axis of the plane, under its edge rule; the value read has `shape` and
/// `dtype`. A plane with axes is read through its view's call form (a value of rank above zero is read component by component
/// into an item), a plane without axes (one runtime axis) through the sample index.
pub fn at(
    role: &str,
    name: &str,
    indices: &[String],
    edge: &str,
    value: (&str, &[usize]),
    axes: bool,
) -> Result<String, Refusal> {
    let (dtype, shape) = value;
    let binding = binding_name(role, shape.len(), name)?;
    if axes && !shape.is_empty() {
        let (args, test) = axis_args(&binding, shape.len(), indices, edge)?;
        let read = format!(
            "hawk_abi::view_item<{}, {}>({binding}, {})",
            elem(dtype)?,
            ext(shape),
            args.join(", ")
        );
        return guarded(read, test, dtype, shape);
    }
    let (element, test) = element_at(&binding, indices, edge, axes)?;
    let read = match (axes, shape.len()) {
        (true, _) => element,
        (false, 0) => format!("{element}.eval()"),
        (false, _) => format!("{element}.get()"),
    };
    guarded(read, test, dtype, shape)
}

/// `cast`: a plain static_cast between Real, Int and bool.
/// A helper function the spelled text calls, which the unit around it must define.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Helper {
    /// An integer helper (`hawk_abi::int_..`).
    Integers,
    /// A random draw (`aether::random::..`).
    Random,
    /// The spacing helper.
    Spacing,
    /// The scalar blend helper.
    Blend,
    /// The item select helper.
    SelectItem,
}

/// The cast of `arg` from `source` to `target`; the helpers its text calls are pushed on `used`.
pub fn cast_op(
    arg: &str,
    target: &str,
    source: &str,
    used: &mut Vec<Helper>,
) -> Result<String, Refusal> {
    let (e, float_src) = (elem(target)?, matches!(source, "f64" | "f32"));
    elem(source)?;
    Ok(if int_plane(target) && float_src {
        used.push(Helper::Integers);
        format!("hawk_abi::int_from_real<{e}>({arg})")
    } else {
        format!("static_cast<{e}>({arg})")
    })
}

/// The name of the cell a by-value parameter of `role` is read from.
pub fn param_cell(role: &str, name: &str) -> Result<String, Refusal> {
    Ok(format!("{}_cell", binding_name(role, 0, name)?))
}

/// The value a constant node holds, as far as a constant can be spelled from it.
pub enum Held<'a> {
    /// An integer.
    Int(i64),
    /// A real number.
    Real(f64),
    /// A flag.
    Flag(bool),
    /// A value no constant is spelled from, named by its kind (`nothing`, `text`, a tuple).
    Other(&'a str),
}

/// The C++ text of a constant of the given element type and rank, spelled from the value it holds.
/// A value the element type cannot hold (a real for an integer type, an integer other than 0 or 1
/// or real other than 0 and 1 for a flag, a flag for a number) is not in this build.
pub fn constant(dtype: &str, rank: usize, held: &Held) -> Result<String, Refusal> {
    if rank > 0 {
        return Err(nib("literal of rank > 0"));
    }
    let refused = || {
        let kind = match held {
            Held::Int(_) => "an integer".to_string(),
            Held::Real(_) => "a real".to_string(),
            Held::Flag(_) => "a flag".to_string(),
            Held::Other(kind) => kind.to_string(),
        };
        nib(format!("literal of {kind} for dtype {dtype}"))
    };
    match (dtype, held) {
        ("f64" | "f32", Held::Real(v)) => Ok(format!("static_cast<Real>({})", real_literal(*v))),
        ("f64" | "f32", Held::Int(v)) => {
            Ok(format!("static_cast<Real>({})", real_literal(*v as f64)))
        }
        ("i32" | "i64", Held::Int(v)) => Ok(format!("static_cast<Int>({v})")),
        (d, Held::Int(v)) if int_plane(d) => Ok(format!(
            "static_cast<{}>({v}LL)",
            elem(d).map_err(|_| refused())?
        )),
        ("bool", Held::Flag(v)) => Ok(v.to_string()),
        ("bool", Held::Int(v @ (0 | 1))) => Ok((*v == 1).to_string()),
        ("bool", Held::Real(v)) if *v == 0.0 || *v == 1.0 => Ok((*v == 1.0).to_string()),
        _ => Err(refused()),
    }
}

/// The sample index as the device entry declares it (a `long long`).
pub const SAMPLE_INDEX_IDENT: &str = "hawk_i";
/// The sample index as an `Int` expression.
pub const SAMPLE_INDEX: &str = "static_cast<Int>(hawk_i)";

/// The lane's place in its warp: the team prologue binds it.
pub const WARP_LANE: &str = "hawk_wl";

/// A select. `host` is the blend setting with both arms free to evaluate: a blend at rank 0, a held
/// Item above; otherwise a
/// ternary at rank 0 and aether::select above.
/// The select of `args`; the helpers its text calls are pushed on `used`.
pub fn select(
    used: &mut Vec<Helper>,
    host: bool,
    cond_rank: usize,
    dtype: &str,
    shape: &[usize],
    args: &[String],
    hoist: &mut dyn FnMut(String) -> String,
) -> Result<String, Refusal> {
    let (mut c, a, b) = (args[0].clone(), &args[1], &args[2]);
    if shape.is_empty() {
        return Ok(if host && (dtype == "f64" || dtype == "f32") {
            used.push(Helper::Blend);
            format!("hawk_abi::blend({c}, {a}, {b})")
        } else {
            format!("({c} ? {a} : {b})")
        });
    }
    if cond_rank == 0 && host {
        used.push(Helper::SelectItem);
        return Ok(hoist(format!(
            "hawk_abi::select_item<{}, {}>({c}, {a}, {b})",
            element(dtype)?,
            ext(shape)
        )));
    }
    if cond_rank == 0 {
        c = hoist(format!("aether::constant<aether::extents<>>({c})"))
    }
    Ok(format!("aether::select({c}, {a}, {b})"))
}

/// The names of the unary math functions the emitted code calls through the library.
const MATH_UNARY: [OpKind; 33] = {
    use OpKind::*;
    [
        Abs, Sqrt, Rsqrt, Exp, Log, Sin, Cos, Tan, Tanh, Asin, Acos, Atan, Floor, Exp2, Expm1,
        Log2, Log10, Log1p, Cbrt, Sinh, Cosh, Asinh, Acosh, Atanh, Ceil, Trunc, Round, Rint, Erf,
        Erfc, Isnan, Isinf, Isfinite,
    ]
};
/// The binary math functions, as (operation, library function).
const MATH_BINARY: [(OpKind, &str); 9] = [
    (OpKind::Pow, "pow"),
    (OpKind::Min, "fmin"),
    (OpKind::Max, "fmax"),
    (OpKind::Atan2, "atan2"),
    (OpKind::Hypot, "hypot"),
    (OpKind::Copysign, "copysign"),
    (OpKind::Fmod, "fmod"),
    (OpKind::Remainder, "remainder"),
    (OpKind::Fdim, "fdim"),
];
/// The infix operators, as (operation, C++ operator).
const INFIX: [(OpKind, &str); 12] = [
    (OpKind::Add, "+"),
    (OpKind::Sub, "-"),
    (OpKind::Mul, "*"),
    (OpKind::Div, "/"),
    (OpKind::Lt, "<"),
    (OpKind::Le, "<="),
    (OpKind::Gt, ">"),
    (OpKind::Ge, ">="),
    (OpKind::Eq, "=="),
    (OpKind::Ne, "!="),
    (OpKind::Land, "&&"),
    (OpKind::Lor, "||"),
];

/// The operations a banded exp, sin or cos may leave raw for: the readers that take a raw band.
pub fn raw_reader(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Neg | OpKind::Div
    )
}

/// Whether the reader `kind` takes an unnormalised band operand (a chain operator or a door that normalises at entry).
pub fn reads_raw(kind: OpKind) -> bool {
    raw_reader(kind)
        || matches!(
            kind,
            OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Cos | OpKind::Sqrt
        )
}

/// The raw-exit twin of a banded exp, sin or cos of `arg`; none for another operation.
pub fn banded_raw_twin(kind: OpKind, arg: &str) -> Option<String> {
    let twin = match kind {
        OpKind::Exp => "expRaw",
        OpKind::Sin => "sinRaw",
        OpKind::Cos => "cosRaw",
        _ => return None,
    };
    Some(format!(
        "aether::banded::detail::{twin}(aether::banded::Band({arg}))"
    ))
}

/// Whether a banded operation's result is a normalised `Band` (a facade door or a division): the math functions and `div`.
pub fn banded_door(kind: OpKind) -> bool {
    kind == OpKind::Div
        || MATH_UNARY.contains(&kind)
        || MATH_BINARY.iter().any(|m| m.0 == kind)
        || matches!(kind, OpKind::Fma | OpKind::Clip)
}

/// The fused multiply-add of a product `a * b` and a normalised addend `c`.
pub fn banded_fma(a: &str, b: &str, c: &str) -> String {
    format!("aether::banded::detail::fma({a}, {b}, aether::banded::Band({c}))")
}

/// The line that binds a banded sine and cosine of one operand, and the line that computes them.
pub fn banded_sincos(sin: &str, cos: &str, arg: &str) -> [String; 2] {
    [
        format!("aether::banded::Band {sin}, {cos};"),
        format!("aether::math::sincos(aether::banded::Band({arg}), &{sin}, &{cos});"),
    ]
}

/// A rank-0 floating-point operation over the operand texts.
/// One element of an operation over scalar texts; the helpers its text calls are pushed on `used`.
pub fn scalar_op(
    kind: OpKind,
    args: &[String],
    dtype: &str,
    used: &mut Vec<Helper>,
) -> Result<String, Refusal> {
    let int_call = [
        (OpKind::Floordiv, "int_floordiv"),
        (OpKind::Mod, "int_mod"),
        (OpKind::Band, "int_and"),
        (OpKind::Bor, "int_or"),
        (OpKind::Bxor, "int_xor"),
        (OpKind::Shl, "int_shl"),
        (OpKind::Shr, "int_shr"),
        (OpKind::Invert, "int_invert"),
    ];
    let int_wrap = [
        (OpKind::Add, "int_add"),
        (OpKind::Sub, "int_sub"),
        (OpKind::Mul, "int_mul"),
        (OpKind::Neg, "int_neg"),
        (OpKind::Abs, "int_abs"),
        (OpKind::Min, "int_min"),
        (OpKind::Max, "int_max"),
    ];
    if let Some((_, f)) = int_call.iter().find(|m| m.0 == kind) {
        used.push(Helper::Integers);
        return Ok(format!("hawk_abi::{f}({})", args.join(", ")));
    }
    if int_plane(dtype) {
        if let Some((_, f)) = int_wrap.iter().find(|m| m.0 == kind) {
            used.push(Helper::Integers);
            return Ok(format!("hawk_abi::{f}({})", args.join(", ")));
        }
    }
    if matches!(kind, OpKind::Min | OpKind::Max) && (dtype == "i32" || dtype == "i64") {
        used.push(Helper::Integers);
        return Ok(format!("hawk_abi::int_{kind}<Int>({})", args.join(", ")));
    }
    // HAWK's (seed, counter) draw: aether's stateless detail::uniform01 / standardNormal under the
    // seed XOR-ed with a per-kind salt
    if args.len() == 2 && matches!(kind, OpKind::RandomUniform | OpKind::RandomNormal) {
        let (f, salt) = if kind == OpKind::RandomUniform {
            ("uniform01", "0x243F6A8885A308D3ULL")
        } else {
            ("standardNormal", "0x13198A2E03707344ULL")
        };
        used.push(Helper::Random);
        return Ok(format!("aether::random::detail::{f}<Real>(static_cast<std::uint64_t>({}) ^ {salt}, static_cast<aether::offset_t>(hawk_i), static_cast<std::uint64_t>({}))", args[0], args[1]));
    }
    if kind == OpKind::Spacing && args.len() == 1 {
        used.push(Helper::Spacing);
        return Ok(format!("hawk_abi::spacing({})", args[0]));
    }
    if matches!(
        kind,
        OpKind::Dot
            | OpKind::Norm
            | OpKind::Sum
            | OpKind::Component
            | OpKind::Sign
            | OpKind::RandomUniform
            | OpKind::RandomNormal
    ) {
        return Err(nib(format!("scalar op {kind}")));
    }
    if MATH_UNARY.contains(&kind) && args.len() == 1 {
        return Ok(format!("aether::math::{kind}({})", args[0]));
    }
    if let Some((_, f)) = MATH_BINARY.iter().find(|m| m.0 == kind) {
        if args.len() == 2 {
            return Ok(format!("aether::math::{f}({}, {})", args[0], args[1]));
        }
    }
    if let Some((_, s)) = INFIX.iter().find(|m| m.0 == kind) {
        if args.len() == 2 {
            return Ok(format!("({} {s} {})", args[0], args[1]));
        }
    }
    match (kind, args.len()) {
        (OpKind::Spacing, 1) => {
            used.push(Helper::Spacing);
            Ok(format!("hawk_abi::spacing({})", args[0]))
        }
        (OpKind::Neg, 1) => Ok(format!("(-{})", args[0])),
        (OpKind::Lnot, 1) => Ok(format!("(!{})", args[0])),
        (OpKind::Fma, 3) | (OpKind::Clip, 3) => {
            Ok(format!("aether::math::{kind}({})", args.join(", ")))
        }
        _ => Err(nib(format!("op {kind}"))),
    }
}

/// The C++ element type of a plane of `dtype`, as a fixed text.
fn element(dtype: &str) -> Result<&'static str, Refusal> {
    match dtype {
        "f64" | "f32" => Ok("Real"),
        "i32" | "i64" => Ok("Int"),
        "bool" => Ok("bool"),
        d => Err(nib(format!("element type {d}"))),
    }
}
/// INT_PLANE_DTYPES (raptor ir core.json consts): the integer plane dtypes, spelled std::<dtype>_t.
pub fn int_plane(dtype: &str) -> bool {
    matches!(dtype, "int32" | "uint32")
}
/// The C++ element type of `dtype`.
fn elem(dtype: &str) -> Result<String, Refusal> {
    if int_plane(dtype) {
        Ok(format!("std::{dtype}_t"))
    } else {
        element(dtype).map(String::from)
    }
}
/// A cast of `text` to the element type of `dtype`.
fn cast(text: &str, dtype: &str) -> Result<String, Refusal> {
    let e = element(dtype)?;
    Ok(if text.starts_with(&format!("static_cast<{e}>")) {
        text.to_string()
    } else {
        format!("static_cast<{e}>({text})")
    })
}
/// The extents of a shape as a C++ list.
fn ext(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The type of a banded float scalar.
pub const BAND_TYPE: &str = "aether::banded::Band";

/// The scalar mode a local is spelled for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// A plain float: the type of a local is inferred.
    Plain,
    /// A banded float: a float scalar local is a band.
    Banded,
}

/// The C++ type `declare` gives a local: `auto`, the `aether::Item` of a held vector, or the band
/// of a float scalar in the banded mode.
pub fn declared_type(
    dtype: &str,
    shape: &[usize],
    held: bool,
    mode: Mode,
) -> Result<String, Refusal> {
    if held && shape.len() == 1 {
        return Ok(format!("aether::Item<{}, {}>", element(dtype)?, shape[0]));
    }
    if mode == Mode::Banded && shape.is_empty() && matches!(dtype, "f64" | "f32") {
        return Ok(BAND_TYPE.to_string());
    }
    Ok("auto".to_string())
}

/// A declaration of a named local; a vector held by more than one reader is an `aether::Item`,
/// evaluated once.
pub fn declare(
    name: &str,
    dtype: &str,
    shape: &[usize],
    text: &str,
    held: bool,
    mode: Mode,
) -> Result<String, Refusal> {
    if held && shape.len() == 1 {
        return Ok(format!(
            "const aether::Item<{}, {}> {name} = {text};",
            element(dtype)?,
            shape[0]
        ));
    }
    let cty = declared_type(dtype, shape, held, mode)?;
    Ok(format!("const {cty} {name} = {text};"))
}

/// A leaf of rank >= 1 read whole (a row of the sample): materialised as an Item.
pub fn row_leaf(role: &str, rank: usize, name: &str) -> Result<String, Refusal> {
    let b = binding_name(role, rank, name)?;
    match role {
        "plane_in" | "lookup" | "mutable" => Ok(format!("{b}[i].get()")),
        _ => Err(nib(format!("leaf role {role} of rank > 0"))),
    }
}

/// What the spelling of an op over tensors needs: the op, its operand texts and ranks, the result's
/// type and the
/// proxy source of a component read.
pub struct TensorOp<'a> {
    /// The op's kind.
    pub kind: OpKind,
    /// The operands, already spelled.
    pub args: &'a [String],
    /// The rank of each operand.
    pub ranks: &'a [usize],
    /// The element type of the result.
    pub dtype: &'a str,
    /// The extents of the result.
    pub shape: &'a [usize],
    /// The static index of a component read; empty for any other op.
    pub component: &'a [i64],
    /// For a component read of a proxy source: whether the source is a mutable plane.
    pub lazy_source: Option<bool>,
}

/// Binds a rank-0 operand to a tensor of the result's shape, ahead of the current line.
fn broadcast(
    op: &TensorOp,
    hoist: &mut dyn FnMut(String) -> String,
    scalar: &str,
) -> Result<String, Refusal> {
    Ok(hoist(format!(
        "aether::constant<aether::extents<{}>>({})",
        ext(op.shape),
        cast(scalar, op.dtype)?
    )))
}

/// An op with an operand or a result of rank >= 1. `hoist` binds a text to a fresh local ahead of
/// the current line.
pub fn tensor_op(
    op: &TensorOp,
    hoist: &mut dyn FnMut(String) -> String,
) -> Result<String, Refusal> {
    structure_op(op)
        .or_else(|| arithmetic_op(op, hoist))
        .or_else(|| unary_op(op))
        .or_else(|| product_op(op))
        .unwrap_or_else(|| Err(nib(format!("op {} of rank > 0", op.kind))))
}

/// Component reads, splats and the construction of a vector; none for another op.
fn structure_op(op: &TensorOp) -> Option<Result<String, Refusal>> {
    let (args, shape, dtype) = (op.args, op.shape, op.dtype);
    Some(match op.kind {
        OpKind::Component => {
            let ks: Vec<String> = op.component.iter().map(i64::to_string).collect();
            // a proxy source is read in place: `.at` on a mutable plane, `.eval` on a read-only one
            Ok(match op.lazy_source {
                Some(true) => format!("{}.at({})", args[0], ks.join(", ")),
                Some(false) => format!("{}.eval<{}>()", args[0], ks.join(", ")),
                None => format!("{}.eval<{}>(i)", args[0], ks.join(", ")),
            })
        }
        OpKind::Splat => cast(&args[0], dtype)
            .map(|c| format!("aether::constant<aether::extents<{}>>({c})", ext(shape))),
        OpKind::Vec if !args.is_empty() => element(dtype)
            .map(|e| format!("aether::Item<{e}, {}>{{{}}}", ext(shape), args.join(", "))),
        _ => return None,
    })
}

/// Sums, differences, products, quotients, minima, maxima and powers of two operands; none for
/// another op.
fn arithmetic_op(
    op: &TensorOp,
    hoist: &mut dyn FnMut(String) -> String,
) -> Option<Result<String, Refusal>> {
    let (kind, args, ranks) = (op.kind, op.args, op.ranks);
    if args.len() != 2 {
        return None;
    }
    Some(match kind {
        OpKind::Add | OpKind::Sub => (|| {
            let sym = if kind == OpKind::Add { "+" } else { "-" };
            let l = if ranks[0] == 0 {
                broadcast(op, hoist, &args[0])?
            } else {
                args[0].clone()
            };
            let r = if ranks[1] == 0 {
                broadcast(op, hoist, &args[1])?
            } else {
                args[1].clone()
            };
            Ok(format!("({l} {sym} {r})"))
        })(),
        OpKind::Mul | OpKind::Div => product_or_quotient(op, hoist),
        OpKind::Min | OpKind::Max | OpKind::Pow => (|| {
            let f = match kind {
                OpKind::Min => "cwiseMin",
                OpKind::Max => "cwiseMax",
                _ => "cwisePow",
            };
            let l = if ranks[0] > 0 {
                args[0].clone()
            } else {
                broadcast(op, hoist, &args[0])?
            };
            Ok(format!("aether::{f}({l}, {})", args[1]))
        })(),
        _ => return None,
    })
}

/// A product or a quotient of two operands, by the ranks of the two.
fn product_or_quotient(
    op: &TensorOp,
    hoist: &mut dyn FnMut(String) -> String,
) -> Result<String, Refusal> {
    let (kind, args, ranks) = (op.kind, op.args, op.ranks);
    let sym = if kind == OpKind::Mul { "*" } else { "/" };
    if ranks[0] == 0 && kind == OpKind::Mul {
        return Ok(format!("({} * {})", cast(&args[0], op.dtype)?, args[1]));
    }
    if ranks[1] == 0 {
        return Ok(format!("({} {sym} {})", args[0], cast(&args[1], op.dtype)?));
    }
    if kind == OpKind::Mul && ranks[0] == ranks[1] {
        return Ok(format!("{}.cwiseMul({})", args[0], args[1]));
    }
    if kind == OpKind::Div {
        let left = if ranks[0] > 0 {
            args[0].clone()
        } else {
            broadcast(op, hoist, &args[0])?
        };
        return Ok(format!("aether::cwiseDiv({left}, {})", args[1]));
    }
    Err(nib("component-wise product of tensors of other ranks"))
}

/// Negation, absolute value and the element-wise math functions; none for another op.
fn unary_op(op: &TensorOp) -> Option<Result<String, Refusal>> {
    let (kind, args) = (op.kind, op.args);
    Some(match kind {
        OpKind::Neg => Ok(format!("({} * static_cast<Real>(-1))", args[0])),
        OpKind::Abs => Ok(format!("aether::cwiseAbs({})", args[0])),
        OpKind::Sqrt
        | OpKind::Rsqrt
        | OpKind::Exp
        | OpKind::Log
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Tan
        | OpKind::Tanh => {
            let name = kind.name();
            let mut f = name.to_string();
            f.replace_range(..1, &name[..1].to_uppercase());
            Ok(format!("aether::cwise{f}({})", args[0]))
        }
        _ => return None,
    })
}

/// Norms, sums, transposes and the matrix and vector products; none for another op.
fn product_op(op: &TensorOp) -> Option<Result<String, Refusal>> {
    let (kind, args, ranks) = (op.kind, op.args, op.ranks);
    let one = args.len() == 1;
    let two = args.len() == 2;
    Some(Ok(match kind {
        OpKind::Norm | OpKind::Sum | OpKind::Transpose
            if one && !(kind == OpKind::Sum && ranks[0] == 2) =>
        {
            format!("{}.{kind}()", args[0])
        }
        OpKind::Mv | OpKind::Mm if two => format!("({} * {})", args[0], args[1]),
        OpKind::Outer if two => format!("aether::outer({}, {})", args[0], args[1]),
        OpKind::Dot | OpKind::Cross if two => format!("{}.{kind}({})", args[0], args[1]),
        _ => return None,
    }))
}

/// An AxisExtent node: the extent of one axis of the plane's view; a plane without axes has one axis, its samples.
pub fn axis_extent(
    role: &str,
    rank: usize,
    name: &str,
    axis: i64,
    axes: bool,
) -> Result<String, Refusal> {
    let binding = binding_name(role, rank, name)?;
    match (axes, axis) {
        (true, _) => Ok(format!(
            "static_cast<Int>({binding}.extent({}))",
            rank as i64 + axis
        )),
        (false, 0) => Ok(format!("static_cast<Int>({binding}.samples())")),
        (false, _) => Err(nib("an axis beyond the first of a plane without axes")),
    }
}

/// The C++ type one loop-carried local is declared as: a plain scalar at rank 0, an Item above.
pub fn carry_type(dtype: &str, shape: &[usize]) -> Result<String, Refusal> {
    let e = elem(dtype)?;
    Ok(if shape.is_empty() {
        e
    } else {
        format!("aether::Item<{e}, {}>", ext(shape))
    })
}

/// A loop name made safe to paste into C++ (spelling._ident).
pub fn ident(name: &str) -> String {
    let t: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if t.is_empty() {
        "x".into()
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_exp_sin_cos_spell_raw_twins_and_a_shared_pair() {
        assert_eq!(
            banded_raw_twin(OpKind::Exp, "x").unwrap(),
            "aether::banded::detail::expRaw(aether::banded::Band(x))"
        );
        assert!(banded_raw_twin(OpKind::Add, "x").is_none());
        assert_eq!(
            banded_fma("a", "b", "c"),
            "aether::banded::detail::fma(a, b, aether::banded::Band(c))"
        );
        assert!(banded_door(OpKind::Div) && banded_door(OpKind::Sqrt) && !banded_door(OpKind::Add));
        assert!(raw_reader(OpKind::Neg) && !raw_reader(OpKind::Exp));
        let [bind, call] = banded_sincos("hawk_sin0", "hawk_cos0", "x");
        assert_eq!(bind, "aether::banded::Band hawk_sin0, hawk_cos0;");
        assert_eq!(
            call,
            "aether::math::sincos(aether::banded::Band(x), &hawk_sin0, &hawk_cos0);"
        );
    }

    #[test]
    fn reals_print_in_the_shortest_round_trip_form() {
        for (value, text) in [
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (100.0, "100.0"),
            (2.0, "2.0"),
            (0.5, "0.5"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (-2.5e-7, "-2.5e-07"),
            (0.1 + 0.2, "0.30000000000000004"),
            (f64::NAN, "NAN"),
            (f64::INFINITY, "INFINITY"),
            (f64::NEG_INFINITY, "-INFINITY"),
        ] {
            assert_eq!(real_literal(value), text, "{value:e}");
        }
    }

    #[test]
    fn a_constant_is_spelled_from_the_value_it_holds() {
        let spell = |dtype: &str, held: Held| constant(dtype, 0, &held);
        assert_eq!(
            spell("f64", Held::Real(1e-5)).unwrap(),
            "static_cast<Real>(1e-05)"
        );
        assert_eq!(
            spell("f32", Held::Int(3)).unwrap(),
            "static_cast<Real>(3.0)"
        );
        assert_eq!(spell("i64", Held::Int(-7)).unwrap(), "static_cast<Int>(-7)");
        assert_eq!(
            spell("int32", Held::Int(4)).unwrap(),
            "static_cast<std::int32_t>(4LL)"
        );
        assert_eq!(spell("bool", Held::Flag(true)).unwrap(), "true");
        assert_eq!(spell("bool", Held::Flag(false)).unwrap(), "false");
        assert_eq!(spell("bool", Held::Int(1)).unwrap(), "true");
        assert_eq!(spell("bool", Held::Int(0)).unwrap(), "false");
        assert_eq!(spell("bool", Held::Real(0.0)).unwrap(), "false");
        assert_eq!(spell("bool", Held::Real(1.0)).unwrap(), "true");
    }

    #[test]
    fn a_value_the_type_cannot_hold_is_refused_by_name() {
        let refused = |dtype: &str, held: Held| constant(dtype, 0, &held).unwrap_err();
        for (dtype, held) in [
            ("i64", Held::Real(100.0)),
            ("i32", Held::Real(2.5)),
            ("int32", Held::Flag(true)),
            ("bool", Held::Int(2)),
            ("bool", Held::Real(0.5)),
            ("bool", Held::Other("nothing")),
            ("f64", Held::Flag(true)),
            ("f64", Held::Other("text")),
        ] {
            let message = refused(dtype, held);
            assert!(
                message.reason.starts_with("not in this build: "),
                "{message}"
            );
        }
        assert!(constant("f64", 1, &Held::Real(1.0))
            .unwrap_err()
            .reason
            .starts_with("not in this build: "));
    }

    /// Oracle: the rule of each edge stated in words, spelled by hand for a table of two axes read at `(i, k)`.
    #[test]
    fn a_read_at_two_indices_is_the_views_call_form_under_each_edge_rule() {
        let at = |edge: &str| {
            at(
                "lookup",
                "p",
                &["i".into(), "k".into()],
                edge,
                ("f64", &[]),
                true,
            )
            .unwrap()
        };
        assert_eq!(
            at("clamp"),
            "lut_p(aether::edge::clamp(i, lut_p.extent(0)), aether::edge::clamp(k, lut_p.extent(1)))"
        );
        assert_eq!(
            at("wrap"),
            "lut_p(aether::edge::wrap(i, lut_p.extent(0)), aether::edge::wrap(k, lut_p.extent(1)))"
        );
        assert_eq!(
            at("unchecked"),
            "lut_p(aether::edge::unchecked(i, lut_p.extent(0)), aether::edge::unchecked(k, lut_p.extent(1)))"
        );
        let zero = at("zero");
        assert!(zero.starts_with("((aether::edge::inside(i, lut_p.extent(0)) && aether::edge::inside(k, lut_p.extent(1))) ? lut_p("));
        assert!(zero.ends_with("{})"));
        assert!(at_refused("diagonal"));
    }

    /// Whether a read under an unknown edge rule is refused.
    fn at_refused(edge: &str) -> bool {
        at("lookup", "p", &["i".into()], edge, ("f64", &[]), true).is_err()
    }

    /// Oracle: the spelling stated by hand: a value of rank above zero out of a plane with axes is read into an item component
    /// by component, and a zero edge yields a constant of the extents held as the same item type.
    #[test]
    fn a_value_of_rank_above_zero_is_read_into_an_item_and_a_zero_edge_gives_a_constant_of_its_extents(
    ) {
        let shape: &[usize] = &[3];
        let vector = |edge: &str, axes: bool, indices: &[String]| {
            at("lookup", "p", indices, edge, ("f64", shape), axes).unwrap()
        };
        let two = ["i".to_string(), "k".to_string()];
        assert_eq!(
            vector("clamp", true, &two),
            "hawk_abi::view_item<Real, 3>(lut_p, aether::edge::clamp(i, lut_p.extent(1)), aether::edge::clamp(k, lut_p.extent(2)))"
        );
        let zero = vector("zero", true, &two);
        assert!(zero.ends_with(": aether::Item<Real, 3>(aether::constant<aether::extents<3>>(static_cast<Real>(0))))"), "{zero}");
        let flat = vector("zero", false, &["j".to_string()]);
        assert!(
            flat.contains(".get() : aether::Item<Real, 3>(aether::constant"),
            "{flat}"
        );
    }

    /// Oracle: a plane without axes keeps the sample-index text, and `unchecked` adds no helper to it.
    #[test]
    fn a_read_of_a_plane_without_axes_keeps_the_sample_index_text_when_unchecked() {
        assert_eq!(
            at(
                "lookup",
                "p",
                &["j".into()],
                "unchecked",
                ("f64", &[]),
                false
            )
            .unwrap(),
            "lut_p[aether::SampleIndex::make(static_cast<std::size_t>(j))].eval()"
        );
        assert!(at(
            "lookup",
            "p",
            &["j".into(), "k".into()],
            "clamp",
            ("f64", &[]),
            false
        )
        .is_err());
    }

    /// Oracle: the axis extent of a plane with axes asks the view for that axis; a plane without axes has its samples as axis 0.
    #[test]
    fn an_axis_extent_asks_the_view_for_the_axis() {
        assert_eq!(
            axis_extent("lookup", 0, "p", 1, true).unwrap(),
            "static_cast<Int>(lut_p.extent(1))"
        );
        assert_eq!(
            axis_extent("lookup", 0, "p", 0, false).unwrap(),
            "static_cast<Int>(lut_p.samples())"
        );
        assert!(axis_extent("lookup", 0, "p", 1, false).is_err());
    }
}
