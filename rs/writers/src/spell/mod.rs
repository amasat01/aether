//! The C++ expression spelling of the aether writer (rank-0 subset): literals, leaves, scalar
//! operations, stores. A construct
//! it does not spell yet is a refusal "not in this build: <construct>".
use crate::refuse::{nib, Refusal};

/// The prefix of the names a role's slots are bound under, none for a role that has no binding.
pub fn ident_prefix(role: &str) -> Option<&'static str> {
    Some(match role {
        "mutable" => "mut",
        "out" => "out",
        "wide_out" => "wout",
        "accum_out" => "aout",
        "vec_in" => "vin",
        "mat_in" => "mtx",
        "wide_in" => "win",
        "per_sample" => "psc",
        "lookup" => "lut",
        "terminated" => "trm",
        "uniform" => "uni",
        "nsamples" => "nsm",
        _ => return None,
    })
}

/// The C++ name a slot of `role` and `name` is bound under.
pub fn binding_name(role: &str, name: &str) -> Result<String, Refusal> {
    ident_prefix(role)
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
    let b = binding_name(role, name)?;
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

/// A plane read at a computed index (`index` is the rendered index operand).
pub fn at(role: &str, name: &str, index: &str, rank: usize) -> Result<String, Refusal> {
    Ok(format!(
        "{}[aether::SampleIndex::make(static_cast<std::size_t>({index}))]{}",
        binding_name(role, name)?,
        if rank > 0 { ".get()" } else { ".eval()" }
    ))
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
    Ok(format!("{}_cell", binding_name(role, name)?))
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
const MATH_UNARY: [&str; 33] = [
    "abs", "sqrt", "rsqrt", "exp", "log", "sin", "cos", "tan", "tanh", "asin", "acos", "atan",
    "floor", "exp2", "expm1", "log2", "log10", "log1p", "cbrt", "sinh", "cosh", "asinh", "acosh",
    "atanh", "ceil", "trunc", "round", "rint", "erf", "erfc", "isnan", "isinf", "isfinite",
];
/// The binary math functions, as (operation, library function).
const MATH_BINARY: [(&str, &str); 9] = [
    ("pow", "pow"),
    ("min", "fmin"),
    ("max", "fmax"),
    ("atan2", "atan2"),
    ("hypot", "hypot"),
    ("copysign", "copysign"),
    ("fmod", "fmod"),
    ("remainder", "remainder"),
    ("fdim", "fdim"),
];
/// The infix operators, as (operation, C++ operator).
const INFIX: [(&str, &str); 12] = [
    ("add", "+"),
    ("sub", "-"),
    ("mul", "*"),
    ("div", "/"),
    ("lt", "<"),
    ("le", "<="),
    ("gt", ">"),
    ("ge", ">="),
    ("eq", "=="),
    ("ne", "!="),
    ("land", "&&"),
    ("lor", "||"),
];

/// A rank-0 floating-point operation over the operand texts.
/// One element of an operation over scalar texts; the helpers its text calls are pushed on `used`.
pub fn scalar_op(
    kind: &str,
    args: &[String],
    dtype: &str,
    used: &mut Vec<Helper>,
) -> Result<String, Refusal> {
    let int_call = [
        ("floordiv", "int_floordiv"),
        ("mod", "int_mod"),
        ("band", "int_and"),
        ("bor", "int_or"),
        ("bxor", "int_xor"),
        ("shl", "int_shl"),
        ("shr", "int_shr"),
        ("invert", "int_invert"),
    ];
    let int_wrap = [
        ("add", "int_add"),
        ("sub", "int_sub"),
        ("mul", "int_mul"),
        ("neg", "int_neg"),
        ("abs", "int_abs"),
        ("min", "int_min"),
        ("max", "int_max"),
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
    if (kind == "min" || kind == "max") && (dtype == "i32" || dtype == "i64") {
        used.push(Helper::Integers);
        return Ok(format!("hawk_abi::int_{kind}<Int>({})", args.join(", ")));
    }
    // HAWK's (seed, counter) draw: aether's stateless detail::uniform01 / standardNormal under the
    // seed XOR-ed with a per-kind salt
    if args.len() == 2 && matches!(kind, "random_uniform" | "random_normal") {
        let (f, salt) = if kind == "random_uniform" {
            ("uniform01", "0x243F6A8885A308D3ULL")
        } else {
            ("standardNormal", "0x13198A2E03707344ULL")
        };
        used.push(Helper::Random);
        return Ok(format!("aether::random::detail::{f}<Real>(static_cast<std::uint64_t>({}) ^ {salt}, static_cast<aether::offset_t>(hawk_i), static_cast<std::uint64_t>({}))", args[0], args[1]));
    }
    if kind == "spacing" && args.len() == 1 {
        used.push(Helper::Spacing);
        return Ok(format!("hawk_abi::spacing({})", args[0]));
    }
    if matches!(kind, "dot" | "norm" | "sum" | "component" | "sign") || kind.starts_with("random") {
        return Err(nib(format!("scalar op {kind}")));
    }
    if MATH_UNARY.contains(&kind) && args.len() == 1 && kind != "sign" {
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
        ("spacing", 1) => {
            used.push(Helper::Spacing);
            Ok(format!("hawk_abi::spacing({})", args[0]))
        }
        ("neg", 1) => Ok(format!("(-{})", args[0])),
        ("lnot", 1) => Ok(format!("(!{})", args[0])),
        ("fma", 3) | ("clip", 3) => Ok(format!("aether::math::{kind}({})", args.join(", "))),
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
        return Ok("aether::banded::Band".to_string());
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
pub fn row_leaf(role: &str, name: &str) -> Result<String, Refusal> {
    let b = binding_name(role, name)?;
    match role {
        "vec_in" | "mat_in" | "per_sample" | "lookup" | "mutable" => Ok(format!("{b}[i].get()")),
        _ => Err(nib(format!("leaf role {role} of rank > 0"))),
    }
}

/// What the spelling of an op over tensors needs: the op, its operand texts and ranks, the result's
/// type and the
/// proxy source of a component read.
pub struct TensorOp<'a> {
    /// The op's name.
    pub kind: &'a str,
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
        "component" => {
            let ks: Vec<String> = op.component.iter().map(i64::to_string).collect();
            // a proxy source is read in place: `.at` on a mutable plane, `.eval` on a read-only one
            Ok(match op.lazy_source {
                Some(true) => format!("{}.at({})", args[0], ks.join(", ")),
                Some(false) => format!("{}.eval<{}>()", args[0], ks.join(", ")),
                None => format!("{}.eval<{}>(i)", args[0], ks.join(", ")),
            })
        }
        "splat" => cast(&args[0], dtype)
            .map(|c| format!("aether::constant<aether::extents<{}>>({c})", ext(shape))),
        "vec" if !args.is_empty() => element(dtype)
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
        "add" | "sub" => (|| {
            let sym = if kind == "add" { "+" } else { "-" };
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
        "mul" | "div" => product_or_quotient(op, hoist),
        "min" | "max" | "pow" => (|| {
            let f = match kind {
                "min" => "cwiseMin",
                "max" => "cwiseMax",
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
    let sym = if kind == "mul" { "*" } else { "/" };
    if ranks[0] == 0 && kind == "mul" {
        return Ok(format!("({} * {})", cast(&args[0], op.dtype)?, args[1]));
    }
    if ranks[1] == 0 {
        return Ok(format!("({} {sym} {})", args[0], cast(&args[1], op.dtype)?));
    }
    if kind == "mul" && ranks[0] == ranks[1] {
        return Ok(format!("{}.cwiseMul({})", args[0], args[1]));
    }
    if kind == "div" {
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
        "neg" => Ok(format!("({} * static_cast<Real>(-1))", args[0])),
        "abs" => Ok(format!("aether::cwiseAbs({})", args[0])),
        "sqrt" | "rsqrt" | "exp" | "log" | "sin" | "cos" | "asin" | "acos" | "atan" | "tan"
        | "tanh" => {
            let mut f = kind.to_string();
            f.replace_range(..1, &kind[..1].to_uppercase());
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
        "norm" | "sum" | "transpose" if one && !(kind == "sum" && ranks[0] == 2) => {
            format!("{}.{kind}()", args[0])
        }
        "mv" | "mm" if two => format!("({} * {})", args[0], args[1]),
        "outer" if two => format!("aether::outer({}, {})", args[0], args[1]),
        "dot" | "cross" if two => format!("{}.{kind}({})", args[0], args[1]),
        _ => return None,
    }))
}

/// A PlaneExtent node: the element count of the plane's view.
pub fn plane_extent(role: &str, name: &str) -> Result<String, Refusal> {
    Ok(format!(
        "static_cast<Int>({}.samples())",
        binding_name(role, name)?
    ))
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
}
