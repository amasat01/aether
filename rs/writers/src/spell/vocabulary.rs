//! The vocabulary table of the plain scalar operations: for each kind spelled as an infix operator
//! or as one call `aether::math::f(..)`, its arity and its spelling. The table is plain data, so
//! the writer spells from it and another module can read the same rows.
use super::{int_plane, Helper};
use crate::kind::OpKind;
use crate::refuse::{nib, Refusal};

/// How a row's operation is written in C++.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// `(a <operator> b)`, the operator text given.
    Infix(&'static str),
    /// Not written by this build: the refusal names the text given.
    Refused(&'static str),
    /// The text given, with `{0}`, `{1}` replaced by the operand texts.
    Template(&'static str),
    /// `hawk_abi::<helper>(args)`, the helper name given (with its template arguments, if any).
    Helper(&'static str),
    /// `(<operator>a)`, the operator text given.
    Prefix(&'static str),
    /// `aether::math::<function>(args)`, the function name given.
    Call(&'static str),
}

/// The kind of value an operation works on; a kind may have a row of its own per class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueClass {
    /// Every class that has no row of its own for the kind.
    Any,
    /// An integer plane (`int32`, `uint32`), whose arithmetic wraps.
    IntegerPlane,
    /// The wide integer scalar `Int` (`i32`, `i64`).
    WideInteger,
}

impl ValueClass {
    /// The class of values of a plane of `dtype`: its own class, or `Any` when it has none.
    pub fn of(dtype: &str) -> ValueClass {
        if int_plane(dtype) {
            ValueClass::IntegerPlane
        } else if matches!(dtype, "i32" | "i64") {
            ValueClass::WideInteger
        } else {
            ValueClass::Any
        }
    }
}

/// One operation of the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// The kind of operation the row spells.
    pub kind: OpKind,
    /// The number of operands the spelling takes.
    pub arity: usize,
    /// The class of values the row spells; a class without a row of its own uses the `Any` row.
    pub class: ValueClass,
    /// The way the operation is written; its text is also the name an author writes.
    pub spelling: Spelling,
    /// The helper the unit must define for the spelling, if it calls one.
    pub helper: Option<Helper>,
}

impl Row {
    /// The C++ text of the operation over operand texts, one per operand of the row; the refusal
    /// of a row that is not written.
    pub fn apply(&self, args: &[String]) -> Result<String, Refusal> {
        Ok(match self.spelling {
            Spelling::Refused(what) => return Err(nib(what)),
            Spelling::Template(text) => {
                args.iter().enumerate().fold(text.to_string(), |t, (i, a)| {
                    t.replace(&format!("{{{i}}}"), a)
                })
            }
            Spelling::Helper(f) => format!("hawk_abi::{f}({})", args.join(", ")),
            Spelling::Infix(op) => format!("({} {op} {})", args[0], args[1]),
            Spelling::Prefix(op) => format!("({op}{})", args[0]),
            Spelling::Call(f) => format!("aether::math::{f}({})", args.join(", ")),
        })
    }
}

/// A row for an infix operator over two operands.
const fn infix(kind: OpKind, op: &'static str) -> Row {
    Row {
        kind,
        arity: 2,
        class: ValueClass::Any,
        spelling: Spelling::Infix(op),
        helper: None,
    }
}

/// A row for a prefix operator over one operand.
const fn prefix(kind: OpKind, op: &'static str) -> Row {
    Row {
        kind,
        arity: 1,
        class: ValueClass::Any,
        spelling: Spelling::Prefix(op),
        helper: None,
    }
}

/// A row for a math call of `arity` operands.
const fn call(kind: OpKind, arity: usize, function: &'static str) -> Row {
    Row {
        kind,
        arity,
        class: ValueClass::Any,
        spelling: Spelling::Call(function),
        helper: None,
    }
}

/// A row for an integer helper `hawk_abi::<name>` over `arity` operands of `class`.
const fn integer(kind: OpKind, arity: usize, class: ValueClass, name: &'static str) -> Row {
    Row {
        kind,
        arity,
        class,
        spelling: Spelling::Helper(name),
        helper: Some(Helper::Integers),
    }
}

/// A row for an operation this build refuses, with the text its refusal names.
const fn refused(kind: OpKind, arity: usize, what: &'static str) -> Row {
    Row {
        kind,
        arity,
        class: ValueClass::Any,
        spelling: Spelling::Refused(what),
        helper: None,
    }
}

/// A row whose spelling needs `helper`, whatever the value class.
const fn with_helper(kind: OpKind, arity: usize, spelling: Spelling, helper: Helper) -> Row {
    Row {
        kind,
        arity,
        class: ValueClass::Any,
        spelling,
        helper: Some(helper),
    }
}

/// HAWK's (seed, counter) draw: aether's stateless generator under the seed XOR-ed with a salt
/// that is different for each distribution.
const UNIFORM: &str = "aether::random::detail::uniform01<Real>(static_cast<std::uint64_t>({0}) ^ 0x243F6A8885A308D3ULL, static_cast<aether::offset_t>(hawk_i), static_cast<std::uint64_t>({1}))";
/// The normal draw, as `UNIFORM` with its own salt.
const NORMAL: &str = "aether::random::detail::standardNormal<Real>(static_cast<std::uint64_t>({0}) ^ 0x13198A2E03707344ULL, static_cast<aether::offset_t>(hawk_i), static_cast<std::uint64_t>({1}))";

/// Every infix, prefix and math-call operation.
pub static VOCABULARY: &[Row] = &[
    integer(OpKind::Floordiv, 2, ValueClass::Any, "int_floordiv"),
    integer(OpKind::Mod, 2, ValueClass::Any, "int_mod"),
    integer(OpKind::Band, 2, ValueClass::Any, "int_and"),
    integer(OpKind::Bor, 2, ValueClass::Any, "int_or"),
    integer(OpKind::Bxor, 2, ValueClass::Any, "int_xor"),
    integer(OpKind::Shl, 2, ValueClass::Any, "int_shl"),
    integer(OpKind::Shr, 2, ValueClass::Any, "int_shr"),
    integer(OpKind::Invert, 1, ValueClass::Any, "int_invert"),
    integer(OpKind::Add, 2, ValueClass::IntegerPlane, "int_add"),
    integer(OpKind::Sub, 2, ValueClass::IntegerPlane, "int_sub"),
    integer(OpKind::Mul, 2, ValueClass::IntegerPlane, "int_mul"),
    integer(OpKind::Neg, 1, ValueClass::IntegerPlane, "int_neg"),
    integer(OpKind::Abs, 1, ValueClass::IntegerPlane, "int_abs"),
    integer(OpKind::Min, 2, ValueClass::IntegerPlane, "int_min"),
    integer(OpKind::Max, 2, ValueClass::IntegerPlane, "int_max"),
    integer(OpKind::Min, 2, ValueClass::WideInteger, "int_min<Int>"),
    integer(OpKind::Max, 2, ValueClass::WideInteger, "int_max<Int>"),
    with_helper(
        OpKind::RandomUniform,
        2,
        Spelling::Template(UNIFORM),
        Helper::Random,
    ),
    with_helper(
        OpKind::RandomNormal,
        2,
        Spelling::Template(NORMAL),
        Helper::Random,
    ),
    with_helper(
        OpKind::Spacing,
        1,
        Spelling::Helper("spacing"),
        Helper::Spacing,
    ),
    refused(OpKind::Dot, 2, "scalar op dot"),
    refused(OpKind::Norm, 1, "scalar op norm"),
    refused(OpKind::Sum, 1, "scalar op sum"),
    refused(OpKind::Component, 1, "scalar op component"),
    refused(OpKind::Sign, 1, "scalar op sign"),
    infix(OpKind::Add, "+"),
    infix(OpKind::Sub, "-"),
    infix(OpKind::Mul, "*"),
    infix(OpKind::Div, "/"),
    infix(OpKind::Lt, "<"),
    infix(OpKind::Le, "<="),
    infix(OpKind::Gt, ">"),
    infix(OpKind::Ge, ">="),
    infix(OpKind::Eq, "=="),
    infix(OpKind::Ne, "!="),
    infix(OpKind::Land, "&&"),
    infix(OpKind::Lor, "||"),
    prefix(OpKind::Neg, "-"),
    prefix(OpKind::Lnot, "!"),
    call(OpKind::Fma, 3, "fma"),
    call(OpKind::Clip, 3, "clip"),
    call(OpKind::Abs, 1, "abs"),
    call(OpKind::Sqrt, 1, "sqrt"),
    call(OpKind::Rsqrt, 1, "rsqrt"),
    call(OpKind::Exp, 1, "exp"),
    call(OpKind::Log, 1, "log"),
    call(OpKind::Sin, 1, "sin"),
    call(OpKind::Cos, 1, "cos"),
    call(OpKind::Tan, 1, "tan"),
    call(OpKind::Tanh, 1, "tanh"),
    call(OpKind::Asin, 1, "asin"),
    call(OpKind::Acos, 1, "acos"),
    call(OpKind::Atan, 1, "atan"),
    call(OpKind::Floor, 1, "floor"),
    call(OpKind::Exp2, 1, "exp2"),
    call(OpKind::Expm1, 1, "expm1"),
    call(OpKind::Log2, 1, "log2"),
    call(OpKind::Log10, 1, "log10"),
    call(OpKind::Log1p, 1, "log1p"),
    call(OpKind::Cbrt, 1, "cbrt"),
    call(OpKind::Sinh, 1, "sinh"),
    call(OpKind::Cosh, 1, "cosh"),
    call(OpKind::Asinh, 1, "asinh"),
    call(OpKind::Acosh, 1, "acosh"),
    call(OpKind::Atanh, 1, "atanh"),
    call(OpKind::Ceil, 1, "ceil"),
    call(OpKind::Trunc, 1, "trunc"),
    call(OpKind::Round, 1, "round"),
    call(OpKind::Rint, 1, "rint"),
    call(OpKind::Erf, 1, "erf"),
    call(OpKind::Erfc, 1, "erfc"),
    call(OpKind::Isnan, 1, "isnan"),
    call(OpKind::Isinf, 1, "isinf"),
    call(OpKind::Isfinite, 1, "isfinite"),
    call(OpKind::Pow, 2, "pow"),
    call(OpKind::Min, 2, "fmin"),
    call(OpKind::Max, 2, "fmax"),
    call(OpKind::Atan2, 2, "atan2"),
    call(OpKind::Hypot, 2, "hypot"),
    call(OpKind::Copysign, 2, "copysign"),
    call(OpKind::Fmod, 2, "fmod"),
    call(OpKind::Remainder, 2, "remainder"),
    call(OpKind::Fdim, 2, "fdim"),
];

/// The row of `kind` taking `arity` operands over values of `class`: the row of that class, else
/// the `Any` row.
pub fn row(kind: OpKind, arity: usize, class: ValueClass) -> Option<&'static Row> {
    let of_kind = || {
        VOCABULARY
            .iter()
            .filter(|r| r.kind == kind && r.arity == arity)
    };
    of_kind()
        .find(|r| r.class == class)
        .or_else(|| of_kind().find(|r| r.class == ValueClass::Any))
}

/// Whether `kind` is spelled as a math call at any arity.
pub fn is_call(kind: OpKind) -> bool {
    VOCABULARY
        .iter()
        .any(|r| r.kind == kind && matches!(r.spelling, Spelling::Call(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spell::scalar_op;

    #[test]
    fn no_kind_has_two_rows_of_one_arity_and_class() {
        for (i, a) in VOCABULARY.iter().enumerate() {
            for b in &VOCABULARY[i + 1..] {
                assert!(
                    !(a.kind == b.kind && a.arity == b.arity && a.class == b.class),
                    "two rows for {}",
                    a.kind
                );
            }
        }
    }

    #[test]
    fn every_row_kind_is_a_known_kind_and_a_unary_call_is_named_as_its_kind() {
        for r in VOCABULARY {
            assert_eq!(OpKind::parse(r.kind.name()), Some(r.kind));
            if let (1, Spelling::Call(f)) = (r.arity, r.spelling) {
                assert_eq!(f, r.kind.name());
            }
        }
    }

    #[test]
    fn the_writer_spells_each_row_as_the_row_says() {
        let args: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        for r in VOCABULARY {
            let args = &args[..r.arity];
            let dtype = match r.class {
                ValueClass::Any => "f64",
                ValueClass::IntegerPlane => "int32",
                ValueClass::WideInteger => "i64",
            };
            let mut used = Vec::new();
            let spelled = scalar_op(r.kind, args, dtype, &mut used);
            assert_eq!(used.first().copied(), r.helper, "{}", r.kind);
            let expected = match r.spelling {
                Spelling::Refused(what) => {
                    let reason = spelled.unwrap_err().reason;
                    assert_eq!(reason, format!("not in this build: {what}"));
                    continue;
                }
                Spelling::Infix(op) => format!("(a {op} b)"),
                Spelling::Prefix(op) => format!("({op}a)"),
                Spelling::Template(t) => t.replace("{0}", "a").replace("{1}", "b"),
                Spelling::Helper(f) if r.arity == 1 => format!("hawk_abi::{f}(a)"),
                Spelling::Helper(f) => format!("hawk_abi::{f}(a, b)"),
                Spelling::Call(f) if r.arity == 1 => format!("aether::math::{f}(a)"),
                Spelling::Call(f) if r.arity == 2 => format!("aether::math::{f}(a, b)"),
                Spelling::Call(f) => format!("aether::math::{f}(a, b, c)"),
            };
            assert_eq!(spelled.unwrap(), expected, "{}", r.kind);
            assert_eq!(r.apply(args).unwrap(), expected);
        }
    }
}
