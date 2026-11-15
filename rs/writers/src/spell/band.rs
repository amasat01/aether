//! Banded literals: a real as the three-limb `Band(hi, mid, lo)` register value, and the literal-operand forms of
//! `+ - * /`. Every limb is the float32 nearest the exact remainder, computed in integers (a value is `q * 2^s`).
use super::{real_literal, OpKind, BAND_TYPE as BAND};
use crate::refuse::{nib, Refusal};

/// The namespace of the raw band forms.
const DETAIL: &str = "aether::banded::detail::";
/// The least exponent of a float32 limb (the unit of its subnormals).
const LIMB_FLOOR: i32 = -149;

/// A value `q * 2^s` and the sign it carries, the remainder a split leaves.
struct Rest {
    /// The value is negative.
    neg: bool,
    /// The magnitude is `q * 2^s`.
    q: u128,
    /// The scale exponent.
    s: i32,
    /// The true value is a little above `q * 2^s` (the quotient was cut).
    sticky: bool,
}

impl Rest {
    /// The float32 nearest the rest (round half to even, subnormal grid below 2^-126), as the value it holds and the
    /// rest it leaves; none when the limb overflows float32.
    fn take(&mut self) -> Option<f64> {
        if self.q == 0 {
            return Some(0.0);
        }
        let bits = 128 - self.q.leading_zeros() as i32;
        let shift = (bits - 24).max(LIMB_FLOOR - self.s).max(0);
        let limb = if shift >= 127 {
            0
        } else {
            let (mask, half) = ((1u128 << shift) - 1, (1u128 << shift) >> 1);
            let (mut l, rem) = (self.q >> shift, self.q & mask);
            if shift > 0 && (rem > half || (rem == half && (self.sticky || l & 1 == 1))) {
                l += 1;
            }
            l
        };
        let kept = if shift >= 127 { 0 } else { limb << shift };
        let magnitude = limb as f64 * 2f64.powi(self.s + shift);
        let value = if self.neg { -magnitude } else { magnitude };
        if kept <= self.q {
            self.q -= kept;
        } else {
            // the limb went past the rest: what is left has the other sign, and the cut tail takes from it
            self.q = kept - self.q - u128::from(self.sticky);
            self.neg = !self.neg;
        }
        (value as f32).is_finite().then_some(value)
    }
}

/// The three limbs of `q * 2^s` (negated when `neg`); none when a limb overflows float32.
fn limbs(neg: bool, q: u128, s: i32, sticky: bool) -> Option<([f64; 3], bool)> {
    let mut rest = Rest { neg, q, s, sticky };
    let mut out = [0.0; 3];
    for l in &mut out {
        *l = rest.take()?;
    }
    Some((out, rest.q == 0 && !rest.sticky))
}

/// A finite double as `m * 2^e` with `m` an integer.
fn parts(x: f64) -> (bool, u128, i32) {
    let bits = x.to_bits();
    let (field, frac) = (
        ((bits >> 52) & 0x7ff) as i32,
        (bits & ((1 << 52) - 1)) as u128,
    );
    let neg = bits >> 63 == 1;
    if field == 0 {
        (neg, frac, -1074)
    } else {
        (neg, frac | (1 << 52), field - 1075)
    }
}

/// The `Band(hi, mid, lo)` text of three limbs.
fn limb_text(limbs: [f64; 3]) -> String {
    let t: Vec<String> = limbs
        .iter()
        .map(|&h| format!("{}f", real_literal(h)))
        .collect();
    format!("{BAND}({})", t.join(", "))
}

/// `value` as `Band(hi, mid, lo)`; a value the three limbs cannot hold exactly is refused by name.
pub fn band_literal(value: f64) -> Result<String, Refusal> {
    if !value.is_finite() {
        return Ok(format!("{BAND}({}, 0.0f, 0.0f)", real_literal(value)));
    }
    if value == 0.0 {
        return Ok(format!("{BAND}({}f, 0.0f, 0.0f)", real_literal(value)));
    }
    let (neg, m, e) = parts(value);
    match limbs(neg, m, e, false) {
        Some((l, true)) => Ok(limb_text(l)),
        _ => Err(nib(format!(
            "a banded literal {value:e} outside the banded range (the FP32 exponent envelope)"
        ))),
    }
}

/// The three limbs of `1 / c`, split from the exact quotient; none out of the float32 range.
fn reciprocal_limbs(c: f64) -> Option<[f64; 3]> {
    let (neg, m, e) = parts(c);
    let k = (128 - m.leading_zeros()) as i32 + 112;
    let (mut q, mut r) = (0u128, 1u128);
    for _ in 0..k {
        r <<= 1;
        q <<= 1;
        if r >= m {
            r -= m;
            q |= 1;
        }
    }
    let (l, _) = limbs(neg, q, -k - e, r != 0)?;
    (l[0].abs() >= 2f64.powi(-94) && l[0].abs() < 2f64.powi(126)).then_some(l)
}

/// Whether `|x|` is a power of two (as a fraction: its significand has one bit).
fn is_pow2(x: f64) -> bool {
    x != 0.0 && parts(x).1.count_ones() == 1
}

/// The float32 equal to `x` exactly, else none.
fn float_exact(x: f64) -> Option<f64> {
    let f = x as f32;
    (f.is_finite() && f as f64 == x).then_some(f as f64)
}

/// A float32 value as a literal.
fn f32_text(x: f64) -> String {
    format!("{}f", real_literal(x))
}

/// `x <op> c` with a literal `c` (`left` when the literal is the left operand), spelled by its exact class; none when the
/// operator on the three-limb `Band` is the spelling.
pub fn literal_op(kind: OpKind, x: &str, c: f64, left: bool) -> Option<String> {
    let c = c + 0.0;
    if matches!(kind, OpKind::Sub | OpKind::Div) && left {
        return None;
    }
    let helper = |name: &str, f: f64| format!("{DETAIL}{name}({x}, {})", f32_text(f));
    if matches!(kind, OpKind::Add | OpKind::Sub) {
        if c == 0.0 {
            return Some(x.to_string());
        }
        return float_exact(if kind == OpKind::Sub { -c } else { c }).map(|f| helper("addRawS", f));
    }
    // the factor: c, or for a quotient its reciprocal (a power of two when c is one)
    let factor = if kind == OpKind::Div {
        if c == 0.0 {
            return None;
        }
        if is_pow2(c) {
            let (_, m, e) = parts(c);
            let k = -(e + m.trailing_zeros() as i32);
            (k.abs() < 1000).then(|| c.signum() * 2f64.powi(k))
        } else {
            None
        }
    } else {
        Some(c)
    };
    if let Some(fac) = factor {
        if fac == 1.0 {
            return Some(x.to_string());
        }
        if fac == -1.0 {
            return Some(format!("(-{x})"));
        }
        if let Some(f) = float_exact(fac) {
            return Some(helper(
                if is_pow2(fac) {
                    "scalePow2f"
                } else {
                    "mulRawS"
                },
                f,
            ));
        }
    }
    if kind == OpKind::Div {
        return reciprocal_limbs(c).map(|l| format!("({x} * {})", limb_text(l)));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_split_into_exact_limbs() {
        assert_eq!(
            band_literal(1.0 / 3.0).unwrap(),
            "aether::banded::Band(0.3333333432674408f, -9.934107758624577e-09f, 2.7755575615628914e-16f)"
        );
        assert_eq!(
            band_literal(0.1).unwrap(),
            "aether::banded::Band(0.10000000149011612f, -1.4901161415892261e-09f, 2.7755575615628914e-17f)"
        );
        assert_eq!(
            band_literal(2.0).unwrap(),
            "aether::banded::Band(2.0f, 0.0f, 0.0f)"
        );
        assert_eq!(
            band_literal(-0.0).unwrap(),
            "aether::banded::Band(-0.0f, 0.0f, 0.0f)"
        );
        assert_eq!(
            band_literal(1e-30).unwrap(),
            "aether::banded::Band(1.0000000031710769e-30f, -3.171076767634631e-39f, 0.0f)"
        );
        assert!(band_literal(1e300).is_err());
        assert!(band_literal(f64::INFINITY)
            .unwrap()
            .ends_with("(INFINITY, 0.0f, 0.0f)"));
    }

    #[test]
    fn literal_operands_take_their_exact_class() {
        let op = |k, c, left| literal_op(k, "x", c, left);
        assert_eq!(op(OpKind::Mul, 1.0, false).unwrap(), "x");
        assert_eq!(op(OpKind::Mul, -1.0, true).unwrap(), "(-x)");
        assert_eq!(op(OpKind::Add, 0.0, true).unwrap(), "x");
        assert_eq!(
            op(OpKind::Mul, 2.0, false).unwrap(),
            "aether::banded::detail::scalePow2f(x, 2.0f)"
        );
        assert_eq!(
            op(OpKind::Mul, 3.0, false).unwrap(),
            "aether::banded::detail::mulRawS(x, 3.0f)"
        );
        assert_eq!(
            op(OpKind::Sub, 0.5, false).unwrap(),
            "aether::banded::detail::addRawS(x, -0.5f)"
        );
        assert_eq!(
            op(OpKind::Div, 4.0, false).unwrap(),
            "aether::banded::detail::scalePow2f(x, 0.25f)"
        );
        assert_eq!(
            op(OpKind::Div, 3.0, false).unwrap(),
            "(x * aether::banded::Band(0.3333333432674408f, -9.934107758624577e-09f, 2.9605948205663494e-16f))"
        );
        assert_eq!(
            op(OpKind::Div, 7.0, false).unwrap(),
            "(x * aether::banded::Band(0.1428571492433548f, -6.38621200366174e-09f, 6.34413185289139e-17f))"
        );
        assert_eq!(
            op(OpKind::Div, -10.0, false).unwrap(),
            "(x * aether::banded::Band(-0.10000000149011612f, 1.4901161415892261e-09f, -2.2204460823375376e-17f))"
        );
        assert!(op(OpKind::Div, 3.0, true).is_none());
        assert!(op(OpKind::Mul, 0.1, false).is_none());
    }
}
