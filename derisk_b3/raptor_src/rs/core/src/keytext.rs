//! The KEY TEXT rule (GA/SPEC.md section 10): how a value that reaches a node's dedup identity is written. It reproduces what
//! Python's `repr` writes for each kind below, with no call into Python. A kind outside this list is refused by name.
use crate::nonprintable::NONPRINTABLE;

#[derive(Clone, Debug, PartialEq)]
pub enum V { None, Bool(bool), Int(i128), Float(f64), Str(String), Text(String), Tuple(Vec<V>), List(Vec<V>) }

/// A float as the shortest text that reads back to the same value, in Python's layout: `inf`, `-inf`, `nan`, `-0.0`, a plain
/// decimal when the decimal exponent lies in (-4, 16], else `d.ddde+XX` (exponent sign always, at least two digits).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() { return "nan".into() }
    if x.is_infinite() { return if x < 0.0 { "-inf" } else { "inf" }.into() }
    let a = x.abs();
    let mut e = format!("{:e}", a); // shortest round-trip digit COUNT: "d[.ddd]e<exp>"
    // among the strings of that length, the one closest to the value is the correctly rounded one (the shortest-digits routine may pick
    // another that also reads back, e.g. at 17 digits); use it when it reads back, as CPython's does
    let n = e.split_once('e').unwrap().0.chars().filter(|c| *c != '.').count();
    let exact = format!("{:.*e}", n - 1, a);
    if exact.parse::<f64>() == Ok(a) { e = exact }
    let (m, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 { return format!("{sign}0.0") }
    let decpt = ex + 1; // position of the decimal point relative to the digits
    let n = digits.len() as i32;
    let body = if -4 < decpt && decpt <= 16 {
        if decpt <= 0 { format!("0.{}{}", "0".repeat((-decpt) as usize), digits) }
        else if decpt >= n { format!("{}{}.0", digits, "0".repeat((decpt - n) as usize)) }
        else { format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..]) }
    } else {
        let mant = if n == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let e = decpt - 1;
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    };
    format!("{sign}{body}")
}

fn printable(c: u32) -> bool {
    if c == 0x20 { return true }
    if c < 0x20 || c == 0x7f { return false }
    if c < 0x7f { return true }
    NONPRINTABLE.binary_search_by(|&(a, b)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }).is_err()
}

/// A string in single quotes (double quotes when it holds a single quote and no double quote); `\\`, the quote, `\t \n \r` escaped;
/// any other non-printable code point as `\xNN`, `\uNNNN` or `\UNNNNNNNN` (lower-case hex), the rest as it is.
pub fn str_repr(s: &str, out: &mut String) {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(q);
    for c in s.chars() {
        let u = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => { out.push('\\'); out.push(c) }
            c if printable(u) => out.push(c),
            _ if u < 0x100 => out.push_str(&format!("\\x{u:02x}")),
            _ if u < 0x10000 => out.push_str(&format!("\\u{u:04x}")),
            _ => out.push_str(&format!("\\U{u:08x}")),
        }
    }
    out.push(q);
}

fn seq(v: &[V], open: char, close: char, one_comma: bool, out: &mut String) {
    out.push(open);
    for (i, x) in v.iter().enumerate() { if i > 0 { out.push_str(", ") } write(x, out) }
    if one_comma && v.len() == 1 { out.push(',') }
    out.push(close);
}

pub fn write(v: &V, out: &mut String) {
    match v {
        V::None => out.push_str("None"),
        V::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        V::Int(i) => out.push_str(&i.to_string()),
        V::Float(f) => out.push_str(&float_repr(*f)),
        V::Str(s) => str_repr(s, out),
        V::Text(t) => out.push_str(t),
        V::Tuple(t) => seq(t, '(', ')', true, out),
        V::List(l) => seq(l, '[', ']', false, out),
    }
}

pub fn text(v: &V) -> String { let mut o = String::new(); write(v, &mut o); o }

/// Python `repr` of a str (the same rule as `str_repr`).
pub fn py_str(s: &str) -> String { let mut o = String::new(); str_repr(s, &mut o); o }
