//! The build contract: ONE JSON text, read with the small reader below (numbers keep their text, so
//! no integer is rounded).
/// A JSON value, as the contract reader makes it.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// The value `null`.
    Null,
    /// A truth value.
    Bool(bool),
    /// A number, kept as its text so no integer is rounded.
    Num(String),
    /// A string, escapes resolved.
    Str(String),
    /// A list.
    Arr(Vec<Json>),
    /// An object: its members in the order they were written.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// The member of an object under `k`, none for a missing key or a value that is not an object.
    pub fn get(&self, k: &str) -> Option<&Json> {
        if let Json::Obj(v) = self {
            v.iter().find(|x| x.0 == k).map(|x| &x.1)
        } else {
            None
        }
    }
    /// The text of a string value.
    pub fn str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    /// The items of a list, empty for any other value.
    pub fn arr(&self) -> &[Json] {
        if let Json::Arr(a) = self {
            a
        } else {
            &[]
        }
    }
    /// The integer a number value holds, none when it is not an integer.
    pub fn int(&self) -> Option<i64> {
        if let Json::Num(n) = self {
            n.parse().ok()
        } else {
            None
        }
    }
}

/// The parsed contract; its keys are those of the recorded render inputs.
#[derive(Debug, Clone)]
pub struct Contract(pub Json);

impl Contract {
    /// Reads one JSON value from `text`; anything after it is refused.
    pub fn parse(text: &str) -> Result<Contract, String> {
        let (b, mut i) = (text.as_bytes(), 0);
        let v = value(b, &mut i)?;
        ws(b, &mut i);
        if i != b.len() {
            return Err(format!("contract: trailing text at {i}"));
        }
        Ok(Contract(v))
    }
    /// The member of the contract's top object under `k`.
    pub fn get(&self, k: &str) -> Option<&Json> {
        self.0.get(k)
    }
}

/// Skips white space.
fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1
    }
}

/// The byte at `i`, or the refusal for the end of the text.
fn at(b: &[u8], i: usize) -> Result<u8, String> {
    b.get(i)
        .copied()
        .ok_or_else(|| "contract: unexpected end".to_string())
}

/// The literal `w` (`true`, `false`, `null`) as the value `v`.
fn lit(b: &[u8], i: &mut usize, w: &str, v: Json) -> Result<Json, String> {
    if b[*i..].starts_with(w.as_bytes()) {
        *i += w.len();
        Ok(v)
    } else {
        Err(format!("contract: bad literal at {i}"))
    }
}

/// An object or a list: the items up to the closing bracket, separated by commas.
fn collection(b: &[u8], i: &mut usize) -> Result<Json, String> {
    let (obj, close) = (b[*i] == b'{', if b[*i] == b'{' { b'}' } else { b']' });
    *i += 1;
    let (mut o, mut a) = (vec![], vec![]);
    loop {
        ws(b, i);
        if at(b, *i)? == close {
            *i += 1;
            return Ok(if obj { Json::Obj(o) } else { Json::Arr(a) });
        }
        if obj {
            let k = match value(b, i)? {
                Json::Str(s) => s,
                _ => return Err("contract: object key".into()),
            };
            ws(b, i);
            if at(b, *i)? != b':' {
                return Err(format!("contract: ':' expected at {i}"));
            }
            *i += 1;
            o.push((k, value(b, i)?));
        } else {
            a.push(value(b, i)?)
        }
        ws(b, i);
        match at(b, *i)? {
            b',' => *i += 1,
            c if c == close => {}
            _ => return Err(format!("contract: ',' expected at {i}")),
        }
    }
}

/// The four hex digits at `at_`, as a code unit.
fn hex4(b: &[u8], at_: usize) -> Result<u32, &'static str> {
    let digits = b.get(at_..at_ + 4).ok_or("contract: \\u")?;
    std::str::from_utf8(digits)
        .ok()
        .and_then(|h| u32::from_str_radix(h, 16).ok())
        .ok_or("contract: \\u")
}

/// The character of a `\u` escape (`i` is just past the `u`), joining a surrogate pair; the bytes
/// go to `s`.
fn unicode_escape(b: &[u8], i: &mut usize, s: &mut Vec<u8>) -> Result<(), String> {
    let mut cp = hex4(b, *i)?;
    *i += 4;
    if (0xD800..0xDC00).contains(&cp) && b.get(*i..*i + 2) == Some(b"\\u") {
        let lo = hex4(b, *i + 2)?;
        *i += 6;
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
    }
    let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
    s.extend_from_slice(ch.to_string().as_bytes());
    Ok(())
}

/// A string literal: the bytes up to the closing quote, escapes resolved.
fn string(b: &[u8], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut s = Vec::new();
    loop {
        match at(b, *i)? {
            b'"' => {
                *i += 1;
                return String::from_utf8(s)
                    .map(Json::Str)
                    .map_err(|e| e.to_string());
            }
            b'\\' => {
                let e = at(b, *i + 1)?;
                *i += 2;
                match e {
                    b'n' => s.push(b'\n'),
                    b't' => s.push(b'\t'),
                    b'r' => s.push(b'\r'),
                    b'b' => s.push(8),
                    b'f' => s.push(12),
                    b'"' | b'\\' | b'/' => s.push(e),
                    b'u' => unicode_escape(b, i, &mut s)?,
                    _ => return Err(format!("contract: escape at {i}")),
                }
            }
            c => {
                s.push(c);
                *i += 1
            }
        }
    }
}

/// A number: the run of digits and sign, point and exponent characters, kept as text.
fn number(b: &[u8], i: &mut usize) -> Result<Json, String> {
    let st = *i;
    while *i < b.len() && (b[*i].is_ascii_digit() || b"+-.eE".contains(&b[*i])) {
        *i += 1
    }
    if st == *i {
        return Err(format!("contract: unexpected byte at {i}"));
    }
    Ok(Json::Num(String::from_utf8_lossy(&b[st..*i]).into_owned()))
}

/// One value, by the kind its first byte names.
fn value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    ws(b, i);
    match at(b, *i)? {
        b'{' | b'[' => collection(b, i),
        b'"' => string(b, i),
        b't' => lit(b, i, "true", Json::Bool(true)),
        b'f' => lit(b, i, "false", Json::Bool(false)),
        b'n' => lit(b, i, "null", Json::Null),
        _ => number(b, i),
    }
}
