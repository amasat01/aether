//! A C++ lexer that keeps the byte span of every token, so that re-emitting all spans gives the
//! source back byte for byte. It needs no type information and never fails: malformed input
//! (an unterminated string or comment) becomes a token that runs to the end of its line or file.

/// What a token is. Whitespace, comments and preprocessor lines are tokens too, so the spans tile
/// the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokKind {
    /// A run of whitespace.
    Space,
    /// A `//` comment up to, not including, the newline.
    LineComment,
    /// A `/* */` comment.
    BlockComment,
    /// A string literal, with its encoding prefix.
    Str,
    /// A character literal, with its encoding prefix.
    Char,
    /// A raw string literal, with its prefix.
    RawStr,
    /// A whole preprocessor line, continuations included, kept opaque.
    Preproc,
    /// An identifier or keyword.
    Ident,
    /// A preprocessing number.
    Number,
    /// An operator or punctuator.
    Punct,
}

/// One token: its kind and its byte span in the source.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    /// What the token is.
    pub kind: TokKind,
    /// Byte offset of the first byte.
    pub start: usize,
    /// Byte offset one past the last byte.
    pub end: usize,
}

/// Operators lexed as one token, longest first, so `<=` is never read as `<` then `=`.
const PUNCT: [&str; 27] = [
    "<<=", ">>=", "...", "<=>", "->*", "::", "->", "++", "--", "==", "!=", "<=", ">=", "&&", "||",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", ".*", "##",
];

/// Splits `src` into tokens whose spans cover it exactly, in order.
#[must_use]
pub fn lex(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut pos = 0;
    // True while only whitespace and comments precede `pos` on its line: a `#` there starts a
    // preprocessor line.
    let mut line_start = true;
    while pos < bytes.len() {
        let (kind, end) = next_token(bytes, pos, line_start);
        match kind {
            TokKind::Space => {
                line_start |= bytes[pos..end].contains(&b'\n');
            }
            TokKind::LineComment | TokKind::BlockComment => {}
            _ => line_start = false,
        }
        tokens.push(Token {
            kind,
            start: pos,
            end,
        });
        pos = end;
    }
    tokens
}

/// Re-emits the tokens' spans; equals the source when the spans tile it.
#[must_use]
pub fn reemit(src: &str, tokens: &[Token]) -> String {
    tokens.iter().map(|t| &src[t.start..t.end]).collect()
}

fn next_token(b: &[u8], pos: usize, line_start: bool) -> (TokKind, usize) {
    let c = b[pos];
    if c.is_ascii_whitespace() {
        let len = b[pos..]
            .iter()
            .take_while(|x| x.is_ascii_whitespace())
            .count();
        return (TokKind::Space, pos + len);
    }
    if b[pos..].starts_with(b"//") {
        return (TokKind::LineComment, line_comment_end(b, pos));
    }
    if b[pos..].starts_with(b"/*") {
        return (TokKind::BlockComment, block_comment_end(b, pos));
    }
    match c {
        b'#' if line_start => (TokKind::Preproc, directive_end(b, pos)),
        b'"' => (TokKind::Str, quoted_end(b, pos, b'"')),
        b'\'' => (TokKind::Char, quoted_end(b, pos, b'\'')),
        _ if is_ident_byte(c) && !c.is_ascii_digit() => ident_token(b, pos),
        _ if c.is_ascii_digit()
            || (c == b'.' && b.get(pos + 1).is_some_and(u8::is_ascii_digit)) =>
        {
            (TokKind::Number, number_end(b, pos))
        }
        _ => (TokKind::Punct, pos + punct_len(b, pos)),
    }
}

/// Identifier bytes; bytes of a multi-byte UTF-8 character count, so a span never splits one.
fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

/// An identifier, or a literal whose encoding prefix looks like one (`u8"x"`, `R"(x)"`).
fn ident_token(b: &[u8], pos: usize) -> (TokKind, usize) {
    let len = b[pos..].iter().take_while(|&&x| is_ident_byte(x)).count();
    let end = pos + len;
    let id = &b[pos..end];
    let encoding = matches!(id, b"L" | b"u" | b"U" | b"u8");
    match b.get(end) {
        Some(b'"') if matches!(id, b"R" | b"LR" | b"uR" | b"UR" | b"u8R") => {
            (TokKind::RawStr, raw_string_end(b, end))
        }
        Some(b'"') if encoding => (TokKind::Str, quoted_end(b, end, b'"')),
        Some(b'\'') if encoding => (TokKind::Char, quoted_end(b, end, b'\'')),
        _ => (TokKind::Ident, end),
    }
}

/// A newline that is escaped by a backslash does not end the line.
fn escaped_newline(b: &[u8], i: usize) -> usize {
    match (b.get(i), b.get(i + 1), b.get(i + 2)) {
        (Some(b'\\'), Some(b'\n'), _) => 2,
        (Some(b'\\'), Some(b'\r'), Some(b'\n')) => 3,
        _ => 0,
    }
}

fn line_comment_end(b: &[u8], start: usize) -> usize {
    let mut i = start + 2;
    while i < b.len() && b[i] != b'\n' {
        let skip = escaped_newline(b, i);
        i += if skip > 0 { skip } else { 1 };
    }
    i
}

fn block_comment_end(b: &[u8], start: usize) -> usize {
    let mut i = start + 2;
    while i < b.len() {
        if b[i..].starts_with(b"*/") {
            return i + 2;
        }
        i += 1;
    }
    b.len()
}

/// A preprocessor line: to the first newline that is not escaped, skipping comments and strings
/// so that a block comment spanning lines stays inside the directive.
fn directive_end(b: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < b.len() {
        let skip = escaped_newline(b, i);
        if skip > 0 {
            i += skip;
        } else if b[i] == b'\n' {
            break;
        } else if b[i..].starts_with(b"/*") {
            i = block_comment_end(b, i);
        } else if b[i..].starts_with(b"//") {
            i = line_comment_end(b, i);
        } else if b[i] == b'"' {
            i = quoted_end(b, i, b'"');
        } else {
            i += 1;
        }
    }
    i
}

/// A string or character literal starting at the opening quote `at`. An unterminated one stops
/// before the newline.
fn quoted_end(b: &[u8], at: usize, quote: u8) -> usize {
    let mut i = at + 1;
    while i < b.len() {
        let skip = escaped_newline(b, i);
        if skip > 0 {
            i += skip;
        } else if b[i] == b'\\' {
            i += 2;
        } else if b[i] == quote {
            return i + 1;
        } else if b[i] == b'\n' {
            return i;
        } else {
            i += 1;
        }
    }
    b.len()
}

/// A raw string whose opening quote is at `quote_at`: `"delim( ... )delim"`. A malformed opening
/// falls back to an ordinary string.
fn raw_string_end(b: &[u8], quote_at: usize) -> usize {
    let delim_start = quote_at + 1;
    let open = b[delim_start..].iter().take(17).position(|&x| x == b'(');
    let Some(open) = open else {
        return quoted_end(b, quote_at, b'"');
    };
    let mut closing = vec![b')'];
    closing.extend_from_slice(&b[delim_start..delim_start + open]);
    closing.push(b'"');
    let body = delim_start + open + 1;
    b[body..]
        .windows(closing.len())
        .position(|w| w == closing.as_slice())
        .map_or(b.len(), |p| body + p + closing.len())
}

/// A preprocessing number: digits, letters, dots, digit separators and signed exponents.
fn number_end(b: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < b.len() {
        let c = b[i];
        let after_exponent = i > start && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P');
        let keep = c.is_ascii_alphanumeric()
            || c == b'_'
            || c == b'.'
            || (c == b'\'' && b.get(i + 1).is_some_and(u8::is_ascii_alphanumeric))
            || (matches!(c, b'+' | b'-') && after_exponent);
        if !keep {
            break;
        }
        i += 1;
    }
    i
}

fn punct_len(b: &[u8], pos: usize) -> usize {
    PUNCT
        .iter()
        .find(|p| b[pos..].starts_with(p.as_bytes()))
        .map_or(1, |p| p.len())
}
