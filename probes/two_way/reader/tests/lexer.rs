//! Lexer properties: every byte lands in exactly one token, and literals and comments are
//! read as the language reads them.

use reader::lex::{lex, reemit, TokKind};

fn kinds(src: &str) -> Vec<(TokKind, &str)> {
    lex(src)
        .iter()
        .map(|t| (t.kind, &src[t.start..t.end]))
        .collect()
}

fn significant(src: &str) -> Vec<(TokKind, &str)> {
    kinds(src)
        .into_iter()
        .filter(|(k, _)| !matches!(k, TokKind::Space))
        .collect()
}

#[test]
fn re_emitting_all_spans_gives_the_file_back_byte_for_byte() {
    let src =
        "#include <a>\nint f() { return \"x{\\\"\" + R\"x(a)\"b)x\"; } // end\r\n/* c */ 0x1'0e+3";
    assert_eq!(reemit(src, &lex(src)), src);
}

#[test]
fn spans_tile_the_file_without_gaps_or_overlap() {
    let src = "a /* b */ \"c\" 'd' #x\n  .5e-3 -> <<= é;";
    let mut at = 0;
    for t in lex(src) {
        assert_eq!(t.start, at);
        at = t.end;
    }
    assert_eq!(at, src.len());
}

#[test]
fn a_brace_inside_a_string_is_part_of_the_string() {
    let toks = significant("s = \"{ if (x) }\";");
    assert!(toks.contains(&(TokKind::Str, "\"{ if (x) }\"")));
}

#[test]
fn comment_markers_inside_a_string_do_not_start_a_comment() {
    let toks = significant("a = \"// not /* a comment\"; b");
    assert_eq!(toks.last(), Some(&(TokKind::Ident, "b")));
    assert!(toks
        .iter()
        .all(|(k, _)| !matches!(k, TokKind::LineComment | TokKind::BlockComment)));
}

#[test]
fn a_raw_string_runs_to_its_own_delimiter() {
    let src = "x = R\"d(a)\" \"quote\" ) d\")d\"; y";
    let toks = significant(src);
    assert!(toks.contains(&(TokKind::RawStr, "R\"d(a)\" \"quote\" ) d\")d\"")));
    assert_eq!(toks.last(), Some(&(TokKind::Ident, "y")));
}

#[test]
fn an_escaped_quote_does_not_end_a_string() {
    let toks = significant(r#"a = "q\"{"; b"#);
    assert!(toks.contains(&(TokKind::Str, r#""q\"{""#)));
}

#[test]
fn a_character_literal_holding_a_brace_is_one_token() {
    let toks = significant("c = '{'; d = '\\'';");
    assert!(toks.contains(&(TokKind::Char, "'{'")));
    assert!(toks.contains(&(TokKind::Char, "'\\''")));
}

#[test]
fn an_encoding_prefix_belongs_to_its_literal() {
    let toks = significant("u8\"a\" L'b' uR\"(c)\"");
    assert_eq!(toks[0].0, TokKind::Str);
    assert_eq!(toks[1].0, TokKind::Char);
    assert_eq!(toks[2].0, TokKind::RawStr);
}

#[test]
fn a_preprocessor_line_with_continuations_and_a_comment_is_one_opaque_token() {
    let src = "#define M(x) \\\n  if (x) { /* multi\nline */ } \\\n  else {}\nint y;";
    let toks = significant(src);
    assert_eq!(toks[0].0, TokKind::Preproc);
    assert!(toks[0].1.ends_with("else {}"));
    assert_eq!(toks[1], (TokKind::Ident, "int"));
}

#[test]
fn a_hash_in_the_middle_of_a_line_is_punctuation() {
    let toks = significant("a # b");
    assert_eq!(toks[1], (TokKind::Punct, "#"));
}

#[test]
fn numbers_keep_separators_and_signed_exponents() {
    let toks = significant("1'000 0x1p-3 1e+5 .5 a");
    let numbers: Vec<_> = toks
        .iter()
        .filter(|(k, _)| *k == TokKind::Number)
        .map(|t| t.1)
        .collect();
    assert_eq!(numbers, ["1'000", "0x1p-3", "1e+5", ".5"]);
}

#[test]
fn compound_operators_are_single_tokens() {
    let toks = significant("a <= b == c += d :: e");
    let ops: Vec<_> = toks
        .iter()
        .filter(|(k, _)| *k == TokKind::Punct)
        .map(|t| t.1)
        .collect();
    assert_eq!(ops, ["<=", "==", "+=", "::"]);
}

#[test]
fn unterminated_literals_and_comments_do_not_panic() {
    for src in [
        "\"abc",
        "'a",
        "/* open",
        "R\"x(open",
        "#define A \\",
        "// x \\",
        "'",
    ] {
        assert_eq!(reemit(src, &lex(src)), src);
    }
}
