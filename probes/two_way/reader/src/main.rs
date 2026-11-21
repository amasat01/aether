//! Command line of the reader prototype: `reader rewrite <in> <out>` and `reader census <file>...`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use reader::analyse;
use reader::lex::{lex, reemit};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [cmd, input, output] if cmd == "rewrite" => rewrite(input, output),
        [cmd, files @ ..] if cmd == "census" && !files.is_empty() => census(files),
        _ => {
            Err("usage: reader rewrite <in file> <out file> | reader census <file>...".to_string())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("reader: {message}");
            ExitCode::FAILURE
        }
    }
}

fn read(file: &str) -> Result<String, String> {
    fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))
}

fn rewrite(input: &str, output: &str) -> Result<(), String> {
    let analysis = analyse(&read(input)?);
    if let Some(dir) = Path::new(output).parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    fs::write(output, &analysis.output).map_err(|e| format!("{output}: {e}"))?;
    for r in &analysis.reports {
        println!("{input}:{}: left untouched: {}", r.line, r.kind);
    }
    println!(
        "{input}: {} statements rewritten, {} left",
        analysis.rewritten.values().sum::<usize>(),
        analysis.reports.len()
    );
    Ok(())
}

/// Totals over the files of a census.
#[derive(Default)]
struct Totals {
    files: usize,
    tokens: usize,
    identity_failures: usize,
    functions: usize,
    rewritten: BTreeMap<&'static str, usize>,
    statements: BTreeMap<&'static str, usize>,
    refusals: BTreeMap<&'static str, usize>,
}

fn census(files: &[String]) -> Result<(), String> {
    let mut totals = Totals::default();
    for file in files {
        let text = read(file)?;
        if reemit(&text, &lex(&text)) != text {
            totals.identity_failures += 1;
            println!("{file}: IDENTITY FAILURE: re-emitting the tokens differs from the file");
        }
        let a = analyse(&text);
        totals.files += 1;
        totals.tokens += a.tokens;
        totals.functions += a.functions;
        for (kind, n) in &a.rewritten {
            *totals.rewritten.entry(kind).or_insert(0) += n;
        }
        for (kind, n) in &a.statements {
            *totals.statements.entry(kind).or_insert(0) += n;
        }
        for r in &a.reports {
            println!("{file}:{}: {}", r.line, r.kind);
            *totals.refusals.entry(r.kind).or_insert(0) += 1;
        }
    }
    print_totals(&totals);
    Ok(())
}

fn print_totals(t: &Totals) {
    println!("files read: {}", t.files);
    println!("tokens: {}", t.tokens);
    println!("identity failures: {}", t.identity_failures);
    println!("function bodies read: {}", t.functions);
    println!("statements per kind:");
    t.statements
        .iter()
        .for_each(|(k, n)| println!("  {k}: {n}"));
    println!("rewritten per kind:");
    t.rewritten.iter().for_each(|(k, n)| println!("  {k}: {n}"));
    println!("left untouched per kind:");
    t.refusals.iter().for_each(|(k, n)| println!("  {k}: {n}"));
}
