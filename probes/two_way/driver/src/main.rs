//! `raise [@FILE]... --entry FILE --kernel NAME... --root DIR... [-I DIR]... [-D NAME[=V]]...
//! --cxx PATH [--std c++20] --trace-include DIR --ir DIR --out DIR [--keep]`

use std::process::ExitCode;

use raise::pipeline::run;
use raise::request::{expand_response, parse};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = expand_response(&args)
        .and_then(|a| parse(&a))
        .and_then(|req| run(&req));
    match result {
        Ok(s) => {
            println!(
                "raised {} kernels; {} statements left untouched (see left.txt); {} of {} closure files were project files",
                s.kernels, s.left, s.project, s.closure
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
