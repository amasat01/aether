//! One test per construct the rewriter knows, and per construct it leaves and reports.

use reader::{analyse, Analysis};

/// Analyses `body` placed inside a function.
fn run(body: &str) -> Analysis {
    analyse(&format!("void f() {{\n{body}\n}}\n"))
}

fn kinds(a: &Analysis) -> Vec<&'static str> {
    a.reports.iter().map(|r| r.kind).collect()
}

/// Collapses whitespace so assertions do not depend on layout.
fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn if_else_becomes_a_branch_with_two_callables() {
    let a = run("if (v > lim) { y = 1; } else { y = 2; }");
    assert!(flat(&a.output)
        .contains("aether::trace::detail::branch(v > lim, [&] { y = 1; }, [&] { y = 2; });"));
    assert_eq!(a.rewritten_count("if"), 1);
}

#[test]
fn if_without_else_gets_an_empty_second_callable() {
    let a = run("if (c) { y = 1; }");
    assert!(flat(&a.output).contains("branch(c, [&] { y = 1; }, [&] {});"));
}

#[test]
fn else_if_nests_inside_the_else_callable() {
    let a = run("if (a) { x = 1; } else if (b) { x = 2; } else { x = 3; }");
    let out = flat(&a.output);
    assert!(out.contains(
        "branch(a, [&] { x = 1; }, [&] { aether::trace::detail::branch(b, [&] { x = 2; }, [&] { x = 3; }); });"
    ));
    assert_eq!(a.rewritten_count("if"), 2);
}

#[test]
fn braceless_bodies_are_accepted() {
    let a = run("if (c) x = 1; else x = 2;");
    assert!(flat(&a.output).contains("branch(c, [&] { x = 1; }, [&] { x = 2; });"));
}

#[test]
fn if_constexpr_is_left_alone() {
    let a = run("if constexpr (N > 1) { x = 1; }");
    assert!(a.output.contains("if constexpr (N > 1) { x = 1; }"));
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(a.statements["if-constexpr"], 1);
}

#[test]
fn if_with_an_init_statement_is_left_and_reported() {
    let a = run("if (int t = g(); t > 0) { x = t; }");
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["if-init"]);
}

#[test]
fn if_declaring_its_condition_variable_is_left_and_reported() {
    let a = run("if (auto p = find(k)) { x = *p; }");
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["if-condition-declaration"]);
}

#[test]
fn a_condition_comparing_with_double_equals_is_not_a_declaration() {
    assert_eq!(run("if (a == b) { x = 1; }").rewritten_count("if"), 1);
}

#[test]
fn for_with_prefix_increment_becomes_a_loop_naming_the_assigned_variables() {
    let a = run("Real root = v; for (int i = 0; i < 8; ++i) { root = root * 0.5; }");
    assert!(flat(&a.output).contains(
        "aether::trace::detail::loop(0, 8, 1, \"i\", {{&root, \"root\"}}, [&]([[maybe_unused]] auto i, [[maybe_unused]] auto& loopExit) { root = root * 0.5; });"
    ));
    assert_eq!(a.rewritten_count("for"), 1);
}

#[test]
fn for_with_postfix_increment_and_stepped_forms_become_loops() {
    let post = run("for (int i = 2; i < n; i++) { s += i; }");
    assert!(flat(&post.output).contains("loop(2, n, 1, \"i\", {{&s, \"s\"}}"));
    let stepped = run("for (std::size_t i = 0; i < n; i += 4) { s += i; }");
    assert!(flat(&stepped.output).contains("loop(0, n, 4, \"i\", {{&s, \"s\"}}"));
}

#[test]
fn names_list_has_every_statement_level_assignment_not_declared_in_the_body() {
    let a = run(
        "for (int i = 0; i < 4; ++i) { Real t = a; t = t * 2; x += t; if (c) { y = 1; } z *= 2; x -= 1; }",
    );
    let out = flat(&a.output);
    assert!(
        out.contains("{{&x, \"x\"}, {&y, \"y\"}, {&z, \"z\"}}"),
        "{out}"
    );
    assert!(!out.contains("&t"));
}

#[test]
fn assignments_in_nested_loops_count_for_the_outer_loop() {
    let a = run("for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) { acc += j; } }");
    let out = flat(&a.output);
    assert!(
        out.contains("loop(0, 4, 1, \"i\", {{&acc, \"acc\"}}"),
        "{out}"
    );
    assert!(
        out.contains("loop(0, 4, 1, \"j\", {{&acc, \"acc\"}}"),
        "{out}"
    );
    assert_eq!(a.rewritten_count("for"), 2);
}

#[test]
fn if_break_with_a_block_becomes_a_loop_exit_returning_from_the_body() {
    let a = run("for (int i = 0; i < 8; ++i) { r = r + 1; if (r > 3) { done = r; break; } }");
    let out = flat(&a.output);
    assert!(
        out.contains("if (loopExit.when(r > 3)) { done = r; return; }"),
        "{out}"
    );
    assert!(out.contains("{&r, \"r\"}, {&done, \"done\"}"), "{out}");
}

#[test]
fn braceless_if_break_becomes_a_loop_exit() {
    let a = run("for (int i = 0; i < 8; ++i) { r = r + 1; if (r > 3) break; }");
    assert!(flat(&a.output).contains("if (loopExit.when(r > 3)) return;"));
}

#[test]
fn for_with_a_braceless_body_is_braced() {
    let a = run("for (int i = 0; i < 3; ++i) s += i;");
    assert!(flat(&a.output)
        .contains("[&]([[maybe_unused]] auto i, [[maybe_unused]] auto& loopExit) { s += i; });"));
}

#[test]
fn a_for_whose_header_is_not_the_known_form_is_left_and_reported() {
    for header in [
        "int i = 0; i <= 8; ++i",
        "int i = 0; i < 8; i *= 2",
        "Real x = 0; x < 1; ++x",
        ";;",
        "int i = 0, j = 5; i < 4; ++i, --j",
    ] {
        let a = run(&format!("for ({header}) {{ s += 1; }}"));
        assert_eq!(a.rewritten_count("for"), 0, "{header}");
        assert_eq!(kinds(&a), ["for-header"], "{header}");
    }
}

#[test]
fn range_based_for_is_left_alone_and_not_reported() {
    let a = run("for (auto& e : items) { s += e; }");
    assert_eq!(a.rewritten_count("for"), 0);
    assert_eq!(a.reports, []);
    assert_eq!(a.statements["for-range"], 1);
}

#[test]
fn an_if_inside_a_loop_body_is_rewritten_inside_the_loop_callable() {
    let a = run("for (int i = 0; i < 4; ++i) { if (c) { s = 1; } else { s = 2; } }");
    assert!(flat(&a.output)
        .contains("{ aether::trace::detail::branch(c, [&] { s = 1; }, [&] { s = 2; }); }"));
}

#[test]
fn a_break_that_is_not_the_exit_form_leaves_the_loop_and_is_reported() {
    let a = run("for (int i = 0; i < 4; ++i) { if (c) { s = 1; break; } else { s = 2; } }");
    assert_eq!(a.rewritten_count("for"), 0);
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["break-not-exit"]);
}

#[test]
fn a_return_at_the_top_level_of_the_body_is_not_reported() {
    assert_eq!(run("x = 1; return;").reports, []);
}

#[test]
fn a_return_inside_a_loop_leaves_the_loop_and_is_reported() {
    let a = run("for (int i = 0; i < 4; ++i) { s += i; return; }");
    assert_eq!(a.rewritten_count("for"), 0);
    assert_eq!(kinds(&a), ["return-in-loop"]);
}

#[test]
fn continue_and_goto_are_reported_and_their_loop_is_left() {
    let a = run("for (int i = 0; i < 4; ++i) { if (c) continue; s += i; }\ngoto out;\nout: ;");
    assert_eq!(a.rewritten_count("for"), 0);
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["continue", "goto"]);
}

#[test]
fn try_is_reported() {
    let a = run("try { x = 1; } catch (...) { x = 2; }");
    assert_eq!(kinds(&a), ["try"]);
}

#[test]
fn a_control_statement_cut_by_a_preprocessor_conditional_is_left_and_reported() {
    let a = run("if (a) {\n  x = 1;\n}\n#if FLAG\nelse {\n  x = 2;\n}\n#endif");
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["cut-by-preprocessor"]);
}

#[test]
fn a_header_split_by_a_preprocessor_conditional_is_left_and_reported() {
    let a = run("if (a\n#if FLAG\n && b\n#endif\n) { x = 1; }");
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(kinds(&a), ["cut-by-preprocessor"]);
}

#[test]
fn a_whole_conditional_inside_a_braced_body_does_not_cut_the_statement() {
    let a = run("if (a) {\n#if FLAG\n  x = 1;\n#endif\n}");
    assert_eq!(a.rewritten_count("if"), 1);
    assert!(a.output.contains("#if FLAG"));
    assert_eq!(a.reports, []);
}

#[test]
fn text_outside_function_bodies_is_copied_byte_for_byte() {
    let src = "#define M(c) if (c) { x = 1; }\nstruct S { int a{0}; static constexpr int k = 3; };\nconst char* t = \"if (a) { b; }\";\nint g[] = { 1, 2 };\nvoid f() { if (a) { b = 1; } }\n";
    let a = analyse(src);
    let tail = "void f() {";
    let at = src.find(tail).unwrap();
    assert_eq!(&a.output[..at], &src[..at]);
    assert_eq!(a.functions, 1);
}

#[test]
fn member_functions_in_classes_and_namespaces_are_rewritten() {
    let src = "namespace n { template <class T> struct S { T get(T v) const noexcept { if (v) { r = 1; } return r; }\n auto h() -> int { if (v) { r = 2; } return 0; } }; }";
    let a = analyse(src);
    assert_eq!(a.functions, 2);
    assert_eq!(a.rewritten_count("if"), 2);
}

#[test]
fn a_lambda_body_inside_a_function_is_read_and_rewritten() {
    let a = run("auto g = [&](int k) { if (k) { x = 1; } };\ng(1);");
    assert_eq!(a.rewritten_count("if"), 1);
    assert!(a
        .output
        .contains("auto g = [&](int k) { aether::trace::detail::branch(k,"));
}

#[test]
fn a_brace_inside_a_string_does_not_end_a_body() {
    let a = run("s = \"}\"; if (c) { t = \"{\"; }");
    assert_eq!(a.rewritten_count("if"), 1);
    assert!(a.output.contains("t = \"{\";"));
}

#[test]
fn a_raw_string_holding_control_text_is_not_read_as_statements() {
    let a = run("s = R\"x(if (a) { b = 1; } else { })x\";");
    assert_eq!(a.rewritten_count("if"), 0);
    assert_eq!(a.statements["expr"], 1);
}

#[test]
fn comment_slashes_inside_a_string_do_not_hide_the_condition() {
    let a = run("if (s == \"//\") { x = 1; } // tail");
    assert!(flat(&a.output).contains("branch(s == \"//\", [&] { x = 1; }, [&] {}); // tail"));
}

#[test]
fn an_if_inside_a_comment_is_not_a_statement() {
    let a = run("// if (a) { b; }\n/* if (c) { d; } */ x = 1;");
    assert_eq!(a.rewritten_count("if"), 0);
}

#[test]
fn a_function_with_unbalanced_braces_is_reported_and_copied_untouched() {
    let src = "void f() {\n#if A\n if (a) {\n#else\n if (b) {\n#endif\n x = 1; }\n}\nvoid g() { if (c) { y = 1; } }\n";
    let a = analyse(src);
    assert!(a.reports.iter().any(|r| r.kind.starts_with("unbalanced")));
    assert_eq!(a.output, src, "an unreadable body is never guessed at");
}

#[test]
fn statement_counts_name_every_kind_read() {
    let a =
        run("if (a) { x = 1; }\nfor (int i = 0; i < 2; ++i) { y += i; }\nwhile (b) { }\nreturn;");
    assert_eq!(a.statements["if"], 1);
    assert_eq!(a.statements["for"], 1);
    assert_eq!(a.statements["while"], 1);
    assert_eq!(a.statements["return"], 1);
}

#[test]
fn an_early_return_that_cannot_be_brought_to_the_known_shape_is_left_and_reported() {
    // The rest of the body does not end in a returned value.
    let a = run("if (c) { return 1; } x = 2;");
    assert_eq!(a.rewritten_count("early-return"), 0);
    assert_eq!(kinds(&a), ["return-in-branch"]);
    // The return sits in an `else`, not at the top level of the body.
    let b = run("if (c) { x = 1; } else { return; } y = 2;");
    assert_eq!(b.rewritten_count("early-return"), 0);
    assert_eq!(kinds(&b), ["return-in-branch"]);
}

#[test]
fn an_early_return_of_a_value_becomes_a_choice_between_two_callables() {
    let a = run("Real t = v;\nif (t > w) { s = 1; return t - w; }\nreturn 0.0;");
    assert!(flat(&a.output).contains(
        "return aether::trace::detail::choose(t > w, [&] { s = 1; return t - w; }, [&] { return 0.0; });"
    ));
    assert_eq!(a.rewritten_count("early-return"), 1);
    assert_eq!(a.reports, []);
}

#[test]
fn early_returns_in_a_row_nest_in_the_second_callable() {
    let a = run("if (v > w) { return v - w; }\nif (-v > w) { return v + w; }\nreturn 0.0;");
    assert!(flat(&a.output).contains(
        "return aether::trace::detail::choose(v > w, [&] { return v - w; }, [&] { return aether::trace::detail::choose(-v > w, [&] { return v + w; }, [&] { return 0.0; }); });"
    ));
    assert_eq!(a.rewritten_count("early-return"), 2);
}

#[test]
fn a_braceless_early_return_is_accepted() {
    let a = run("if (c) return 1;\nreturn 2;");
    assert!(flat(&a.output).contains("choose(c, [&] { return 1; }, [&] { return 2; });"));
}

#[test]
fn an_early_return_of_nothing_becomes_a_branch() {
    let a = run("if (c) { s = 1; return; }\ns = 2;");
    assert!(flat(&a.output)
        .contains("aether::trace::detail::branch(c, [&] { s = 1; }, [&] { s = 2; });"));
    let b = run("if (c) { s = 1; return; }");
    assert!(flat(&b.output).contains("branch(c, [&] { s = 1; }, [&] {});"));
    assert_eq!(a.reports, []);
}

#[test]
fn a_bounded_while_becomes_a_loop_that_tests_its_condition_first() {
    let a = run("while (aether::bounded(abs(r * r - v) > 1e-12, 8)) { r = 0.5 * (r + v / r); }");
    assert!(flat(&a.output).contains(
        "aether::trace::detail::loop(0, 8, 1, \"it\", {{&r, \"r\"}}, [&]([[maybe_unused]] auto it, auto& loopExit) { if (loopExit.when(!(abs(r * r - v) > 1e-12))) { return; } r = 0.5 * (r + v / r); });"
    ));
    assert_eq!(a.rewritten_count("while"), 1);
    assert_eq!(a.reports, []);
}

#[test]
fn a_while_without_bounded_is_left_and_reported() {
    let a = run("while (abs(r) > 1) { r = r / 2; }");
    assert_eq!(a.rewritten_count("while"), 0);
    assert_eq!(kinds(&a), ["while"]);
}

#[test]
fn a_bounded_while_whose_body_breaks_is_left_and_reported() {
    let a = run("while (aether::bounded(c, 4)) { if (d) { break; } x += 1; }");
    assert_eq!(a.rewritten_count("while"), 0);
    assert_eq!(kinds(&a), ["while"]);
}

#[test]
fn bounded_with_a_wrong_argument_count_is_not_the_form() {
    let a = run("while (aether::bounded(c)) { x += 1; }");
    assert_eq!(kinds(&a), ["while"]);
}

#[test]
fn a_ternary_becomes_a_choice_of_two_returning_callables() {
    let a = run("x = a > b ? p : q;");
    assert!(flat(&a.output).contains(
        "x = aether::trace::detail::choose(a > b, [&] { return p; }, [&] { return q; });"
    ));
    assert_eq!(a.rewritten_count("ternary"), 1);
    assert_eq!(a.reports, []);
}

#[test]
fn a_ternary_takes_the_whole_condition_up_to_the_assignment() {
    let a = run("Real y = u > 0 || w < 1 ? u : w;");
    assert!(flat(&a.output).contains("Real y = aether::trace::detail::choose(u > 0 || w < 1,"));
}

#[test]
fn a_ternary_in_a_return_or_an_argument_is_delimited_by_its_boundary() {
    let r = run("return a ? b : c;");
    assert!(flat(&r.output).contains(
        "return aether::trace::detail::choose(a, [&] { return b; }, [&] { return c; });"
    ));
    let call = run("g(a ? b : c, d, e ? f : h);");
    assert!(flat(&call.output).contains(
        "g(aether::trace::detail::choose(a, [&] { return b; }, [&] { return c; }), d, aether::trace::detail::choose(e, [&] { return f; }, [&] { return h; }));"
    ));
    assert_eq!(call.rewritten_count("ternary"), 2);
}

#[test]
fn nested_ternaries_nest_their_choices() {
    let right = run("x = a ? b : c ? d : e;");
    assert!(flat(&right.output).contains(
        "choose(a, [&] { return b; }, [&] { return aether::trace::detail::choose(c, [&] { return d; }, [&] { return e; }); })"
    ));
    let left = run("x = a ? b ? 1 : 2 : 3;");
    assert!(flat(&left.output).contains(
        "choose(a, [&] { return aether::trace::detail::choose(b, [&] { return 1; }, [&] { return 2; }); }, [&] { return 3; })"
    ));
    assert_eq!(right.rewritten_count("ternary"), 2);
    assert_eq!(left.rewritten_count("ternary"), 2);
}

#[test]
fn a_ternary_in_parentheses_is_a_condition_of_a_wider_ternary() {
    let a = run("x = (a ? b : c) ? d : e;");
    assert_eq!(a.rewritten_count("ternary"), 2);
    assert_eq!(a.reports, []);
}

#[test]
fn a_ternary_whose_extent_is_not_certain_is_left_and_reported() {
    // A ternary in an `if` condition, and one without its `:`.
    let a = run("if (a ? b : c) { x = 1; }\ny = p ? q;");
    assert_eq!(a.rewritten_count("ternary"), 0);
    assert_eq!(kinds(&a), ["ternary", "ternary"]);
    assert!(a.output.contains("y = p ? q;"));
}

#[test]
fn a_switch_whose_cases_end_in_break_becomes_a_chain_of_branches() {
    let a = run("switch (k) {\n case 1: { x = 1; break; }\n case 2: { x = 2; break; }\n default: { x = 3; }\n}");
    let out = flat(&a.output);
    assert!(out.contains(
        "aether::trace::detail::branch(k == 1, [&] { x = 1; }, [&] { aether::trace::detail::branch(k == 2, [&] { x = 2; }, [&] { x = 3; }); });"
    ), "{out}");
    assert_eq!(a.rewritten_count("switch"), 1);
    assert_eq!(a.reports, []);
}

#[test]
fn a_switch_without_default_ends_in_an_empty_callable() {
    let a = run("switch (k) { case 1: { x = 1; break; } }");
    assert!(flat(&a.output).contains("branch(k == 1, [&] { x = 1; }, [&] {});"));
}

#[test]
fn braceless_cases_are_delimited_by_the_next_label() {
    let a = run("switch (k) { case 1: x = 1; y = 2; break; case 2: break; }");
    assert!(flat(&a.output).contains("branch(k == 1, [&] { x = 1; y = 2; }, [&] { aether::trace::detail::branch(k == 2, [&] {}, [&] {}); });"));
}

#[test]
fn a_compound_switch_key_or_case_value_is_parenthesised() {
    let a = run("switch (m & 3) { case A | B: { x = 1; break; } }");
    assert!(flat(&a.output).contains("branch((m & 3) == (A | B),"));
}

#[test]
fn a_switch_that_falls_through_or_returns_is_left_and_reported() {
    for body in [
        "case 1: case 2: { x = 1; break; }",
        "case 1: { x = 1; }  case 2: { x = 2; break; }",
        "case 1: { return; } default: { break; }",
        "default: { x = 1; break; } case 1: { x = 2; break; }",
        "case 1: { if (c) { break; } x = 1; break; }",
    ] {
        let a = run(&format!("switch (k) {{ {body} }}"));
        assert_eq!(a.rewritten_count("switch"), 0, "{body}");
        assert_eq!(kinds(&a), ["switch"], "{body}");
    }
}

#[test]
fn a_switch_statement_keeps_its_nested_statements_rewritten() {
    let a = run("switch (k) { case 1: { if (c) { x = 1; } break; } }");
    assert_eq!(a.rewritten_count("if"), 1);
    assert_eq!(a.rewritten_count("switch"), 1);
}

#[test]
fn do_and_goto_stay_untouched_and_are_reported_with_their_lines() {
    let a = run("do { x = 2; } while (b);\ngoto out;\nout: ;");
    assert_eq!(kinds(&a), ["do", "goto"]);
    let lines: Vec<_> = a.reports.iter().map(|r| r.line).collect();
    assert_eq!(lines, [2, 3]);
    assert!(a.output.contains("do { x = 2; } while (b);"));
}

#[test]
fn an_early_return_chain_holding_a_preprocessor_conditional_is_left_and_reported() {
    let a = run("if (c) { return; }\n#if A\n x = 1;\n#else\n x = 2;\n#endif");
    assert_eq!(a.rewritten_count("early-return"), 0);
    assert_eq!(kinds(&a), ["return-in-branch"]);
}

#[test]
fn text_between_an_early_return_and_the_rest_is_kept() {
    let a = run("if (c) { return; }\n// why\nx = 1;");
    assert!(a.output.contains("// why"));
}

#[test]
fn a_ternary_in_a_constant_expression_is_left_and_reported() {
    let a = run("static_assert(K > 1 ? true : false);");
    assert_eq!(a.rewritten_count("ternary"), 0);
    assert_eq!(kinds(&a), ["ternary"]);
}

#[test]
fn a_ternary_split_by_a_preprocessor_conditional_is_left_and_reported() {
    let a = run("x = a ?\n#if A\n b\n#else\n c\n#endif\n : d;");
    assert_eq!(a.rewritten_count("ternary"), 0);
    assert_eq!(kinds(&a), ["ternary"]);
}

#[test]
fn a_switch_holding_a_preprocessor_conditional_is_left_and_reported() {
    let a = run("switch (k) {\n#if A\n case 1: { x = 1; break; }\n#endif\n}");
    assert_eq!(a.rewritten_count("switch"), 0);
    assert_eq!(kinds(&a), ["switch"]);
}

#[test]
fn an_early_return_of_a_value_in_a_function_returning_a_reference_is_left_and_reported() {
    let a = analyse("T& f(T& o) { if (this == &o) { return *this; } return o; }");
    assert_eq!(a.rewritten_count("early-return"), 0);
    assert_eq!(kinds(&a), ["return-in-branch"]);
    let by_value = analyse("T f(T o) { if (c) { return o; } return o; }");
    assert_eq!(by_value.rewritten_count("early-return"), 1);
}

#[test]
fn signatures_give_qualified_names_and_parameter_names() {
    let src = "namespace a::b {\ninline void k(PerSample<Real> v, Uniform<std::array<int, 3>> w = 0, Mutable<Real>) { x = 1; }\nstruct S { int m(int p) const { return p; } };\n}\nAETHER_DEVICEHOST() inline Real g(Real q) noexcept { return q; }\nvoid ns::h(int z) { }";
    let sigs = reader::signatures(src);
    let got: Vec<(&str, Vec<&str>)> = sigs
        .iter()
        .map(|s| {
            (
                s.name.as_str(),
                s.params.iter().map(|p| p.name.as_str()).collect(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("a::b::k", vec!["v", "w", ""]),
            ("a::b::S::m", vec!["p"]),
            ("g", vec!["q"]),
            ("ns::h", vec!["z"]),
        ]
    );
}

/// The line each line of `output` has once the `#line` directives are applied, by content.
fn lines_after_directives(output: &str) -> Vec<(usize, String)> {
    let mut at = 1;
    let mut out = Vec::new();
    for line in output.lines() {
        if let Some(n) = line.strip_prefix("#line ") {
            at = n.trim().parse().unwrap();
        } else {
            out.push((at, line.trim().to_string()));
            at += 1;
        }
    }
    out
}

#[test]
fn a_line_marker_follows_a_rewrite_that_changes_the_line_count() {
    let src =
        "void f() {\n  if (c)\n  {\n    x = 1;\n  }\n  else\n  {\n    x = 2;\n  }\n  z = 3;\n}\n";
    let a = analyse(src);
    assert!(a.output.contains("#line "), "{}", a.output);
    let lines = lines_after_directives(&a.output);
    let line_of = |text: &str| lines.iter().find(|(_, l)| l.contains(text)).unwrap().0;
    assert_eq!(line_of("z = 3;"), 10);
    assert_eq!(line_of("x = 2;"), 8);
}

#[test]
fn text_that_keeps_its_line_count_gets_no_marker() {
    let src = "void f() {\n  x = 1;\n  if (c) {\n    y = 2;\n  }\n  z = 3;\n}\n";
    assert!(!analyse(src).output.contains("#line"));
    let plain = "int g;\n// c\n";
    assert_eq!(analyse(plain).output, plain);
}

#[test]
fn a_marker_never_lands_inside_a_literal_or_a_directive() {
    let src = "void f() {\n  if (c)\n  {\n    x = 1;\n  }\n  else\n  {\n    x = 2;\n  }\n  s = R\"(a\nb)\";\n#define M(x) \\\n  x\n  z = 3;\n}\n";
    let a = analyse(src);
    assert!(a.output.contains("R\"(a\nb)\""));
    assert!(a.output.contains("#define M(x) \\\n  x\n"));
}

/// The `(name, role, type)` of each parameter of the one function in `src`.
fn roles(src: &str) -> Vec<(String, Option<&'static str>, String)> {
    let sigs = reader::signatures(src);
    sigs[0]
        .params
        .iter()
        .map(|p| (p.name.clone(), p.role, p.type_text.clone()))
        .collect()
}

#[test]
fn a_role_is_read_from_the_type_with_or_without_the_namespace() {
    let got = roles(
        "void k(aether::PerSample<Real> a, Uniform<Real> b, aether::Mutable<Real> c, ::aether::Uniform<Real> d) { }",
    );
    let roles: Vec<_> = got.iter().map(|g| g.1).collect();
    assert_eq!(
        roles,
        [
            Some("per_sample"),
            Some("uniform"),
            Some("mutable"),
            Some("uniform")
        ]
    );
    assert_eq!(got[0].2, "aether::PerSample<Real>");
}

#[test]
fn a_const_or_a_default_argument_does_not_hide_the_role() {
    let got = roles("void k(const PerSample<Real> a, Uniform<Real> b = 1.0, Mutable<Real> c) { }");
    assert_eq!(
        got[0],
        (
            "a".into(),
            Some("per_sample"),
            "const PerSample<Real>".into()
        )
    );
    assert_eq!(
        got[1],
        ("b".into(), Some("uniform"), "Uniform<Real>".into())
    );
}

#[test]
fn a_parameter_list_over_several_lines_reads_each_parameter_and_its_line() {
    let src = "void k(\n    PerSample<Real> a,\n    // the limit\n    Uniform<Real> b,\n    Mutable<Real> c) { }";
    let sigs = reader::signatures(src);
    let lines: Vec<usize> = sigs[0].params.iter().map(|p| p.line).collect();
    assert_eq!(lines, [2, 4, 5]);
    assert_eq!(roles(src)[2].1, Some("mutable"));
}

#[test]
fn a_type_that_is_no_role_gives_none_even_if_an_argument_names_one() {
    let got = roles("void k(Real a, std::array<PerSample<Real>, 2> b, Speed c, int) { }");
    let roles: Vec<_> = got.iter().map(|g| g.1).collect();
    assert_eq!(roles, [None, None, None, None]);
    assert_eq!(got[2].2, "Speed");
    assert_eq!(got[3], (String::new(), None, "int".into()));
}

#[test]
fn nested_template_arguments_keep_the_outer_role() {
    let got =
        roles("void k(PerSample<std::array<Real, 3>> a, Mutable<std::pair<int, Real>> b) { }");
    assert_eq!(
        got[0],
        (
            "a".into(),
            Some("per_sample"),
            "PerSample<std::array<Real, 3>>".into()
        )
    );
    assert_eq!(got[1].1, Some("mutable"));
}
