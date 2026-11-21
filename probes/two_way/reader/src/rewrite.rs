//! The rewriter and the census over the statement tree. A statement is rewritten only when it is
//! exactly a form that is known; every other statement is copied byte for byte (its nested
//! statements may still be rewritten) and, when it matters, reported with its line.

use std::collections::{BTreeMap, HashSet};

use crate::read::{
    is_name, read_functions, Condition, Ctl, Expr, ForStmt, Function, IfStmt, Kind, ParseError,
    Source, Stmt, Ternary, ASSIGN_OPS,
};

/// A construct that was left untouched or could not be read, with where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// 1-based line.
    pub line: usize,
    /// The construct, such as `while`, `switch`, `ternary` or `return-in-branch`.
    pub kind: &'static str,
    /// The function the construct is in; empty when the body could not be read.
    pub function: String,
    /// First and last line of the code that is left as written, where a compiler may then
    /// complain: the statement, or for a `return` the `if` around it.
    pub span: (usize, usize),
}

impl Report {
    /// What the author can do about the construct.
    #[must_use]
    pub fn way_out(&self) -> &'static str {
        match self.kind {
            "ternary" => {
                "a `?:` whose operands cannot be delimited is left; write it as an if that assigns in each branch"
            }
            "ternary-as-lvalue" => {
                "a `?:` that is assigned to, addressed or bound to a reference is left; write it as an if that assigns in each branch"
            }
            "return-in-branch" => {
                "a `return` that cannot move into the branch callables (not at the top level of the function body, no value on every path, or a function that returns a reference) is left; assign to a local in the branches and return it once at the end"
            }
            "return-in-loop" => {
                "a `return` inside a loop is left; assign to a local, leave through the loop's own exit and return after the loop"
            }
            "cut-by-preprocessor" => {
                "a statement cut by a preprocessor conditional is left; keep each `#if` branch around whole statements"
            }
            "while" => "a `while` without `aether::bounded(condition, bound)` is left; state its bound",
            "switch" | "switch-return-and-break" => {
                "a switch the reader cannot write is left; end every case in `break`, or every case in `return` of a value, with `default` last"
            }
            _ => "the statement is left as written; rewrite it in a form the reader knows",
        }
    }
}

/// The result of reading one file.
#[derive(Debug)]
pub struct Analysis {
    /// The rewritten text.
    pub output: String,
    /// Number of tokens, whitespace and comments included.
    pub tokens: usize,
    /// Function bodies read.
    pub functions: usize,
    /// Statements read, per kind.
    pub statements: BTreeMap<&'static str, usize>,
    /// What was left untouched or unreadable, in source order.
    pub reports: Vec<Report>,
    /// Statements rewritten into a recording call, per kind (`if`, `for`, `while`, `switch`,
    /// `ternary`, `early-return`).
    pub rewritten: BTreeMap<&'static str, usize>,
}

impl Analysis {
    /// How many statements of `kind` were rewritten.
    #[must_use]
    pub fn rewritten_count(&self, kind: &str) -> usize {
        self.rewritten.get(kind).copied().unwrap_or(0)
    }
}

/// Reads `text`, rewrites what it can inside function bodies and reports the rest.
#[must_use]
pub fn analyse(text: &str) -> Analysis {
    let src = Source::new(text);
    let (funcs, errors) = read_functions(&src);
    let mut census = Census {
        src: &src,
        statements: BTreeMap::new(),
        reports: errors.iter().map(parse_report).collect(),
        consumed: HashSet::new(),
        function: String::new(),
        current: (0, 0),
        if_spans: Vec::new(),
    };
    let mut exits = HashSet::new();
    for f in &funcs {
        mark_exits(&f.body, &mut exits);
        census.mark_returns(f);
    }
    for f in &funcs {
        census.function.clone_from(&f.name);
        census.visit(&f.body, &mut Vec::new(), &exits);
    }
    census.reports.sort_by_key(|r| r.line);
    let mut renderer = Renderer {
        src: &src,
        rewritten: BTreeMap::new(),
    };
    let output = mark_lines(&renderer.top(&funcs));
    Analysis {
        output,
        tokens: src.tokens.len(),
        functions: funcs.len(),
        statements: census.statements,
        reports: census.reports,
        rewritten: renderer.rewritten,
    }
}

fn parse_report(e: &ParseError) -> Report {
    Report {
        line: e.line,
        kind: e.message,
        function: String::new(),
        span: (e.line, e.line),
    }
}

// ---- what a statement lets escape ----------------------------------------------------------

/// True when `s` can leave the callable it would be wrapped into: a `return` or `goto`, a
/// `break` that is not caught by a loop or `switch` inside `s`, a `continue` not caught by a loop
/// inside `s`.
fn escapes(s: &Stmt, in_loop: bool, in_switch: bool) -> bool {
    match &s.kind {
        Kind::Break => !in_loop && !in_switch,
        Kind::Continue => !in_loop,
        Kind::Return(_) | Kind::Goto => true,
        Kind::Block(v) | Kind::Try(v) => v.iter().any(|c| escapes(c, in_loop, in_switch)),
        Kind::If(i) => {
            escapes(&i.then_, in_loop, in_switch)
                || i.else_
                    .as_deref()
                    .is_some_and(|e| escapes(e, in_loop, in_switch))
        }
        Kind::For(f) => escapes(&f.ctl.body, true, in_switch),
        Kind::While(c) | Kind::Do(c) => escapes(&c.body, true, in_switch),
        Kind::Switch(c) => escapes(&c.body, in_loop, true),
        Kind::Label | Kind::Expr(_) => false,
    }
}

fn if_convertible(i: &IfStmt) -> bool {
    !i.constexpr
        && i.condition == Condition::Plain
        && !i.cut
        && !escapes(&i.then_, false, false)
        && !i.else_.as_deref().is_some_and(|e| escapes(e, false, false))
}

/// The `break` that makes `s` a loop exit: `if (C) break;` or `if (C) {S; break;}` with no
/// `else` and nothing else in `S` that escapes.
fn exit_break(s: &Stmt) -> Option<&Stmt> {
    let Kind::If(i) = &s.kind else {
        return None;
    };
    if i.constexpr || i.condition != Condition::Plain || i.cut || i.else_.is_some() {
        return None;
    }
    match &i.then_.kind {
        Kind::Break => Some(&i.then_),
        Kind::Block(v) => v
            .split_last()
            .filter(|(last, rest)| {
                matches!(last.kind, Kind::Break) && !rest.iter().any(|c| escapes(c, false, false))
            })
            .map(|(last, _)| last),
        _ => None,
    }
}

/// The statements of a loop body; a braceless body is one statement.
fn body_stmts(body: &Stmt) -> &[Stmt] {
    match &body.kind {
        Kind::Block(v) => v,
        _ => std::slice::from_ref(body),
    }
}

fn loop_convertible(f: &ForStmt) -> bool {
    f.header.is_some()
        && !f.ctl.cut
        && body_stmts(&f.ctl.body)
            .iter()
            .all(|c| exit_break(c).is_some() || !escapes(c, false, false))
}

/// A `while (aether::bounded(C, N))` whose body leaves only by running out.
fn while_convertible(c: &Ctl) -> bool {
    c.bound.is_some() && !c.cut && !body_stmts(&c.body).iter().any(|s| escapes(s, false, false))
}

/// Records the start of every `break` that is the exit of a convertible loop.
fn mark_exits(s: &Stmt, exits: &mut HashSet<usize>) {
    if let Kind::For(f) = &s.kind {
        if loop_convertible(f) {
            exits.extend(
                body_stmts(&f.ctl.body)
                    .iter()
                    .filter_map(exit_break)
                    .map(|b| b.start),
            );
        }
    }
    for c in s.children() {
        mark_exits(c, exits);
    }
}

// ---- early returns -------------------------------------------------------------------------

fn returns_value(src: &Source, s: &Stmt) -> bool {
    src.words_in(s.start, s.end).len() > 2
}

/// The shape of one statement `if (C) { S; return E; }` (or `return;`) with no `else`: whether
/// it returns a value.
fn early_shape(src: &Source, s: &Stmt) -> Option<bool> {
    let Kind::If(i) = &s.kind else {
        return None;
    };
    if i.constexpr || i.condition != Condition::Plain || i.cut || i.else_.is_some() {
        return None;
    }
    let then: Vec<&Stmt> = match &i.then_.kind {
        Kind::Block(v) => v.iter().collect(),
        _ => vec![&*i.then_],
    };
    let (last, before) = then.split_last()?;
    let ends_in_return = matches!(last.kind, Kind::Return(_));
    let quiet = !before.iter().any(|c| escapes(c, false, false));
    (ends_in_return && quiet).then(|| returns_value(src, last))
}

/// True when the statements after an early return can move into the other callable: they
/// reach the end of the function by returning a value (or, for a void function, by not
/// returning one), through plain statements and further early returns.
fn tail_ok(src: &Source, rest: &[&Stmt], value: bool) -> bool {
    for (r, s) in rest.iter().enumerate() {
        if let Some(v) = early_shape(src, s) {
            return v == value && tail_ok(src, &rest[r + 1..], value);
        }
        if r + 1 == rest.len() && matches!(s.kind, Kind::Return(_)) {
            return returns_value(src, s) == value;
        }
        if escapes(s, false, false) {
            return false;
        }
    }
    !value
}

/// Whether `stmts[j]` starts an early-return chain that reaches the end of the function body:
/// `Some(true)` for a value, `Some(false)` for a void function.
fn early_return(src: &Source, stmts: &[&Stmt], j: usize, by_reference: bool) -> Option<bool> {
    let value = early_shape(src, stmts[j])?;
    // A callable returns by value, which would copy what a reference-returning function hands out.
    if value && by_reference {
        return None;
    }
    // Statements are moved into a callable; a conditional directive among them could be
    // split from its partner.
    let last = stmts[stmts.len() - 1];
    if src.has_conditional(stmts[j].start, last.end) {
        return None;
    }
    tail_ok(src, &stmts[j + 1..], value).then_some(value)
}

// ---- switch --------------------------------------------------------------------------------

/// One `case` (or the `default`) of a switch that ends in `break`.
struct Group<'a> {
    /// Byte range of the case value; `None` for `default`.
    value: Option<(usize, usize)>,
    /// The statements of the case before its `break`.
    stmts: Vec<&'a Stmt>,
    /// The `break`; the last case, `default`, may end without one.
    brk: Option<&'a Stmt>,
    /// The block that holds the whole case, when it is braced.
    block: Option<&'a Stmt>,
}

/// Reads a switch whose every case ends in `break` and holds nothing else that escapes.
fn switch_plan<'a>(src: &Source, c: &'a Ctl) -> Option<Vec<Group<'a>>> {
    let Kind::Block(items) = &c.body.kind else {
        return None;
    };
    if c.cut || src.has_conditional(c.body.start, c.body.end) {
        return None;
    }
    let mut groups = Vec::new();
    let mut at = 0;
    while at < items.len() {
        let label = &items[at];
        let words = src.words_in(label.start, label.end);
        let value = match (&label.kind, words.first().copied()) {
            (Kind::Label, Some("case")) => Some((label.start + 4, label.end - 1)),
            (Kind::Label, Some("default")) if words.len() == 2 => None,
            _ => return None,
        };
        let next = items[at + 1..]
            .iter()
            .position(|s| matches!(s.kind, Kind::Label))
            .map_or(items.len(), |p| at + 1 + p);
        if value.is_none() && next != items.len() {
            return None;
        }
        let body: Vec<&Stmt> = items[at + 1..next].iter().collect();
        groups.push(case_group(value, &body)?);
        at = next;
    }
    (!groups.is_empty()).then_some(groups)
}

fn case_group<'a>(value: Option<(usize, usize)>, body: &[&'a Stmt]) -> Option<Group<'a>> {
    let (block, inner): (Option<&Stmt>, Vec<&Stmt>) = match body {
        [b] => match &b.kind {
            Kind::Block(v) => (Some(*b), v.iter().collect()),
            _ => (None, body.to_vec()),
        },
        _ => (None, body.to_vec()),
    };
    let (last, before) = inner.split_last()?;
    let (brk, stmts) = if matches!(last.kind, Kind::Break) {
        (Some(*last), before.to_vec())
    } else if value.is_none() {
        (None, inner)
    } else {
        return None;
    };
    let quiet = !stmts.iter().any(|s| escapes(s, false, false));
    quiet.then_some(Group {
        value,
        stmts,
        brk,
        block,
    })
}

// ---- switch whose cases return ---------------------------------------------------------------

/// What a `case` or `default` label says.
enum CaseLabel {
    /// `case value:` with the byte range of the value.
    Value((usize, usize)),
    /// `default:`.
    Default,
}

/// The `case` or `default` a label statement is; `None` for any other label.
fn label_value(src: &Source, label: &Stmt) -> Option<CaseLabel> {
    let words = src.words_in(label.start, label.end);
    match (&label.kind, words.first().copied()) {
        (Kind::Label, Some("case")) => Some(CaseLabel::Value((label.start + 4, label.end - 1))),
        (Kind::Label, Some("default")) if words.len() == 2 => Some(CaseLabel::Default),
        _ => None,
    }
}

/// A run of labels and the statements under them, up to the next label.
struct LabelRun<'a> {
    labels: Vec<&'a Stmt>,
    body: Vec<&'a Stmt>,
}

/// The labelled runs of a switch body, or `None` when it holds anything before its first label.
fn label_runs<'a>(src: &Source, c: &'a Ctl) -> Option<Vec<LabelRun<'a>>> {
    let Kind::Block(items) = &c.body.kind else {
        return None;
    };
    let mut runs: Vec<LabelRun> = Vec::new();
    for item in items {
        let is_label = label_value(src, item).is_some();
        match (is_label, runs.last_mut()) {
            (true, Some(run)) if run.body.is_empty() => run.labels.push(item),
            (true, _) => runs.push(LabelRun {
                labels: vec![item],
                body: Vec::new(),
            }),
            (false, Some(run)) => run.body.push(item),
            (false, None) => return None,
        }
    }
    Some(runs)
}

/// The statements of a run's body, looking through a single braced block.
fn run_statements<'a>(run: &LabelRun<'a>) -> Vec<&'a Stmt> {
    match run.body.as_slice() {
        [only] => match &only.kind {
            Kind::Block(v) => v.iter().collect(),
            _ => run.body.clone(),
        },
        _ => run.body.clone(),
    }
}

/// One case of a switch whose cases all end in `return` of a value.
struct ReturnCase<'a> {
    /// Byte ranges of the case values; empty for `default`.
    values: Vec<(usize, usize)>,
    default: bool,
    /// The statements of the case, its `return` last.
    stmts: Vec<&'a Stmt>,
    /// The block that holds the whole case, when it is braced.
    block: Option<&'a Stmt>,
}

/// Reads a switch in which every case, including a fall-through group of labels, ends in
/// `return` of a value and holds nothing else that escapes; `default` may be missing but not
/// come before another case.
fn return_cases<'a>(src: &Source, c: &'a Ctl) -> Option<Vec<ReturnCase<'a>>> {
    if c.cut || src.has_conditional(c.body.start, c.body.end) {
        return None;
    }
    let runs = label_runs(src, c)?;
    let mut cases = Vec::new();
    for (n, run) in runs.iter().enumerate() {
        let stmts = run_statements(run);
        let (last, before) = stmts.split_last()?;
        let returns = matches!(last.kind, Kind::Return(_)) && returns_value(src, last);
        if !returns || before.iter().any(|s| escapes(s, false, false)) {
            return None;
        }
        let kinds: Vec<_> = run
            .labels
            .iter()
            .filter_map(|l| label_value(src, l))
            .collect();
        let default = kinds.iter().any(|k| matches!(k, CaseLabel::Default));
        if default && n + 1 != runs.len() {
            return None;
        }
        let block = match run.body.as_slice() {
            [only] if matches!(only.kind, Kind::Block(_)) => Some(*only),
            _ => None,
        };
        let values = kinds
            .into_iter()
            .filter_map(|k| match k {
                CaseLabel::Value(range) => Some(range),
                CaseLabel::Default => None,
            })
            .collect();
        cases.push(ReturnCase {
            values,
            default,
            stmts,
            block,
        });
    }
    (!cases.is_empty()).then_some(cases)
}

/// True when a switch has cases that end in `return` of a value and cases that end in `break`: both ways
/// out of one switch need the statements after it in each callable, which is not written.
fn mixes_return_and_break(src: &Source, c: &Ctl) -> bool {
    let Some(runs) = label_runs(src, c) else {
        return false;
    };
    let ends: Vec<_> = runs
        .iter()
        .filter_map(|r| {
            run_statements(r)
                .last()
                .map(|s| matches!(s.kind, Kind::Return(_)) && returns_value(src, s))
        })
        .collect();
    ends.contains(&true) && ends.contains(&false)
}

/// The cases of a function-level switch that can be written as a chain of choices: its cases
/// return values, the function does not return a reference, and either `default` ends the chain
/// or the statements `rest` after the switch reach a returned value.
fn return_switch<'a>(
    src: &Source,
    c: &'a Ctl,
    rest: &[&Stmt],
    by_reference: bool,
) -> Option<Vec<ReturnCase<'a>>> {
    let cases = return_cases(src, c).filter(|_| !by_reference)?;
    let ends_in_default = cases.iter().any(|k| k.default);
    (ends_in_default || tail_ok(src, rest, true)).then_some(cases)
}

// ---- the census ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Ancestor {
    If,
    /// A classic `for`; the flag tells whether its header is the convertible form.
    For(bool),
    /// A range-based `for`, `while`, `do` or `switch`: owns the `break`s inside it.
    Breakable,
}

struct Census<'s, 'a> {
    src: &'s Source<'a>,
    statements: BTreeMap<&'static str, usize>,
    reports: Vec<Report>,
    /// Starts of the `return`s that the early-return rewrite takes in.
    consumed: HashSet<usize>,
    /// The function being visited.
    function: String,
    /// First and last line of the statement being visited.
    current: (usize, usize),
    /// First and last line of each `if` around the statement being visited.
    if_spans: Vec<(usize, usize)>,
}

impl Census<'_, '_> {
    fn report(&mut self, line: usize, kind: &'static str) {
        self.report_span(line, self.current, kind);
    }

    fn report_span(&mut self, line: usize, span: (usize, usize), kind: &'static str) {
        let function = self.function.clone();
        self.reports.push(Report {
            line,
            kind,
            function,
            span,
        });
    }

    /// Notes the `return` that ends each early-return chain of a function body.
    fn mark_returns(&mut self, function: &Function) {
        let Kind::Block(v) = &function.body.kind else {
            return;
        };
        let stmts: Vec<&Stmt> = v.iter().collect();
        for j in 0..stmts.len() {
            if let Kind::Switch(c) = &stmts[j].kind {
                let rest = &stmts[j + 1..];
                if return_switch(self.src, c, rest, function.returns_reference).is_some() {
                    self.consumed.insert(stmts[j].start);
                }
            }
            if early_return(self.src, &stmts, j, function.returns_reference).is_some() {
                if let Kind::If(i) = &stmts[j].kind {
                    let last = match &i.then_.kind {
                        Kind::Block(b) => b.last(),
                        _ => Some(&*i.then_),
                    };
                    self.consumed.extend(last.map(|l| l.start));
                }
            }
        }
    }

    fn visit(&mut self, s: &Stmt, anc: &mut Vec<Ancestor>, exits: &HashSet<usize>) {
        *self.statements.entry(kind_name(s)).or_insert(0) += 1;
        let line = s.line;
        self.current = (line, self.src.line(s.end.saturating_sub(1)));
        match &s.kind {
            Kind::Block(v) | Kind::Try(v) => {
                if matches!(s.kind, Kind::Try(_)) {
                    self.report(line, "try");
                }
                for c in v {
                    self.visit(c, anc, exits);
                }
            }
            Kind::If(i) => self.visit_if(i, line, anc, exits),
            Kind::For(f) => {
                if !f.range && f.header.is_none() {
                    self.report(line, "for-header");
                }
                let own = if f.range {
                    Ancestor::Breakable
                } else {
                    Ancestor::For(f.header.is_some())
                };
                self.visit_ctl(&f.ctl, line, own, anc, exits);
            }
            Kind::While(c) => {
                if !while_convertible(c) {
                    self.report(line, "while");
                }
                self.visit_ctl(c, line, Ancestor::Breakable, anc, exits);
            }
            Kind::Do(c) => {
                self.report(line, "do");
                self.visit_ctl(c, line, Ancestor::Breakable, anc, exits);
            }
            Kind::Switch(c) => {
                if !self.consumed.contains(&s.start) && switch_plan(self.src, c).is_none() {
                    let mixed = mixes_return_and_break(self.src, c);
                    self.report(
                        line,
                        if mixed {
                            "switch-return-and-break"
                        } else {
                            "switch"
                        },
                    );
                }
                self.visit_ctl(c, line, Ancestor::Breakable, anc, exits);
            }
            Kind::Break => self.visit_break(line, s.start, anc, exits),
            Kind::Continue => self.report(line, "continue"),
            Kind::Goto => self.report(line, "goto"),
            Kind::Return(e) => {
                if self.consumed.contains(&s.start) {
                } else if anc.contains(&Ancestor::If) {
                    let span = self.if_spans.first().copied().unwrap_or(self.current);
                    self.report_span(line, span, "return-in-branch");
                } else if anc.iter().any(|a| matches!(a, Ancestor::For(_))) {
                    self.report(line, "return-in-loop");
                }
                self.visit_expr(e, exits, true);
            }
            Kind::Expr(e) => self.visit_expr(e, exits, true),
            Kind::Label => {}
        }
    }

    /// Notes the `?:` of a region that is left, and reads its lambda bodies as functions of
    /// their own. Only the expression of a statement or a `return` is rewritten.
    fn visit_expr(&mut self, e: &Expr, exits: &HashSet<usize>, rewritable: bool) {
        for t in &e.ternaries {
            *self.statements.entry("ternary").or_insert(0) += 1;
            if !(rewritable && self.src.ternary_extent(t).is_some()) {
                let lvalue = rewritable && self.src.ternary_is_lvalue(t);
                self.report(
                    t.line,
                    if lvalue {
                        "ternary-as-lvalue"
                    } else {
                        "ternary"
                    },
                );
            }
        }
        for l in &e.lambdas {
            self.visit(l, &mut Vec::new(), exits);
        }
    }

    fn visit_if(
        &mut self,
        i: &IfStmt,
        line: usize,
        anc: &mut Vec<Ancestor>,
        exits: &HashSet<usize>,
    ) {
        for (flag, kind) in [
            (i.condition == Condition::Init, "if-init"),
            (
                i.condition == Condition::Declaration,
                "if-condition-declaration",
            ),
            (i.cut, "cut-by-preprocessor"),
        ] {
            if flag {
                self.report(line, kind);
            }
        }
        self.visit_expr(&i.head, exits, false);
        anc.push(Ancestor::If);
        self.if_spans.push(self.current);
        self.visit(&i.then_, anc, exits);
        if let Some(e) = &i.else_ {
            self.visit(e, anc, exits);
        }
        anc.pop();
        self.if_spans.pop();
    }

    fn visit_ctl(
        &mut self,
        c: &Ctl,
        line: usize,
        own: Ancestor,
        anc: &mut Vec<Ancestor>,
        exits: &HashSet<usize>,
    ) {
        if c.cut {
            self.report(line, "cut-by-preprocessor");
        }
        self.visit_expr(&c.head, exits, false);
        anc.push(own);
        self.visit(&c.body, anc, exits);
        anc.pop();
    }

    /// A `break` is fine natively unless the loop it belongs to is a convertible `for` and it is
    /// not that loop's exit.
    fn visit_break(&mut self, line: usize, start: usize, anc: &[Ancestor], exits: &HashSet<usize>) {
        let owner = anc.iter().rev().find(|a| !matches!(a, Ancestor::If));
        if matches!(owner, Some(Ancestor::For(true))) && !exits.contains(&start) {
            self.report(line, "break-not-exit");
        }
    }
}

fn kind_name(s: &Stmt) -> &'static str {
    match &s.kind {
        Kind::Block(_) => "block",
        Kind::If(i) if i.constexpr => "if-constexpr",
        Kind::If(_) => "if",
        Kind::For(f) if f.range => "for-range",
        Kind::For(_) => "for",
        Kind::While(_) => "while",
        Kind::Do(_) => "do",
        Kind::Switch(_) => "switch",
        Kind::Try(_) => "try",
        Kind::Break => "break",
        Kind::Continue => "continue",
        Kind::Return(_) => "return",
        Kind::Goto => "goto",
        Kind::Label => "label",
        Kind::Expr(_) => "expr",
    }
}

// ---- the rewriter --------------------------------------------------------------------------

/// Where a run of statements sits, which decides what is rewritten at its top level.
#[derive(Clone, Copy, PartialEq)]
enum Level {
    Plain,
    /// The top level of a loop body: `if (C) break;` becomes the loop's exit.
    Loop,
    /// The top level of a function body: an early return takes in the statements after it.
    Function {
        /// The function's declared return type is a reference.
        by_reference: bool,
    },
}

struct Renderer<'s, 'a> {
    src: &'s Source<'a>,
    rewritten: BTreeMap<&'static str, usize>,
}

impl Renderer<'_, '_> {
    /// The source text `[from, to)`, with a marker after its first line break that tells where
    /// the next line stands in the source (see `mark_lines`).
    fn copy(&self, from: usize, to: usize) -> String {
        let text = &self.src.text[from..to];
        match self.src.line_break_in(from, to) {
            Some(at) => {
                let (head, tail) = text.split_at(at + 1 - from);
                let line = self.src.line(at + 1);
                format!("{head}{MARK}{line}{MARK_END}{tail}")
            }
            None => text.to_string(),
        }
    }

    fn count(&mut self, kind: &'static str) {
        *self.rewritten.entry(kind).or_insert(0) += 1;
    }

    /// The whole file: text outside function bodies is copied, bodies are rendered.
    fn top(&mut self, functions: &[Function]) -> String {
        let text = self.src.text;
        let mut out = String::new();
        let mut pos = 0;
        for f in functions {
            let b = &f.body;
            out.push_str(&self.copy(pos, b.start));
            let kids: Vec<&Stmt> = b.children();
            let level = Level::Function {
                by_reference: f.returns_reference,
            };
            out.push_str(&self.span(b.start, b.end, &kids, level));
            pos = b.end;
        }
        out.push_str(&self.copy(pos, text.len()));
        out
    }

    /// The text of `[from, to)` with each statement of `kids` that lies inside replaced by its
    /// rendering; `level` says what the top level of the run rewrites.
    fn span(&mut self, from: usize, to: usize, kids: &[&Stmt], level: Level) -> String {
        let mut out = String::new();
        let mut pos = from;
        for (j, k) in kids.iter().enumerate() {
            if k.start < from || k.end > to {
                continue;
            }
            out.push_str(&self.copy(pos, k.start));
            if let Level::Function { by_reference } = level {
                if let Some(value) = early_return(self.src, kids, j, by_reference) {
                    out.push_str(&self.render_early(&kids[j..], value, level));
                    pos = kids[kids.len() - 1].end;
                    break;
                }
                if let Kind::Switch(c) = &k.kind {
                    if let Some(cases) = return_switch(self.src, c, &kids[j + 1..], by_reference) {
                        let (text, consumed_rest) =
                            self.render_return_switch(c, &cases, &kids[j + 1..], level);
                        out.push_str(&text);
                        pos = if consumed_rest {
                            kids[kids.len() - 1].end
                        } else {
                            k.end
                        };
                        if consumed_rest {
                            break;
                        }
                        continue;
                    }
                }
            }
            out.push_str(&if level == Level::Loop && exit_break(k).is_some() {
                self.render_exit(k)
            } else {
                self.render(k)
            });
            pos = k.end;
        }
        out.push_str(&self.copy(pos, to));
        out
    }

    /// The rendering of one statement. Inside an open parenthesis or initialiser no directive can
    /// restore the line numbers, so a rewrite that lost lines gets them back as blank lines.
    fn render(&mut self, s: &Stmt) -> String {
        let text = self.render_kind(s);
        self.keep_line_count(s.start, s.end, text)
    }

    fn keep_line_count(&self, start: usize, end: usize, mut text: String) -> String {
        if self.src.open_at(start) > 0 {
            let wanted = self.src.text[start..end].matches('\n').count();
            let lost = wanted.saturating_sub(text.matches('\n').count());
            text.push_str(&"\n".repeat(lost));
        }
        text
    }

    fn render_kind(&mut self, s: &Stmt) -> String {
        match &s.kind {
            Kind::If(i) if if_convertible(i) => self.render_if(i),
            Kind::For(f) if loop_convertible(f) => self.render_for(f),
            Kind::While(c) if while_convertible(c) => self.render_while(c),
            Kind::Switch(c) => match switch_plan(self.src, c) {
                Some(groups) => self.render_switch(c, &groups),
                None => self.span(s.start, s.end, &s.children(), Level::Plain),
            },
            Kind::Expr(e) | Kind::Return(e) if !e.ternaries.is_empty() => {
                let ts: Vec<&Ternary> = e.ternaries.iter().collect();
                self.render_ternaries(s.start, s.end, &ts, &s.children())
            }
            _ => self.span(s.start, s.end, &s.children(), Level::Plain),
        }
    }

    /// A branch body as the text of a callable body: braces are added around a single statement.
    fn callable_body(&mut self, body: &Stmt) -> String {
        let text = self.render(body);
        if matches!(body.kind, Kind::Block(_)) {
            text
        } else {
            format!("{{ {text} }}")
        }
    }

    fn condition(&mut self, i: &IfStmt) -> String {
        let lambdas: Vec<&Stmt> = i.head.lambdas.iter().collect();
        self.span(i.cond.0, i.cond.1, &lambdas, Level::Plain)
            .trim()
            .to_string()
    }

    fn render_if(&mut self, i: &IfStmt) -> String {
        self.count("if");
        let cond = self.condition(i);
        let then_ = self.callable_body(&i.then_);
        let else_ = i
            .else_
            .as_deref()
            .map_or_else(|| "{}".to_string(), |e| self.callable_body(e));
        format!("aether::trace::detail::branch({cond}, [&] {then_}, [&] {else_});")
    }

    /// The block of `body` without its last statement: the closing `return;` or `break;` is
    /// dropped. A braceless body is empty then.
    fn without_last(&mut self, body: &Stmt) -> String {
        let Kind::Block(v) = &body.kind else {
            return "{}".to_string();
        };
        let (last, rest) = v.split_last().expect("a body that ends in a jump holds it");
        let kids: Vec<&Stmt> = rest.iter().collect();
        let head = self.span(body.start, last.start, &kids, Level::Plain);
        format!("{head}{}", self.copy(last.end, body.end))
    }

    /// `if (C) {S; return E;} REST` as `return choose(C, {S; return E;}, {REST'});`, or, for
    /// a void function, `branch(C, {S;}, {REST'});`. `chain` starts at the `if`.
    fn render_early(&mut self, chain: &[&Stmt], value: bool, level: Level) -> String {
        self.count("early-return");
        let Kind::If(i) = &chain[0].kind else {
            unreachable!("an early-return chain starts at an `if`");
        };
        let cond = self.condition(i);
        let rest = &chain[1..];
        let rest_text = match rest.last() {
            Some(last) => {
                format!("{{ {} }}", self.span(chain[0].end, last.end, rest, level))
            }
            None => "{}".to_string(),
        };
        if value {
            let then_ = self.callable_body(&i.then_);
            format!("return aether::trace::detail::choose({cond}, [&] {then_}, [&] {rest_text});")
        } else {
            let then_ = self.without_last(&i.then_);
            format!("aether::trace::detail::branch({cond}, [&] {then_}, [&] {rest_text});")
        }
    }

    /// `if (C) {S; break;}` as `if (loopExit.when(C)) {S; return;}`.
    fn render_exit(&mut self, s: &Stmt) -> String {
        let Kind::If(i) = &s.kind else {
            return self.render(s);
        };
        let cond = self.condition(i);
        let then_ = match &i.then_.kind {
            Kind::Block(v) => {
                let (last, rest) = v.split_last().expect("an exit block holds its break");
                let kids: Vec<&Stmt> = rest.iter().collect();
                let head = self.span(i.then_.start, last.start, &kids, Level::Plain);
                format!("{head}return;{}", self.copy(last.end, i.then_.end))
            }
            _ => "return;".to_string(),
        };
        format!("if (loopExit.when({cond})) {then_}")
    }

    /// The loop body as the text between the braces of its callable.
    fn loop_body(&mut self, body: &Stmt, level: Level) -> String {
        let kids: Vec<&Stmt> = body_stmts(body).iter().collect();
        match &body.kind {
            Kind::Block(_) => self.span(body.start + 1, body.end - 1, &kids, level),
            _ => self.span(body.start, body.end, &kids, level),
        }
    }

    /// The list `{&x, "x"}, ...` of the variables a loop body assigns.
    fn carried(&self, body: &Stmt) -> String {
        let names: Vec<String> = assigned_names(self.src, body)
            .iter()
            .map(|n| format!("{{&{n}, \"{n}\"}}"))
            .collect();
        names.join(", ")
    }

    fn render_for(&mut self, f: &ForStmt) -> String {
        self.count("for");
        let h = f.header.as_ref().expect("a convertible loop has a header");
        let text = self.src.text;
        let body = &*f.ctl.body;
        let inner = self.loop_body(body, Level::Loop);
        let step = h.step.map_or("1", |(a, b)| text[a..b].trim());
        // A macro in the body may use the index by name, so the parameter is always named.
        format!(
            "aether::trace::detail::loop({}, {}, {step}, \"{index}\", {{{}}}, [&]([[maybe_unused]] auto {index}, [[maybe_unused]] auto& loopExit) {{ {inner} }});",
            text[h.start.0..h.start.1].trim(),
            text[h.stop.0..h.stop.1].trim(),
            self.carried(body),
            index = h.index,
        )
    }

    /// `while (aether::bounded(C, N)) BODY` as a loop of `N` trips whose body first tests `C`.
    fn render_while(&mut self, c: &Ctl) -> String {
        self.count("while");
        let b = c.bound.as_ref().expect("a convertible while is bounded");
        let text = self.src.text;
        let inner = self.loop_body(&c.body, Level::Plain);
        format!(
            "aether::trace::detail::loop(0, {}, 1, \"it\", {{{}}}, [&]([[maybe_unused]] auto it, auto& loopExit) {{ if (loopExit.when(!({}))) {{ return; }} {inner} }});",
            text[b.limit.0..b.limit.1].trim(),
            self.carried(&c.body),
            text[b.cond.0..b.cond.1].trim(),
        )
    }

    /// A switch as a chain of branches on `K == value`.
    fn render_switch(&mut self, c: &Ctl, groups: &[Group]) -> String {
        self.count("switch");
        let text = self.src.text;
        let key = text[c.cond.0..c.cond.1].trim();
        let key = self.operand(key, c.cond);
        self.chain(&key, groups)
    }

    /// `text`, in parentheses when it is more than one token.
    fn operand(&self, text: &str, range: (usize, usize)) -> String {
        if self.src.words_in(range.0, range.1).len() > 1 {
            format!("({text})")
        } else {
            text.to_string()
        }
    }

    fn case_body(&mut self, g: &Group) -> String {
        if let (Some(block), Some(brk)) = (g.block, g.brk) {
            let head = self.span(block.start, brk.start, &g.stmts, Level::Plain);
            return format!("{head}{}", self.copy(brk.end, block.end));
        }
        if let Some(block) = g.block {
            return self.render(block);
        }
        match (g.stmts.first(), g.stmts.last()) {
            (Some(first), Some(last)) => {
                format!(
                    "{{ {} }}",
                    self.span(first.start, last.end, &g.stmts, Level::Plain)
                )
            }
            _ => "{}".to_string(),
        }
    }

    fn chain(&mut self, key: &str, groups: &[Group]) -> String {
        let Some((first, rest)) = groups.split_first() else {
            return "{}".to_string();
        };
        let Some(value) = first.value else {
            return self.case_body(first);
        };
        let text = self.src.text;
        let value = self.operand(text[value.0..value.1].trim(), value);
        let then_ = self.case_body(first);
        let else_ = match rest {
            [] => "{}".to_string(),
            [only] if only.value.is_none() => self.case_body(only),
            _ => format!("{{ {} }}", self.chain(key, rest)),
        };
        format!("aether::trace::detail::branch({key} == {value}, [&] {then_}, [&] {else_});")
    }

    /// A switch whose cases return values as `return choose(K == a, {case a}, {return choose(...)})`.
    /// A fall-through group of labels repeats its body for each value. The last callable holds
    /// `default`, or else the statements after the switch (which then move into it); the flag
    /// says whether those were taken in.
    fn render_return_switch(
        &mut self,
        c: &Ctl,
        cases: &[ReturnCase],
        rest: &[&Stmt],
        level: Level,
    ) -> (String, bool) {
        self.count("switch");
        let text = self.src.text;
        let key = self.operand(text[c.cond.0..c.cond.1].trim(), c.cond);
        let mut tests: Vec<(String, String)> = Vec::new();
        let mut last = None;
        for case in cases {
            if case.default {
                last = Some(self.return_case_statements(case));
                continue;
            }
            let body = self.return_case_body(case);
            for &value in &case.values {
                let operand = self.operand(text[value.0..value.1].trim(), value);
                tests.push((format!("{key} == {operand}"), body.clone()));
            }
        }
        let (mut statements, consumed_rest) = if let Some(statements) = last {
            (statements, false)
        } else {
            let (first, end) = (rest[0].start, rest[rest.len() - 1].end);
            (self.span(first, end, rest, level), true)
        };
        for (test, body) in tests.iter().rev() {
            statements = format!(
                "return aether::trace::detail::choose({test}, [&] {body}, [&] {{ {statements} }});"
            );
        }
        (statements, consumed_rest)
    }

    /// The statements of one case without the braces around them.
    fn return_case_statements(&mut self, case: &ReturnCase) -> String {
        if let Some(block) = case.block {
            let text = self.render(block);
            return text[1..text.len() - 1].trim().to_string();
        }
        match (case.stmts.first(), case.stmts.last()) {
            (Some(first), Some(last)) => {
                self.span(first.start, last.end, &case.stmts, Level::Plain)
            }
            _ => String::new(),
        }
    }

    /// The braced text of one case: its block as written, or its statements put in braces.
    fn return_case_body(&mut self, case: &ReturnCase) -> String {
        if let Some(block) = case.block {
            return self.render(block);
        }
        match (case.stmts.first(), case.stmts.last()) {
            (Some(first), Some(last)) => {
                format!(
                    "{{ {} }}",
                    self.span(first.start, last.end, &case.stmts, Level::Plain)
                )
            }
            _ => "{}".to_string(),
        }
    }

    /// The text of `[from, to)` with each `C ? A : B` whose operands are certain written as a
    /// choice; operands are rewritten in turn.
    fn render_ternaries(
        &mut self,
        from: usize,
        to: usize,
        ts: &[&Ternary],
        kids: &[&Stmt],
    ) -> String {
        let mut found: Vec<((usize, usize), crate::read::Extent)> = ts
            .iter()
            .filter_map(|t| self.src.ternary_extent(t))
            .map(|x| (self.src.bytes_of(x.cond.0, x.else_.1), x))
            .collect();
        found.sort_by_key(|(span, _)| (span.0, std::cmp::Reverse(span.1)));
        let mut out = String::new();
        let mut pos = from;
        for (span, x) in found {
            if span.0 < pos || span.1 > to {
                continue;
            }
            out.push_str(&self.span(pos, span.0, kids, Level::Plain));
            let part = |this: &mut Self, r: (usize, usize)| {
                let (a, b) = this.src.bytes_of(r.0, r.1);
                this.render_ternaries(a, b, ts, kids)
            };
            let (c, a, b) = (part(self, x.cond), part(self, x.then_), part(self, x.else_));
            let choice = format!(
                "aether::trace::detail::choose({}, [&] {{ return {}; }}, [&] {{ return {}; }})",
                c.trim(),
                a.trim(),
                b.trim()
            );
            let choice = self.keep_line_count(span.0, span.1, choice);
            out.push_str(&choice);
            self.count("ternary");
            pos = span.1;
        }
        out.push_str(&self.span(pos, to, kids, Level::Plain));
        out
    }
}

// ---- the names a loop body assigns -----------------------------------------------------------

/// Identifiers that are the target of an assignment statement anywhere in `body` and are not
/// declared there, in order of first assignment.
fn assigned_names(src: &Source, body: &Stmt) -> Vec<String> {
    let (mut assigned, mut declared) = (Vec::new(), HashSet::new());
    collect_names(src, body, &mut assigned, &mut declared);
    let mut seen = HashSet::new();
    assigned
        .into_iter()
        .filter(|n| !declared.contains(n) && seen.insert(n.clone()))
        .collect()
}

fn collect_names(
    src: &Source,
    s: &Stmt,
    assigned: &mut Vec<String>,
    declared: &mut HashSet<String>,
) {
    match &s.kind {
        Kind::Expr(_) => {
            let words = src.words_in(s.start, s.end);
            if words.len() >= 2 && is_name(words[0]) && ASSIGN_OPS.contains(&words[1]) {
                assigned.push(words[0].to_string());
            } else {
                declared.extend(declared_names(&words).into_iter().map(str::to_string));
            }
        }
        Kind::For(f) => {
            if let Some(&(a, b)) = f.parts.first() {
                declared.extend(
                    declared_names(&src.words_in(a, b))
                        .into_iter()
                        .map(str::to_string),
                );
            }
        }
        _ => {}
    }
    for c in s.children() {
        collect_names(src, c, assigned, declared);
    }
}

/// The variables a declaration statement declares: `Type a = x, *b;` gives `a` and `b`. A name
/// preceded by a type-like word, `>`, `*` or `&` is taken as declared; without types this reads
/// `a * b;` as a declaration too.
fn declared_names<'a>(words: &[&'a str]) -> Vec<&'a str> {
    let stops = ["=", "(", "{", ";", ",", "[", ":"];
    let head_end = words
        .iter()
        .position(|w| stops.contains(w))
        .unwrap_or(words.len());
    let head = &words[..head_end];
    let [.., before, last] = head else {
        return Vec::new();
    };
    if !is_name(last) || !(is_name(before) || [">", "*", "&", "&&"].contains(before)) {
        return Vec::new();
    }
    let mut names = vec![*last];
    let mut depth = 0usize;
    for (i, w) in words.iter().enumerate() {
        match *w {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => depth = depth.saturating_sub(1),
            "," if depth == 0 => {
                let rest = &words[i + 1..];
                let skip = rest.iter().take_while(|x| ["*", "&"].contains(x)).count();
                if let Some(&name) = rest.get(skip).filter(|n| is_name(n)) {
                    names.push(name);
                }
            }
            _ => {}
        }
    }
    names
}

/// Starts the line number marker `copy` writes after a line break.
const MARK: char = '\u{1}';
/// Ends it.
const MARK_END: char = '\u{2}';

/// Turns the markers into `#line N` directives, written only where the line a marker stands at
/// differs from the source line it was copied from: after a rewrite that changed the number of
/// lines, so that a compiler's line numbers are the author's. `#line N` without a file name keeps
/// the current one. The output starts at line 1.
fn mark_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut expected = 1usize;
    for line in text.split_inclusive('\n') {
        let mut clean = String::with_capacity(line.len());
        let mut target = None;
        let mut rest = line;
        while let Some(at) = rest.find(MARK) {
            clean.push_str(&rest[..at]);
            let after = &rest[at + MARK.len_utf8()..];
            let end = after.find(MARK_END).unwrap_or(after.len());
            if clean.is_empty() {
                target = target.or(after[..end].parse::<usize>().ok());
            }
            rest = &after[(end + MARK_END.len_utf8()).min(after.len())..];
        }
        clean.push_str(rest);
        if let Some(n) = target.filter(|&n| n != expected) {
            let directive = format!("#line {n}\n");
            out.push_str(&directive);
            expected = n;
        }
        out.push_str(&clean);
        if clean.ends_with('\n') {
            expected += 1;
        }
    }
    out
}
