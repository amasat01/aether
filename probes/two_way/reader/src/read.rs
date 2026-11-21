//! The statement reader: finds function bodies in a token stream and reads the control statements
//! inside them as a tree. Conditions, headers and bodies stay text (byte spans); only the shape
//! of the statements is read, so no type information is needed.

use crate::lex::{lex, TokKind, Token};

/// A source file with its tokens, ready to be read.
pub struct Source<'a> {
    /// The file text.
    pub text: &'a str,
    /// Every token, spans tiling the text.
    pub tokens: Vec<Token>,
    /// Indices into `tokens` of the tokens a reader looks at: no space, comment or directive.
    sig: Vec<usize>,
    /// Byte starts of the conditional directives (`#if`, `#else`, `#endif`, ...).
    conditionals: Vec<usize>,
    line_starts: Vec<usize>,
    /// For each significant token: how many parentheses, brackets and brace initialisers are
    /// open after it. A statement block does not count.
    open_after: Vec<usize>,
}

impl<'a> Source<'a> {
    /// Lexes `text`.
    #[must_use]
    pub fn new(text: &'a str) -> Self {
        let tokens = lex(text);
        let mut sig = Vec::new();
        let mut conditionals = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            match t.kind {
                TokKind::Space | TokKind::LineComment | TokKind::BlockComment => {}
                TokKind::Preproc => {
                    if is_conditional_directive(&text[t.start..t.end]) {
                        conditionals.push(t.start);
                    }
                }
                _ => sig.push(i),
            }
        }
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        let open_after = open_depths(text, &tokens, &sig);
        Source {
            text,
            tokens,
            sig,
            conditionals,
            line_starts,
            open_after,
        }
    }

    /// The 1-based line of a byte offset.
    #[must_use]
    pub fn line(&self, byte: usize) -> usize {
        self.line_starts.partition_point(|&s| s <= byte)
    }

    /// Text of the `k`-th significant token, or "" past the end.
    fn word(&self, k: usize) -> &'a str {
        self.sig.get(k).map_or("", |&i| {
            &self.text[self.tokens[i].start..self.tokens[i].end]
        })
    }

    fn token(&self, k: usize) -> &Token {
        &self.tokens[self.sig[k.min(self.sig.len() - 1)]]
    }

    fn kind(&self, k: usize) -> Option<TokKind> {
        self.sig.get(k).map(|&i| self.tokens[i].kind)
    }

    /// The texts of the significant tokens whose start lies in `[from, to)`.
    #[must_use]
    pub fn words_in(&self, from: usize, to: usize) -> Vec<&'a str> {
        let first = self.sig.partition_point(|&i| self.tokens[i].start < from);
        (first..self.sig.len())
            .take_while(|&k| self.tokens[self.sig[k]].start < to)
            .map(|k| self.word(k))
            .collect()
    }

    /// Byte range from the start of significant token `from` to the end of token `to - 1`.
    #[must_use]
    pub fn bytes_of(&self, from: usize, to: usize) -> (usize, usize) {
        (self.token(from).start, self.token(to.max(from + 1) - 1).end)
    }

    /// Operands of the `?` of `t`, found by bracket depth and the statement or argument
    /// boundary; `None` when the extent is not certain.
    #[must_use]
    pub fn ternary_extent(&self, t: &Ternary) -> Option<Extent> {
        // A constant expression cannot hold the callables a choice is made of.
        if matches!(
            self.word(t.lo),
            "static_assert" | "constexpr" | "consteval" | "constinit"
        ) {
            return None;
        }
        let colon = self.matching_colon(t)?;
        let start = self.operand_start(t);
        let end = self.operand_end(t, colon);
        if self.is_lvalue_use(start, end) {
            return None;
        }
        let (from, to) = self.bytes_of(start, end);
        let certain = start < t.tok && t.tok + 1 < colon && colon + 1 < end;
        // Text between the operands is dropped, so a directive there cannot be kept.
        (certain && !self.has_conditional(from, to)).then_some(Extent {
            cond: (start, t.tok),
            then_: (t.tok + 1, colon),
            else_: (colon + 1, end),
        })
    }

    /// True when the extent of `t` is certain but its value is used as an lvalue (see
    /// `is_lvalue_use`), so a choice that returns a copy would change what the code means.
    #[must_use]
    pub fn ternary_is_lvalue(&self, t: &Ternary) -> bool {
        let Some(colon) = self.matching_colon(t) else {
            return false;
        };
        self.is_lvalue_use(self.operand_start(t), self.operand_end(t, colon))
    }

    /// A `?:` over the tokens `[start, end)` that is assigned to or incremented, has a member or
    /// element taken, has its address taken, or initialises a reference: `(c ? a : b) = x`,
    /// `(c ? a : b).m`, `&(c ? a : b)`, `auto& r = c ? a : b`.
    fn is_lvalue_use(&self, start: usize, end: usize) -> bool {
        let wrapped = start > 0 && self.word(start - 1) == "(" && self.word(end) == ")";
        if wrapped {
            let after = self.word(end + 1);
            let before = if start >= 2 { self.word(start - 2) } else { "" };
            if ASSIGN_OPS.contains(&after) || matches!(after, "." | "->" | "[" | "++" | "--") {
                return true;
            }
            return matches!(before, "&" | "++" | "--");
        }
        let binds_reference = start >= 3
            && self.word(start - 1) == "="
            && is_name(self.word(start - 2))
            && matches!(self.word(start - 3), "&" | "&&");
        binds_reference
    }

    /// The `:` that pairs with the `?` of `t`.
    fn matching_colon(&self, t: &Ternary) -> Option<usize> {
        let (mut depth, mut nested) = (0usize, 0usize);
        for j in t.tok + 1..t.hi {
            match self.word(j) {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => depth = depth.checked_sub(1)?,
                "?" if depth == 0 => nested += 1,
                ":" if depth == 0 && nested == 0 => return Some(j),
                ":" if depth == 0 => nested -= 1,
                _ => {}
            }
        }
        None
    }

    /// First token of the condition: scanning left to the start of the region, an enclosing
    /// bracket, or an operator that binds looser than `?:`.
    fn operand_start(&self, t: &Ternary) -> usize {
        let mut depth = 0usize;
        for j in (t.lo..t.tok).rev() {
            let w = self.word(j);
            match w {
                ")" | "]" | "}" => depth += 1,
                "(" | "[" | "{" if depth == 0 => return j + 1,
                "(" | "[" | "{" => depth -= 1,
                ";" | "," | ":" | "?" | "return" | "co_return" | "co_yield" | "throw" | "case"
                    if depth == 0 =>
                {
                    return j + 1
                }
                _ if depth == 0 && ASSIGN_OPS.contains(&w) => return j + 1,
                _ => {}
            }
        }
        t.lo
    }

    /// One past the last token of the value when false: to the end of the region, an enclosing
    /// bracket, a `,`, or the `:` of an enclosing `?:`.
    fn operand_end(&self, t: &Ternary, colon: usize) -> usize {
        let (mut depth, mut nested) = (0usize, 0usize);
        for j in colon + 1..t.hi {
            match self.word(j) {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" | "," | ";" if depth == 0 => return j,
                ")" | "]" | "}" => depth -= 1,
                "?" if depth == 0 => nested += 1,
                ":" if depth == 0 && nested == 0 => return j,
                ":" if depth == 0 => nested -= 1,
                _ => {}
            }
        }
        t.hi
    }

    /// How many parentheses, brackets and brace initialisers are open at byte offset `at`.
    #[must_use]
    pub fn open_at(&self, at: usize) -> usize {
        let seen = self.sig.partition_point(|&i| self.tokens[i].start < at);
        seen.checked_sub(1).map_or(0, |k| self.open_after[k])
    }

    /// Byte offset of the first line break in `[from, to)` that lies in whitespace, so not inside
    /// a literal, a comment or a directive, and outside every open parenthesis, bracket and
    /// brace initialiser: a directive may stand there.
    #[must_use]
    pub fn line_break_in(&self, from: usize, to: usize) -> Option<usize> {
        let first = self.tokens.partition_point(|t| t.end <= from);
        self.tokens[first..]
            .iter()
            .take_while(|t| t.start < to)
            .filter(|t| t.kind == TokKind::Space)
            .find_map(|t| {
                let (a, b) = (t.start.max(from), t.end.min(to));
                let breaks = self.text[a..b].match_indices('\n').map(|(p, _)| a + p);
                breaks.into_iter().find(|&at| self.open_at(at) == 0)
            })
    }

    /// True when a conditional directive starts in `[from, to)`.
    #[must_use]
    pub fn has_conditional(&self, from: usize, to: usize) -> bool {
        let first = self.conditionals.partition_point(|&b| b < from);
        self.conditionals.get(first).is_some_and(|&b| b < to)
    }
}

/// Words after which a `{` opens a statement block or a type, not an initialiser.
const BLOCK_PRECEDERS: [&str; 12] = [
    ")", "]", ";", "{", "}", ":", "else", "do", "try", "mutable", "noexcept", "const",
];

/// How many parentheses, brackets and brace initialisers are open after each significant token.
/// A `{` is an initialiser unless a block or a type body can follow the word before it.
fn open_depths(text: &str, tokens: &[Token], sig: &[usize]) -> Vec<usize> {
    let word = |k: usize| &text[tokens[sig[k]].start..tokens[sig[k]].end];
    let mut stack: Vec<bool> = Vec::new(); // true: counts as open (not a statement block)
    let mut depths = Vec::with_capacity(sig.len());
    for k in 0..sig.len() {
        match word(k) {
            "(" | "[" => stack.push(true),
            "{" => {
                let prev = if k > 0 { word(k - 1) } else { ";" };
                let before = if k > 1 { word(k - 2) } else { ";" };
                let in_initialiser = stack.last().copied().unwrap_or(false);
                let type_body =
                    matches!(before, "struct" | "class" | "union" | "namespace" | "enum");
                let block = BLOCK_PRECEDERS.contains(&prev) || type_body;
                // A brace directly inside an initialiser is a nested initialiser.
                let nested = in_initialiser && matches!(prev, "{" | ",");
                stack.push(!block || nested);
            }
            ")" | "]" | "}" => {
                stack.pop();
            }
            _ => {}
        }
        depths.push(stack.iter().filter(|&&open| open).count());
    }
    depths
}

fn is_conditional_directive(line: &str) -> bool {
    let name: String = line[1..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    matches!(
        name.as_str(),
        "if" | "ifdef" | "ifndef" | "elif" | "elifdef" | "elifndef" | "else" | "endif"
    )
}

/// A read statement: its byte span, its first line and what it is.
#[derive(Debug)]
pub struct Stmt {
    /// What the statement is, with its parts.
    pub kind: Kind,
    /// Byte offset of its first byte.
    pub start: usize,
    /// Byte offset one past its last byte.
    pub end: usize,
    /// 1-based line of its first byte.
    pub line: usize,
}

/// What an expression region contains that the reader reports or reads: lambda bodies (whose
/// statements are read like any other) and the lines of `?:` operators.
#[derive(Debug, Default)]
pub struct Expr {
    /// Lambda bodies inside the region, as blocks.
    pub lambdas: Vec<Stmt>,
    /// Each `?` outside the lambda bodies.
    pub ternaries: Vec<Ternary>,
}

/// A `?` operator and the region of significant tokens it was found in.
#[derive(Debug, Clone, Copy)]
pub struct Ternary {
    /// 1-based line of the `?`.
    pub line: usize,
    /// Index of the `?` among the significant tokens.
    pub tok: usize,
    /// First significant token of the region (inclusive).
    pub lo: usize,
    /// End of the region (exclusive).
    pub hi: usize,
}

/// Where the three operands of a `?:` lie, as ranges `[from, to)` of significant tokens.
#[derive(Debug, Clone, Copy)]
pub struct Extent {
    /// The condition.
    pub cond: (usize, usize),
    /// The value when true.
    pub then_: (usize, usize),
    /// The value when false.
    pub else_: (usize, usize),
}

/// The two arguments of `aether::bounded(C, N)`.
#[derive(Debug)]
pub struct Bounded {
    /// Byte range of the condition.
    pub cond: (usize, usize),
    /// Byte range of the bound.
    pub limit: (usize, usize),
}

/// A statement with a condition or header, and a body (`while`, `switch`, and the body of `for`).
#[derive(Debug)]
pub struct Ctl {
    /// Byte range between the parentheses of the header.
    pub cond: (usize, usize),
    /// The bounded-loop form of a `while`, when it is that.
    pub bound: Option<Bounded>,
    /// The header or condition region.
    pub head: Expr,
    /// The body statement.
    pub body: Box<Stmt>,
    /// True when a conditional preprocessor directive sits inside the statement's own structure
    /// (header, or between its parts), so the statement is cut by it.
    pub cut: bool,
}

/// What the parentheses of an `if` hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// An expression.
    Plain,
    /// An init statement then a condition: `if (init; cond)`.
    Init,
    /// A declaration or an assignment (a top-level `=`) used as the condition.
    Declaration,
}

/// `if`, with its optional `else` (which is itself an `if` for `else if`).
#[derive(Debug)]
pub struct IfStmt {
    /// Byte range between the parentheses of the condition.
    pub cond: (usize, usize),
    /// The condition region.
    pub head: Expr,
    /// `if constexpr` or `if consteval`.
    pub constexpr: bool,
    /// What the parentheses hold.
    pub condition: Condition,
    /// The taken branch.
    pub then_: Box<Stmt>,
    /// The `else` branch.
    pub else_: Option<Box<Stmt>>,
    /// A conditional directive sits in the structure of the statement.
    pub cut: bool,
}

/// The pieces of a classic `for (int i = a; i < b; ++i)` header.
#[derive(Debug)]
pub struct ForHeader {
    /// The index variable.
    pub index: String,
    /// Byte range of the start expression.
    pub start: (usize, usize),
    /// Byte range of the stop expression.
    pub stop: (usize, usize),
    /// Byte range of the step expression; `None` for `++i` and `i++` (step one).
    pub step: Option<(usize, usize)>,
}

/// A `for` statement.
#[derive(Debug)]
pub struct ForStmt {
    /// Header and body.
    pub ctl: Ctl,
    /// Range-based (`for (x : range)`).
    pub range: bool,
    /// Present when the classic header is exactly the convertible form.
    pub header: Option<ForHeader>,
    /// Byte ranges of the header parts: init, condition, step (classic) or declaration, range.
    pub parts: Vec<(usize, usize)>,
}

/// The kinds of statement the reader tells apart.
#[derive(Debug)]
pub enum Kind {
    /// `{ ... }`.
    Block(Vec<Stmt>),
    /// `if`.
    If(IfStmt),
    /// `for`, classic or range-based.
    For(ForStmt),
    /// `while`.
    While(Ctl),
    /// `do ... while (...)`; the body and the condition.
    Do(Ctl),
    /// `switch`.
    Switch(Ctl),
    /// `try` with its handlers, as blocks.
    Try(Vec<Stmt>),
    /// `break;`
    Break,
    /// `continue;`
    Continue,
    /// `return ...;`
    Return(Expr),
    /// `goto label;`
    Goto,
    /// `case X:`, `default:` or a named label.
    Label,
    /// Any other statement, up to its `;`.
    Expr(Expr),
}

impl Stmt {
    /// The statements read inside this one, in source order: nested bodies, branches and lambda
    /// bodies of the regions it holds.
    #[must_use]
    pub fn children(&self) -> Vec<&Stmt> {
        match &self.kind {
            Kind::Block(v) | Kind::Try(v) => v.iter().collect(),
            Kind::If(i) => {
                let mut out = i.head.lambdas.iter().collect::<Vec<_>>();
                out.push(&i.then_);
                out.extend(i.else_.as_deref());
                out
            }
            Kind::For(f) => ctl_children(&f.ctl),
            Kind::While(c) | Kind::Switch(c) => ctl_children(c),
            Kind::Do(c) => {
                let mut out = vec![&*c.body];
                out.extend(c.head.lambdas.iter());
                out
            }
            Kind::Return(e) | Kind::Expr(e) => e.lambdas.iter().collect(),
            Kind::Break | Kind::Continue | Kind::Goto | Kind::Label => Vec::new(),
        }
    }
}

fn ctl_children(c: &Ctl) -> Vec<&Stmt> {
    let mut out: Vec<&Stmt> = c.head.lambdas.iter().collect();
    out.push(&c.body);
    out
}

/// A construct the reader could not read; the enclosing function body is left untouched.
#[derive(Debug)]
pub struct ParseError {
    /// 1-based line where reading stopped.
    pub line: usize,
    /// What was wrong.
    pub message: &'static str,
}

type Res<T> = Result<T, ParseError>;

/// Qualifiers that may sit between a function's `)` and its `{`.
const QUALIFIERS: [&str; 9] = [
    "const",
    "noexcept",
    "override",
    "final",
    "mutable",
    "volatile",
    "&",
    "&&",
    "constexpr",
];

/// The integer types a convertible loop index may be declared with.
const INDEX_TYPES: [&str; 20] = [
    "int",
    "long",
    "short",
    "unsigned",
    "signed",
    "size_t",
    "ptrdiff_t",
    "auto",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "intptr_t",
    "uintptr_t",
    "std",
    "::",
];

/// Operators that assign to their left operand.
pub const ASSIGN_OPS: [&str; 11] = [
    "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
];

/// A function body and what its head says about the returned value.
#[derive(Debug)]
pub struct Function {
    /// The body block.
    pub body: Stmt,
    /// The declared return type has a `&` or `&&`: a callable that returns by value would copy.
    pub returns_reference: bool,
    /// The name with its enclosing namespaces and classes (`ns::name`); empty for a lambda.
    pub name: String,
    /// The parameters in order.
    pub params: Vec<Param>,
}

/// One parameter of a function, as its declaration spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// The name; "" when the parameter is unnamed.
    pub name: String,
    /// The role its type spells: `per_sample`, `uniform` or `mutable`; none for any other type.
    pub role: Option<&'static str>,
    /// The type as written, without the name and any default value.
    pub type_text: String,
    /// 1-based line of the parameter.
    pub line: usize,
}

/// Words that may precede a parenthesis without being a function name.
const NOT_NAMES: [&str; 5] = [
    "noexcept",
    "alignas",
    "requires",
    "decltype",
    "__attribute__",
];

/// Reads the function bodies of a source. Bodies that cannot be read are returned as errors.
#[must_use]
pub fn read_functions(src: &Source) -> (Vec<Function>, Vec<ParseError>) {
    let mut parser = Parser {
        s: src,
        k: 0,
        scopes: Vec::new(),
    };
    let mut funcs = Vec::new();
    let mut errors = Vec::new();
    parser.scan_scope(&mut funcs, &mut errors, false);
    (funcs, errors)
}

struct Parser<'s, 'a> {
    s: &'s Source<'a>,
    k: usize,
    /// Names of the namespaces and classes around the cursor.
    scopes: Vec<String>,
}

#[derive(PartialEq)]
enum Brace {
    /// A namespace or class body, with its name (empty when it has none).
    Scope(String),
    Function,
    Opaque,
}

impl Parser<'_, '_> {
    fn word(&self) -> &str {
        self.s.word(self.k)
    }

    fn err<T>(&self, message: &'static str) -> Res<T> {
        let byte = if self.s.sig.is_empty() {
            0
        } else {
            self.s.token(self.k).start
        };
        Err(ParseError {
            line: self.s.line(byte),
            message,
        })
    }

    /// Index of the bracket that closes the one at `k`.
    fn matching(&self, k: usize) -> Option<usize> {
        let mut depth = 0usize;
        for j in k..self.s.sig.len() {
            match self.s.word(j) {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(j);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Ranges of the tokens in `[from, to)` split at top-level occurrences of `sep`.
    fn split_top(&self, from: usize, to: usize, sep: &str) -> Vec<(usize, usize)> {
        let mut parts = Vec::new();
        let (mut depth, mut begin) = (0usize, from);
        for j in from..to {
            match self.s.word(j) {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => depth = depth.saturating_sub(1),
                w if w == sep && depth == 0 => {
                    parts.push((begin, j));
                    begin = j + 1;
                }
                _ => {}
            }
        }
        parts.push((begin, to));
        parts
    }

    fn scan_scope(
        &mut self,
        funcs: &mut Vec<Function>,
        errors: &mut Vec<ParseError>,
        nested: bool,
    ) {
        let mut head = self.k;
        while self.k < self.s.sig.len() {
            match self.word() {
                "}" => {
                    self.k += 1;
                    if nested {
                        return;
                    }
                    head = self.k;
                }
                ";" => {
                    self.k += 1;
                    head = self.k;
                }
                "{" => {
                    self.scan_brace(head, funcs, errors);
                    head = self.k;
                }
                ":" if self.k == head + 1
                    && matches!(self.s.word(head), "public" | "private" | "protected") =>
                {
                    self.k += 1;
                    head = self.k;
                }
                _ => self.k += 1,
            }
        }
    }

    /// Handles the `{` at the cursor of a declaration scope: a scope is entered, a function body
    /// is read, anything else is skipped.
    fn scan_brace(&mut self, head: usize, funcs: &mut Vec<Function>, errors: &mut Vec<ParseError>) {
        let open = self.k;
        match self.classify_brace(head, open) {
            Brace::Scope(name) => {
                self.k += 1;
                self.scopes.push(name);
                self.scan_scope(funcs, errors, true);
                self.scopes.pop();
            }
            Brace::Function => match self.parse_block() {
                Ok(body) => {
                    let returns_reference = self.head_returns_reference(head, open);
                    let (name, params) = self.signature(head, open);
                    funcs.push(Function {
                        body,
                        returns_reference,
                        name,
                        params,
                    });
                }
                Err(e) => {
                    errors.push(e);
                    self.k = self.matching(open).map_or(self.s.sig.len(), |c| c + 1);
                }
            },
            Brace::Opaque => {
                self.k = self.matching(open).map_or(self.s.sig.len(), |c| c + 1);
            }
        }
    }

    /// True when the declaration `[head, open)` names a `&` or `&&` before its parameter list or
    /// after a trailing `->`.
    fn head_returns_reference(&self, head: usize, open: usize) -> bool {
        let is_ref = |j: usize| matches!(self.s.word(j), "&" | "&&");
        let params = (head..open)
            .find(|&j| self.s.word(j) == "(")
            .unwrap_or(open);
        let arrow = (head..open).find(|&j| self.s.word(j) == "->");
        (head..params).any(is_ref) || arrow.is_some_and(|a| (a..open).any(is_ref))
    }

    /// The name a namespace or class declaration gives: `a::b` for `namespace a::b {`.
    fn scope_name(&self, keyword: usize, open: usize) -> String {
        let stop = |w: &str| matches!(w, ":" | "final" | "<" | "[" | "{");
        let words: Vec<&str> = (keyword + 1..open)
            .map(|j| self.s.word(j))
            .take_while(|w| !stop(w))
            .collect();
        let is_namespace = self.s.word(keyword) == "namespace";
        match (is_namespace, words.first()) {
            (true, _) => words.concat(),
            (false, Some(w)) if is_name(w) => (*w).to_string(),
            _ => String::new(),
        }
    }

    /// The qualified name and the parameter names of the function declared by `[head, open)`:
    /// the last parenthesis at depth zero that follows a name (not a macro's, not `noexcept`).
    fn signature(&self, head: usize, open: usize) -> (String, Vec<Param>) {
        let mut first = head;
        while self.s.word(first) == "template" && self.s.word(first + 1) == "<" {
            first = self.skip_angles(first + 1);
        }
        let mut group = None;
        let mut j = first;
        while j < open {
            let w = self.s.word(j);
            if matches!(w, "->" | ":") {
                break;
            }
            if matches!(w, "(" | "[") {
                let Some(close) = self.matching(j) else { break };
                let named = j > first && is_name(self.s.word(j - 1));
                if w == "(" && named && !NOT_NAMES.contains(&self.s.word(j - 1)) {
                    group = Some((j, close));
                }
                j = close + 1;
            } else {
                j += 1;
            }
        }
        let Some((paren, close)) = group else {
            return (String::new(), Vec::new());
        };
        let mut at = paren - 1;
        let mut parts = vec![self.s.word(at)];
        while at >= first + 2 && self.s.word(at - 1) == "::" && is_name(self.s.word(at - 2)) {
            at -= 2;
            parts.insert(0, self.s.word(at));
        }
        let mut qualified: Vec<&str> = self.scopes.iter().map(String::as_str).collect();
        qualified.retain(|n| !n.is_empty());
        qualified.extend(parts);
        (qualified.join("::"), self.parameter_names(paren + 1, close))
    }

    /// The name of each parameter in `[from, to)`: the last word before a default value.
    fn parameter_names(&self, from: usize, to: usize) -> Vec<Param> {
        let mut params = Vec::new();
        let (mut depth, mut angles, mut begin) = (0usize, 0usize, from);
        for j in from..=to {
            let at_end = j == to;
            match self.s.word(j) {
                "(" | "[" | "{" if !at_end => depth += 1,
                ")" | "]" | "}" if !at_end => depth = depth.saturating_sub(1),
                "<" if !at_end => angles += 1,
                ">" if !at_end => angles = angles.saturating_sub(1),
                ">>" if !at_end => angles = angles.saturating_sub(2),
                "," if depth == 0 && angles == 0 && !at_end => {
                    params.push(self.parameter(begin, j));
                    begin = j + 1;
                }
                _ => {}
            }
            if at_end && begin < to {
                params.push(self.parameter(begin, to));
            }
        }
        params
    }

    /// The parameter in tokens `[from, to)`: its name, the role its type spells and its type.
    fn parameter(&self, from: usize, to: usize) -> Param {
        let end = (from..to).find(|&j| self.s.word(j) == "=").unwrap_or(to);
        let named = end >= from + 2 && is_name(self.s.word(end - 1));
        let type_end = if named { end - 1 } else { end };
        // The role is the word before the first `<`: `aether::PerSample<T>` gives `PerSample`,
        // while `std::array<PerSample<T>, 2>` gives `array`, which is no role.
        let role = (from..type_end)
            .find(|&j| self.s.word(j) == "<")
            .filter(|&j| j > from)
            .and_then(|j| match self.s.word(j - 1) {
                "PerSample" => Some("per_sample"),
                "Uniform" => Some("uniform"),
                "Mutable" => Some("mutable"),
                _ => None,
            });
        let type_text = if from < type_end {
            let (a, b) = self.byte_range(from, type_end);
            self.s.text[a..b]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            String::new()
        };
        let name = if named {
            self.s.word(end - 1).to_string()
        } else {
            String::new()
        };
        let line = self.s.line(self.s.token(from).start);
        Param {
            name,
            role,
            type_text,
            line,
        }
    }

    /// What the declaration `[head, open)` that ends in a `{` opens.
    fn classify_brace(&self, head: usize, open: usize) -> Brace {
        let mut i = head;
        while self.s.word(i) == "template" && self.s.word(i + 1) == "<" {
            i = self.skip_angles(i + 1);
        }
        let first = i;
        let mut depth = 0usize;
        while i < open {
            match self.s.word(i) {
                "(" | "[" => depth += 1,
                ")" | "]" => depth = depth.saturating_sub(1),
                "namespace" | "class" | "struct" | "union" if depth == 0 => {
                    return Brace::Scope(self.scope_name(i, open));
                }
                "enum" if depth == 0 => return Brace::Opaque,
                "extern" if i + 2 == open && self.s.kind(i + 1) == Some(TokKind::Str) => {
                    return Brace::Scope(String::new())
                }
                _ => {}
            }
            i += 1;
        }
        if self.ends_in_call_parens(first, open) {
            Brace::Function
        } else {
            Brace::Opaque
        }
    }

    /// Past the `<...>` that opens at `k`.
    fn skip_angles(&self, k: usize) -> usize {
        let mut depth = 0usize;
        for j in k..self.s.sig.len() {
            match self.s.word(j) {
                "<" => depth += 1,
                ">" => depth = depth.saturating_sub(1),
                ">>" => depth = depth.saturating_sub(2),
                _ => {}
            }
            if depth == 0 {
                return j + 1;
            }
        }
        self.s.sig.len()
    }

    /// True when `[first, open)` ends in `)` once qualifiers and a trailing return type are set
    /// aside: a function or lambda header.
    fn ends_in_call_parens(&self, first: usize, open: usize) -> bool {
        let skip_qualifiers = |mut j: usize| {
            while j > first && QUALIFIERS.contains(&self.s.word(j - 1)) {
                j -= 1;
            }
            j
        };
        let end = skip_qualifiers(open);
        if end > first && self.s.word(end - 1) == ")" {
            return true;
        }
        (first..open)
            .find(|&j| self.s.word(j) == "->")
            .is_some_and(|arrow| {
                let before = skip_qualifiers(arrow);
                before > first && self.s.word(before - 1) == ")"
            })
    }

    fn parse_block(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        self.k += 1;
        let mut stmts = Vec::new();
        loop {
            if self.k >= self.s.sig.len() {
                self.k = k0;
                return self.err("unbalanced brace: the block is never closed");
            }
            if self.word() == "}" {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        let close = self.k;
        self.k += 1;
        Ok(self.make(Kind::Block(stmts), k0, close))
    }

    /// A statement spanning the significant tokens `first..=last`.
    fn make(&self, kind: Kind, first: usize, last: usize) -> Stmt {
        let start = self.s.token(first).start;
        Stmt {
            kind,
            start,
            end: self.s.token(last).end,
            line: self.s.line(start),
        }
    }

    fn parse_stmt(&mut self) -> Res<Stmt> {
        if self.k >= self.s.sig.len() {
            return self.err("a statement was expected before the end of the file");
        }
        match self.word() {
            "{" => self.parse_block(),
            "if" => self.parse_if(),
            "for" => self.parse_for(),
            "while" => self.parse_while(),
            "do" => self.parse_do(),
            "switch" => self.parse_switch(),
            "try" => self.parse_try(),
            "break" => Ok(self.parse_jump(Kind::Break)),
            "continue" => Ok(self.parse_jump(Kind::Continue)),
            "goto" => Ok(self.parse_jump(Kind::Goto)),
            "return" => self.parse_return(),
            "case" => self.parse_label(),
            "default" if self.s.word(self.k + 1) == ":" => self.parse_label(),
            "else" => self.err("`else` without a matching `if`"),
            w if is_name(w) && self.s.word(self.k + 1) == ":" => self.parse_label(),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_jump(&mut self, kind: Kind) -> Stmt {
        let k0 = self.k;
        let end = self.statement_end(k0);
        self.k = end + 1;
        self.make(kind, k0, end.min(self.s.sig.len() - 1))
    }

    fn parse_return(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let end = self.statement_end(k0);
        let expr = self.scan_range(k0 + 1, end)?;
        self.k = end + 1;
        Ok(self.make(Kind::Return(expr), k0, end.min(self.s.sig.len() - 1)))
    }

    fn parse_label(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let mut j = k0;
        while j < self.s.sig.len() && self.s.word(j) != ":" {
            j += 1;
        }
        if j >= self.s.sig.len() {
            return self.err("a label without its `:`");
        }
        self.k = j + 1;
        Ok(self.make(Kind::Label, k0, j))
    }

    fn parse_expr_stmt(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let end = self.statement_end(k0);
        let expr = self.scan_range(k0, end)?;
        let terminated = self.s.word(end) == ";";
        self.k = if terminated { end + 1 } else { end };
        let last = if terminated || end == k0 {
            end
        } else {
            end - 1
        };
        Ok(self.make(Kind::Expr(expr), k0, last.min(self.s.sig.len() - 1)))
    }

    /// Index of the `;` that ends the statement starting at `k`, or of the `}` or the end of
    /// the file when it has none.
    fn statement_end(&self, k: usize) -> usize {
        let mut depth = 0usize;
        for j in k..self.s.sig.len() {
            match self.s.word(j) {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" => depth = depth.saturating_sub(1),
                "}" | ";" if depth == 0 => return j,
                "}" => depth -= 1,
                _ => {}
            }
        }
        self.s.sig.len()
    }

    /// Reads the expression region `[from, to)`: lambda bodies become blocks, `?` is noted.
    fn scan_range(&mut self, from: usize, to: usize) -> Res<Expr> {
        let mut expr = Expr::default();
        let mut j = from;
        while j < to {
            match self.s.word(j) {
                "?" => {
                    let line = self.s.line(self.s.token(j).start);
                    expr.ternaries.push(Ternary {
                        line,
                        tok: j,
                        lo: from,
                        hi: to,
                    });
                }
                "{" if self.is_lambda_body(j) => {
                    self.k = j;
                    expr.lambdas.push(self.parse_block()?);
                    j = self.k;
                    continue;
                }
                _ => {}
            }
            j += 1;
        }
        Ok(expr)
    }

    /// A `{` after `)`, `]` or a trailing return type opens a lambda body inside an expression.
    fn is_lambda_body(&self, open: usize) -> bool {
        let mut j = open;
        while j > 0 && QUALIFIERS.contains(&self.s.word(j - 1)) {
            j -= 1;
        }
        if j > 0 && matches!(self.s.word(j - 1), ")" | "]") {
            return true;
        }
        let from = j.saturating_sub(12);
        (from..j).any(|a| self.s.word(a) == "->")
            && (from..j).all(|a| !matches!(self.s.word(a), ";" | "{" | "}" | "="))
    }

    /// Reads `( ... )` at the cursor: the index of `(`, of its `)`, and the cursor moved past it.
    fn parenthesis(&mut self) -> Res<(usize, usize)> {
        let open = self.k;
        if self.word() != "(" {
            return self.err("`(` expected after the keyword");
        }
        let Some(close) = self.matching(open) else {
            return self.err("unbalanced parenthesis");
        };
        self.k = close + 1;
        Ok((open, close))
    }

    /// True when a conditional directive lies in `[from, to)` of the structure.
    fn cut(&self, ranges: &[(usize, usize)]) -> bool {
        ranges.iter().any(|&(a, b)| self.s.has_conditional(a, b))
    }

    fn parse_if(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        self.k += 1;
        let consteval = self.word() == "consteval"
            || (self.word() == "!" && self.s.word(self.k + 1) == "consteval");
        let constexpr = consteval || self.word() == "constexpr";
        if constexpr && !consteval {
            self.k += 1;
        }
        let (mut cond, mut head, mut condition) = ((0, 0), Expr::default(), Condition::Plain);
        let mut header_end = self.s.token(k0).end;
        if consteval {
            while self.word() != "{" && self.k < self.s.sig.len() {
                self.k += 1;
            }
        } else {
            let (open, close) = self.parenthesis()?;
            if self.split_top(open + 1, close, ";").len() > 1 {
                condition = Condition::Init;
            } else if self.split_top(open + 1, close, "=").len() > 1 {
                condition = Condition::Declaration;
            }
            cond = (self.s.token(open).end, self.s.token(close).start);
            head = self.scan_range(open + 1, close)?;
            self.k = close + 1;
            header_end = self.s.token(close).end;
        }
        let then_ = Box::new(self.parse_stmt()?);
        let mut gaps = vec![(self.s.token(k0).start, header_end)];
        if !matches!(then_.kind, Kind::Block(_)) {
            gaps.push((then_.start, then_.end));
        }
        let mut else_ = None;
        if self.word() == "else" {
            let kw = self.s.token(self.k);
            gaps.push((then_.end, kw.end));
            self.k += 1;
            let body = self.parse_stmt()?;
            if !matches!(body.kind, Kind::Block(_) | Kind::If(_)) {
                gaps.push((body.start, body.end));
            }
            else_ = Some(Box::new(body));
        }
        let end = else_.as_ref().map_or(then_.end, |e| e.end);
        let cut = self.cut(&gaps);
        let kind = Kind::If(IfStmt {
            cond,
            head,
            constexpr,
            condition,
            then_,
            else_,
            cut,
        });
        Ok(Stmt {
            kind,
            start: self.s.token(k0).start,
            end,
            line: self.s.line(self.s.token(k0).start),
        })
    }

    /// Reads the header and the body of `for`, `while` or `switch`; also gives the indices of the
    /// header's parentheses.
    fn parse_headed(&mut self) -> Res<(Ctl, usize, usize)> {
        let k0 = self.k;
        self.k += 1;
        let (open, close) = self.parenthesis()?;
        let head = self.scan_range(open + 1, close)?;
        self.k = close + 1;
        let body = Box::new(self.parse_stmt()?);
        let mut gaps = vec![(self.s.token(k0).start, self.s.token(close).end)];
        if !matches!(body.kind, Kind::Block(_)) {
            gaps.push((body.start, body.end));
        }
        let cut = self.cut(&gaps);
        let cond = (self.s.token(open).end, self.s.token(close).start);
        Ok((
            Ctl {
                cond,
                bound: None,
                head,
                body,
                cut,
            },
            open,
            close,
        ))
    }

    fn finish(&self, kind: Kind, k0: usize, end: usize) -> Stmt {
        let start = self.s.token(k0).start;
        Stmt {
            kind,
            start,
            end,
            line: self.s.line(start),
        }
    }

    fn parse_for(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let (ctl, open, close) = self.parse_headed()?;
        let semicolons = self.split_top(open + 1, close, ";");
        let classic = semicolons.len() > 1;
        let colons = self.split_top(open + 1, close, ":");
        let range = !classic && colons.len() > 1;
        let header = if semicolons.len() == 3 {
            self.convertible_header(&semicolons)
        } else {
            None
        };
        let parts = if classic { semicolons } else { colons };
        let parts = parts.iter().map(|&(a, b)| self.byte_range(a, b)).collect();
        let end = ctl.body.end;
        Ok(self.finish(
            Kind::For(ForStmt {
                ctl,
                range,
                header,
                parts,
            }),
            k0,
            end,
        ))
    }

    /// Byte range of the tokens `[a, b)`; empty at the first token's start when there are none.
    fn byte_range(&self, a: usize, b: usize) -> (usize, usize) {
        if a >= b {
            let at = self.s.token(a).start;
            return (at, at);
        }
        (self.s.token(a).start, self.s.token(b - 1).end)
    }

    /// The exact form `T i = a; i < b; ++i | i++ | i += s`, with `T` an integer type.
    fn convertible_header(&self, parts: &[(usize, usize)]) -> Option<ForHeader> {
        if parts
            .iter()
            .any(|&(a, b)| self.split_top(a, b, ",").len() > 1)
        {
            return None;
        }
        let w = |p: (usize, usize)| (p.0..p.1).map(|j| self.s.word(j)).collect::<Vec<_>>();
        let (init, cond, step) = (w(parts[0]), w(parts[1]), w(parts[2]));
        let eq = init.iter().position(|&x| x == "=")?;
        let index = *init.get(eq.checked_sub(1)?)?;
        let type_words = &init[..eq - 1];
        let integer = !type_words.is_empty() && type_words.iter().all(|t| INDEX_TYPES.contains(t));
        if !integer || !is_name(index) || eq + 1 >= init.len() {
            return None;
        }
        if cond.len() < 3 || cond[0] != index || cond[1] != "<" {
            return None;
        }
        let step_range = match step.as_slice() {
            ["++", i] | [i, "++"] if *i == index => None,
            [i, "+=", _, ..] if *i == index => Some(self.byte_range(parts[2].0 + 2, parts[2].1)),
            _ => return None,
        };
        Some(ForHeader {
            index: index.to_string(),
            start: self.byte_range(parts[0].0 + eq + 1, parts[0].1),
            stop: self.byte_range(parts[1].0 + 2, parts[1].1),
            step: step_range,
        })
    }

    /// `aether::bounded(C, N)` filling the header `[from, to)` exactly.
    fn bounded_args(&self, from: usize, to: usize) -> Option<Bounded> {
        let named = self.s.word(from) == "aether"
            && self.s.word(from + 1) == "::"
            && self.s.word(from + 2) == "bounded"
            && self.s.word(from + 3) == "(";
        if !named || self.matching(from + 3) != Some(to - 1) {
            return None;
        }
        let args = self.split_top(from + 4, to - 1, ",");
        match args.as_slice() {
            [(a, b), (c, d)] if a < b && c < d => Some(Bounded {
                cond: self.byte_range(*a, *b),
                limit: self.byte_range(*c, *d),
            }),
            _ => None,
        }
    }

    fn parse_while(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let (mut ctl, open, close) = self.parse_headed()?;
        ctl.bound = self.bounded_args(open + 1, close);
        let end = ctl.body.end;
        Ok(self.finish(Kind::While(ctl), k0, end))
    }

    fn parse_switch(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        let (ctl, _, _) = self.parse_headed()?;
        let end = ctl.body.end;
        Ok(self.finish(Kind::Switch(ctl), k0, end))
    }

    fn parse_do(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        self.k += 1;
        let body = Box::new(self.parse_stmt()?);
        if self.word() != "while" {
            return self.err("`while` expected after the body of `do`");
        }
        let while_kw = self.k;
        self.k += 1;
        let (open, close) = self.parenthesis()?;
        let head = self.scan_range(open + 1, close)?;
        let mut end = close;
        if self.word() == ";" {
            end = self.k;
            self.k += 1;
        }
        let gaps = [
            (self.s.token(k0).start, body.start),
            (body.end, self.s.token(while_kw).end),
            (self.s.token(open).start, self.s.token(close).end),
        ];
        let cut = self.cut(&gaps);
        let last = self.s.token(end).end;
        let cond = (self.s.token(open).end, self.s.token(close).start);
        Ok(self.finish(
            Kind::Do(Ctl {
                cond,
                bound: None,
                head,
                body,
                cut,
            }),
            k0,
            last,
        ))
    }

    fn parse_try(&mut self) -> Res<Stmt> {
        let k0 = self.k;
        self.k += 1;
        let mut blocks = vec![self.parse_block()?];
        while self.word() == "catch" {
            self.k += 1;
            self.parenthesis()?;
            blocks.push(self.parse_block()?);
        }
        let end = blocks.last().map_or(0, |b| b.end);
        Ok(self.finish(Kind::Try(blocks), k0, end))
    }
}

/// True for a word that can name a variable or label: an identifier that is no keyword that
/// starts a statement or an expression of its own.
#[must_use]
pub fn is_name(w: &str) -> bool {
    let starts = w
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_');
    starts
        && !matches!(
            w,
            "return"
                | "break"
                | "continue"
                | "goto"
                | "case"
                | "default"
                | "else"
                | "new"
                | "delete"
                | "throw"
                | "co_return"
                | "co_yield"
                | "sizeof"
                | "typedef"
                | "using"
                | "static_assert"
        )
}
