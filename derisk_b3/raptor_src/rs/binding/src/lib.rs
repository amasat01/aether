//! The binding of the raptor core: pyo3 only. The construction checks are DECLARED in raptor/ir/core.json and run here by one
//! generic interpreter (one call per node, from the generated __init__); the walk/digest/verify run once per canonical() call.
use pyo3::exceptions::PyException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rcore::json::{self, J};
use rcore::keytext::V;
use rcore::{Dedup, Item, Rec, CLASSES, CONSTS, STR_CONSTS};
mod canon;
mod loops;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pyo3::create_exception!(raptor.ir.nodes, HawkError, PyException);

struct Reg { ttype: Py<PyAny>, node: Py<PyAny>, leaf: Py<PyAny>, by_ptr: HashMap<usize, usize>, leafish: Vec<bool>, by_name: HashMap<String, Py<PyAny>> }
static REG: OnceLock<Reg> = OnceLock::new();
static LOOP: Mutex<Option<Py<PyAny>>> = Mutex::new(None);
static MAN: OnceLock<Man> = OnceLock::new();

fn reg() -> &'static Reg { REG.get().expect("raptor._core.setup has not run") }
fn he(m: String) -> PyErr { HawkError::new_err(m) }
fn rep(o: &Bound<'_, PyAny>) -> String { o.repr().map(|s| s.to_string()).unwrap_or_else(|_| "<unprintable>".into()) }
fn tupv<'py>(py: Python<'py>, v: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, v) }
fn s<'py>(py: Python<'py>, v: &str) -> PyResult<Bound<'py, PyAny>> { Ok(v.into_pyobject(py)?.into_any()) }
fn str_const(name: &str) -> &'static str { STR_CONSTS.iter().find(|c| c.0 == name).map(|c| c.1).expect("manifest constant") }
fn const_of(name: &str) -> Option<&'static [&'static str]> { CONSTS.iter().find(|c| c.0 == name).map(|c| c.1) }

// ---------------- the declared constructor programs ----------------
// The manifest's `ctor` steps are compiled once (names resolved to slots, constants to cached tuples, builtins to numbers) and run by
// this interpreter. Nothing here knows a class: every refusal, its condition and its text come from raptor/ir/core.json.
#[derive(Clone, Copy, PartialEq)]
enum Cmp { Eq, Ne, Lt, Gt, In, NotIn, Is, IsNot }
enum E { Name(String), Var(usize), ConstT(usize), Int(i64), Str(String), Nil, True, False, Tuple(Vec<E>), Attr(Box<E>, String), Index(Box<E>, Box<E>),
         Call(usize, Vec<E>), Not(Box<E>), And(Box<E>, Box<E>), Or(Box<E>, Box<E>), Cmp(Box<E>, Cmp, Box<E>), Class(String), RawCall(String, Vec<E>) }
enum Piece { Lit(String), Val(E, bool) }
enum Step { Let(usize, E), Check(E, Vec<Piece>), Node(E, E), Call(usize, Vec<E>) }
struct Prog { params: usize, nvars: usize, steps: Vec<Step>, out: Vec<E>, slots: Vec<String>, slot_objs: OnceLock<Vec<Py<PyAny>>> }
struct Man { progs: Vec<Prog>, by_class: HashMap<String, usize>, named: Vec<Prog>, named_idx: HashMap<String, usize>, node_base: usize, const_names: Vec<String>, key_types: Vec<String> }
static CONST_OBJS: OnceLock<Vec<Py<PyAny>>> = OnceLock::new();
const BUILTINS: &[&str] = &["len", "isinstance", "tt", "width_of", "rem", "int", "tuple", "cat", "opt", "head", "first_ne_ttype", "first_not_node", "cut_in_loop", "coalesce", "orelse", "auto_or_int", "name_of", "bool", "trip", "first_mismatch", "first_not_instance", "carry_fault"];
const RESERVED: &[&str] = &["_cls", "_t", "_o"];
const V_CLS: usize = 0; const V_T: usize = 1; const V_O: usize = 2;   // offsets after the parameters

struct P<'a> { c: &'a [char], i: usize }
impl<'a> P<'a> {
    fn ws(&mut self) { while self.i < self.c.len() && self.c[self.i] == ' ' { self.i += 1 } }
    fn kw(&mut self, w: &str) -> bool {
        self.ws();
        let n = w.chars().count();
        if self.i + n <= self.c.len() && self.c[self.i..self.i + n].iter().collect::<String>() == w
            && !self.c.get(self.i + n).map(|x| x.is_alphanumeric() || *x == '_').unwrap_or(false) { self.i += n; true } else { false }
    }
    fn sym(&mut self, w: &str) -> bool {
        self.ws();
        let n = w.chars().count();
        if self.i + n <= self.c.len() && self.c[self.i..self.i + n].iter().collect::<String>() == w { self.i += n; true } else { false }
    }
    fn or(&mut self) -> Result<E, String> { let mut l = self.and()?; while self.kw("or") { l = E::Or(Box::new(l), Box::new(self.and()?)) } Ok(l) }
    fn and(&mut self) -> Result<E, String> { let mut l = self.not()?; while self.kw("and") { l = E::And(Box::new(l), Box::new(self.not()?)) } Ok(l) }
    fn not(&mut self) -> Result<E, String> { if self.kw("not") { Ok(E::Not(Box::new(self.not()?))) } else { self.cmp() } }
    fn cmp(&mut self) -> Result<E, String> {
        let l = self.post()?;
        let op = if self.sym("==") { Cmp::Eq } else if self.sym("!=") { Cmp::Ne } else if self.sym("<") { Cmp::Lt } else if self.sym(">") { Cmp::Gt }
            else if self.kw("not") { if self.kw("in") { Cmp::NotIn } else { return Err("expected `in` after `not`".into()) } }
            else if self.kw("in") { Cmp::In } else if self.kw("is") { if self.kw("not") { Cmp::IsNot } else { Cmp::Is } } else { return Ok(l) };
        Ok(E::Cmp(Box::new(l), op, Box::new(self.post()?)))
    }
    fn post(&mut self) -> Result<E, String> {
        let mut e = self.atom()?;
        loop {
            if self.sym(".") { let n = self.name()?; e = E::Attr(Box::new(e), n) }
            else if self.sym("[") { let i = self.or()?; if !self.sym("]") { return Err("expected ]".into()) } e = E::Index(Box::new(e), Box::new(i)) }
            else { return Ok(e) }
        }
    }
    fn name(&mut self) -> Result<String, String> {
        self.ws(); let st = self.i;
        while self.i < self.c.len() && (self.c[self.i].is_alphanumeric() || self.c[self.i] == '_') { self.i += 1 }
        if st == self.i { return Err(format!("name expected at {}", self.i)) }
        Ok(self.c[st..self.i].iter().collect())
    }
    fn atom(&mut self) -> Result<E, String> {
        self.ws();
        match self.c.get(self.i) {
            Some('(') => {
                self.i += 1; let mut items = vec![]; let mut comma = false;
                loop {
                    if self.sym(")") { break }
                    items.push(self.or()?);
                    if self.sym(",") { comma = true } else if !self.sym(")") { return Err("expected , or )".into()) } else { break }
                }
                if items.len() == 1 && !comma { Ok(items.pop().unwrap()) } else { Ok(E::Tuple(items)) }
            }
            Some('\'') => {
                self.i += 1; let st = self.i;
                while self.i < self.c.len() && self.c[self.i] != '\'' { self.i += 1 }
                let t: String = self.c[st..self.i].iter().collect(); self.i += 1; Ok(E::Str(t))
            }
            Some(c) if c.is_ascii_digit() => { let st = self.i; while self.i < self.c.len() && self.c[self.i].is_ascii_digit() { self.i += 1 } Ok(E::Int(self.c[st..self.i].iter().collect::<String>().parse().unwrap())) }
            _ => {
                let n = self.name()?;
                match n.as_str() { "None" => return Ok(E::Nil), "True" => return Ok(E::True), "False" => return Ok(E::False), _ => {} }
                if self.sym("(") {
                    let mut args = vec![];
                    loop { if self.sym(")") { break } args.push(self.or()?); if !self.sym(",") { if !self.sym(")") { return Err("expected )".into()) } break } }
                    Ok(E::RawCall(n, args))
                } else { Ok(E::Name(n)) }
            }
        }
    }
}
struct Scope<'a> { vars: Vec<String>, consts: &'a [String], classes: &'a [String], errors: Vec<String>, who: String }
impl<'a> Scope<'a> {
    fn resolve(&mut self, e: E) -> E {
        match e {
            E::Name(n) => {
                if let Some(i) = self.vars.iter().position(|v| *v == n) { E::Var(i) }
                else if let Some(i) = self.consts.iter().position(|c| *c == n) { E::ConstT(i) }
                else if n == "TensorType" || n == "Node" || self.classes.contains(&n) { E::Class(n) }
                else { self.errors.push(format!("{}: unknown name {n}", self.who)); E::Nil }
            }
            E::Tuple(v) => E::Tuple(v.into_iter().map(|x| self.resolve(x)).collect()),
            E::Attr(a, n) => E::Attr(Box::new(self.resolve(*a)), n),
            E::Index(a, b) => E::Index(Box::new(self.resolve(*a)), Box::new(self.resolve(*b))),
            E::Not(a) => E::Not(Box::new(self.resolve(*a))),
            E::And(a, b) => E::And(Box::new(self.resolve(*a)), Box::new(self.resolve(*b))),
            E::Or(a, b) => E::Or(Box::new(self.resolve(*a)), Box::new(self.resolve(*b))),
            E::Cmp(a, o, b) => E::Cmp(Box::new(self.resolve(*a)), o, Box::new(self.resolve(*b))),
            E::RawCall(n, a) => {
                let args: Vec<E> = a.into_iter().map(|x| self.resolve(x)).collect();
                match BUILTINS.iter().position(|b| *b == n) { Some(i) => E::Call(i, args), None => { self.errors.push(format!("{}: unknown function {n}", self.who)); E::Nil } }
            }
            x => x,
        }
    }
    fn expr(&mut self, t: &str) -> E { match parse_expr(t) { Ok(e) => self.resolve(e), Err(m) => { self.errors.push(format!("{}: {m}", self.who)); E::Nil } } }
}
fn parse_expr(t: &str) -> Result<E, String> {
    let c: Vec<char> = t.chars().collect();
    let mut p = P { c: &c, i: 0 };
    let e = p.or()?; p.ws();
    if p.i != c.len() { return Err(format!("trailing text in `{t}` at {}", p.i)) }
    Ok(e)
}
fn tmpl(sc: &mut Scope, t: &str) -> Vec<Piece> {
    let mut out = vec![]; let mut lit = String::new(); let c: Vec<char> = t.chars().collect(); let mut i = 0;
    while i < c.len() {
        if c[i] == '{' {
            let mut j = i + 1; let mut depth = 1;
            while j < c.len() { if c[j] == '{' { depth += 1 } if c[j] == '}' { depth -= 1; if depth == 0 { break } } j += 1 }
            let mut inner: String = c[i + 1..j.min(c.len())].iter().collect(); let r = inner.ends_with("!r");
            if r { inner.truncate(inner.len() - 2) }
            if !lit.is_empty() { out.push(Piece::Lit(std::mem::take(&mut lit))) }
            out.push(Piece::Val(sc.expr(&inner), r)); i = j + 1;
        } else { lit.push(c[i]); i += 1 }
    }
    if !lit.is_empty() { out.push(Piece::Lit(lit)) }
    out
}
fn param_names(p: &str) -> Vec<String> {
    p.split(',').map(|x| x.trim()).filter(|x| !x.is_empty() && *x != "*").map(|x| x.split('=').next().unwrap().split(':').next().unwrap().trim().to_string()).collect()
}
fn compile_prog(who: &str, params: Vec<String>, steps: &[J], out: &[String], slots: Vec<String>, named: &HashMap<String, (usize, usize)>, consts: &[String], classes: &[String], errors: &mut Vec<String>) -> Prog {
    let mut sc = Scope { vars: params.clone(), consts, classes, errors: vec![], who: who.to_string() };
    for r in RESERVED { sc.vars.push(r.to_string()) }
    let mut cs = vec![];
    for st in steps {
        let a = st.arr();
        let t = |i: usize| a.get(i).and_then(|x| x.str()).unwrap_or("");
        match t(0) {
            "let" => { let e = sc.expr(t(2)); sc.vars.push(t(1).to_string()); cs.push(Step::Let(sc.vars.len() - 1, e)) }
            "check" => { let e = sc.expr(t(1)); let m = tmpl(&mut sc, t(2)); cs.push(Step::Check(e, m)) }
            "node" => { let x = sc.expr(t(1)); let y = sc.expr(t(2)); cs.push(Step::Node(x, y)) }
            "call" => {
                let args: Vec<E> = (2..a.len()).map(|i| sc.expr(t(i))).collect();
                match named.get(t(1)) {
                    None => sc.errors.push(format!("{who}: calls an undeclared check {}", t(1))),
                    Some((ix, np)) => { if *np != args.len() { sc.errors.push(format!("{who}: check {} takes {np} arguments, got {}", t(1), args.len())) } cs.push(Step::Call(*ix, args)) }
                }
            }
            x => sc.errors.push(format!("{who}: unknown step {x}")),
        }
    }
    let outs: Vec<E> = out.iter().map(|e| sc.expr(e)).collect();
    errors.extend(sc.errors);
    Prog { params: params.len(), nvars: sc.vars.len(), steps: cs, out: outs, slots, slot_objs: OnceLock::new() }
}
fn compile(text: &str) -> Result<Man, String> {
    let m = json::parse(text)?;
    let consts: Vec<String> = m.get("consts").map(|c| c.obj().iter().filter(|x| matches!(x.1, J::Arr(_))).map(|x| x.0.clone()).collect()).unwrap_or_default();
    let classes: Vec<String> = m.get("kinds").unwrap().arr().iter().filter_map(|k| k.get("class").and_then(|c| c.str()).map(|s| s.to_string())).collect();
    let mut named_map: HashMap<String, (usize, usize)> = HashMap::new();
    let chk = m.get("checks").map(|c| c.obj()).unwrap_or(&[]);
    for (i, (k, v)) in chk.iter().enumerate() { named_map.insert(k.clone(), (i, v.get("params").map(|p| p.arr().len()).unwrap_or(0))); }
    let mut errors = vec![];
    let mut named = vec![]; let mut named_idx = HashMap::new();
    for (k, v) in chk {
        named_idx.insert(k.clone(), named.len());
        named.push(compile_prog(k, v.get("params").unwrap().strs(), v.get("steps").unwrap().arr(), &[], vec![], &named_map, &consts, &classes, &mut errors));
    }
    let mut progs = vec![]; let mut by_class = HashMap::new();
    for k in m.get("kinds").unwrap().arr() {
        if let Some(c) = k.get("ctor") {
            let cls = k.get("class").unwrap().str().unwrap().to_string();
            by_class.insert(cls.clone(), progs.len());
            let mut slots = vec!["ttype".to_string(), "operands".to_string()];
            slots.extend(k.get("assigns").map(|a| a.strs()).unwrap_or_default());
            progs.push(compile_prog(&cls, param_names(k.get("params").and_then(|p| p.str()).unwrap_or("")), c.get("steps").unwrap().arr(), &c.get("out").unwrap().strs(), slots, &named_map, &consts, &classes, &mut errors));
        }
    }
    if !errors.is_empty() { errors.sort(); return Err(errors.join("\n")) }
    let node_base = *named_idx.get("node_base").ok_or("the manifest declares no `node_base` check")?;
    Ok(Man { progs, by_class, named, named_idx, node_base, const_names: consts, key_types: m.get("key_text").map(|k| k.strs()).unwrap_or_default() })
}
/// Every problem a manifest text has: parse errors, unknown names, unknown functions, unknown checks, a wrong argument count.
#[pyfunction]
fn check_manifest(text: &str) -> Vec<String> {
    match compile(text) { Ok(_) => vec![], Err(e) => e.lines().map(|l| l.to_string()).collect() }
}

type Env<'py> = Vec<Option<Bound<'py, PyAny>>>;
fn truthy(v: &Bound<'_, PyAny>) -> PyResult<bool> { v.is_truthy() }
fn class_named<'py>(py: Python<'py>, n: &str) -> PyResult<Bound<'py, PyAny>> {
    let g = reg(); if n == "TensorType" { return Ok(g.ttype.bind(py).clone()) } if n == "Node" { return Ok(g.node.bind(py).clone()) }
    g.by_name.get(n).map(|c| c.bind(py).clone()).ok_or_else(|| he(format!("manifest: class {n} is not generated")))
}
fn cget<'a, 'py>(env: &'a Env<'py>, i: usize) -> PyResult<&'a Bound<'py, PyAny>> { env[i].as_ref().ok_or_else(|| he("manifest: a name is used before it is set".into())) }
fn eval<'py>(py: Python<'py>, e: &E, env: &Env<'py>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyBool;
    Ok(match e {
        E::Var(i) => cget(env, *i)?.clone(),
        E::ConstT(i) => CONST_OBJS.get().unwrap()[*i].bind(py).clone(),
        E::Class(n) => class_named(py, n)?,
        E::Int(i) => i.into_pyobject(py)?.into_any(),
        E::Str(t) => s(py, t)?,
        E::Nil => py.None().into_bound(py),
        E::True => PyBool::new(py, true).to_owned().into_any(),
        E::False => PyBool::new(py, false).to_owned().into_any(),
        E::Tuple(v) => tupv(py, v.iter().map(|x| eval(py, x, env)).collect::<PyResult<Vec<_>>>()?)?.into_any(),
        E::Attr(a, n) => eval(py, a, env)?.getattr(n.as_str())?,
        E::Index(a, i) => eval(py, a, env)?.get_item(eval(py, i, env)?)?,
        E::Not(a) => PyBool::new(py, !truthy(&eval(py, a, env)?)?).to_owned().into_any(),
        E::And(a, b) => PyBool::new(py, truthy(&eval(py, a, env)?)? && truthy(&eval(py, b, env)?)?).to_owned().into_any(),
        E::Or(a, b) => PyBool::new(py, truthy(&eval(py, a, env)?)? || truthy(&eval(py, b, env)?)?).to_owned().into_any(),
        E::Cmp(a, op, b) => PyBool::new(py, cmp(py, a, *op, b, env)?).to_owned().into_any(),
        E::Call(n, a) => call(py, *n, a, env)?,
        E::Name(_) | E::RawCall(..) => return Err(he("manifest: unresolved expression".into())),
    })
}
fn cond<'py>(py: Python<'py>, e: &E, env: &Env<'py>) -> PyResult<bool> {
    match e {
        E::Cmp(a, op, b) => cmp(py, a, *op, b, env),
        E::And(a, b) => Ok(cond(py, a, env)? && cond(py, b, env)?),
        E::Or(a, b) => Ok(cond(py, a, env)? || cond(py, b, env)?),
        E::Not(a) => Ok(!cond(py, a, env)?),
        _ => truthy(&eval(py, e, env)?),
    }
}
fn cmp<'py>(py: Python<'py>, a: &E, op: Cmp, b: &E, env: &Env<'py>) -> PyResult<bool> {
    let l = eval(py, a, env)?;
    Ok(match op {
        Cmp::In | Cmp::NotIn => {
            let hit = if let E::ConstT(i) = b {
                let man = MAN.get().unwrap(); let name = &man.const_names[*i];
                let c = const_of(name).unwrap();
                l.extract::<&str>().map(|x| c.contains(&x)).unwrap_or(false)
            } else { eval(py, b, env)?.contains(&l)? };
            hit == (op == Cmp::In)
        }
        _ => { let rv = eval(py, b, env)?; match op { Cmp::Eq => l.eq(&rv)?, Cmp::Ne => l.ne(&rv)?, Cmp::Lt => l.lt(&rv)?, Cmp::Gt => l.gt(&rv)?, Cmp::Is => l.is(&rv), _ => !l.is(&rv) } }
    })
}
fn call<'py>(py: Python<'py>, n: usize, a: &[E], env: &Env<'py>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyBool;
    match n {
        13 => { let x = eval(py, &a[0], env)?; return if x.is_none() { eval(py, &a[1], env) } else { Ok(x) } }   // coalesce (lazy)
        14 => { let x = eval(py, &a[0], env)?; return if truthy(&x)? { Ok(x) } else { eval(py, &a[1], env) } } // orelse (lazy)
        _ => {}
    }
    let v = a.iter().map(|x| eval(py, x, env)).collect::<PyResult<Vec<_>>>()?;
    let boolv = |b: bool| PyBool::new(py, b).to_owned().into_any();
    Ok(match n {
        0 => v[0].len()?.into_pyobject(py)?.into_any(),
        1 => boolv(v[0].is_instance(&v[1])?),
        2 => reg().ttype.bind(py).call1((PyTuple::empty(py), &v[0]))?,
        3 => { let mut w: i64 = 0; for x in v[0].try_iter()? { let mut k: i64 = 1; for e in x?.getattr("ttype")?.getattr("shape")?.try_iter()? { k *= e?.extract::<i64>()? } w += k }
            w.into_pyobject(py)?.into_any() }
        4 => v[0].rem(&v[1])?,
        5 => py.get_type::<pyo3::types::PyInt>().call1((&v[0],))?,
        6 => py.get_type::<PyTuple>().call1((&v[0],))?,
        7 => { let mut o: Vec<Bound<'py, PyAny>> = vec![]; for x in &v { for i in x.try_iter()? { o.push(i?) } } tupv(py, o)?.into_any() }
        8 => if v[0].is_none() { PyTuple::empty(py).into_any() } else { tupv(py, vec![v[0].clone()])?.into_any() },
        9 => { let t: String = v[0].extract()?; s(py, t.split('_').next().unwrap_or(""))? }
        10 => { let first = v[0].get_item(0)?.getattr("ttype")?; for b in v[0].try_iter()?.skip(1) { let b = b?; if b.getattr("ttype")?.ne(&first)? { return Ok(b) } } py.None().into_bound(py) }
        11 => { let node = reg().node.bind(py); for o in v[0].try_iter()? { let o = o?; if !o.is_instance(node)? { return Ok(o) } } py.None().into_bound(py) }
        12 => { let mut cut = false;
            for o in v[0].try_iter()? { if o?.getattr(intern!(py, "kind"))?.extract::<&str>().map(|k| k == "reduce").unwrap_or(false) { cut = true; break } }
            let mut hit = false;
            if cut { for c in v[0].try_iter()? { let k = c?.getattr(intern!(py, "kind")); if k.ok().and_then(|k| k.extract::<&str>().ok().map(|k| k.starts_with("loop"))).unwrap_or(false) { hit = true } } }
            boolv(hit) }
        15 => if v[0].eq("auto")? { v[0].clone() } else { py.get_type::<pyo3::types::PyInt>().call1((&v[0],))? },
        16 => v[0].getattr("__name__")?,
        17 => boolv(v[0].is_truthy()?),
        18 => { let (a, b, c): (i64, i64, i64) = (v[0].extract()?, v[1].extract()?, v[2].extract()?);
            let n = if c > 0 { if b > a { (b - a + c - 1) / c } else { 0 } } else if c < 0 { if a > b { (a - b - c - 1) / (-c) } else { 0 } } else { 0 };
            n.into_pyobject(py)?.into_any() }
        19 => loops::first_mismatch(&v[0], &v[1])?.into_pyobject(py)?.into_any(),
        20 => { let mut at: i64 = -1; for (j, x) in v[0].try_iter()?.enumerate() { if !x?.is_instance(&v[1])? { at = j as i64; break } } at.into_pyobject(py)?.into_any() }
        21 => loops::carry_fault(py, &v[0], &v[1], &v[2])?,
        _ => return Err(he("manifest: unknown function".into())),
    })
}
fn render(py: Python<'_>, t: &[Piece], env: &Env<'_>) -> PyResult<String> {
    let mut o = String::new();
    for p in t { match p { Piece::Lit(x) => o.push_str(x), Piece::Val(e, r) => { let v = eval(py, e, env)?; o.push_str(&if *r { rep(&v) } else { v.str()?.to_string() }) } } }
    Ok(o)
}
fn run<'py>(py: Python<'py>, man: &Man, p: &Prog, env: &mut Env<'py>) -> PyResult<()> {
    for st in &p.steps {
        match st {
            Step::Let(i, e) => { let v = eval(py, e, env)?; env[*i] = Some(v); }
            Step::Check(c, t) => if cond(py, c, env)? { return Err(he(render(py, t, env)?)) },
            Step::Node(tt, ops) => {
                let ttype = eval(py, tt, env)?;
                let o = eval(py, ops, env)?;
                let ops = match o.cast::<PyTuple>() { Ok(t) => t.clone(), Err(_) => PyTuple::new(py, o.try_iter()?.collect::<PyResult<Vec<_>>>()?)? };
                let base = &man.named[man.node_base];
                let mut e2: Env<'py> = vec![None; base.nvars];
                e2[0] = env[p.params + V_CLS].clone(); e2[1] = Some(ttype.clone()); e2[2] = Some(ops.clone().into_any());
                run(py, man, base, &mut e2)?;
                env[p.params + V_T] = Some(ttype); env[p.params + V_O] = Some(ops.into_any());
            }
            Step::Call(ix, args) => {
                let c = &man.named[*ix];
                let mut e2: Env<'py> = vec![None; c.nvars];
                for (k, a) in args.iter().enumerate() { e2[k] = Some(eval(py, a, env)?); }
                e2[c.params + V_CLS] = env[p.params + V_CLS].clone();
                run(py, man, c, &mut e2)?;
                for off in [V_T, V_O] { if let Some(v) = &e2[c.params + off] { env[p.params + off] = Some(v.clone()); } }
            }
        }
    }
    Ok(())
}
/// One construction: the declared program of ``idx`` over the arguments; sets the node's slots itself (no Python hop per slot).
#[pyfunction]
#[pyo3(signature = (idx, this, *args))]
fn construct<'py>(py: Python<'py>, idx: usize, this: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>) -> PyResult<()> {
    let man = MAN.get().expect("raptor._core.setup has not run");
    let p = &man.progs[idx];
    let mut env: Env<'py> = vec![None; p.nvars];
    for (k, a) in args.iter().enumerate().take(p.params) { env[k] = Some(a); }
    env[p.params + V_CLS] = Some(this.get_type().into_any());
    run(py, man, p, &mut env)?;
    let names = p.slot_objs.get_or_init(|| p.slots.iter().map(|n| pyo3::types::PyString::intern(py, n).into_any().unbind()).collect());
    let set = |k: usize, v: &Bound<'py, PyAny>| -> PyResult<()> {
        let rc = unsafe { pyo3::ffi::PyObject_GenericSetAttr(this.as_ptr(), names[k].as_ptr(), v.as_ptr()) };
        if rc < 0 { Err(PyErr::fetch(py)) } else { Ok(()) }
    };
    set(0, cget(&env, p.params + V_T)?)?; set(1, cget(&env, p.params + V_O)?)?;
    for (k, e) in p.out.iter().enumerate() { set(k + 2, &eval(py, e, &env)?)?; }
    Ok(())
}
/// A named declared check (``refuse_factor_arms`` ...), run over its arguments.
#[pyfunction]
#[pyo3(signature = (name, *args))]
fn run_check<'py>(py: Python<'py>, name: &str, args: &Bound<'py, PyTuple>) -> PyResult<()> {
    let man = MAN.get().expect("setup");
    let c = &man.named[*man.named_idx.get(name).ok_or_else(|| he(format!("no check {name}")))?];
    let mut env: Env<'py> = vec![None; c.nvars];
    for (k, a) in args.iter().enumerate().take(c.params) { env[k] = Some(a); }
    run(py, man, c, &mut env)
}
#[pyfunction]
fn ctor_index(class: &str) -> Option<usize> { MAN.get().and_then(|m| m.by_class.get(class).copied()) }

// ---------------- the dedup identity, from the table ----------------
/// A Python value as a key-text value (rcore::keytext): the exact types None, bool, int, float, str, tuple, list; any other type is
/// refused by name (no fallback to repr).
fn conv(o: &Bound<'_, PyAny>) -> PyResult<V> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    if o.is_none() { return Ok(V::None) }
    if o.is_exact_instance_of::<PyBool>() { return Ok(V::Bool(o.is_truthy()?)) }
    if o.is_exact_instance_of::<PyInt>() {
        return o.extract::<i128>().map(V::Int).map_err(|_| he("key text: an int beyond 128 bits has no written rule".into()));
    }
    if o.is_exact_instance_of::<PyFloat>() { return Ok(V::Float(o.extract::<f64>()?)) }
    if o.is_exact_instance_of::<PyString>() { return Ok(V::Str(o.extract::<String>()?)) }
    if let Ok(t) = o.cast_exact::<PyTuple>() { return Ok(V::Tuple(t.iter().map(|x| conv(&x)).collect::<PyResult<_>>()?)) }
    if let Ok(l) = o.cast_exact::<PyList>() { return Ok(V::List(l.iter().map(|x| conv(&x)).collect::<PyResult<_>>()?)) }
    let t = o.get_type();
    // an object that states its own key text (the `__key_text__` hook of its class: the text must itself follow the rule, section 10)
    // only a type the manifest DECLARES (core.json `key_text`) may state its own text; any other object that has the method is refused by name
    let tn = t.name()?.to_string();
    let declared = MAN.get().map_or(false, |m| m.key_types.iter().any(|k| *k == tn));
    if t.hasattr("__key_text__")? {
        if !declared { return Err(he(format!("key text: type {tn} has a __key_text__ method but is not declared in the manifest (key_text), so it has no written rule"))) }
        return Ok(V::Text(o.call_method0("__key_text__")?.extract::<String>().map_err(|_| he("key text: __key_text__ must return a str".into()))?))
    }
    if declared { return Err(he(format!("key text: type {tn} is declared in the manifest but has no __key_text__ method"))) }
    Err(he(format!("key text: a value of type {} has no written rule (None, bool, int, float, str, tuple and list have; any other type states its own text through a __key_text__ method)", t.name()?)))
}
/// The key text of any value of those kinds (the rule of GA/SPEC.md section 10).
#[pyfunction]
fn keytext(o: &Bound<'_, PyAny>) -> PyResult<String> { Ok(rcore::keytext::text(&conv(o)?)) }


// ---------------- the schema / manifest validator (rcore::manifest), one implementation ----------------
use rcore::pjson::Json;
fn to_json(o: &Bound<'_, PyAny>) -> PyResult<Json> {
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};
    if o.is_none() { return Ok(Json::Null) }
    if o.is_instance_of::<PyBool>() { return Ok(Json::Bool(o.is_truthy()?)) }
    if o.is_instance_of::<PyInt>() { if let Ok(i) = o.extract::<i128>() { return Ok(Json::Int(i)) } }
    else if o.is_instance_of::<PyFloat>() { return Ok(Json::Float(o.extract::<f64>()?)) }
    else if o.is_instance_of::<PyString>() { return Ok(Json::Str(o.extract::<String>()?)) }
    else if let Ok(d) = o.cast::<PyDict>() {
        let mut v = Vec::with_capacity(d.len());
        for (k, x) in d.iter() { let ks = if k.is_instance_of::<PyString>() { k.extract::<String>()? } else { k.str()?.extract::<String>()? }; v.push((ks, to_json(&x)?)) }
        return Ok(Json::Obj(v))
    }
    else if let Ok(l) = o.cast::<PyList>() { return Ok(Json::Arr(l.iter().map(|x| to_json(&x)).collect::<PyResult<_>>()?)) }
    else if let Ok(t) = o.cast::<PyTuple>() { return Ok(Json::Tup(t.iter().map(|x| to_json(&x)).collect::<PyResult<_>>()?)) }
    Ok(Json::Other(o.get_type().name()?.to_string(), o.repr()?.extract::<String>()?))
}
fn refuse(m: String) -> PyErr {
    match m.split_once(": ") {
        Some(("TypeError", t)) => pyo3::exceptions::PyTypeError::new_err(t.to_string()),
        Some((_, t)) => pyo3::exceptions::PyValueError::new_err(t.to_string()),
        None => pyo3::exceptions::PyValueError::new_err(m),
    }
}
/// `which` names one validator of raptor.schema.manifest (or "document" for raptor.schema.validate_manifest); raises ValueError / TypeError
/// with the text the Python validators gave. Returns the schema version for "schema_version", else None.
#[pyfunction]
#[pyo3(signature = (which, meta, name="<document>", version=1, legacy=false))]
fn schema_check(which: &str, meta: &Bound<'_, PyAny>, name: &str, version: i64, legacy: bool) -> PyResult<Option<i64>> {
    use rcore::manifest as m;
    let j = to_json(meta)?;
    let nm = rcore::keytext::py_str(name);
    let r = match which {
        "document" => m::validate(&j),
        "block_descriptor" => m::validate_block_descriptor(&j, name),
        "schema_version" => return m::check_schema_version(&j, &nm, legacy).map(Some).map_err(refuse),
        "execution_axis" => m::check_execution_axis(&j, version, &nm),
        "lane_space" => m::check_lane_space(&j, &nm),
        "launch_fields" => m::check_launch_fields(&j, &nm),
        "team_fields" => m::check_team_fields(&j, &nm),
        "index_planes" => m::check_index_planes(&j, &nm),
        _ => return Err(he(format!("schema_check: no validator {which}"))),
    };
    r.map(|_| None).map_err(refuse)
}

fn item<'py>(py: Python<'py>, node: &Bound<'py, PyAny>, it: &Item) -> PyResult<Option<V>> {
    Ok(Some(match it {
        Item::Field(f) => conv(&node.getattr(*f)?)?,
        Item::ReprOf(f) => V::Str(rcore::keytext::text(&conv(&node.getattr(*f)?)?)),
        Item::Lit(l) => V::Str(l.to_string()),
        Item::Names(f) => { let mut o = vec![]; for x in node.getattr(*f)?.try_iter()? { o.push(conv(&x?.getattr("name")?)?) } V::Tuple(o) }
        Item::SinkKeys(f) => { let mut o = vec![]; for x in node.getattr(*f)?.try_iter()? { let x = x?; o.push(V::Tuple(vec![conv(&x.getattr("kind")?)?, conv(&x.getattr("role")?)?, conv(&x.getattr("name")?)?])) } V::Tuple(o) }
        Item::Body => V::Str(loops::body_key(py, node)?),
        Item::LitIfSet(l, f) => { if node.getattr(*f)?.is_none() { return Ok(None) } V::Str(l.to_string()) }
        Item::FieldIfSet(f, g) => { if node.getattr(*g)?.is_none() { return Ok(None) } conv(&node.getattr(*f)?)? }
        Item::K => V::Int(node.getattr(intern!(py, "operands"))?.len()? as i128 - 1),
        Item::FieldUnless(f, set) => {
            let v = node.getattr(*f)?;
            if v.extract::<String>().map(|x| set.contains(&x.as_str())).unwrap_or(false) { return Ok(None) }
            conv(&v)?
        }
    }))
}
fn spec_of(node: &Bound<'_, PyAny>) -> Option<usize> { reg().by_ptr.get(&(node.get_type().as_ptr() as usize)).copied() }

fn dedup_value<'py>(py: Python<'py>, node: &Bound<'py, PyAny>, idx: Option<usize>) -> PyResult<V> {
    let d = match idx { Some(i) => &CLASSES[i].dedup, None => &Dedup::Py };
    match d {
        Dedup::None => Ok(V::None),
        Dedup::Py => conv(&node.getattr(intern!(py, "dedup_extra"))?),
        Dedup::One(i) => Ok(item(py, node, i)?.unwrap()),
        Dedup::Tup(items) => {
            let mut v = vec![];
            for i in items.iter() { if let Some(x) = item(py, node, i)? { v.push(x) } }
            Ok(V::Tuple(v))
        }
    }
}
#[pyfunction]
fn dedup_extra<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // the class of the node may be a subclass built elsewhere: walk its MRO for the first table class
    let mut idx = spec_of(node);
    if idx.is_none() {
        for c in node.get_type().getattr("__mro__")?.try_iter()? {
            if let Some(i) = reg().by_ptr.get(&(c?.as_ptr() as usize)) { idx = Some(*i); break }
        }
    }
    match idx.map(|i| &CLASSES[i].dedup) {
        Some(Dedup::Py) | None => Err(HawkError::new_err("dedup_extra: no table recipe")),
        _ => { let v = dedup_value(py, node, idx)?; value_obj(py, &v) }
    }
}

/// Compile the declared programs of the compiled-in manifest (idempotent).
#[pyfunction]
fn load(py: Python<'_>) -> PyResult<()> {
    if MAN.get().is_none() {
        let m = compile(rcore::MANIFEST).map_err(he)?;
        let objs = m.const_names.iter().map(|n| PyTuple::new(py, const_of(n).unwrap().iter().map(|x| x.to_string())).map(|t| t.into_any().unbind())).collect::<PyResult<Vec<_>>>()?;
        let _ = CONST_OBJS.set(objs);
        let _ = MAN.set(m);
    }
    Ok(())
}
#[pyfunction]
fn setup(py: Python<'_>, ttype: Py<PyAny>, node: Py<PyAny>, leaf: Py<PyAny>, classes: &Bound<'_, PyDict>) -> PyResult<()> {
    load(py)?;
    let mut by_ptr = HashMap::new();
    let mut by_name = HashMap::new();
    for (k, v) in classes.iter() {
        let name: String = k.extract()?;
        if let Some(i) = CLASSES.iter().position(|c| c.name == name) { by_ptr.insert(v.as_ptr() as usize, i); }
        by_name.insert(name, v.unbind());
    }
    let leafish = CLASSES.iter().map(|c| { let mut n = c.name; loop { if n == "Leaf" { return true } match CLASSES.iter().find(|x| x.name == n) { Some(x) if !x.base.is_empty() => n = x.base, _ => return false } } }).collect();
    let _ = REG.set(Reg { ttype, node, leaf, by_ptr, leafish, by_name });
    Ok(())
}
#[pyfunction]
fn set_loop(cls: Py<PyAny>) { *LOOP.lock().unwrap() = Some(cls); }
#[pyfunction]
fn manifest() -> &'static str { rcore::MANIFEST }

/// What the generated classes, the specs compiled into the core and a candidate table text agree on: [] when equal.
#[pyfunction]
fn spec_dump() -> Vec<(String, String, Vec<String>, bool)> {
    CLASSES.iter().map(|c| (c.name.to_string(), c.base.to_string(), c.slots.iter().map(|s| s.to_string()).collect(), c.binds)).collect()
}

/// The rule of a dialect kind: the operand bounds, the plane-leaf families and the result rule of its declaration (None: no declaration).
fn decl_rule(b: &Bound<'_, PyAny>) -> PyResult<Option<rcore::Rule>> {
    let py = b.py();
    let d = match b.getattr(intern!(py, "decl")) { Ok(d) if !d.is_none() => d, _ => return Ok(None) };
    let ops = d.getattr(intern!(py, "operands"))?;
    let (lo, hi, fams): (u32, Option<i32>, Vec<String>) = ops.extract()?;
    let res = d.getattr(intern!(py, "result"))?;
    let res = res.cast::<PyTuple>()?;
    let code: String = res.get_item(0)?.extract()?;
    let mut u = rcore::Rule { min: lo, max: hi.unwrap_or(-1), ..Default::default() };
    for (i, f) in fams.iter().enumerate() { if f == "plane-leaf" && i < 32 { u.plane |= 1 << i } }
    match code.as_str() {
        "same_as" => { u.res = 1; u.idx = res.get_item(1)?.extract()? }
        "scalar_of" => { u.res = 2; u.idx = res.get_item(1)?.extract()? }
        "fixed" => { u.res = 3; u.dtype = res.get_item(1)?.extract()?; u.shape = res.get_item(2)?.extract()? }
        _ => {}
    }
    Ok(Some(u))
}

// ---------------- the walk: one crossing ----------------
fn read<'py>(b: &Bound<'py, PyAny>) -> PyResult<(Rec, Vec<Bound<'py, PyAny>>, u8)> {
    // special: 1 binds a slot, 2 is a loop; bit 4 watched by the access/dispatch passes, bit 8 a leaf
    let py = b.py();
    let g = reg();
    let idx = spec_of(b);
    let ttype = b.getattr(intern!(py, "ttype"))?;
    let mut r = Rec {
        kind: b.getattr(intern!(py, "kind"))?.extract()?,
        dtype: ttype.getattr(intern!(py, "dtype"))?.extract()?,
        shape: ttype.getattr(intern!(py, "shape"))?.cast::<PyTuple>()?.extract()?,
        tag: ttype.getattr(intern!(py, "tag"))?.extract()?,
        ..Default::default()
    };
    let mut special = 0u8;
    r.rule = match idx { Some(i) => Some(rcore::core_rule(i)), None => decl_rule(b)? };
    match idx {
        Some(i) => {
            r.leaf = g.leafish[i];
            if CLASSES[i].binds { special = 1 }
            match CLASSES[i].watch { 1 => special |= 4, 2 => if r.kind == "raw_device" { special |= 4 }, _ => {} }
            if let Some(l) = LOOP.lock().unwrap().as_ref() { if b.is_instance(l.bind(py))? { special = (special & 4) | 2 } }
        }
        None => {
            r.leaf = b.is_instance(g.leaf.bind(py))?;
            if !b.getattr(intern!(py, "binding"))?.is_none() { special = 1 }
            special |= 4; // a class the table does not know: the passes look at it
            if let Some(l) = LOOP.lock().unwrap().as_ref() { if b.is_instance(l.bind(py))? { special = (special & 4) | 2 } }
        }
    }
    if r.leaf {
        special |= 8;
        r.role = b.getattr(intern!(py, "role"))?.extract()?;
        r.name = b.getattr(intern!(py, "name"))?.extract()?;
    } else {
        r.extra = rcore::keytext::text(&dedup_value(py, b, idx)?);
    }
    let kids = b.getattr(intern!(py, "operands"))?.cast::<PyTuple>()?.iter().collect::<Vec<_>>();
    Ok((r, kids, special))
}

/// ``(order, position, merges, preimage)``: the canonical order; ``id(node) -> position`` for every reached node; the
/// binding-bearing (and loop) nodes in visit order with their canonical node; the node part of the digest preimage.
#[pyfunction]
fn walk<'py>(py: Python<'py>, sinks: Vec<Bound<'py, PyAny>>) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>, Bound<'py, PyList>, String, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    let mut ids: HashMap<usize, u32> = HashMap::new();
    let (mut recs, mut objs, mut kids, mut special): (Vec<Rec>, Vec<Bound<'py, PyAny>>, Vec<Vec<Bound<'py, PyAny>>>, Vec<u8>) = (vec![], vec![], vec![], vec![]);
    let mut stack: Vec<Bound<'py, PyAny>> = sinks.to_vec();
    while let Some(b) = stack.pop() {
        let p = b.as_ptr() as usize;
        if ids.contains_key(&p) { continue }
        ids.insert(p, recs.len() as u32);
        let (r, ch, sp) = read(&b)?;
        for c in &ch { stack.push(c.clone()) }
        recs.push(r); objs.push(b); kids.push(ch); special.push(sp);
    }
    for (i, ch) in kids.iter().enumerate() { recs[i].ops = ch.iter().map(|c| ids[&(c.as_ptr() as usize)]).collect(); }
    let roots: Vec<u32> = sinks.iter().map(|s| ids[&(s.as_ptr() as usize)]).collect();
    let w = rcore::walk_full(&recs, &roots).map_err(he)?;
    let order = PyTuple::new(py, w.order.iter().map(|&i| objs[i as usize].clone()))?;
    let position = PyDict::new(py);
    for (i, o) in objs.iter().enumerate() { position.set_item(o.as_ptr() as usize, w.pos[i])?; }
    let merges = PyList::empty(py);
    for &v in &w.visits {
        let sp = special[v as usize];
        if sp & 3 == 0 { continue }
        let _ = special[v as usize];
        let canon = objs[w.order[w.pos[v as usize] as usize] as usize].clone();
        merges.append((objs[v as usize].clone(), canon))?;
    }
    let watched = PyTuple::new(py, w.order.iter().filter(|&&i| special[i as usize] & 4 != 0).map(|&i| objs[i as usize].clone()).collect::<Vec<_>>())?;
    let leaves = PyTuple::new(py, w.order.iter().filter(|&&i| special[i as usize] & 8 != 0).map(|&i| objs[i as usize].clone()).collect::<Vec<_>>())?;
    Ok((order, position, merges, w.pre, watched, leaves))
}

#[pyfunction]
fn digest(pre: &str, tail: &[u8]) -> String { rcore::digest(pre, tail) }

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HawkError", m.py().get_type::<HawkError>())?;
    macro_rules! f { ($($n:path),*) => { $( m.add_function(wrap_pyfunction!($n, m)?)?; )* } }
    f!(canon::canonical, canon::body_nodes, canon::quantity_expand, canon::companion, loops::body_roots, loops::exit_form, loops::body_key_widths, canon::set_loop_carry, canon::merge_bindings, canon::infer_access, canon::dispatch_nodes, canon::dispatch_listing, canon::written, canon::digest_of, keytext, schema_check, load, construct, ctor_index, run_check, check_manifest, dedup_extra, setup, set_loop, manifest, spec_dump, walk, digest);
    Ok(())
}

/// A key-text value back as the Python object (what `dedup_extra` returns to Python callers).
fn value_obj<'py>(py: Python<'py>, v: &V) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        V::None => py.None().into_bound(py),
        V::Bool(b) => pyo3::types::PyBool::new(py, *b).to_owned().into_any(),
        V::Int(i) => i.into_pyobject(py)?.into_any(),
        V::Float(f) => f.into_pyobject(py)?.into_any(),
        V::Text(x) => x.as_str().into_pyobject(py)?.into_any(),
        V::Str(x) => x.as_str().into_pyobject(py)?.into_any(),
        V::Tuple(t) => PyTuple::new(py, t.iter().map(|x| value_obj(py, x)).collect::<PyResult<Vec<_>>>()?)?.into_any(),
        V::List(t) => PyList::new(py, t.iter().map(|x| value_obj(py, x)).collect::<PyResult<Vec<_>>>()?)?.into_any(),
    })
}
