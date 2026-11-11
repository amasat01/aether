//! W2 step 1: the crossing microbenchmark core. Zero dependencies. One node record, the canonical walk and the
//! digest preimage, reproducing hawk.ir.walk._post_order / _key / _digest for the node part byte for byte.
pub mod sha256;
pub mod keytext;
pub mod nonprintable;
use std::collections::HashMap;

/// ONE declaration table: the Rust records and the generated Python classes (design D2) both come from it.
/// Line = Name|Base|own fields.
/// The attribute names the converter of design D2 reads from a Python node (checked against DECL by a test).
pub const READS: &str = "kind,ttype,operands,extra,role,name,dtype,shape,tag";
pub const DTYPES: [&str; 7] = ["f64", "f32", "i32", "i64", "bool", "int32", "uint32"];

pub const DECL: &str = "TType||shape,dtype,tag\nNode||kind,ttype,operands,extra\nOp|Node|literal\nConst|Node|literal\nLeaf|Node|role,name\n";

/// The manifest of raptor's core kinds (raptor/ir/core.json), compiled in; the generated specs below come from it by build.rs.
pub const MANIFEST: &str = include_str!("../../../raptor/ir/core.json");
pub mod json;
pub mod pjson;
pub mod manifest;
pub enum Item { Field(&'static str), ReprOf(&'static str), Lit(&'static str), K, FieldUnless(&'static str, &'static [&'static str]),
    /// a tuple of the `name` of each element of a field
    Names(&'static str),
    /// a tuple of (kind, role, name) of each sink of a field
    SinkKeys(&'static str),
    /// the structural digest of a loop body
    Body,
    /// a literal, only when a field is set (not None)
    LitIfSet(&'static str, &'static str),
    /// a field's value, only when a gate field is set
    FieldIfSet(&'static str, &'static str) }
pub enum Dedup { None, Py, One(Item), Tup(&'static [Item]) }
pub struct ClassSpec { pub name: &'static str, pub base: &'static str, pub slots: &'static [&'static str], pub binds: bool, pub dedup: Dedup, pub watch: u8 }
include!(concat!(env!("OUT_DIR"), "/specs.rs"));

/// The operand-count and result-type rule of one kind (core.json rows and dialect manifests alike): operands min..max (max -1 = many),
/// result code 0 carried, 1 same_as(idx), 2 scalar_of(idx), 3 fixed(dtype, shape); `plane` bit i = operand i must be a plane leaf.
#[derive(Clone, Default)]
pub struct Rule { pub min: u32, pub max: i32, pub res: u8, pub idx: u32, pub dtype: String, pub shape: Vec<i64>, pub plane: u32 }
pub fn core_rule(i: usize) -> Rule {
    let r = &RULES[i];
    Rule { min: r.0, max: r.1, res: r.2, idx: r.3, dtype: r.4.to_string(), shape: r.5.to_vec(), plane: 0 }
}

#[derive(Clone, Default)]
pub struct Rec {
    pub rule: Option<Rule>,
    pub leaf: bool,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub tag: Option<String>,
    pub ops: Vec<u32>,
    pub extra: String, // Python repr of dedup_extra, already text
    pub role: String,
    pub name: String,
}

pub use keytext::str_repr as repr_str;

fn repr_ints(v: &[i64], out: &mut String) {
    out.push('(');
    for (i, x) in v.iter().enumerate() {
        if i > 0 { out.push_str(", ") }
        out.push_str(&x.to_string());
    }
    if v.len() == 1 { out.push(',') }
    out.push(')');
}

fn key_of(r: &Rec, pos: &[u32], out: &mut String) {
    if r.leaf {
        out.push_str("('leaf', ");
        repr_str(&r.role, out); out.push_str(", "); repr_str(&r.name, out); out.push(')');
        return;
    }
    out.push('(');
    repr_str(&r.kind, out); out.push_str(", ");
    repr_str(&r.dtype, out); out.push_str(", ");
    repr_ints(&r.shape, out); out.push_str(", ");
    match &r.tag { None => out.push_str("None"), Some(t) => repr_str(t, out) }
    out.push_str(", ");
    let p: Vec<i64> = r.ops.iter().map(|&c| pos[c as usize] as i64).collect();
    repr_ints(&p, out);
    out.push_str(", ");
    out.push_str(&r.extra);
    out.push(')');
}

/// Post-order walk with structural dedup. Returns (order = rec ids in canonical order, preimage of the node lines).
pub struct WalkOut { pub order: Vec<u32>, pub pre: String, pub pos: Vec<u32>, pub visits: Vec<u32> }

pub fn walk(recs: &[Rec], roots: &[u32]) -> Result<(Vec<u32>, String), String> {
    let w = walk_full(recs, roots)?;
    Ok((w.order, w.pre))
}

/// The tuple form of a list of strings, as Python prints it: ('a', 'b').
pub fn repr_tuple_str(v: &[&str]) -> String {
    let mut o = String::from("(");
    for (i, x) in v.iter().enumerate() { if i > 0 { o.push_str(", ") } repr_str(x, &mut o) }
    if v.len() == 1 { o.push(',') }
    o.push(')');
    o
}

pub fn walk_full(recs: &[Rec], roots: &[u32]) -> Result<WalkOut, String> {
    verify(recs)?;
    let mut live = vec![false; recs.len()];
    const UNSET: u32 = u32::MAX;
    let mut pos = vec![UNSET; recs.len()];
    let mut by_key: HashMap<String, u32> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut pre = String::new();
    let mut visits: Vec<u32> = Vec::new();
    let mut stack: Vec<(u32, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    let mut key = String::new();
    while let Some((n, expanded)) = stack.pop() {
        if pos[n as usize] != UNSET { continue }
        let r = &recs[n as usize];
        if !expanded {
            stack.push((n, true));
            live[n as usize] = true;
            for &c in r.ops.iter().rev() {
                if live[c as usize] { return Err(format!("the graph has a cycle through a {} node", recs[c as usize].kind)) }
                if pos[c as usize] == UNSET { stack.push((c, false)) }
            }
            continue;
        }
        live[n as usize] = false;
        key.clear();
        key_of(r, &pos, &mut key);
        let p = match by_key.get(&key) {
            Some(&p) => p,
            None => {
                let p = order.len() as u32;
                order.push(n);
                pre.push_str(&p.to_string()); pre.push('|'); pre.push_str(&key); pre.push('\n');
                by_key.insert(key.clone(), p);
                p
            }
        };
        pos[n as usize] = p;
        visits.push(n);
    }
    Ok(WalkOut { order, pre, pos, visits })
}

pub fn digest(pre: &str, tail: &[u8]) -> String {
    let mut b = Vec::with_capacity(pre.len() + tail.len() + 16);
    b.extend_from_slice(b"hawk.ir.walk/1\n");
    b.extend_from_slice(pre.as_bytes());
    b.extend_from_slice(tail);
    sha256::hex(&b)
}

/// The checks the Rust side enforces at the boundary, on the records of ONE crossing.
pub fn verify(recs: &[Rec]) -> Result<(), String> {
    for (i, r) in recs.iter().enumerate() {
        if r.kind.is_empty() { return Err(format!("node {i}: empty kind")) }
        if !DTYPES.contains(&r.dtype.as_str()) { return Err(format!("node {i} ({}): unknown dtype {:?}", r.kind, r.dtype)) }
        if r.shape.iter().any(|&d| d < 0) { return Err(format!("node {i} ({}): negative extent in shape {:?}", r.kind, r.shape)) }
        if r.ops.iter().any(|&c| c as usize >= recs.len()) { return Err(format!("node {i}: operand out of range")) }
        if r.leaf && (!r.ops.is_empty() || r.role.is_empty() || r.name.is_empty()) {
            return Err(format!("node {i}: a leaf has no operands and carries a role and a name"))
        }
        if let Some(u) = &r.rule { check_rule(i, r, u, recs)? }
    }
    Ok(())
}

/// The operand count, the plane-leaf families and the result-type rule of one record (the declared rule of its kind).
fn check_rule(i: usize, r: &Rec, u: &Rule, recs: &[Rec]) -> Result<(), String> {
    let n = r.ops.len();
    if (n as u32) < u.min || (u.max >= 0 && n as i32 > u.max) {
        let hi = if u.max < 0 { "many".to_string() } else { u.max.to_string() };
        return Err(format!("node {i} ({}): takes {}..{hi} operand(s), got {n}", r.kind, u.min));
    }
    for (j, &c) in r.ops.iter().enumerate() {
        if j < 32 && u.plane >> j & 1 == 1 && !recs[c as usize].leaf {
            return Err(format!("node {i} ({}): operand {j} must be a plane leaf", r.kind));
        }
    }
    let same = |o: &Rec| o.dtype == r.dtype && o.shape == r.shape && o.tag == r.tag;
    let ok = match u.res {
        1 => same(&recs[r.ops[u.idx as usize] as usize]),
        2 => { let o = &recs[r.ops[u.idx as usize] as usize]; r.shape.is_empty() && o.dtype == r.dtype }
        3 => r.dtype == u.dtype && r.shape == u.shape,
        _ => true,
    };
    if !ok {
        let want = match u.res {
            1 => format!("the type of operand {}", u.idx),
            2 => format!("a scalar of operand {}'s dtype", u.idx),
            _ => format!("{} {:?}", u.dtype, u.shape),
        };
        return Err(format!("node {i} ({}): type {} {:?} is not what its result rule derives, {want}", r.kind, r.dtype, r.shape));
    }
    Ok(())
}
