//! The loop family in the core: the body roots, the structural body keys and digest (the dedup identity of a Loop), the break form, and the
//! checks of a carried slot. All of it reads the node slots; nothing here calls a Python method of the loop classes.
use super::*;
use crate::canon::{body_nodes, is_a};
use rcore::keytext::{self, V};
use std::collections::HashMap;

const BODY_KEY_HEX: usize = 32;

/// What the body evaluates, in emission order: the carried results, every sink operand, then the break condition and values.
pub fn body_roots_of<'py>(lp: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut roots: Vec<Bound<'py, PyAny>> = vec![];
    for n in lp.getattr("nexts")?.try_iter()? { roots.push(n?) }
    for sk in lp.getattr("body_sinks")?.try_iter()? { for o in sk?.getattr("operands")?.try_iter()? { roots.push(o?) } }
    let ec = lp.getattr("exit_cond")?;
    if !ec.is_none() { roots.push(ec); for v in lp.getattr("exit_values")?.try_iter()? { roots.push(v?) } }
    Ok(roots)
}
#[pyfunction]
pub fn body_roots<'py>(lp: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> { body_roots_of(lp) }

/// Where the loop's break sits, from its values: "head", "tail", "middle", or None without a break.
#[pyfunction]
pub fn exit_form<'py>(py: Python<'py>, lp: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if lp.getattr("exit_cond")?.is_none() { return Ok(py.None().into_bound(py)) }
    let vals: Vec<_> = lp.getattr("exit_values")?.try_iter()?.collect::<PyResult<_>>()?;
    let same = |other: &str| -> PyResult<bool> {
        let o: Vec<_> = lp.getattr(other)?.try_iter()?.collect::<PyResult<_>>()?;
        Ok(vals.iter().zip(o.iter()).all(|(v, c)| v.is(c)))
    };
    let r = if same("carries")? { "head" } else if same("nexts")? { "tail" } else { "middle" };
    Ok(pyo3::types::PyString::new(py, r).into_any())
}

fn key_of_child(c: &Bound<'_, PyAny>, key_of: &mut HashMap<usize, String>) -> PyResult<String> {
    let id = c.as_ptr() as usize;
    if let Some(h) = key_of.get(&id) { return Ok(h.clone()) }
    let held = if is_a(c, "LoopIndex")? { "index".to_string() }
        else if is_a(c, "LoopCarry")? { format!("carry{}", c.getattr("slot")?.str()?) }
        else { return Err(he(format!("Loop body: {} is neither a boundary value, a placeholder nor a body node — the boundary computation and the body traversal disagree", rep(c)))) };
    key_of.insert(id, held.clone());
    Ok(held)
}

/// `(id -> structural key, the body's node list)`: boundary values by position in the loop's operands, placeholders by their slot, every other node
/// by a fixed-width hash of (kind, dtype, shape, tag, child keys, dedup identity).
fn body_keys<'py>(lp: &Bound<'py, PyAny>) -> PyResult<(HashMap<usize, String>, Vec<Bound<'py, PyAny>>)> {
    let mut key_of: HashMap<usize, String> = HashMap::new();
    for (k, o) in lp.getattr("operands")?.try_iter()?.enumerate() { key_of.entry(o?.as_ptr() as usize).or_insert(format!("free{k}")); }
    let body = body_nodes(lp)?;
    for node in &body {
        let mut parts = vec![];
        for c in node.getattr("operands")?.try_iter()? { parts.push(V::Str(key_of_child(&c?, &mut key_of)?)) }
        let t = node.getattr("ttype")?;
        let raw = keytext::text(&V::Tuple(vec![conv(&node.getattr("kind")?)?, conv(&t.getattr("dtype")?)?, conv(&t.getattr("shape")?)?, conv(&t.getattr("tag")?)?,
            V::Tuple(parts), conv(&node.getattr("dedup_extra")?)?]));
        key_of.insert(node.as_ptr() as usize, rcore::sha256::hex(raw.as_bytes())[..BODY_KEY_HEX].to_string());
    }
    Ok((key_of, body))
}
/// The byte width of every structural key of the loop's body, in body order (the diagnostic that sees an O(2**depth) key).
#[pyfunction]
pub fn body_key_widths<'py>(lp: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
    let (key_of, body) = body_keys(lp)?;
    Ok(body.iter().map(|n| key_of[&(n.as_ptr() as usize)].len()).collect())
}
/// A process-stable structural hash of the loop's body.
pub fn body_digest(lp: &Bound<'_, PyAny>) -> PyResult<String> {
    let (mut key_of, body) = body_keys(lp)?;
    let mut roots = vec![];
    for r in body_roots_of(lp)? {
        roots.push(match key_of.get(&(r.as_ptr() as usize)) { Some(h) => V::Str(h.clone()),
            None => if is_a(&r, "LoopIndex")? || is_a(&r, "LoopCarry")? { V::Str(key_of_child(&r, &mut key_of)?) } else { V::None } });
    }
    let mut data = b"hawk.ir.loop/2\n".to_vec();
    for n in &body { data.extend_from_slice(key_of[&(n.as_ptr() as usize)].as_bytes()); data.push(b'\n') }
    data.extend_from_slice(format!("roots|{}\n", keytext::text(&V::List(roots))).as_bytes());
    Ok(rcore::sha256::hex(&data)[..32].to_string())
}
/// The cached body digest of a loop (`_body_key`).
pub fn body_key<'py>(py: Python<'py>, lp: &Bound<'py, PyAny>) -> PyResult<String> {
    let c = lp.getattr(intern!(py, "_body_key"))?;
    if !c.is_none() { return c.extract() }
    let d = body_digest(lp)?;
    lp.setattr(intern!(py, "_body_key"), &d)?;
    Ok(d)
}

fn same_type(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> { Ok(a.getattr("ttype")?.eq(b.getattr("ttype")?)?) }
/// Index of the first element of `b` whose type differs from the type of the element of `a` at the same position, or -1.
pub fn first_mismatch(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<i64> {
    for (j, (x, y)) in a.try_iter()?.zip(b.try_iter()?).enumerate() { if !same_type(&x?, &y?)? { return Ok(j as i64) } }
    Ok(-1)
}
/// First fault of the carried slots, in the old scan order: (slot, code) with code 0 = initial value type, 1 = body result type, 2 = rank above 2.
pub fn carry_fault<'py>(py: Python<'py>, carries: &Bound<'py, PyAny>, inits: &Bound<'py, PyAny>, nexts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let c: Vec<_> = carries.try_iter()?.collect::<PyResult<_>>()?;
    let i: Vec<_> = inits.try_iter()?.collect::<PyResult<_>>()?;
    let n: Vec<_> = nexts.try_iter()?.collect::<PyResult<_>>()?;
    for j in 0..c.len() {
        if !same_type(&c[j], &i[j])? { return Ok(PyTuple::new(py, [j as i64, 0])?.into_any()) }
        if !same_type(&c[j], &n[j])? { return Ok(PyTuple::new(py, [j as i64, 1])?.into_any()) }
        if c[j].getattr("ttype")?.getattr("shape")?.len()? > 2 { return Ok(PyTuple::new(py, [j as i64, 2])?.into_any()) }
    }
    Ok(py.None().into_bound(py))
}
