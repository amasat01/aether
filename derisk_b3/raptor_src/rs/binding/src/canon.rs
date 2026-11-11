//! The parts of `canonical()` that used to run in Python over the short lists the walk hands back: the total merge of the bindings,
//! the access classification, the dispatch listing and the digest tail. They read nodes through the same attributes the Python code did.
use super::*;
use rcore::keytext::{self, V};
use std::collections::HashSet;

static LOOP_CARRY: Mutex<Option<Py<PyAny>>> = Mutex::new(None);
#[pyfunction]
pub fn set_loop_carry(cls: Py<PyAny>) { *LOOP_CARRY.lock().unwrap() = Some(cls); }

pub(crate) fn is_a(b: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    match reg().by_name.get(name) { Some(c) => b.is_instance(c.bind(b.py())), None => Ok(false) }
}
pub(crate) fn is_loop(b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let g = LOOP.lock().unwrap(); match g.as_ref() { Some(c) => b.is_instance(c.bind(b.py())), None => Ok(false) }
}
fn is_carry(b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let g = LOOP_CARRY.lock().unwrap(); match g.as_ref() { Some(c) => b.is_instance(c.bind(b.py())), None => Ok(false) }
}
pub(crate) fn st<'py>(b: &Bound<'py, PyAny>, a: &str) -> PyResult<String> { b.getattr(a)?.extract() }

/// The TOTAL merge: ``merges`` is the walk's list of (node, canonical node); returns {(role, name): the binding node}, a Loop's body sinks included.
#[pyfunction]
pub fn merge_bindings<'py>(py: Python<'py>, merges: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    fn one<'py>(out: &Bound<'py, PyDict>, node: &Bound<'py, PyAny>, canon: &Bound<'py, PyAny>) -> PyResult<()> {
        let slot = node.getattr("binding")?;
        if slot.is_none() { return Ok(()) }
        let key = (st(&slot, "role")?, st(&slot, "name")?);
        match out.get_item(&key)? {
            None => out.set_item(key, canon)?,
            Some(held) => {
                let (ht, nt) = (held.getattr("ttype")?, node.getattr("ttype")?);
                if ht.ne(&nt)? {
                    return Err(he(format!("leaf identity collision on (role={}, name={}): {} is typed {} but {} is typed {}. A slot is keyed on \
(role, name) ALONE, so these two would share one slot; the merge is TOTAL and the disagreement is refused, never silently collapsed.",
                        rep(&key.0.as_str().into_pyobject(out.py())?.into_any()), rep(&key.1.as_str().into_pyobject(out.py())?.into_any()),
                        held.get_type().name()?, rep(&ht), node.get_type().name()?, rep(&nt))))
                }
            }
        }
        Ok(())
    }
    for (node, canon) in &merges {
        one(&out, node, canon)?;
        if is_loop(node)? { for s in node.getattr("body_sinks")?.try_iter()? { let s = s?; one(&out, &s, &s)? } }
    }
    Ok(out)
}

fn with_loop_bodies<'py>(order: &[Bound<'py, PyAny>], out: &mut Vec<Bound<'py, PyAny>>) -> PyResult<()> {
    for n in order {
        out.push(n.clone());
        if is_loop(n)? {
            for s in n.getattr("body_sinks")?.try_iter()? { out.push(s?) }
            let body = body_nodes(n)?;
            with_loop_bodies(&body, out)?;
        }
    }
    Ok(())
}
fn step_word<'py>(count: &Bound<'py, PyAny>) -> PyResult<Option<usize>> {
    if count.is_none() { return Ok(None) }
    let (mut stack, mut seen) = (vec![count.clone()], HashSet::new());
    while let Some(n) = stack.pop() {
        if !seen.insert(n.as_ptr() as usize) { continue }
        if is_a(&n, "At")? {
            let plane = n.getattr("plane")?;
            if st(&plane, "role")? == "lookup" && st(&plane, "name")? == str_const("FUSED_STEPS_PLANE")
                && is_a(&n.getattr("index")?, "Const")? { return Ok(Some(n.as_ptr() as usize)) }
        }
        for c in n.getattr("operands")?.try_iter()? { stack.push(c?) }
    }
    Ok(None)
}
fn param_cell(n: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_a(n, "At")? { return Ok(false) }
    let i = n.getattr("index")?;
    Ok(is_a(&i, "RoleConst")? && st(&i, "label")? == str_const("PARAM_CELL"))
}
const RANK: [&str; 3] = ["sample_local", "cross_sample_read", "cross_sample_write"];
fn rank(s: &str) -> usize { RANK.iter().position(|x| *x == s).unwrap_or(0) }

/// The access class of a canonical node order, loop bodies descended into (see raptor/ir/access.py: the same rules).
#[pyfunction]
pub fn infer_access(order: Vec<Bound<'_, PyAny>>) -> PyResult<&'static str> {
    let mut all = vec![];
    with_loop_bodies(&order, &mut all)?;
    let mut seam = HashSet::new();
    for n in &all { if is_loop(n)? { if let Some(w) = step_word(&n.getattr("count")?)? { seam.insert(w); } } }
    let mut cls = 0usize;
    for n in &all {
        if seam.contains(&(n.as_ptr() as usize)) || param_cell(n)? { continue }
        let new = if is_a(n, "At")? { 1 }
            else if (is_a(n, "WideWrite")? || is_a(n, "AccumWrite")?) && !n.getattr("index")?.is_none() { 2 }
            else if is_a(n, "Op")? && st(n, "kind")? == "raw_device" {
                let lit = n.getattr("literal")?;
                let d = lit.getattr("access").ok().and_then(|a| a.extract::<String>().ok());
                match d.as_deref() { Some(d) if RANK.contains(&d) => rank(d),
                    _ => return Err(he(format!("a spliced raw block reached the classifier declaring access={}: @raw_device requires an explicit access= annotation \\
drawn from {:?} — the walk cannot see inside the text, and a mapreduce class is a sink property, not a block's",
                        lit.getattr("access").map(|a| rep(&a)).unwrap_or("None".into()), RANK))) }
            } else { 0 };
        cls = cls.max(new);
    }
    Ok(RANK[cls])
}

fn written_names(order: &[Bound<'_, PyAny>]) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    for n in order {
        if is_a(n, "Sink")? { out.insert(st(n, "name")?); }
        if let Ok(b) = n.getattr("body_sinks") { if b.is_truthy()? { for s in b.try_iter()? { let s = s?; if is_a(&s, "Sink")? { out.insert(st(&s, "name")?); } } } }
    }
    Ok(out)
}
fn selector_reads(node: &Bound<'_, PyAny>) -> PyResult<Vec<(String, String)>> {
    let (mut seen, mut reads) = (HashSet::new(), std::collections::BTreeSet::new());
    let mut stack = vec![node.getattr("selector")?];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.as_ptr() as usize) { continue }
        if is_a(&n, "Leaf")? { reads.insert((st(&n, "role")?, st(&n, "name")?)); }
        else if is_carry(&n)? { reads.insert(("mutable".to_string(), st(&n, "name")?)); }
        for c in n.getattr("operands")?.try_iter()? { stack.push(c?) }
    }
    Ok(reads.into_iter().collect())
}
/// Every body-subgraph node of a Loop, post-order, boundary values (the loop's operands) and the loop's own placeholders excluded
/// (iterative: the emitter, access inference and the digest all ask for it). Replaces the Python `Loop.body_nodes`.
#[pyfunction]
pub fn body_nodes<'py>(lp: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut boundary: HashSet<usize> = HashSet::new();
    for o in lp.getattr("operands")?.try_iter()? { boundary.insert(o?.as_ptr() as usize); }
    let mut roots: Vec<Bound<'py, PyAny>> = vec![];
    for n in lp.getattr("nexts")?.try_iter()? { roots.push(n?) }
    for sk in lp.getattr("body_sinks")?.try_iter()? { for o in sk?.getattr("operands")?.try_iter()? { roots.push(o?) } }
    let ec = lp.getattr("exit_cond")?;
    if !ec.is_none() { roots.push(ec); for v in lp.getattr("exit_values")?.try_iter()? { roots.push(v?) } }
    let mut seen: HashSet<usize> = HashSet::new(); let mut order = vec![];
    let mut stack: Vec<(Bound<'py, PyAny>, bool)> = roots.into_iter().rev().map(|n| (n, false)).collect();
    while let Some((node, expanded)) = stack.pop() {
        let id = node.as_ptr() as usize;
        if seen.contains(&id) || boundary.contains(&id) { continue }
        if is_a(&node, "LoopIndex")? || is_a(&node, "LoopCarry")? { continue }
        if !expanded {
            stack.push((node.clone(), true));
            let ops: Vec<Bound<'py, PyAny>> = node.getattr("operands")?.try_iter()?.collect::<PyResult<_>>()?;
            for c in ops.into_iter().rev() { if !seen.contains(&(c.as_ptr() as usize)) { stack.push((c, false)) } }
            continue
        }
        seen.insert(id); order.push(node);
    }
    Ok(order)
}
/// The slots of a Quantity, one per wire in declaration order: `Slot(role, <prefix>__<wire>, TensorType(shape, dtype, tag))`.
#[pyfunction]
pub fn quantity_expand<'py>(q: &Bound<'py, PyAny>, slot_c: &Bound<'py, PyAny>, ttype_c: &Bound<'py, PyAny>, sep: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let slug: String = q.getattr("slug")?.extract()?; let nm = q.getattr("name")?;
    let prefix = if nm.is_none() { slug } else { format!("{slug}{sep}{}", nm.extract::<String>()?) };
    let mut out = vec![];
    for w in q.getattr("wires")?.try_iter()? {
        let w = w?;
        let shape = PyTuple::new(w.py(), w.getattr("shape")?.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let tt = ttype_c.call1((shape, w.getattr("dtype")?, w.getattr("tag")?))?;
        out.push(slot_c.call1((w.getattr("role")?, format!("{prefix}{sep}{}", st(&w, "name")?), tt))?);
    }
    Ok(out)
}
/// The compensation plane's slot of a sum: the sink's role and type under `<name><suffix>`.
#[pyfunction]
pub fn companion<'py>(sink: &Bound<'py, PyAny>, slot_c: &Bound<'py, PyAny>, suffix: &str) -> PyResult<Bound<'py, PyAny>> {
    slot_c.call1((sink.getattr("role")?, format!("{}{suffix}", st(sink, "name")?), sink.getattr("ttype")?))
}

/// The Dispatch nodes of a canonical order: the top-level ones, then those inside lowered loop bodies (first-in first-out).
#[pyfunction]
pub fn dispatch_nodes<'py>(order: Vec<Bound<'py, PyAny>>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut top = vec![]; let mut seen = HashSet::new(); let mut loops: std::collections::VecDeque<Bound<'py, PyAny>> = Default::default();
    for n in &order { if is_a(n, "Dispatch")? { seen.insert(n.as_ptr() as usize); top.push(n.clone()) } if is_loop(n)? { loops.push_back(n.clone()) } }
    while let Some(l) = loops.pop_front() {
        for n in body_nodes(&l)?.into_iter().map(Ok::<_, PyErr>) {
            let n = n?;
            if is_a(&n, "Dispatch")? && !seen.contains(&(n.as_ptr() as usize)) { seen.insert(n.as_ptr() as usize); top.push(n.clone()) }
            else if is_loop(&n)? { loops.push_back(n) }
        }
    }
    Ok(top)
}
/// Per Dispatch node (in `dispatch_nodes` order): (name, K, policy, selector reads, static selector).
#[pyfunction]
pub fn dispatch_listing(order: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<(String, i64, String, Vec<(String, String)>, bool)>> {
    let written = written_names(&order)?;
    let mut out = vec![];
    for (i, n) in dispatch_nodes(order)?.iter().enumerate() {
        let reads = selector_reads(n)?;
        let stat = !reads.iter().any(|(_, nm)| written.contains(nm));
        out.push((format!("dispatch{i}"), n.getattr("K")?.extract()?, st(n, "policy")?, reads, stat));
    }
    Ok(out)
}
#[pyfunction]
pub fn written(order: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<String>> { let mut v: Vec<String> = written_names(&order)?.into_iter().collect(); v.sort(); Ok(v) }

fn record(name: &str, fields: &[(&str, V)]) -> String {
    let mut o = format!("{name}(");
    for (i, (f, v)) in fields.iter().enumerate() { if i > 0 { o.push_str(", ") } o.push_str(f); o.push('='); keytext::write(v, &mut o) }
    o.push(')'); o
}
fn ttype_text(t: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(record("TensorType", &[("shape", conv(&t.getattr("shape")?)?), ("dtype", conv(&t.getattr("dtype")?)?), ("tag", conv(&t.getattr("tag")?)?)]))
}
/// The digest of a walk: the node-part preimage, then the slots, the arg_spec, the access and the quantity spans (all written by the key-text rule).
#[pyfunction]
pub fn digest_of(pre: &str, leaves: Vec<Bound<'_, PyAny>>, access_cls: &str, access_op: Option<String>, quantities: &Bound<'_, PyDict>) -> PyResult<String> {
    let mut tail = String::new();
    let mut spec = vec![];
    for s in &leaves {
        let (role, name) = (st(s, "role")?, st(s, "name")?);
        tail.push_str(&format!("leaf|{role}|{name}|{}\n", ttype_text(&s.getattr("ttype")?)?));
        spec.push(V::Tuple(vec![V::Str(role), V::Str(name)]));
    }
    tail.push_str(&format!("arg_spec|{}\n", keytext::text(&V::Tuple(spec))));
    tail.push_str(&format!("access|{access_cls}|{}\n", access_op.unwrap_or("None".into())));
    let mut qs: Vec<(String, Bound<'_, PyAny>)> = quantities.iter().map(|(k, v)| Ok((k.extract::<String>()?, v))).collect::<PyResult<_>>()?;
    qs.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, q) in &qs {
        tail.push_str(&format!("quantity|{}\n", record("QuantitySpan", &[("prefix", conv(&q.getattr("prefix")?)?), ("wires", conv(&q.getattr("wires")?)?), ("slots", conv(&q.getattr("slots")?)?)])));
    }
    Ok(rcore::digest(pre, tail.as_bytes()))
}

// ---------------- canonical(): the whole of it ----------------
fn tup<'py>(py: Python<'py>, v: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, v) }
fn kv<'py>(py: Python<'py>, a: &str, b: &str) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, [a, b]) }
fn cls<'py>(name: &str, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    reg().by_name.get(name).map(|c| c.bind(py).clone()).ok_or_else(|| he(format!("class {name} is not registered")))
}

/// The ONE traversal and everything derived from it (raptor/ir/walk.py's `canonical`, whole): the sink-set check, the finish statement, the total
/// merge of the bindings, declared slots, compensation planes, the positional dispatch names, the role order with the quantity expansion, the
/// spans, the access class (floored by `force`), the digest. `env` = (Slot, QuantitySpan, TensorType, FINISHED_TTYPE, NAMESPACE_SEP, SAMPLE_SUM_COMP). Returns the fields of a Walk.
#[pyfunction]
#[pyo3(signature = (sinks, declared, force, env))]
pub fn canonical<'py>(py: Python<'py>, sinks: Vec<Bound<'py, PyAny>>, declared: Vec<Bound<'py, PyAny>>, force: Option<String>, env: Bound<'py, PyTuple>)
    -> PyResult<Bound<'py, PyTuple>> {
    let (slot_c, span_c, ttype_c, fin_tt) = (env.get_item(0)?, env.get_item(1)?, env.get_item(2)?, env.get_item(3)?);
    let (sep, comp_suffix): (String, String) = (env.get_item(4)?.extract()?, env.get_item(5)?.extract()?);
    if sinks.is_empty() { return Err(he("canonical(): the sink set is empty — the sinks are the walk's root set".into())) }
    for s in &sinks {
        if is_loop(s)? {
            if !s.getattr("body_sinks")?.is_truthy()? {
                return Err(he(format!("canonical(): the loop over {} is in the root set but commits nothing; a loop is a root only when its body commits a sink once per iteration",
                    rep(&s.getattr("index")?.getattr("name")?))))
            }
            continue;
        }
        if !is_a(s, "Sink")? { return Err(he(format!("canonical(): {} is not a sink; the root set is sinks only", rep(s)))) }
    }
    let mut finishes = vec![];
    for s in &sinks { if is_a(s, "Finish")? { finishes.push(s.clone()) } }
    if finishes.len() > 1 {
        let names: Vec<String> = finishes.iter().map(|f| f.getattr("name").map(|n| rep(&n)).unwrap_or_default()).collect();
        return Err(he(format!("canonical(): {} finish statements ([{}]); a kernel finishes its sample through ONE mask", finishes.len(), names.join(", "))))
    }
    let finish: Bound<'py, PyAny> = match finishes.first() { Some(f) => tup(py, vec![f.getattr("name")?, f.getattr("steps")?])?.into_any(), None => py.None().into_bound(py) };
    let mut declared = declared;
    if !finishes.is_empty() { declared.push(slot_c.call1(("lookup", str_const("FINISHED_PLANE"), &fin_tt))?) }

    let (order, position, merges, pre, watched, leaf_nodes) = walk(py, sinks.clone())?;
    let _ = position;
    let merges: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = merges.iter().map(|m| m.extract()).collect::<PyResult<_>>()?;
    let bindings = merge_bindings(py, merges)?;
    let leaf_cls = cls("Leaf", py)?;
    for slot in &declared {
        let (role, name) = (st(slot, "role")?, st(slot, "name")?);
        let key = (role.clone(), name.clone());
        match bindings.get_item(&key)? {
            None => { let kind = if role == "terminated" { "terminated" } else { "vocab_read" };
                      bindings.set_item(&key, leaf_cls.call1((kind, &role, &name, slot.getattr("ttype")?))?)? }
            Some(held) => if held.getattr("ttype")?.ne(slot.getattr("ttype")?)? {
                return Err(he(format!("declared slot (role={}, name={}) is typed {} but the body binds it as {}; a slot is keyed on (role, name) ALONE",
                    rep(&role.as_str().into_pyobject(py)?.into_any()), rep(&name.as_str().into_pyobject(py)?.into_any()),
                    rep(&slot.getattr("ttype")?), rep(&held.getattr("ttype")?))))
            }
        }
    }
    for sink in &sinks {
        if is_a(sink, "SampleSum")? && st(sink, "op")? == "sum" {
            let comp = companion(sink, &slot_c, &comp_suffix)?;
            let key = (st(&comp, "role")?, st(&comp, "name")?);
            if bindings.contains(&key)? {
                return Err(he(format!("sample_sum {}: its compensation plane {} collides with a plane of the same name",
                    rep(&sink.getattr("name")?), rep(&comp.getattr("name")?))))
            }
            let kw = PyDict::new(py); kw.set_item("ttype", comp.getattr("ttype")?)?;
            let node = cls("Sink", py)?.call((sink.getattr("kind")?, &key.0, &key.1, sink.getattr("value")?), Some(&kw))?;
            bindings.set_item(&key, node)?;
        }
    }
    let watched_v: Vec<Bound<'py, PyAny>> = watched.iter().collect();
    let listing = dispatch_listing(watched_v.clone())?;
    for (dname, _, policy, _, _) in &listing {
        if policy == "auto" {
            let name = format!("{dname}_mix");
            let key = ("lookup".to_string(), name.clone());
            if !bindings.contains(&key)? {
                bindings.set_item(&key, leaf_cls.call1(("vocab_read", "lookup", &name, ttype_c.call1((PyTuple::empty(py), "uint32"))?))?)?;
            }
        }
    }

    // the role order, with the all-or-nothing quantity expansion
    let mut reached: Vec<(String, Bound<'py, PyAny>)> = vec![];
    for n in leaf_nodes.iter() {
        let q = match n.getattr("quantity") { Ok(q) if !q.is_none() => q, _ => continue };
        let prefix = st(&q, "prefix")?;
        match reached.iter().find(|r| r.0 == prefix) {
            None => reached.push((prefix, q)),
            Some((_, held)) => {
                let (hw, qw) = (held.getattr("wires")?, q.getattr("wires")?);
                if hw.ne(&qw)? {
                    let names = |w: &Bound<'py, PyAny>| -> PyResult<String> { Ok(w.try_iter()?.map(|x| x.and_then(|x| x.getattr("name")).map(|n| rep(&n))).collect::<PyResult<Vec<_>>>()?.join(", ")) };
                    return Err(he(format!("two quantities share the namespace {} but declare different wires ([{}] vs [{}]) — a namespace identifies a quantity ",
                        rep(&prefix.as_str().into_pyobject(py)?.into_any()), names(&hw)?, names(&qw)?)))
                }
            }
        }
    }
    let mut owner: HashMap<(String, String), (usize, usize)> = HashMap::new();
    let mut slots: Vec<((String, String), Bound<'py, PyAny>)> = vec![];
    let mut at: HashMap<(String, String), usize> = HashMap::new();
    for (qi, (_, q)) in reached.iter().enumerate() {
        for (i, s) in quantity_expand(q, &slot_c, &ttype_c, &sep)?.into_iter().map(Ok::<_, PyErr>).enumerate() {
            let s = s?; let key = (st(&s, "role")?, st(&s, "name")?);
            owner.insert(key.clone(), (qi, i));
            match at.get(&key) { Some(&p) => slots[p].1 = s, None => { at.insert(key.clone(), slots.len()); slots.push((key, s)) } }
        }
    }
    for (k, node) in bindings.iter() {
        let key: (String, String) = k.extract()?;
        if !at.contains_key(&key) { at.insert(key.clone(), slots.len()); slots.push((key, node.getattr("binding")?)) }
    }
    let sep = str_const("NAMESPACE_SEP");
    for (key, _) in &slots {
        if key.1.contains(sep) && !owner.contains_key(key) {
            return Err(he(format!("wire {} (role {}) is bound with no owning quantity: the {} namespace is reserved for compound-quantity wires, which only a Quantity declaration may mint ",
                rep(&key.1.as_str().into_pyobject(py)?.into_any()), rep(&key.0.as_str().into_pyobject(py)?.into_any()), rep(&sep.into_pyobject(py)?.into_any()))))
        }
    }
    let roles = const_of("ROLE_ORDER").expect("ROLE_ORDER in the manifest");
    let mut ordered: Vec<Bound<'py, PyAny>> = vec![];
    for role in roles {
        let mut owned: Vec<(&(usize, usize), &Bound<'py, PyAny>)> = vec![];
        let mut free: Vec<(&String, &Bound<'py, PyAny>)> = vec![];
        for (k, s) in &slots { if k.0 == *role { match owner.get(k) { Some(o) => owned.push((o, s)), None => free.push((&k.1, s)) } } }
        owned.sort_by_key(|x| *x.0); free.sort_by(|a, b| a.0.cmp(b.0));
        ordered.extend(owned.into_iter().map(|x| x.1.clone())); ordered.extend(free.into_iter().map(|x| x.1.clone()));
    }
    if let Some(((r, _), _)) = slots.iter().find(|(k, _)| !roles.contains(&k.0.as_str())) {
        return Err(he(format!("binding with a role outside the canonical order: {r}")))
    }
    let (arg_spec, slot_of, slot_types) = (PyList::empty(py), PyDict::new(py), PyDict::new(py));
    for (i, s) in ordered.iter().enumerate() {
        let k = kv(py, &st(s, "role")?, &st(s, "name")?)?;
        arg_spec.append(&k)?; slot_of.set_item(&k, i)?; slot_types.set_item(&k, s.getattr("ttype")?)?;
    }
    // the spans: one consecutive, declaration-ordered run per role
    let quantities = PyDict::new(py);
    for (prefix, q) in &reached {
        let expanded = quantity_expand(q, &slot_c, &ttype_c, &sep)?;
        let mut idx = vec![]; let mut per: Vec<(String, Vec<i64>)> = vec![];
        for s in &expanded {
            let k = kv(py, &st(s, "role")?, &st(s, "name")?)?;
            let i: i64 = slot_of.get_item(&k)?.ok_or_else(|| he("quantity slot missing".into()))?.extract()?;
            idx.push(i);
            let role = st(s, "role")?;
            match per.iter_mut().find(|p| p.0 == role) { Some(p) => p.1.push(i), None => per.push((role, vec![i])) }
        }
        for (role, run) in &per {
            let mut sorted = run.clone(); sorted.sort();
            if *run != sorted || run.iter().enumerate().any(|(j, v)| *v != run[0] + j as i64) {
                return Err(he(format!("quantity {} occupies a non-contiguous or mis-ordered span in role {}: arg_spec indices {:?} (require one consecutive, declaration-ordered run per role)",
                    rep(&prefix.as_str().into_pyobject(py)?.into_any()), rep(&role.as_str().into_pyobject(py)?.into_any()), run)))
            }
        }
        let names = tup(py, expanded.iter().map(|s| s.getattr("name")).collect::<PyResult<Vec<_>>>()?)?;
        quantities.set_item(prefix, span_c.call1((prefix, names, PyTuple::new(py, idx)?))?)?;
    }
    let mut acc = infer_access(watched_v.clone())?;
    if let Some(f) = &force { if rank(f) > rank(acc) { acc = RANK[rank(f)] } }
    let digest = digest_of(&pre, ordered.clone(), acc, None, &quantities)?;
    let prior: Vec<String> = { let mut v = vec![]; for n in leaf_nodes.iter() { if is_a(&n, "Leaf")? && st(&n, "kind")? == "prior_read" { v.push(st(&n, "name")?) } } v };
    let listing_py: Vec<Bound<'py, PyAny>> = listing.into_iter().map(|(a, b, c, d, e)| Ok((a, b, c, d, e).into_pyobject(py)?.into_any())).collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, [order.into_any(), tup(py, ordered)?.into_any(), arg_spec.to_tuple().into_any(), slot_of.into_any(), slot_types.into_any(),
        acc.into_pyobject(py)?.into_any(), quantities.into_any(), digest.into_pyobject(py)?.into_any(), tup(py, listing_py)?.into_any(),
        PyList::new(py, prior)?.into_any(), finish])?)
}
