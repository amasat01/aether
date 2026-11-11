//! Manifest shape validation: a port of `raptor.schema.validate_manifest` and the `check_*` functions of
//! `raptor.schema.manifest`. Returns `Err("ValueError: ...")` / `Err("TypeError: ...")` in Python's own words.

use crate::pjson::Json;

pub const SCHEMA_VERSION: i64 = 1;
pub const MAX_SCHEMA_VERSION: i64 = 3;
pub const AETHER_ABI_V1: &str = "aether-abi/1";
pub const AETHER_ABI_V2: &str = "aether-abi/2";

const EXEC_KEYS: [&str; 3] = ["exec_targets", "exec_access", "exec_op"];
const EXEC_TARGETS: [&str; 2] = ["device", "host"];
const EXEC_ACCESS: [&str; 4] = ["cross_sample_read", "cross_sample_write", "mapreduce", "sample_local"];
const EXEC_OPS: [&str; 4] = ["land", "max", "sum", "times"];
const ALL_PATTERNS: [&str; 3] = ["graph_block", "pure", "vector"];
const TEAM_GRIDS: [&str; 4] = ["chunks", "keys", "lanes", "targets"];
const TEAM_DTYPES: [&str; 2] = ["int32", "real"];
const ORDER_V1: [&str; 4] = ["schema_version", "pattern", "aether_abi", "plugins"];
const ORDER_V2: [&str; 6] = ["schema_version", "pattern", "aether_abi", "exec_targets", "exec_access", "exec_op"];
const ORDER_BLOCKS: [&str; 5] = ["schema_version", "pattern", "aether_abi", "blocks", "plugins"];

type R = Result<(), String>;

fn ve<T>(m: String) -> Result<T, String> {
    Err(format!("ValueError: {m}"))
}
fn lst(v: &[&str]) -> String {
    Json::Arr(v.iter().map(|s| Json::Str((*s).into())).collect()).repr()
}
fn slst(v: &[String]) -> String {
    Json::Arr(v.iter().map(|s| Json::Str(s.clone())).collect()).repr()
}
fn is_int(v: &Json) -> bool {
    matches!(v, Json::Int(_))
}
fn int_ge(v: &Json, low: i128) -> bool {
    matches!(v, Json::Int(i) if *i >= low)
}
/// `v in frozenset_of_str`: unhashable values raise TypeError, as in Python.
fn in_set(v: &Json, set: &[&str]) -> Result<bool, String> {
    match v {
        Json::Arr(_) | Json::Obj(_) => Err(format!("TypeError: cannot use '{}' as a set element (unhashable type: '{}')", v.type_name(), v.type_name())),
        Json::Str(s) => Ok(set.contains(&s.as_str())),
        _ => Ok(false),
    }
}
fn is_str(v: &Json) -> bool {
    matches!(v, Json::Str(_))
}
fn nonempty_str(v: &Json) -> bool {
    matches!(v, Json::Str(s) if !s.is_empty())
}

pub fn validate(doc: &Json) -> R {
    let nm = "'<document>'";
    if !matches!(doc, Json::Obj(_)) {
        return Err(format!("TypeError: expected a dict, got {}", doc.type_name()));
    }
    let version = check_schema_version(doc, nm, true)?;
    check_execution_axis(doc, version, nm)?;
    check_lane_space(doc, nm)?;
    check_launch_fields(doc, nm)?;
    check_team_fields(doc, nm)?;
    check_index_planes(doc, nm)?;
    let pattern = doc.get("pattern");
    if let Some(Json::Str(s)) = pattern {
        if s == "neural_block" {
            return ve("pattern 'neural_block' was renamed to 'graph_block' in 0.7.3".into());
        }
    }
    if let Some(p) = pattern {
        if *p != Json::Null && !in_set(p, &ALL_PATTERNS)? {
            return ve(format!("pattern {} is not recognized; expected one of {}", p.repr(), lst(&ALL_PATTERNS)));
        }
    }
    if doc.has("plugins") {
        let mut v2 = ORDER_V2.to_vec();
        v2.push("plugins");
        let expected: Vec<&str> = if doc.has("blocks") { ORDER_BLOCKS.to_vec() } else if version >= 2 { v2 } else { ORDER_V1.to_vec() };
        let got: Vec<String> = doc.keys().into_iter().filter(|k| expected.contains(k)).map(String::from).collect();
        let want: Vec<String> = expected.iter().filter(|k| doc.has(k)).map(|s| s.to_string()).collect();
        if got != want {
            return ve(format!(
                "manifest top-level key order violates the key-order contract: got {}, expected {}",
                slst(&got), slst(&want)
            ));
        }
    }
    Ok(())
}

pub fn check_schema_version(meta: &Json, nm: &str, legacy: bool) -> Result<i64, String> {
    let v = meta.get("schema_version").or_else(|| if legacy { meta.get("version") } else { None }).cloned().unwrap_or(Json::Int(1));
    match v {
        Json::Int(i) if i <= MAX_SCHEMA_VERSION as i128 => Ok(i.max(i64::MIN as i128) as i64),
        _ => ve(format!(
            "{nm} was built for plugin schema v{}, but this loader supports v{MAX_SCHEMA_VERSION}; upgrade eagle to load it",
            v.pystr()
        )),
    }
}

pub fn check_execution_axis(meta: &Json, version: i64, nm: &str) -> R {
    let tag = meta.get("aether_abi");
    if version == 1 {
        if let Some(t) = tag {
            if *t != Json::Null && *t != Json::Str(AETHER_ABI_V1.into()) {
                return ve(format!("{nm}: schema v1 requires aether_abi='aether-abi/1' when present; got {}", t.repr()));
            }
        }
        let mut present: Vec<String> = EXEC_KEYS.iter().filter(|k| meta.has(k)).map(|s| s.to_string()).collect();
        present.sort();
        if !present.is_empty() {
            return ve(format!(
                "{nm}: schema v1 must not carry execution key(s) {} (the execution axis is schema-v2-only; set schema_version=2 and aether_abi='aether-abi/2' to use it)",
                slst(&present)
            ));
        }
        return Ok(());
    }
    if let Some(t) = tag {
        if *t != Json::Null && *t != Json::Str(AETHER_ABI_V2.into()) {
            return ve(format!("{nm}: schema v2 requires aether_abi='aether-abi/2'; got {}", t.repr()));
        }
    }
    let missing: Vec<String> = ["exec_targets", "exec_access"].iter().filter(|k| !meta.has(k)).map(|s| s.to_string()).collect();
    if !missing.is_empty() {
        return ve(format!("{nm}: schema v2 requires execution key(s) {} (absence = load refused)", slst(&missing)));
    }
    let targets = meta.get("exec_targets").unwrap();
    let mut valid = false;
    if let Json::Arr(a) = targets {
        if !a.is_empty() {
            valid = true;
            for t in a {
                if !in_set(t, &EXEC_TARGETS)? {
                    valid = false;
                    break;
                }
            }
            if valid {
                let mut seen: Vec<&Json> = vec![];
                for t in a {
                    if !seen.contains(&t) {
                        seen.push(t);
                    }
                }
                valid = seen.len() == a.len();
            }
        }
    }
    if !valid {
        return ve(format!(
            "{nm}: exec_targets must be a non-empty list drawn from {} with no duplicates; got {}",
            lst(&EXEC_TARGETS), targets.repr()
        ));
    }
    let access = meta.get("exec_access").unwrap();
    if !in_set(access, &EXEC_ACCESS)? {
        return ve(format!("{nm}: exec_access {} is not a supported access class (supported: {})", access.repr(), lst(&EXEC_ACCESS)));
    }
    let op = meta.get("exec_op");
    if *access == Json::Str("mapreduce".into()) {
        if op.is_none() {
            return ve(format!("{nm}: exec_access='mapreduce' requires exec_op (one of {})", lst(&EXEC_OPS)));
        }
    } else if op.is_some() {
        return ve(format!("{nm}: exec_op is forbidden unless exec_access='mapreduce' (exec_access={})", access.repr()));
    }
    if let Some(o) = op {
        if !in_set(o, &EXEC_OPS)? {
            return ve(format!("{nm}: exec_op {} is not a supported reduction op (supported: {})", o.repr(), lst(&EXEC_OPS)));
        }
    }
    Ok(())
}

pub fn check_lane_space(meta: &Json, nm: &str) -> R {
    let Some(launch @ Json::Obj(_)) = meta.get("launch") else { return Ok(()) };
    let Some(space) = launch.get("lane_space") else { return Ok(()) };
    let bad = |why: String| -> R { ve(format!("{nm}: launch.lane_space {why}")) };
    let exactly = |j: &Json, ks: &[&str]| matches!(j, Json::Obj(_)) && j.keys().len() == ks.len() && ks.iter().all(|k| j.has(k));
    if !exactly(space, &["axes", "group"]) {
        return bad("must be an object with exactly 'axes' and 'group'".into());
    }
    let axes = space.get("axes").unwrap();
    let Json::Arr(axes) = axes else { return bad("axes must be a non-empty list".into()) };
    if axes.is_empty() {
        return bad("axes must be a non-empty list".into());
    }
    let mut seen: Vec<&str> = vec![];
    for (i, ax) in axes.iter().enumerate() {
        if !exactly(ax, &["label", "extent"]) {
            return bad(format!("axes[{i}] must be an object with exactly 'label' and 'extent'"));
        }
        let (label, ext) = (ax.get("label").unwrap(), ax.get("extent").unwrap());
        if *label != Json::Null {
            let Json::Str(l) = label else { return bad(format!("axes[{i}].label must be a string or null; got {}", label.repr())) };
            if seen.contains(&l.as_str()) {
                return bad(format!("axes[{i}].label {} is a duplicate", label.repr()));
            }
            seen.push(l);
        }
        if *ext == Json::Null {
            if i > 0 {
                return bad(format!("axes[{i}].extent is null; only the first axis may be run-time"));
            }
        } else if !int_ge(ext, 1) {
            return bad(format!("axes[{i}].extent must be an integer >= 1 or null; got {}", ext.repr()));
        }
    }
    let group = space.get("group").unwrap();
    if !int_ge(group, 1) {
        return bad(format!("group must be an integer >= 1; got {}", group.repr()));
    }
    Ok(())
}

pub fn check_launch_fields(meta: &Json, nm: &str) -> R {
    let Some(launch @ Json::Obj(_)) = meta.get("launch") else { return Ok(()) };
    let bad = |what: &str, why: String| -> R { ve(format!("{nm}: launch.{what} {why}")) };
    if let Some(b) = launch.get("box") {
        let ok = matches!(b, Json::Obj(_)) && b.keys().len() == 3 && ["axis", "T", "stride"].iter().all(|k| b.has(k));
        if !ok {
            return bad("box", "must be an object with exactly 'axis', 'T' and 'stride'".into());
        }
        for (key, low) in [("axis", 0), ("T", 2), ("stride", 1)] {
            let v = b.get(key).unwrap();
            if !int_ge(v, low) {
                return bad("box", format!("{key} must be an integer >= {low}; got {}", v.repr()));
            }
        }
    }
    if let Some(iset) = launch.get("index_set") {
        let ks = iset.keys();
        if !matches!(iset, Json::Obj(_)) || !iset.has("width") || ks.iter().any(|k| !["width", "prefix", "count"].contains(k)) {
            return bad("index_set", "must be an object with 'width' and optionally 'prefix' and 'count'".into());
        }
        let w = iset.get("width").unwrap();
        if !matches!(w, Json::Int(32) | Json::Int(64)) {
            return bad("index_set", format!("width must be 32 or 64; got {}", w.repr()));
        }
        let prefix = match iset.get("prefix") {
            None => false,
            Some(Json::Bool(b)) => *b,
            Some(p) => return bad("index_set", format!("prefix must be a boolean; got {}", p.repr())),
        };
        if prefix {
            if *w != Json::Int(32) {
                return bad("index_set", "a prefix set has width 32".into());
            }
            if !iset.get("count").is_some_and(nonempty_str) {
                return bad("index_set", "a prefix set names its count plane in 'count'".into());
            }
        } else if iset.has("count") {
            return bad("index_set", "names a 'count' plane only for a prefix set".into());
        }
    }
    let keys = ["local_bytes", "local_source", "local_limit"];
    let present: Vec<&str> = keys.iter().copied().filter(|k| launch.has(k)).collect();
    if !present.is_empty() && present.len() != 3 {
        let missing: Vec<&str> = keys.iter().copied().filter(|k| !launch.has(k)).collect();
        return bad(present[0], format!("comes with {}; missing {}", keys.join(", "), lst(&missing)));
    }
    if !present.is_empty() {
        let (b, s, l) = (launch.get("local_bytes").unwrap(), launch.get("local_source").unwrap(), launch.get("local_limit").unwrap());
        if !int_ge(b, 0) {
            return bad("local_bytes", format!("must be an integer >= 0; got {}", b.repr()));
        }
        if !matches!(s, Json::Str(x) if x == "ptxas" || x == "ptx") {
            return bad("local_source", format!("must be 'ptxas' or 'ptx'; got {}", s.repr()));
        }
        if !int_ge(l, 1) {
            return bad("local_limit", format!("must be an integer >= 1; got {}", l.repr()));
        }
    }
    Ok(())
}

fn keyed_record(rec: &Json, whr: &str, nm: &str) -> R {
    let bad = |why: String| -> R { ve(format!("{nm}: {whr} {why}")) };
    let ks = rec.keys();
    if !matches!(rec, Json::Obj(_)) || !rec.has("width") || !rec.has("axes") || ks.iter().any(|k| !["width", "axes", "payload"].contains(k)) {
        return bad("must be an object with 'width', 'axes' and optionally 'payload'".into());
    }
    let w = rec.get("width").unwrap();
    if !int_ge(w, 1) {
        return bad(format!("width must be an integer >= 1; got {}", w.repr()));
    }
    let axes_ok = match rec.get("axes").unwrap() {
        Json::Arr(a) => a.iter().all(|x| match x {
            Json::Arr(p) if p.len() == 2 => (p[0] == Json::Null || is_str(&p[0])) && (p[1] == Json::Null || int_ge(&p[1], 1)),
            _ => false,
        }),
        _ => false,
    };
    if !axes_ok {
        return bad("axes must be a list of [label | null, extent | null] pairs".into());
    }
    let empty = Json::Arr(vec![]);
    let pay = rec.get("payload").unwrap_or(&empty);
    if !matches!(pay, Json::Arr(a) if a.iter().all(|p| int_ge(p, 1))) {
        return bad(format!("payload must be a list of integers >= 1; got {}", pay.repr()));
    }
    Ok(())
}

pub fn check_team_fields(meta: &Json, nm: &str) -> R {
    let bad = |whr: &str, why: String| -> R { ve(format!("{nm}: {whr} {why}")) };
    if let Some(Json::Arr(stages)) = meta.get("stages") {
        for (k, st) in stages.iter().enumerate() {
            let Some(team) = st.get("team") else { continue };
            let whr = format!("stages[{k}].team");
            if !matches!(team, Json::Obj(_)) || !team.has("threads") {
                return bad(&whr, "must be an object with 'threads'".into());
            }
            let th = team.get("threads").unwrap();
            if !int_ge(th, 1) {
                return bad(&whr, format!("threads must be an integer >= 1; got {}", th.repr()));
            }
            let gj = team.get("grid").cloned().unwrap_or(Json::Str("lanes".into()));
            if !in_set(&gj, &TEAM_GRIDS)? {
                return bad(&whr, format!("grid {} is not one of {}", gj.repr(), lst(&TEAM_GRIDS)));
            }
            let Json::Str(grid) = &gj else { unreachable!() };
            let extra: &[&str] = match grid.as_str() { "lanes" => &[], "keys" => &["keyed"], "chunks" => &["of", "status"], _ => &["of"] };
            let mut surplus: Vec<String> = team.keys().into_iter().filter(|k| !["threads", "grid"].contains(k) && !extra.contains(k)).map(String::from).collect();
            surplus.sort();
            if !surplus.is_empty() {
                return bad(&whr, format!("has keys {} that grid {} does not carry", slst(&surplus), gj.repr()));
            }
            let mut need: Vec<String> = extra.iter().filter(|k| !team.has(k)).map(|s| s.to_string()).collect();
            need.sort();
            if !need.is_empty() {
                return bad(&whr, format!("grid {} needs {}", gj.repr(), slst(&need)));
            }
            if grid == "keys" {
                keyed_record(team.get("keyed").unwrap(), &format!("{whr}.keyed"), nm)?;
            }
            // Python walks a set here: its order is hash-dependent. The port walks sorted order.
            let mut names: Vec<&str> = extra.iter().copied().filter(|k| *k != "keyed").collect();
            names.sort();
            for key in names {
                let v = team.get(key).unwrap();
                if !nonempty_str(v) {
                    return bad(&whr, format!("{key} must be a plane name; got {}", v.repr()));
                }
            }
        }
    }
    if let Some(planes) = meta.get("team_planes") {
        let Json::Obj(items) = planes else { return bad("team_planes", "must be an object of plane records".into()) };
        for (pname, spec) in items {
            let whr = format!("team_planes[{}]", crate::keytext::py_str(pname));
            let ks = spec.keys();
            if !matches!(spec, Json::Obj(_)) || !spec.has("dtype") || !spec.has("per_lane")
                || ks.iter().any(|k| !["dtype", "per_lane", "keys", "per_target", "once", "fill"].contains(k))
            {
                return bad(&whr, "must be an object with 'dtype', 'per_lane' and optionally 'keys', 'per_target', 'once', 'fill'".into());
            }
            let dt = spec.get("dtype").unwrap();
            if !in_set(dt, &TEAM_DTYPES)? {
                return bad(&whr, format!("dtype must be one of {}; got {}", lst(&TEAM_DTYPES), dt.repr()));
            }
            let pl = spec.get("per_lane").unwrap();
            if !int_ge(pl, 1) {
                return bad(&whr, format!("per_lane must be an integer >= 1; got {}", pl.repr()));
            }
            if ["keys", "per_target", "once"].iter().filter(|k| spec.has(k)).count() > 1 {
                return bad(&whr, "sizes by at most one of 'keys', 'per_target' and 'once'".into());
            }
            if let Some(k) = spec.get("keys") {
                keyed_record(k, &format!("{whr}.keys"), nm)?;
            }
            if let Some(pt) = spec.get("per_target") {
                if !nonempty_str(pt) {
                    return bad(&whr, format!("per_target must be a plane name; got {}", pt.repr()));
                }
            }
            if let Some(o) = spec.get("once") {
                if *o != Json::Bool(true) {
                    return bad(&whr, format!("once must be true; got {}", o.repr()));
                }
            }
            if let Some(f) = spec.get("fill") {
                if !is_int(f) {
                    return bad(&whr, format!("fill must be an integer; got {}", f.repr()));
                }
            }
        }
    }
    if let Some(k) = meta.get("keyed") {
        let Json::Obj(items) = k else { return bad("keyed", "must be an object of keyed records".into()) };
        for (kname, rec) in items {
            keyed_record(rec, &format!("keyed[{}]", crate::keytext::py_str(kname)), nm)?;
        }
    }
    Ok(())
}

pub fn check_index_planes(meta: &Json, nm: &str) -> R {
    let Some(planes) = meta.get("index_planes") else { return Ok(()) };
    let names: Option<Vec<&str>> = match planes {
        Json::Arr(a) if !a.is_empty() => a.iter().map(|p| match p { Json::Str(s) if !s.is_empty() => Some(s.as_str()), _ => None }).collect(),
        _ => None,
    };
    let Some(names) = names else { return ve(format!("{nm}: index_planes must be a non-empty list of plane names")) };
    let mut uniq = names.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() != names.len() {
        return ve(format!("{nm}: index_planes names a plane twice"));
    }
    let mut declared: Vec<&Json> = vec![];
    match meta.get("arg_spec") {
        Some(Json::Arr(a)) => {
            for e in a {
                if let Json::Arr(p) = e {
                    if p.len() == 2 {
                        if matches!(p[1], Json::Arr(_) | Json::Obj(_)) {
                            return Err(format!("TypeError: cannot use '{}' as a set element (unhashable type: '{}')", p[1].type_name(), p[1].type_name()));
                        }
                        declared.push(&p[1]);
                    }
                }
            }
        }
        Some(j @ (Json::Int(_) | Json::Float(_) | Json::Bool(_))) => {
            let falsy = matches!(j, Json::Int(0) | Json::Bool(false)) || matches!(j, Json::Float(f) if *f == 0.0);
            if !falsy {
                return Err(format!("TypeError: '{}' object is not iterable", j.type_name()));
            }
        }
        _ => {}
    }
    let missing: Vec<String> = names.iter().filter(|p| !declared.iter().any(|d| matches!(d, Json::Str(s) if s == *p))).map(|s| s.to_string()).collect();
    if !declared.is_empty() && !missing.is_empty() {
        return ve(format!("{nm}: index_planes names {}, which arg_spec does not declare", slst(&missing)));
    }
    Ok(())
}


// ---------------- the graph_block descriptor family (port of raptor.schema.blocks.validate_block_descriptor) ----------------
fn truthy(j: Option<&Json>) -> bool {
    match j {
        None | Some(Json::Null) | Some(Json::Bool(false)) | Some(Json::Int(0)) => false,
        Some(Json::Float(f)) => *f != 0.0,
        Some(Json::Str(s)) => !s.is_empty(),
        Some(Json::Arr(a)) | Some(Json::Tup(a)) => !a.is_empty(),
        Some(Json::Obj(o)) => !o.is_empty(),
        _ => true,
    }
}
fn len_of(j: &Json) -> Result<usize, String> {
    match j {
        Json::Arr(a) | Json::Tup(a) => Ok(a.len()),
        Json::Obj(o) => Ok(o.len()),
        Json::Str(s) => Ok(s.chars().count()),
        o => Err(format!("TypeError: object of type '{}' has no len()", o.type_name())),
    }
}
const BLOCK_EXEC_REF_FIELDS: [&str; 3] = ["forward_exec", "jvp_exec", "vjp_exec"];
const BLOCK_REQUIRED: [&str; 8] = ["forward_exec", "in_degree", "input_width", "out_degree", "output_width", "param_width", "scatter_policy", "state_width"];
const BLOCK_INT: [&str; 6] = ["in_degree", "input_width", "out_degree", "output_width", "param_width", "state_width"];
const BLOCK_FORBIDDEN: [&str; 6] = ["aether_abi", "buffers", "derivative", "host_entry", "mat_shapes", "mutables"];
const SCATTER_POLICIES: [&str; 2] = ["accumulate", "unique_write"];
const EXEC_REF_KINDS: [&str; 1] = ["kernel"];

fn exec_ref(r: &Json, field: &str, name: &str) -> R {
    let Json::Obj(items) = r else { return ve(format!("{name}: graph_block {field} must be an object {{'kind': ..., 'kernel': ...}}; got {}", r.repr())) };
    let mut unknown: Vec<String> = items.iter().map(|(k, _)| k.clone()).filter(|k| k != "kind" && k != "kernel").collect();
    unknown.sort();
    if !unknown.is_empty() {
        return ve(format!("{name}: graph_block {field} carries unknown key(s) {} (exec references are closed to 'kind', 'kernel')", slst(&unknown)));
    }
    let kind = r.get("kind").cloned().unwrap_or(Json::Null);
    if !in_set(&kind, &EXEC_REF_KINDS)? {
        return ve(format!("{name}: graph_block {field}.kind {} is not a supported exec reference kind (supported: {}); upgrade eagle", kind.repr(), EXEC_REF_KINDS.join(", ")));
    }
    let target = r.get("kernel").cloned().unwrap_or(Json::Null);
    if !nonempty_str(&target) {
        return ve(format!("{name}: graph_block {field}.kernel must be a non-empty kernel id; got {}", target.repr()));
    }
    Ok(())
}

/// `raptor.schema.blocks.validate_block_descriptor`: the version gate, then the graph_block clause. `name` is the raw artifact name.
pub fn validate_block_descriptor(meta: &Json, name: &str) -> R {
    check_schema_version(meta, &crate::keytext::py_str(name), false)?;
    if !truthy(meta.get("kernel")) {
        return ve(format!("{name}: a 'graph_block' descriptor must carry a non-empty 'kernel' (its wire identity)"));
    }
    let st = meta.get("scalar_type").cloned().unwrap_or(Json::Null);
    if !(matches!(&st, Json::Str(s) if s == "float32" || s == "float64")) {
        return ve(format!("{name}: graph_block requires scalar_type in (float32, float64); got {} (a graph block's arithmetic is native-float only)", st.repr()));
    }
    let Some(spec) = meta.get("arg_spec") else {
        return ve(format!("{name}: a 'graph_block' descriptor must carry an arg_spec, present and empty ([]); the key is absent"));
    };
    if truthy(Some(spec)) {
        return ve(format!("{name}: a 'graph_block' descriptor must carry an EMPTY arg_spec ([]); got {} entries (a descriptor binds nothing \u{2014} the kernels it references carry their own arg_spec)", len_of(spec)?));
    }
    for f in BLOCK_REQUIRED {
        if !meta.has(f) {
            return ve(format!("{name}: missing required graph_block field {} (schema-v1 graph_block requires {})", crate::keytext::py_str(f), BLOCK_REQUIRED.join(", ")));
        }
    }
    for f in BLOCK_INT {
        let v = meta.get(f).unwrap();
        let nonneg = f == "param_width";
        let ok = matches!(v, Json::Int(i) if (if nonneg { *i >= 0 } else { *i > 0 }) && *i < (1i128 << 32));
        if !ok {
            let what = if nonneg { "a non-negative integer" } else { "a positive integer" };
            return ve(format!("{name}: graph_block {f} must be {what} representable in a uint32 (< 2**32); got {}", v.repr()));
        }
    }
    if let (Some(Json::Int(o)), Some(Json::Int(s))) = (meta.get("output_width"), meta.get("state_width")) {
        if o > s {
            return ve(format!("{name}: graph_block output_width ({o}) must not exceed state_width ({s})"));
        }
    }
    let policy = meta.get("scatter_policy").unwrap();
    if !in_set(policy, &SCATTER_POLICIES)? {
        return ve(format!("{name}: graph_block scatter_policy {} is not a supported terminal-write contract (supported: {}); upgrade eagle", policy.repr(), SCATTER_POLICIES.join(", ")));
    }
    for f in BLOCK_EXEC_REF_FIELDS {
        if let Some(r) = meta.get(f) {
            exec_ref(r, f, name)?;
        }
    }
    for f in BLOCK_FORBIDDEN {
        if truthy(meta.get(f)) {
            return ve(format!("{name}: a 'graph_block' descriptor must not carry {} (forbidden on a descriptor: {}); it belongs on the kernel artifact the descriptor references", crate::keytext::py_str(f), BLOCK_FORBIDDEN.join(", ")));
        }
    }
    Ok(())
}
