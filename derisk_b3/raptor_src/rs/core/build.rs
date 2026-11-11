// Generates the class specs (class names, slots, dedup recipes, watch flags, constants) from raptor/ir/core.json, the manifest of
// the core kinds. No dependency: the reader is src/json.rs, shared with the binding.
#[path = "src/json.rs"]
mod json;
use std::{env, fs, path::Path};
fn q(s: &str) -> String { format!("{:?}", s) }
fn main() {
    println!("cargo:rerun-if-changed=../../raptor/ir/core.json");
    let m = json::parse(&fs::read_to_string("../../raptor/ir/core.json").unwrap()).unwrap();
    let mut out = String::from("pub static CONSTS: &[(&str, &[&str])] = &[\n");
    for (k, v) in m.get("consts").unwrap().obj() {
        if let json::J::Arr(a) = v {
            let items: Vec<String> = a.iter().map(|x| q(x.str().unwrap())).collect();
            out.push_str(&format!("    ({}, &[{}]),\n", q(k), items.join(", ")));
        }
    }
    out.push_str("];\npub static STR_CONSTS: &[(&str, &str)] = &[\n");
    for (k, v) in m.get("consts").unwrap().obj() { if let json::J::Str(x) = v { out.push_str(&format!("    ({}, {}),\n", q(k), q(x))) } }
    out.push_str("];\n");
    let kinds = m.get("kinds").unwrap().arr();
    // (class, base class, slots, binds, dedup code, watch)
    let mut cl: Vec<(String, String, Vec<String>, Option<bool>, Option<String>, Option<u8>)> = vec![];
    for k in kinds {
        let name = k.get("class").unwrap().str().unwrap().to_string();
        let base = match k.get("refines").and_then(|r| r.str()) {
            Some(r) => kinds.iter().find(|x| x.get("name").unwrap().str() == Some(r)).unwrap().get("class").unwrap().str().unwrap().to_string(),
            None => String::new(),
        };
        let slots = k.get("slots").map(|s| s.strs()).unwrap_or_default();
        let binds = k.get("binding").and_then(|b| b.str()).map(|b| b == "slot");
        let item = |x: &str| -> String {
            if x == "k" { return "Item::K".into() }
            let p: Vec<&str> = x.splitn(3, ':').collect();
            match p[0] {
                "f" => format!("Item::Field({})", q(p[1])),
                "r" => format!("Item::ReprOf({})", q(p[1])),
                "s" => format!("Item::Lit({})", q(p[1])),
                "names" => format!("Item::Names({})", q(p[1])),
                "skeys" => format!("Item::SinkKeys({})", q(p[1])),
                "body" => "Item::Body".to_string(),
                "litif" => format!("Item::LitIfSet({}, {})", q(p[1]), q(p[2])),
                "fieldif" => format!("Item::FieldIfSet({}, {})", q(p[1]), q(p[2])),
                "fx" => format!("Item::FieldUnless({}, &[{}])", q(p[1]), p[2].split(',').map(q).collect::<Vec<_>>().join(", ")),
                _ => panic!("dedup item {x}"),
            }
        };
        let dedup = k.get("dedup").map(|d| { let w = d.strs(); match w[0].as_str() {
            "none" => "Dedup::None".to_string(), "py" => "Dedup::Py".to_string(),
            "one" => format!("Dedup::One({})", item(&w[1])),
            "tup" => format!("Dedup::Tup(&[{}])", w[1..].iter().map(|x| item(x)).collect::<Vec<_>>().join(", ")),
            _ => panic!("dedup mode") } });
        let watch = k.get("watch").and_then(|w| w.str()).map(|w| if w == "all" { 1 } else { 2 });
        cl.push((name, base, slots, binds, dedup, watch));
    }
    for i in 0..cl.len() {
        let base = cl[i].1.clone();
        if base.is_empty() { continue }
        let b = cl.iter().find(|c| c.0 == base).unwrap().clone();
        if cl[i].3.is_none() { cl[i].3 = b.3 }
        if cl[i].4.is_none() { cl[i].4 = b.4.clone() }
        if cl[i].5.is_none() { cl[i].5 = b.5 }
    }
    // the operand-count and result-type rules of every class: (min, max or -1, result code, operand index, dtype, shape)
    let rule_of = |k: &json::J| -> (i64, i64, u8, i64, String, Vec<i64>) {
        let (mut lo, mut hi) = (0i64, -1i64);
        if let Some(o) = k.get("operands") {
            lo = o.get("min").map(|x| x.int()).unwrap_or(0);
            hi = match o.get("max") { Some(m) => if m.str() == Some("many") { -1 } else { m.int() }, None => -1 };
        }
        let (mut code, mut idx, mut dt, mut sh) = (0u8, 0i64, String::new(), vec![]);
        if let Some(r) = k.get("result") {
            let a = r.arr();
            match a[0].str().unwrap() {
                "carried" => {}
                "same_as" => { code = 1; idx = a[1].int() }
                "scalar_of" => { code = 2; idx = a[1].int() }
                "fixed" => { code = 3; dt = a[1].str().unwrap().to_string(); sh = a[2].arr().iter().map(|x| x.int()).collect() }
                w => panic!("result rule {w}"),
            }
        }
        (lo, hi, code, idx, dt, sh)
    };
    out.push_str("pub static RULES: &[(u32, i32, u8, u32, &str, &[i64])] = &[\n");
    for k in kinds {
        let (lo, hi, code, idx, dt, sh) = rule_of(k);
        out.push_str(&format!("    ({lo}, {hi}, {code}, {idx}, {}, &[{}]),\n", q(&dt), sh.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")));
    }
    out.push_str("];\n");
    out.push_str("pub static CLASSES: &[ClassSpec] = &[\n");
    for (n, b, s, bind, d, wt) in &cl {
        out.push_str(&format!("    ClassSpec {{ name: {}, base: {}, slots: &[{}], binds: {}, dedup: {}, watch: {} }},\n", q(n), q(b),
            s.iter().map(|x| q(x)).collect::<Vec<_>>().join(", "), bind.unwrap_or(false), d.clone().unwrap_or("Dedup::Py".into()), wt.unwrap_or(0)));
    }
    out.push_str("];\n");
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("specs.rs"), out).unwrap();
}
