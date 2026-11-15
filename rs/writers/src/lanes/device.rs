//! The fast-path entries a device unit of an automatic (two-path) finishing kernel carries beside its own entry: the
//! persistent entry, whose warps claim samples from a shared counter and run the fused step in chunks, and the entry over a
//! contiguous range. Both are cut from the lane split of the fused body (head, one trip, tail).
use super::device_text::*;
use crate::body::frame::buffer::tokens;
use crate::body::Buffer;
use crate::lanes::cut::Carry;
use crate::lanes::split::LaneSplit;
use crate::spell::binding_name;
use crate::spell::carry_type;
use std::collections::BTreeSet;

/// A step of at most this many operations has its chunk loop unrolled by four.
const SHORT_DEVICE_STEP_OPS: usize = 128;

/// A carry the step reads, held in a register across trips.
struct HeldCarry {
    /// The register that holds it.
    store: String,
    /// Its C++ type.
    ctype: String,
    /// The carry's name in the step.
    name: String,
    /// The text of its initial value.
    init: String,
    /// The name of its next value.
    next: String,
    /// Whether it is a vector (set from a constructor call, not a cast).
    shaped: bool,
}

/// A local an entry declares: its type, its name and the text it is set from.
struct Local {
    /// Its C++ type.
    ctype: String,
    /// Its name.
    name: String,
    /// The text it is set from.
    text: String,
}

/// A live value held in a register: the register, its type and the value's name.
struct LiveValue {
    /// The register that holds it.
    store: String,
    /// Its C++ type.
    ctype: String,
    /// The value's name in the step.
    name: String,
}

/// The finish counter's binding.
fn finish_counter() -> Result<String, String> {
    binding_name("lookup", "finished_count")
}

/// One count into the finish counter for a sample finished on entry.
fn finished_on_entry() -> Result<String, String> {
    Ok(format!("hawk_abi::finish_count({});", finish_counter()?))
}

/// The word that is the fused-steps plane read at sample 0.
fn fused_steps_word() -> Result<String, String> {
    Ok(format!(
        "{}[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()",
        binding_name("lookup", "fused_steps")?
    ))
}

/// Whether a value the step reads may be one no sample changes (a parameter and what is computed from it alone): the old
/// writer hoists those to entry. Whether a value is such is the lane split's own analysis, which the split does not report, so
/// a live value whose one-line declarations read no plane and no sample index is taken as possibly such, and the unit is refused.
fn possibly_sample_free(pre: &Buffer, name: &str) -> bool {
    let chain = pre.one_line_chain(&[name.to_string()]);
    let recorded = !chain.is_empty()
        && chain.iter().all(|d| {
            let value = d.value.as_deref().unwrap_or("");
            !value.contains('[') && !tokens(value).any(|w| w == "i")
        });
    recorded
}

/// The indexes of the lines of the step budget `count` in `pre` when it reads the launch's fused-steps word alone, so the
/// persistent entry evaluates them once at entry; none otherwise.
fn launch_budget(pre: &Buffer, count: &Option<String>) -> Vec<usize> {
    let Some(count) = count
        .as_ref()
        .filter(|c| c.chars().all(|x| x.is_alphanumeric() || x == '_'))
    else {
        return vec![];
    };
    let chain = pre.one_line_chain(std::slice::from_ref(count));
    let plane = binding_name("lookup", "fused_steps").unwrap_or_default();
    for decl in &chain {
        let value = decl.value.as_deref().unwrap_or("");
        let parts: Vec<&str> = value.split('[').collect();
        let other_plane = parts[..parts.len().saturating_sub(1)].iter().any(|part| {
            let name: String = part
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            !name.is_empty() && name != plane
        });
        if tokens(value).any(|w| w == "i") || other_plane {
            return vec![];
        }
    }
    chain.iter().map(|d| d.line).collect()
}

/// `(type, name, text)` for each carry the step never reads, spelled from the trip count of a lane whose sample has ended, or
/// `None` when one is not the step's exit flag selecting between two constants.
fn exit_rebuilt(step: &str, exit_flag: &str, fresh: &[Local]) -> Option<Vec<Local>> {
    let flag_line = step.lines().find_map(|line| {
        let rest = line
            .trim_start()
            .strip_prefix(&format!("const bool {exit_flag} = "))?;
        let value = rest.strip_suffix(';')?;
        value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
            .then(|| value.to_string())
    })?;
    let is_const = |text: &str| -> bool {
        let Some(inner) = text.strip_prefix("static_cast<") else {
            return false;
        };
        let Some((ty, rest)) = inner.split_once(">(") else {
            return false;
        };
        let Some(number) = rest.strip_suffix(')') else {
            return false;
        };
        !ty.is_empty()
            && ty.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !number.is_empty()
            && number
                .chars()
                .all(|c| c.is_ascii_digit() || "-+.e".contains(c))
    };
    let mut out = vec![];
    for Local {
        ctype,
        name: cname,
        text: next,
    } in fresh
    {
        let (first, second) = step.lines().find_map(|line| {
            let rest = line.trim_start().strip_prefix("const ")?;
            let (_, after) = rest.split_once(' ')?;
            let after = after
                .strip_prefix(&format!("{next} = ({flag_line} ? "))?
                .strip_suffix(");")?;
            let (a, b) = after.split_once(" : ")?;
            (is_const(a) && is_const(b)).then(|| (a.to_string(), b.to_string()))
        })?;
        out.push(Local {
            ctype: ctype.clone(),
            name: cname.clone(),
            text: format!("(hawk_trip != {ENDED} ? {first} : {second})"),
        });
    }
    Some(out)
}

/// `store = type(value);` for a shaped carry, else `store = static_cast<type>(value);`.
fn put(store: &str, ctype: &str, value: &str, shaped: bool) -> String {
    if shaped {
        format!("{store} = {ctype}({value});")
    } else {
        format!("{store} = static_cast<{ctype}>({value});")
    }
}

/// The signature of a fast entry: the unit's own signature with the entry's extra parameters, under the `_suffix` name.
fn fast_head(signature: &str, name: &str, params: &str, suffix: &str) -> Result<String, String> {
    if !signature.ends_with("    long long nSamples)") {
        return Err("fast entry: the signature does not end with the partition triple".into());
    }
    let head = format!("{},\n{params})", &signature[..signature.len() - 1]);
    Ok(head.replacen(
        &format!("void {name}("),
        &format!("void {name}{suffix}("),
        1,
    ))
}

/// What the two fast entries are cut from.
pub struct Cut<'a> {
    /// The kernel's entry name the fast entries are named after.
    pub name: &'a str,
    /// The own entry's signature line, from which the fast entries' heads are made.
    pub signature: String,
    /// The lines that unpack the entry's arguments, repeated in each fast entry.
    pub unpack: String,
    /// The lane split of the fused body the entries are cut from.
    pub split: &'a LaneSplit,
    /// Whether a guarded sample counts as finished when its guard is false.
    pub counts_finished: bool,
    /// The plain lane prologue of a device entry (a flat lane with no team, set, segment or box), which the range entry opens.
    pub plain_prologue: String,
}

/// What the persistent entry is built from: its head, the carries the step reads (held in registers), the live values, the
/// head's own lines, the launch budget, the trip count, the tail and what writes an ended sample back.
struct PersistParts {
    /// The entry's signature line.
    head: String,
    /// The carries the step reads, held across trips.
    carries: Vec<HeldCarry>,
    /// The live values: register, type and name.
    live: Vec<LiveValue>,
    /// The head's lines, without the launch budget.
    pre: String,
    /// The one-line declarations the entry evaluates once for the launch budget.
    budget: Vec<String>,
    /// The runtime trip count's text.
    count: String,
    /// The tail, with the finish count rewritten and the guard dropped.
    post: String,
    /// The write-back of a carry the step never reads, when the sample's end can be rebuilt.
    rebuilt: Option<Vec<Local>>,
    /// The commit lines of an ended sample.
    commit: Vec<String>,
}

/// The sample index a persistent lane works on.
const PERSIST_INDEX: &str = "aether::SampleIndex::make(static_cast<std::size_t>(base + hawk_s))";

/// The tail with its finish count rewritten to the entry's own counter; refuses when the head or the step counts the finish.
fn counted_post(split: &LaneSplit) -> Result<String, String> {
    let per_sample = finished_on_entry()?;
    let post = split.post.with_finish_counted("++hawk_fin;", &per_sample);
    let marked = |b: &crate::body::Buffer| {
        b.marks
            .iter()
            .any(|m| m.0 == crate::body::Mark::FinishCount)
    };
    if marked(&split.step) || marked(&split.pre) || post.contains(&per_sample) {
        return Err(crate::refuse::nib(
            "a persistent entry whose step or head counts the finish itself",
        ));
    }
    Ok(post)
}

/// The carries the step reads (held in registers) and those it never reads (type, name, next value).
fn held_and_fresh(split: &LaneSplit) -> Result<(Vec<HeldCarry>, Vec<Local>), String> {
    let (mut carries, mut fresh): (Vec<HeldCarry>, Vec<Local>) = (vec![], vec![]);
    for (
        j,
        Carry {
            name: cname,
            ty,
            init,
            next,
            ..
        },
    ) in split.carries.iter().enumerate()
    {
        let ctype = carry_type(&ty.dtype, &ty.shape)?;
        if !split
            .step
            .reads_of(&BTreeSet::from([cname.clone()]))
            .is_empty()
        {
            carries.push(HeldCarry {
                store: format!("hawk_pc{j}"),
                ctype,
                name: cname.clone(),
                init: init.clone(),
                next: next.clone(),
                shaped: !ty.shape.is_empty(),
            })
        } else {
            fresh.push(Local {
                ctype,
                name: cname.clone(),
                text: next.clone(),
            })
        }
    }
    Ok((carries, fresh))
}

/// The commit lines of an ended sample: the held carries, then the rebuilt or the fresh ones.
fn entry_commit(
    carries: &[HeldCarry],
    rebuilt: &Option<Vec<Local>>,
    fresh: &[Local],
) -> Vec<String> {
    let mut commit: Vec<String> = carries
        .iter()
        .map(
            |HeldCarry {
                 store, ctype, name, ..
             }| format!("{ctype} {name} = {store};"),
        )
        .collect();
    match rebuilt {
        Some(items) => commit.extend(
            items
                .iter()
                .map(|Local { ctype, name, text }| format!("{ctype} {name} = {text};")),
        ),
        None => commit.extend(
            fresh
                .iter()
                .map(|Local { ctype, name, text }| format!("{ctype} {name} = {text};")),
        ),
    }
    commit
        .into_iter()
        .map(|line| format!("const {line}"))
        .collect()
}

/// Works out what the persistent entry is built from; refuses a split it cannot carry.
fn persist_parts(cut: &Cut) -> Result<PersistParts, String> {
    let split = cut.split;
    let post = counted_post(split)?;
    let (carries, fresh) = held_and_fresh(split)?;
    if split
        .live_ins
        .iter()
        .any(|(lname, _)| possibly_sample_free(&split.pre, lname))
    {
        return Err(crate::refuse::nib("a persistent entry with a live value no sample changes (the split does not report which are)"));
    }
    let live: Vec<LiveValue> = split
        .live_ins
        .iter()
        .enumerate()
        .map(|(m, (lname, ty))| {
            Ok(LiveValue {
                store: format!("hawk_pl{m}"),
                ctype: carry_type(&ty.dtype, &ty.shape)?,
                name: lname.clone(),
            })
        })
        .collect::<Result<_, String>>()?;
    let budget_at = launch_budget(&split.pre, &split.count);
    let pre = split.pre.text_without(&budget_at);
    let budget: Vec<String> = budget_at
        .iter()
        .map(|&at| split.pre.lines[at].trim().to_string())
        .collect();
    let count = split
        .count
        .clone()
        .ok_or("persistent entry: no runtime trip count")?;
    let post = match (&split.guard, post.is_empty()) {
        (Some(guard), false) => post.replace(&format!("!{guard} && "), ""),
        _ => post,
    };
    let head = fast_head(&cut.signature, cut.name, PERSIST_PARAMS, "_persist")?;
    let rebuilt = if post.is_empty() {
        None
    } else {
        exit_rebuilt(&split.step.text(), &split.exit, &fresh)
    };
    let commit = entry_commit(&carries, &rebuilt, &fresh);
    Ok(PersistParts {
        head,
        carries,
        live,
        pre,
        budget,
        count,
        post,
        rebuilt,
        commit,
    })
}

/// The entry's opening: the held registers, the launch budget, the counters and the head of the claim loop.
fn persist_prologue(cut: &Cut, parts: &PersistParts) -> Vec<String> {
    let PersistParts {
        head,
        carries,
        live,
        budget,
        count,
        rebuilt,
        ..
    } = parts;
    let mut lines: Vec<String> = vec![head.clone(), "{".into(), cut.unpack.clone()];
    lines.extend(
        carries
            .iter()
            .map(|HeldCarry { store, ctype, .. }| format!("    {ctype} {store}{{}};")),
    );
    lines.extend(
        live.iter()
            .map(|LiveValue { store, ctype, .. }| format!("    {ctype} {store}{{}};")),
    );
    lines.extend(budget.iter().map(|d| format!("    {d}")));
    lines.push(if budget.is_empty() { "    unsigned int hawk_budget = 0u;".into() }
               else { format!("    const unsigned int hawk_budget = static_cast<unsigned>(hawk_clamp_count({count}));") });
    lines.extend([
        "    // a lane holds a sample while it has trips left to run (hawk_trip <".into(),
        "    // hawk_budget), and needs one while it holds none to run and its last".into(),
        "    // claim fell inside count (none yet: offset 0); a sample offset is 32-bit,".into(),
        "    // as the claim counter is".into(),
    ]);
    if rebuilt.is_some() {
        lines.push(format!(
            "    // (an ended sample not yet written back: hawk_trip carries {ENDED})"
        ))
    }
    lines.extend([
        "    const unsigned hawk_count = static_cast<unsigned>(count);", "    unsigned int hawk_trip = hawk_budget;",
        "    unsigned hawk_s = 0u;", "    unsigned hawk_maxtrip = 0u;", "    unsigned hawk_fin = 0u;", "    unsigned hawk_ran = 0u;",
        "    (void)nSamples;", "    while (true) {",
        "        // the chunk head: the warp's lanes without a sample to run write back",
        "        // the one that ended, then claim the next ones together (the ballot is",
        "        // the warp's meeting point)",
        "        while (true) {",
        "            const bool hawk_needy = hawk_trip >= hawk_budget && hawk_s < hawk_count;",
        "            const unsigned hawk_need = __ballot_sync(0xffffffffu, hawk_needy);",
        "            if (hawk_need == 0u) break;",
        "            // read here, not hoisted: a value held from entry would cost the step a register",
        "            unsigned hawk_lid, hawk_below;",
        "            asm volatile(\"mov.u32 %0, %%laneid;\" : \"=r\"(hawk_lid));",
        "            asm volatile(\"mov.u32 %0, %%lanemask_lt;\" : \"=r\"(hawk_below));",
        "            const int hawk_lead = __ffs(hawk_need) - 1;",
        "            unsigned hawk_base = 0u;",
        "            if (hawk_lid == static_cast<unsigned>(hawk_lead))",
        "                hawk_base = atomicAdd(hawk_next, static_cast<unsigned>(__popc(hawk_need)));",
        "            hawk_base = __shfl_sync(0xffffffffu, hawk_base, hawk_lead);",
        "            if (hawk_needy) {",
    ].map(String::from));
    lines
}

/// The head of a chunk: the lanes without a sample write the ended one back, then claim the next ones together.
fn persist_claims(cut: &Cut, parts: &PersistParts, lines: &mut Vec<String>) {
    persist_write_back(parts, lines);
    persist_claim(cut, parts, lines);
    persist_chunk_head(lines);
}

/// A lane whose sample ended writes it back (only when the ended sample can be rebuilt).
fn persist_write_back(parts: &PersistParts, lines: &mut Vec<String>) {
    let PersistParts {
        post,
        rebuilt,
        commit,
        ..
    } = parts;
    let index = PERSIST_INDEX;
    if rebuilt.is_some() {
        lines.extend([
            "                if (hawk_trip > hawk_budget) {".to_string(),
            format!("                    const unsigned hawk_t = hawk_trip & ~{ENDED};"),
            "                    const unsigned hawk_ran1 = hawk_t != 0u ? hawk_t : hawk_budget;"
                .into(),
            "                    hawk_maxtrip = max(hawk_maxtrip, hawk_ran1);".into(),
            "                    hawk_ran += hawk_ran1;".into(),
            format!("                    const aether::SampleIndex i = {index};"),
        ]);
        lines.extend(
            commit
                .iter()
                .map(|line| format!("                    {line}")),
        );
        lines.extend([
            post.clone(),
            "                    hawk_trip = hawk_budget;".into(),
            "                }".into(),
        ]);
    }
}

/// A needy lane claims its next sample and sets up its carries and live values.
fn persist_claim(cut: &Cut, parts: &PersistParts, lines: &mut Vec<String>) {
    let PersistParts {
        carries,
        live,
        pre,
        budget,
        count,
        ..
    } = parts;
    let split = cut.split;
    let index = PERSIST_INDEX;
    lines.extend([
        "                // a lane ends at most one sample between two claims, so its".to_string(),
        "                // step count is reported long before it could wrap".into(),
        "                if (hawk_ran >= (1u << 31)) {".into(),
        "                    hawk_persist_flush(hawk_util, hawk_stepsum, hawk_ran);".into(),
        "                    hawk_ran = 0u;".into(),
        "                }".into(),
        "                hawk_s = hawk_base + static_cast<unsigned>(__popc(hawk_need & hawk_below));".into(),
        "                if (hawk_s < hawk_count) {".into(),
        format!("                    const aether::SampleIndex i = {index};"),
    ]);
    let mut inner = "                    ".to_string();
    if let Some(guard) = &split.guard {
        lines.push(format!("{inner}if (!{guard}) {{"));
        inner.push_str("    ");
    }
    if !pre.is_empty() {
        lines.push(pre.clone())
    }
    lines.extend(carries.iter().map(
        |HeldCarry {
             store,
             ctype,
             init,
             shaped,
             ..
         }| { format!("{inner}{}", put(store, ctype, init, *shaped)) },
    ));
    lines.extend(
        live.iter()
            .map(|LiveValue { store, name, .. }| format!("{inner}{store} = {name};")),
    );
    if budget.is_empty() {
        lines.push(format!(
            "{inner}hawk_budget = static_cast<unsigned>(hawk_clamp_count({count}));"
        ))
    }
    lines.push(format!("{inner}hawk_trip = 0u;"));
    if split.guard.is_some() {
        lines.push(if cut.counts_finished {
            "                    } else { ++hawk_fin; }".into()
        } else {
            "                    }".into()
        });
    }
}

/// The close of the claim loop and the check that opens the chunk.
fn persist_chunk_head(lines: &mut Vec<String>) {
    lines.extend([
        "                }".to_string(),
        "            }".into(),
        "        }".into(),
        "        if (!__any_sync(0xffffffffu, hawk_trip < hawk_budget)) break;".into(),
        "        if (hawk_util != nullptr) hawk_persist_slots(hawk_util, hawk_chunk);".into(),
        "        // the chunk: a lane whose sample ends is masked until the next head;".into(),
        "        // a lane holding a sample at a head claimed it at a head, so every such".into(),
        "        // lane reaches the chunk's end at the same trip count".into(),
        "        if (hawk_trip < hawk_budget)".into(),
    ]);
}

/// The chunk: every lane runs the step until its sample ends or the budget is spent.
fn persist_chunk(cut: &Cut, parts: &PersistParts, lines: &mut Vec<String>) {
    let PersistParts { carries, live, .. } = parts;
    let split = cut.split;
    let index = PERSIST_INDEX;
    if split.op_count <= SHORT_DEVICE_STEP_OPS {
        lines.push(PERSIST_UNROLL_LINE.to_string())
    }
    lines.extend([
        "        do {".to_string(),
        format!("            const aether::SampleIndex i = {index};"),
        "            (void)i;".into(),
        format!("            const Int {} = hawk_trip;", split.index),
        format!("            (void){};", split.index),
    ]);
    lines.extend(carries.iter().map(
        |HeldCarry {
             store,
             ctype,
             name: cname,
             ..
         }| { format!("            const {ctype} {cname} = {store};") },
    ));
    lines.extend(live.iter().map(|LiveValue { store, ctype, name }| {
        format!("            const {ctype} {name} = {store};")
    }));
    lines.push(split.step.text());
    lines.extend(
        carries
            .iter()
            .map(|HeldCarry { store, next, .. }| format!("            {store} = {next};")),
    );
    lines.extend([
        "            ++hawk_trip;".to_string(),
        format!(
            "            if ({} || hawk_trip >= hawk_budget) {{",
            split.exit
        ),
    ]);
}

/// The end of a trip, the commit of an ended sample and the closing counters.
fn persist_tail(cut: &Cut, parts: &PersistParts, lines: &mut Vec<String>) -> Result<(), String> {
    let PersistParts {
        post,
        rebuilt,
        commit,
        ..
    } = parts;
    let split = cut.split;
    if rebuilt.is_some() {
        // the sample ends: its lane only marks it, and writes it back at the head
        let mark = if rebuilt.as_ref().is_some_and(|r| !r.is_empty()) {
            format!(
                "                hawk_trip = {} ? (hawk_trip | {ENDED}) : {ENDED};",
                split.exit
            )
        } else {
            format!("                hawk_trip |= {ENDED};")
        };
        lines.push(mark);
    } else {
        lines.extend([
            "                hawk_maxtrip = max(hawk_maxtrip, hawk_trip);".to_string(),
            "                hawk_ran += hawk_trip;".into(),
            "                {".into(),
        ]);
        lines.extend(commit.iter().map(|line| format!("                {line}")));
        if !post.is_empty() {
            lines.push(post.clone())
        }
        lines.extend([
            "                }".to_string(),
            "                hawk_trip = hawk_budget;".into(),
        ]);
    }
    lines.extend([
        "                break;".to_string(),
        "            }".into(),
        "        } while ((hawk_trip & (hawk_chunk - 1u)) != 0u);".into(),
        "    }".into(),
        "    hawk_persist_util(hawk_util, hawk_ran);".into(),
        "    if (hawk_stepsum != nullptr) hawk_steps_sum(hawk_stepsum, hawk_ran);".into(),
        "    if (hawk_steps != nullptr) hawk_steps_max(hawk_steps, hawk_maxtrip);".into(),
        format!("    hawk_finish_sum({}, hawk_fin);", finish_counter()?),
        "}".into(),
    ]);
    Ok(())
}

/// The persistent entry `<name>_persist`.
pub fn persist_entry(cut: &Cut) -> Result<String, String> {
    let parts = persist_parts(cut)?;
    let mut lines = persist_prologue(cut, &parts);
    persist_claims(cut, &parts, &mut lines);
    persist_chunk(cut, &parts, &mut lines);
    persist_tail(cut, &parts, &mut lines)?;
    Ok(lines.join("\n"))
}

/// What the range entry is built from: its head, the lane prologue opened for a range, the live condition, the carries and
/// the trip count.
struct RangeParts {
    /// The entry's signature line.
    head: String,
    /// The lane prologue with its exit line turned into an opening `if`.
    opened: String,
    /// The condition under which the lane's sample is live.
    live: String,
    /// The carries, held across trips.
    carries: Vec<HeldCarry>,
    /// The runtime trip count's text.
    count: String,
}

/// Works out what the range entry is built from; refuses a lane prologue it cannot open.
fn range_parts(cut: &Cut) -> Result<RangeParts, String> {
    let split = cut.split;
    let plain = cut.plain_prologue.clone();
    let exit_line = "    if (hawk_flat >= count) return;\n    {";
    if !plain.contains(exit_line) {
        return Err("range entry: the lane prologue has no exit line".into());
    }
    let opened = plain.replace(exit_line, "    if (hawk_flat < count) {");
    let live = match &split.guard {
        None => "true".to_string(),
        Some(guard) => format!("!({guard})"),
    };
    let mut carries: Vec<HeldCarry> = vec![];
    for (
        j,
        Carry {
            name: cname,
            ty,
            init,
            next,
            ..
        },
    ) in split.carries.iter().enumerate()
    {
        carries.push(HeldCarry {
            store: format!("hawk_pc{j}"),
            ctype: carry_type(&ty.dtype, &ty.shape)?,
            name: cname.clone(),
            init: init.clone(),
            next: next.clone(),
            shaped: !ty.shape.is_empty(),
        })
    }
    let count = split
        .count
        .clone()
        .ok_or("range entry: no runtime trip count")?;
    let head = fast_head(&cut.signature, cut.name, RANGE_PARAMS, "_range")?;
    Ok(RangeParts {
        head,
        opened,
        live,
        carries,
        count,
    })
}

/// The entry's opening and the lane that runs the single step when the launch is one step.
fn range_single(cut: &Cut, parts: &RangeParts, single: &str) -> Result<Vec<String>, String> {
    let RangeParts {
        head, opened, live, ..
    } = parts;
    let mut lines: Vec<String> = vec![
        head.clone(),
        "{".into(),
        cut.unpack.clone(),
        "    unsigned hawk_ran = 0u;".into(),
        format!("    if ({} == 1) {{", fused_steps_word()?),
        opened.clone(),
        format!("        const bool hawk_live = {live};"),
        single.to_string(),
        "        hawk_ran = hawk_live ? 1u : 0u;".into(),
    ];
    if cut.counts_finished {
        lines.push(format!("        if (!hawk_live) {}", finished_on_entry()?))
    }
    lines.extend(["    }".to_string(), "    } else {".into(), opened.clone()]);
    Ok(lines)
}

/// The lane's own head: the guard, the head's lines, the held carries and the step budget.
fn range_start(cut: &Cut, parts: &RangeParts, lines: &mut Vec<String>) {
    let RangeParts { carries, count, .. } = parts;
    let split = cut.split;
    let pad = "            ";
    if let Some(guard) = &split.guard {
        lines.push(format!("        if (!{guard}) {{"))
    }
    if !split.pre.is_empty() {
        lines.push(split.pre.text())
    }
    lines.extend(carries.iter().map(
        |HeldCarry {
             store,
             ctype,
             init,
             shaped,
             ..
         }| {
            if *shaped {
                format!("{pad}{ctype} {store} = {ctype}({init});")
            } else {
                format!("{pad}{ctype} {store} = static_cast<{ctype}>({init});")
            }
        },
    ));
    lines.extend([
        format!("{pad}const int hawk_cnt = hawk_clamp_count({count});"),
        format!("{pad}int hawk_trip = 0;"),
        format!("{pad}if (hawk_cnt > 0)"),
    ]);
    if split.op_count <= SHORT_DEVICE_STEP_OPS {
        lines.push(format!("{pad}#pragma unroll 4"))
    }
}

/// The trip loop of one lane.
fn range_loop(cut: &Cut, parts: &RangeParts, lines: &mut Vec<String>) {
    let RangeParts { carries, .. } = parts;
    let split = cut.split;
    let pad = "            ";
    lines.extend([
        format!("{pad}while (true) {{"),
        format!("{pad}    const Int {} = hawk_trip;", split.index),
        format!("{pad}    (void){};", split.index),
    ]);
    lines.extend(carries.iter().map(
        |HeldCarry {
             store, ctype, name, ..
         }| { format!("{pad}    const {ctype} {name} = {store};") },
    ));
    lines.push(split.step.text());
    lines.extend(
        carries
            .iter()
            .map(|HeldCarry { store, next, .. }| format!("{pad}    {store} = {next};")),
    );
}

/// The end of the loop, the tail over the held carries and the closing counters.
fn range_tail(cut: &Cut, parts: &RangeParts, lines: &mut Vec<String>) -> Result<(), String> {
    let RangeParts { carries, .. } = parts;
    let split = cut.split;
    let pad = "            ";
    lines.extend([
        format!("{pad}    ++hawk_trip;"),
        format!(
            "{pad}    if ({} || hawk_trip >= hawk_cnt) break;",
            split.exit
        ),
        format!("{pad}}}"),
        format!("{pad}{{"),
    ]);
    lines.extend(carries.iter().map(
        |HeldCarry {
             store, ctype, name, ..
         }| format!("{pad}const {ctype} {name} = {store};"),
    ));
    if !split.post.is_empty() {
        lines.push(split.post.text())
    }
    lines.extend([
        format!("{pad}}}"),
        format!("{pad}hawk_ran = static_cast<unsigned>(hawk_trip);"),
    ]);
    if split.guard.is_some() {
        lines.push(if cut.counts_finished {
            format!("        }} else {{ {} }}", finished_on_entry()?)
        } else {
            "        }".into()
        });
    }
    lines.extend([
        "    }".to_string(),
        "    }".into(),
        "    if (hawk_steps != nullptr) hawk_steps_max(hawk_steps, hawk_ran);".into(),
        "    if (hawk_stepsum != nullptr) hawk_steps_sum(hawk_stepsum, hawk_ran);".into(),
        "}".into(),
    ]);
    Ok(())
}

/// The contiguous-range entry `<name>_range`; `single` is the single step's body.
pub fn range_entry(cut: &Cut, single: &str) -> Result<String, String> {
    let parts = range_parts(cut)?;
    let mut lines = range_single(cut, &parts, single)?;
    range_start(cut, &parts, &mut lines);
    range_loop(cut, &parts, &mut lines);
    range_tail(cut, &parts, &mut lines)?;
    Ok(lines.join("\n"))
}

/// The fast entries of a two-path unit, as lines after its own entry: the helpers they share, then the persistent entry
/// (always for a map-free kernel) and the range entry (under the build's fast-entries macro).
pub fn fast_entries(cut: &Cut, single: &str, active: bool) -> Result<Vec<String>, String> {
    let persist = persist_entry(cut)?;
    let ranged = range_entry(cut, single)?;
    let (always, gated): (Vec<String>, Vec<String>) = if active {
        (vec![], vec![ranged, String::new(), persist, String::new()])
    } else {
        (vec![persist, String::new()], vec![ranged, String::new()])
    };
    let mut out = vec![FAST_PRELUDE.to_string(), String::new()];
    out.extend(always);
    out.extend([
        "#ifndef HAWK_FAST_ENTRIES\n#define HAWK_FAST_ENTRIES 0\n#endif".to_string(),
        "#if HAWK_FAST_ENTRIES".into(),
    ]);
    out.extend(gated);
    out.extend(["#endif  // HAWK_FAST_ENTRIES".to_string(), String::new()]);
    Ok(out)
}
