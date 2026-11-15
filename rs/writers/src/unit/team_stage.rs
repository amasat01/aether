//! The body of a team stage: one scope per collective call of the stage's steps, on the lane's
//! slice of the scratch planes.
//! A team stage's walk only names those planes, so its body does not come from the walk: it comes
//! from the stage record.
use crate::body::frame::needs::{Need, Needs, Rendered};
#[cfg(feature = "old-contract")]
use crate::contract::types::{Form, Target};
use crate::contract::types::{MapBits, Scalar, TeamStage, Unit};
#[cfg(feature = "old-contract")]
use crate::contract::Contract;
use crate::ir::door::{self, RxCtx};
use crate::refuse::nib;
use crate::spell::binding_name;

/// The namespace of the collective calls.
const TEAM: &str = "aether::linalg::team::";
/// Virtual lanes one block of an ordered plain sum adds, and the blocks per key.
const FOLD_LANES: usize = 1024;
/// The number of groups a team stage's scratch is split into.
const GROUPS: usize = 64;

/// One collective call of the stage: its operation, sizes and the (role, plane) pairs it works on.
struct Step {
    /// The operation's name.
    op: String,
    /// The first size of the operation.
    n: i64,
    /// The second size of the operation.
    k: i64,
    /// The third size of the operation.
    m: i64,
    /// The most chunks the operation is cut into.
    max_chunks: i64,
    /// The (role, plane) pairs the step works on.
    planes: Vec<(String, String)>,
}

impl Step {
    /// The step as the contract describes it, copied by value.
    fn from_typed(step: &crate::contract::types::Step) -> Step {
        Step {
            op: step.op.name().to_string(),
            n: step.n,
            k: step.k,
            m: step.m,
            max_chunks: step.max_chunks,
            planes: step.planes.clone(),
        }
    }
    /// The scratch plane the step holds in `role`.
    fn plane(&self, role: &str) -> Result<&str, String> {
        self.planes
            .iter()
            .find(|p| p.0 == role)
            .map(|p| p.1.as_str())
            .ok_or_else(|| nib(format!("a team step without plane {role}")))
    }
    /// Whether the step holds a plane in `role`.
    fn has(&self, role: &str) -> bool {
        self.planes.iter().any(|p| p.0 == role)
    }
}

/// The binding of a scratch plane.
fn scratch(name: &str) -> String {
    binding_name("wide_out", name).unwrap_or_default()
}

/// The binding of the plane a stage reads in place when it is one of its inputs, else its scratch.
fn input(name: &str, reads: &[String]) -> String {
    if reads.iter().any(|r| r == name) {
        binding_name("per_sample", name).unwrap_or_default()
    } else {
        scratch(name)
    }
}

/// The pointer to the lane's own slice of a plane, `per_lane` elements wide.
fn pointer(name: &str, role: &str, per_lane: i64) -> String {
    let ctype = if role == "info" || role == "piv" {
        "std::int32_t"
    } else {
        "Real"
    };
    format!("{ctype}* const hawk_{role} = {}.data() + static_cast<std::size_t>(hawk_lane) * {per_lane};", scratch(name))
}

/// The factor's view over the step's planes, as one expression.
fn factor_view(step: &Step) -> Result<String, String> {
    let kind = match step.op.as_str() {
        "lu" | "lu_solve" | "lu_solve_t" | "lu_inverse" | "lu_det" => "lu",
        "chol" | "chol_from" | "cho_solve" | "tri_lower" | "tri_upper_t" => "chol",
        "eigh" => "eigh",
        "qr" | "qr_solve" | "qr_gram_solve" => "qr",
        other => return Err(nib(format!("team step {other}"))),
    };
    let (name, fields) = match kind {
        "lu" => ("TeamLu", "hawk_a, hawk_piv, hawk_info"),
        "eigh" => ("TeamEigh", "hawk_w, hawk_v, hawk_work, hawk_info"),
        "qr" => ("TeamQr", "hawk_a, hawk_tau, hawk_work, hawk_info"),
        _ => ("TeamCholesky", "hawk_a, hawk_info"),
    };
    Ok(format!("{TEAM}{name}<Real, {}>{{{fields}}}", step.n))
}

/// The collective call of a factorisation, a solve or a determinant step.
fn call(step: &Step) -> Result<String, String> {
    let (n, k, m) = (step.n, step.k, step.m);
    Ok(match step.op.as_str() {
        "qr" => format!("{TEAM}qr<{n}>(hawk_tm, hawk_f, hawk_a, {m});"),
        "qr_solve" => format!("{TEAM}qrSolve<{n}>(hawk_tm, hawk_f, {m}, hawk_r, hawk_x, {k});"),
        "qr_gram_solve" => format!("{TEAM}qrGramSolve<{n}>(hawk_tm, hawk_f, hawk_x, hawk_x, {k});"),
        "lu" => format!("{TEAM}lu<{n}>(hawk_tm, hawk_f, hawk_a);"),
        "chol" => format!("{TEAM}cholesky<{n}>(hawk_tm, hawk_f, hawk_a);"),
        "chol_from" => format!("{TEAM}choleskyFrom<{n}>(hawk_tm, hawk_f, hawk_a);"),
        "eigh" => format!("{TEAM}eigh<{n}>(hawk_tm, hawk_f, hawk_a);"),
        "lu_inverse" => format!("{TEAM}luInverse<{n}>(hawk_tm, hawk_f, hawk_x);"),
        "lu_det" => format!("if (hawk_tm.rank == 0) *hawk_x = {TEAM}luDet<{n}>(hawk_f);"),
        "lu_solve" | "lu_solve_t" | "cho_solve" | "tri_lower" | "tri_upper_t" => {
            let name = match step.op.as_str() {
                "lu_solve" => "luSolve",
                "lu_solve_t" => "luSolveT",
                "cho_solve" => "choSolve",
                "tri_lower" => "triLower",
                _ => "triUpperT",
            };
            format!("{TEAM}{name}<{n}>(hawk_tm, hawk_f, hawk_x, hawk_x, {k});")
        }
        other => return Err(nib(format!("team step {other}"))),
    })
}

/// The owned form of a list of lines.
fn strs(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|l| l.to_string()).collect()
}

/// The halving tree over `hawk_sh`, `width` cells, ending with the barrier.
fn halving_tree(width: usize) -> Vec<String> {
    strs(&[&format!("for (std::size_t hawk_w = {}; hawk_w > 0; hawk_w >>= 1) {{", width / 2),
        "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; hawk_i += static_cast<std::size_t>(hawk_tm.size))",
        "        hawk_sh[hawk_i] += hawk_sh[hawk_i + hawk_w];", "    hawk_tm.sync();", "}"])
}

/// The call of a block sum (no index map): the team of key `hawk_lane` sums that key's rows into
/// its block, then copies the
/// block to the output plane when the sum is stored.
fn block_sum(
    step: &Step,
    per_lane: &[(String, i64)],
    active: Option<&(String, String)>,
) -> Result<Vec<String>, String> {
    let (p, q, per) = (step.n, step.k, step.m.max(1));
    let mut lines = strs(&[
        "const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
        "const std::size_t hawk_keys = static_cast<std::size_t>(count);",
        "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / hawk_keys;",
    ]);
    if let Some((map, count)) = active {
        lines.extend(live_row_edges(map, count));
    }
    let stride_v = factor_pointers(step, q, &mut lines)?;
    let call_head = format!("{TEAM}blockSum<{p}, {q}>(hawk_tm, ");
    if step.has("gp") && active.is_none() {
        lines.extend(grouped_store(step)?);
        return Ok(lines);
    }
    let first = "hawk_key * hawk_rows";
    if !step.has("g") || active.is_some() {
        lines.push(format!(
            "Real* const hawk_out = {}.data();",
            scratch(step.plane("out")?)
        ));
        let (range, map) = if active.is_some() {
            ("hawk_edge[0], hawk_edge[1] - hawk_edge[0]", "hawk_map")
        } else {
            ("hawk_key * hawk_rows, hawk_rows", "nullptr")
        };
        lines.push(format!("{call_head}hawk_out + hawk_key, hawk_u, hawk_v, {range}, {p}, {stride_v}, {map}, {q} * hawk_keys, hawk_keys, {per});"));
        return Ok(lines);
    }
    let g = step.plane("g")?;
    let width = per_lane
        .iter()
        .find(|x| x.0 == g)
        .map(|x| x.1)
        .ok_or_else(|| nib("a team plane without width"))?;
    lines.push(pointer(g, "g", width));
    lines.push(format!("{call_head}hawk_g, hawk_u, hawk_v, {first}, hawk_rows, {p}, {stride_v}, nullptr, 0, 0, {per});"));
    if step.has("out") {
        lines.push(format!(
            "Real* const hawk_out = {}.data();",
            scratch(step.plane("out")?)
        ));
        lines.extend(strs(&[&format!("for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {}; hawk_e += static_cast<std::size_t>(hawk_tm.size))", p * q),
            "    hawk_out[hawk_key + hawk_e * hawk_keys] = hawk_g[hawk_e];"]));
    }
    Ok(lines)
}

/// The first live row at or after each of a key's two row boundaries, found by bisection over the
/// live samples' sorted map.
fn live_row_edges(map: &str, count: &str) -> Vec<String> {
    let mut lines = vec![];
    // the keys' row ranges come from the live samples' sorted map
    lines.extend(strs(&[&format!("const std::int32_t* const hawk_map = reinterpret_cast<const std::int32_t*>({map}.data());"),
        &format!("const std::size_t hawk_live = static_cast<std::size_t>(reinterpret_cast<const std::int32_t*>({count}.data())[0]);"),
        "std::size_t hawk_edge[2];",
        "for (std::size_t hawk_s = 0; hawk_s < 2; ++hawk_s) {",
        "    std::size_t hawk_lo = 0, hawk_hi = hawk_live;",
        "    while (hawk_lo < hawk_hi) {",
        "        const std::size_t hawk_mid = (hawk_lo + hawk_hi) / 2;",
        "        if (static_cast<std::size_t>(hawk_map[hawk_mid]) < (hawk_key + hawk_s) * hawk_rows) hawk_lo = hawk_mid + 1;",
        "        else hawk_hi = hawk_mid;",
        "    }",
        "    hawk_edge[hawk_s] = hawk_lo;",
        "}"]));
    lines
}

/// The pointers to the factors a block sum reads (`u`, `v`); returns the stride of `v`, zero when
/// it is the single one.
fn factor_pointers(step: &Step, q: i64, lines: &mut Vec<String>) -> Result<String, String> {
    if step.has("u") {
        lines.push(format!(
            "const Real* const hawk_u = {}.data();",
            scratch(step.plane("u")?)
        ))
    }
    Ok(if step.has("v") {
        lines.push(format!(
            "const Real* const hawk_v = {}.data();",
            scratch(step.plane("v")?)
        ));
        q.to_string()
    } else {
        lines.extend(strs(&[
            "const Real hawk_one = Real(1);",
            "const Real* const hawk_v = &hawk_one;",
        ]));
        "0".to_string()
    })
}

/// The statements of a block sum whose partial groups are stored: each group's value is read,
/// folded by the halving tree and
/// the result written to the output plane.
fn grouped_store(step: &Step) -> Result<Vec<String>, String> {
    let mut lines = vec![];
    lines.push(format!("Real hawk_sh[{GROUPS}];"));
    lines.extend(strs(&[&format!("for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < {GROUPS}; hawk_i += static_cast<std::size_t>(hawk_tm.size))"),
        &format!("    hawk_sh[hawk_i] = {}.data()[hawk_key * {GROUPS} + hawk_i];", scratch(step.plane("gp")?)), "hawk_tm.sync();"]));
    lines.extend(halving_tree(GROUPS));
    lines.extend(strs(&[
        &format!(
            "if (hawk_tm.rank == 0) {}.data()[hawk_key] = hawk_sh[0];",
            scratch(step.plane("out")?)
        ),
        "hawk_tm.sync();",
    ]));
    Ok(lines)
}

/// The first stage of a plain ordered sum: block `hawk_lane` folds its group of virtual lanes by
/// the halving tree and writes
/// the group's value to its cell of the partials plane.
fn block_part(step: &Step, reads: &[String]) -> Result<Vec<String>, String> {
    let virtual_lanes = FOLD_LANES * GROUPS;
    let mut lines = strs(&[
        &format!("const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane) / {GROUPS};"),
        &format!("const std::size_t hawk_grp = static_cast<std::size_t>(hawk_lane) % {GROUPS};"),
        &format!("const std::size_t hawk_keys = static_cast<std::size_t>(count) / {GROUPS};"),
        "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / hawk_keys;",
        &format!(
            "const Real* const hawk_u = {}.data() + hawk_key * hawk_rows;",
            input(step.plane("u")?, reads)
        ),
    ]);
    let term = if step.has("v") {
        lines.push(format!(
            "const Real* const hawk_v = {}.data() + hawk_key * hawk_rows;",
            input(step.plane("v")?, reads)
        ));
        "hawk_u[hawk_r] * hawk_v[hawk_r]"
    } else {
        "hawk_u[hawk_r]"
    };
    lines.extend(strs(&[&format!("Real hawk_sh[{FOLD_LANES}];"),
        &format!("for (std::size_t hawk_t = 0; hawk_t < {FOLD_LANES}; ++hawk_t) hawk_sh[hawk_t] = Real(0);"),
        &format!("for (std::size_t hawk_b = hawk_grp * {FOLD_LANES}; hawk_b < hawk_rows; hawk_b += {virtual_lanes}) {{"),
        &format!("    const std::size_t hawk_m = hawk_rows - hawk_b < {FOLD_LANES} ? hawk_rows - hawk_b : {FOLD_LANES};"),
        "    for (std::size_t hawk_t = 0; hawk_t < hawk_m; ++hawk_t) {", "        const std::size_t hawk_r = hawk_b + hawk_t;",
        &format!("        hawk_sh[hawk_t] += {term};"), "    }", "}", "hawk_tm.sync();"]));
    lines.extend(halving_tree(FOLD_LANES));
    lines.extend(strs(&[
        &format!(
            "if (hawk_tm.rank == 0) {}.data()[hawk_key * {GROUPS} + hawk_grp] = hawk_sh[0];",
            scratch(step.plane("gp")?)
        ),
        "hawk_tm.sync();",
    ]));
    Ok(lines)
}

/// The call of an ordered fold: the team of key `hawk_lane` adds the key's contributions to each of
/// the `n` slots in lane order.
fn ordered_fold(step: &Step) -> Result<Vec<String>, String> {
    let n = step.n;
    Ok(strs(&["const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
        "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / static_cast<std::size_t>(count);",
        &format!("const Real* const hawk_v = {}.data();", scratch(step.plane("v")?)),
        &format!("const Real* const hawk_c = {}.data();", scratch(step.plane("i")?)),
        &format!("Real* const hawk_out = {}.data();", scratch(step.plane("out")?)),
        &format!("for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {n}; hawk_e += static_cast<std::size_t>(hawk_tm.size)) {{"),
        "    Real hawk_s = Real(0);", "    for (std::size_t hawk_r = 0; hawk_r < hawk_rows; ++hawk_r)",
        &format!("        hawk_s += hawk_v[(hawk_key * hawk_rows + hawk_r) * {n} + hawk_e];"),
        &format!("    hawk_out[static_cast<std::size_t>(hawk_c[hawk_key * {n} + hawk_e])] = hawk_s;"), "}", "hawk_tm.sync();"]))
}

/// The chunks of an ordered scatter: `min(max_chunks, max(1, nSamples / cells))`.
fn chunk_count(step: &Step) -> Result<Vec<String>, String> {
    if step.max_chunks == 0 {
        return Err(nib("an ordered scatter step without max_chunks"));
    }
    let m = step.max_chunks;
    Ok(strs(&[
        "const std::size_t hawk_cells = static_cast<std::size_t>(count);",
        "const std::size_t hawk_fit = static_cast<std::size_t>(nSamples) / hawk_cells;",
        &format!(
            "const std::size_t hawk_chunks = hawk_fit < 1 ? 1 : (hawk_fit > {m} ? {m} : hawk_fit);"
        ),
    ]))
}

/// An ordered scatter's first stage: block `hawk_lane` zeroes its cell's partial sums; block 0
/// writes the cell count.
fn scatter_zero(step: &Step) -> Result<Vec<String>, String> {
    let mut lines = chunk_count(step)?;
    lines.extend(strs(&[&format!("Real* const hawk_p = {}.data() + static_cast<std::size_t>(hawk_lane) * hawk_chunks;", scratch(step.plane("part")?)),
        &format!("std::int32_t* const hawk_m = const_cast<std::int32_t*>({}.data());", scratch(step.plane("status")?)),
        "if (hawk_tm.rank == 0) {", "    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_p[hawk_b] = Real(0);",
        "    if (hawk_lane == 0) hawk_m[0] = static_cast<std::int32_t>(hawk_cells);", "}", "hawk_tm.sync();"]));
    Ok(lines)
}

/// An ordered scatter's middle stage: block `hawk_lane` adds the slots of its lane chunk, serially,
/// into its own partial sums.
fn scatter_chunks(step: &Step) -> Result<Vec<String>, String> {
    let n = step.n;
    Ok(strs(&["const std::size_t hawk_chunks = static_cast<std::size_t>(count);",
        "const std::size_t hawk_span = (static_cast<std::size_t>(nSamples) + hawk_chunks - 1) / hawk_chunks;",
        "const std::size_t hawk_lo = static_cast<std::size_t>(hawk_lane) * hawk_span;",
        "const std::size_t hawk_hi = hawk_lo + hawk_span < static_cast<std::size_t>(nSamples) ? hawk_lo + hawk_span : static_cast<std::size_t>(nSamples);",
        &format!("std::int32_t* const hawk_k = const_cast<std::int32_t*>({}.data());", scratch(step.plane("k")?)),
        &format!("const Real* const hawk_v = {}.data();", scratch(step.plane("v")?)),
        &format!("Real* const hawk_p = {}.data();", scratch(step.plane("part")?)),
        &format!("std::int32_t* const hawk_m = const_cast<std::int32_t*>({}.data());", scratch(step.plane("status")?)),
        "const std::int64_t hawk_cells = hawk_m[0];", "if (hawk_tm.rank == 0)",
        &format!("    for (std::size_t hawk_r = hawk_lo * {n}; hawk_r < hawk_hi * {n}; ++hawk_r) {{"),
        "        const std::int32_t hawk_key = hawk_k[hawk_r];", "        if (hawk_key == -1) continue;",
        "        if (hawk_key >= 0 && hawk_key < hawk_cells)",
        "            hawk_p[static_cast<std::size_t>(hawk_key) * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_v[hawk_r];",
        "        else", "            hawk_m[1 + hawk_lane] = 1;", "        hawk_k[hawk_r] = -1;", "    }", "hawk_tm.sync();"]))
}

/// An ordered scatter's last stage: one thread adds the cell's partial sums in chunk order and the
/// total to the cell.
fn scatter_cells(step: &Step) -> Result<Vec<String>, String> {
    let mut lines = chunk_count(step)?;
    lines.extend(strs(&[&format!("const Real* const hawk_p = {}.data() + static_cast<std::size_t>(hawk_lane) * hawk_chunks;", scratch(step.plane("part")?)),
        &format!("Real* const hawk_out = {}.data();", scratch(step.plane("out")?)), "if (hawk_tm.rank == 0) {", "    Real hawk_s = Real(0);",
        "    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_s += hawk_p[hawk_b];",
        "    hawk_out[hawk_lane] += hawk_s;", "}", "hawk_tm.sync();"]));
    Ok(lines)
}

/// The statements of one step, before indentation: pointers to the planes, the factor view and the
/// call, or the step's own form.
fn step_lines(
    step: &Step,
    per_lane: &[(String, i64)],
    reads: &[String],
    active: Option<&(String, String)>,
) -> Result<Vec<String>, String> {
    match step.op.as_str() {
        "ordered_fold" => return ordered_fold(step),
        "scatter_zero" => return scatter_zero(step),
        "scatter_chunks" => return scatter_chunks(step),
        "scatter_cells" => return scatter_cells(step),
        // under an index map the partial blocks are the sums' own: this step writes nothing
        "block_part" if active.is_some() => return Ok(vec![]),
        "block_part" => return block_part(step, reads),
        "block_sum" => return block_sum(step, per_lane, active),
        "eigh_warp" => return Err(nib("a team step eigh_warp")),
        _ => {}
    }
    let mut lines = vec![];
    for (role, name) in &step.planes {
        let width = per_lane
            .iter()
            .find(|x| &x.0 == name)
            .map(|x| x.1)
            .ok_or_else(|| nib("a team plane without width"))?;
        lines.push(pointer(name, role, width));
    }
    lines.push(format!("const auto hawk_f = {};", factor_view(step)?));
    lines.push(call(step)?);
    Ok(lines)
}

/// Whether the recorded call is a team stage.
#[cfg(feature = "old-contract")]
pub fn is_team_stage(contract: &Contract) -> bool {
    let unit = crate::contract::old::typed_of(contract, Target::Host);
    unit.is_ok_and(|unit| matches!(unit.form, Form::TeamStage(_)))
}

/// The body of a team stage: the team and lane bindings, then each step in a scope of its own.
/// Written for a float stage of a
/// host row without an index map; the banded forms and an index-set build are refused.
pub fn stage_body(ctx: *mut RxCtx, unit: &Unit, stage: &TeamStage) -> Result<Rendered, String> {
    let active = match &stage.active {
        None => None,
        Some(a) => {
            if a.map_bits != MapBits::B32 {
                return Err(nib("an index map that is not 32 bits wide"));
            }
            match &a.map_binding {
                Some(map) => Some((map.clone(), a.count_binding.clone())),
                None => return Err(nib("an index map record")),
            }
        }
    };
    if unit.scalar == Scalar::Banded {
        return Err(nib("a team stage in a banded scalar mode"));
    }
    let per_lane = stage.planes.clone();
    let mut reads = vec![];
    for &node in &stage.reads {
        reads.push(door::field_text(ctx, node, "name")?.unwrap_or_default());
    }
    // every stage binds the team through the collective namespace
    let mut needs = Needs::default();
    needs.record(Need::Linalg);
    needs.record(Need::Team);
    let mut lines = vec![
        format!("        const {TEAM}Team hawk_tm = {TEAM}Team::here();"),
        "        const long long hawk_lane = hawk_i;".to_string(),
    ];
    for typed in &stage.steps {
        let step = Step::from_typed(typed);
        lines.push(format!("        {{  // {}, n = {}", step.op, step.n));
        lines.extend(
            step_lines(&step, &per_lane, &reads, active.as_ref())?
                .into_iter()
                .map(|l| format!("            {l}")),
        );
        lines.push("        }".to_string());
    }
    Ok(Rendered {
        text: lines.join("\n"),
        needs,
    })
}
