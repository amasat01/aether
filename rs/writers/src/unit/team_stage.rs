//! The body of a team stage: one scope per collective call of the stage's steps, on the lane's
//! slice of the scratch planes.
//! A team stage's walk only names those planes, so its body does not come from the walk: it comes
//! from the stage record.
use crate::body::frame::needs::{Need, Needs, Rendered};
#[cfg(feature = "old-contract")]
use crate::contract::types::Form;
use crate::contract::types::{MapBits, Scalar, Target, TeamStage, Unit};
#[cfg(feature = "old-contract")]
use crate::contract::Contract;
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, unknown, Refusal};
use crate::spell::binding_name;
use crate::unit::team_banded as band;

/// The namespace of the collective calls.
const TEAM: &str = "aether::linalg::team::";
/// Virtual lanes one block of an ordered plain sum adds, and the blocks per key.
const FOLD_LANES: usize = 1024;

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
    fn plane(&self, role: &str) -> Result<&str, Refusal> {
        self.planes
            .iter()
            .find(|p| p.0 == role)
            .map(|p| p.1.as_str())
            .ok_or_else(|| unknown(format!("a team step without plane {role}")))
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
fn factor_view(step: &Step) -> Result<String, Refusal> {
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
fn call(step: &Step) -> Result<String, Refusal> {
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

/// The declaration of a team's scratch of `width` partial sums: in shared memory on the device.
fn partials_decl(width: usize, device: bool) -> String {
    let shared = if device { "__shared__ " } else { "" };
    format!("{shared}Real hawk_sh[{width}];")
}

/// The groups a team stage's scratch is split into: the width of the step's partials plane, which the plan that made the stage
/// states. The halving tree needs a power of two.
fn groups_of(step: &Step, per_lane: &[(String, i64)]) -> Result<usize, Refusal> {
    let name = step.plane("gp")?;
    let width = per_lane
        .iter()
        .find(|plane| plane.0 == name)
        .map(|plane| plane.1)
        .ok_or_else(|| unknown("a team plane without width"))?;
    match usize::try_from(width) {
        Ok(groups) if groups.is_power_of_two() => Ok(groups),
        _ => Err(nib(format!(
            "a block sum split into {width} groups (a power of two is needed)"
        ))),
    }
}

/// The call of a block sum (no index map): the team of key `hawk_lane` sums that key's rows into
/// its block, then copies the
/// block to the output plane when the sum is stored.
fn block_sum(
    step: &Step,
    per_lane: &[(String, i64)],
    active: Option<&(String, String)>,
    device: bool,
    banded: bool,
) -> Result<Vec<String>, Refusal> {
    let (p, q, per) = (step.n, step.k, step.m.max(1));
    let mut lines = strs(&[
        "const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
        "const std::size_t hawk_keys = static_cast<std::size_t>(count);",
        "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / hawk_keys;",
    ]);
    if let Some((map, count)) = active {
        lines.extend(live_row_edges(map, count));
    }
    let stride_v = factor_pointers(step, q, banded, &mut lines)?;
    let call_head = format!("{TEAM}blockSum<{p}, {q}>(hawk_tm, ");
    if step.has("gp") && active.is_none() {
        lines.extend(grouped_store(
            step,
            groups_of(step, per_lane)?,
            device,
            banded,
        )?);
        return Ok(lines);
    }
    if banded {
        return Err(nib(
            "a team step block_sum without partial groups in a banded scalar mode",
        ));
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
        .ok_or_else(|| unknown("a team plane without width"))?;
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
fn factor_pointers(
    step: &Step,
    q: i64,
    banded: bool,
    lines: &mut Vec<String>,
) -> Result<String, Refusal> {
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
        let one = if banded { band::ONE } else { "Real(1)" };
        lines.extend(strs(&[
            &format!("const Real hawk_one = {one};"),
            "const Real* const hawk_v = &hawk_one;",
        ]));
        "0".to_string()
    })
}

/// The statements of a block sum whose partial groups are stored: each group's value is read,
/// folded by the halving tree and
/// the result written to the output plane.
fn grouped_store(
    step: &Step,
    groups: usize,
    device: bool,
    banded: bool,
) -> Result<Vec<String>, Refusal> {
    if banded {
        return Ok(band::block_groups(
            &scratch(step.plane("gp")?),
            &scratch(step.plane("out")?),
            groups,
        ));
    }
    let mut lines = vec![];
    lines.push(partials_decl(groups, device));
    lines.extend(strs(&[&format!("for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < {groups}; hawk_i += static_cast<std::size_t>(hawk_tm.size))"),
        &format!("    hawk_sh[hawk_i] = {}.data()[hawk_key * {groups} + hawk_i];", scratch(step.plane("gp")?)), "hawk_tm.sync();"]));
    lines.extend(halving_tree(groups));
    lines.extend(strs(&[
        &format!(
            "if (hawk_tm.rank == 0) {}.data()[hawk_key] = hawk_sh[0];",
            scratch(step.plane("out")?)
        ),
        "hawk_tm.sync();",
    ]));
    Ok(lines)
}

/// The fold of a block part on the host: one thread adds the rows of every virtual lane in order.
fn host_fold(term: &str, groups: usize) -> Vec<String> {
    let virtual_lanes = FOLD_LANES * groups;
    strs(&[&format!("for (std::size_t hawk_t = 0; hawk_t < {FOLD_LANES}; ++hawk_t) hawk_sh[hawk_t] = Real(0);"),
        &format!("for (std::size_t hawk_b = hawk_grp * {FOLD_LANES}; hawk_b < hawk_rows; hawk_b += {virtual_lanes}) {{"),
        &format!("    const std::size_t hawk_m = hawk_rows - hawk_b < {FOLD_LANES} ? hawk_rows - hawk_b : {FOLD_LANES};"),
        "    for (std::size_t hawk_t = 0; hawk_t < hawk_m; ++hawk_t) {", "        const std::size_t hawk_r = hawk_b + hawk_t;",
        &format!("        hawk_sh[hawk_t] += {term};"), "    }", "}"])
}

/// The fold of a block part on the device: each thread of the team adds the rows of its own virtual lanes in order.
fn device_fold(term: &str, groups: usize) -> Vec<String> {
    let virtual_lanes = FOLD_LANES * groups;
    strs(&[&format!("for (std::size_t hawk_t = static_cast<std::size_t>(hawk_tm.rank); hawk_t < {FOLD_LANES}; hawk_t += static_cast<std::size_t>(hawk_tm.size)) {{"),
        "    Real hawk_s = Real(0);",
        &format!("    for (std::size_t hawk_r = hawk_grp * {FOLD_LANES} + hawk_t; hawk_r < hawk_rows; hawk_r += {virtual_lanes})"),
        &format!("        hawk_s += {term};"), "    hawk_sh[hawk_t] = hawk_s;", "}"])
}

/// The first stage of a plain ordered sum: block `hawk_lane` folds its group of virtual lanes by
/// the halving tree and writes
/// the group's value to its cell of the partials plane.
fn block_part(
    step: &Step,
    per_lane: &[(String, i64)],
    reads: &[String],
    device: bool,
    banded: bool,
) -> Result<Vec<String>, Refusal> {
    let groups = groups_of(step, per_lane)?;
    let mut lines = strs(&[
        &format!("const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane) / {groups};"),
        &format!("const std::size_t hawk_grp = static_cast<std::size_t>(hawk_lane) % {groups};"),
        &format!("const std::size_t hawk_keys = static_cast<std::size_t>(count) / {groups};"),
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
    if banded {
        lines.extend(band::block_fold(
            &scratch(step.plane("gp")?),
            step.has("v"),
            groups,
        ));
        return Ok(lines);
    }
    lines.push(partials_decl(FOLD_LANES, device));
    lines.extend(if device {
        device_fold(term, groups)
    } else {
        host_fold(term, groups)
    });
    lines.push("hawk_tm.sync();".to_string());
    lines.extend(halving_tree(FOLD_LANES));
    lines.extend(strs(&[
        &format!(
            "if (hawk_tm.rank == 0) {}.data()[hawk_key * {groups} + hawk_grp] = hawk_sh[0];",
            scratch(step.plane("gp")?)
        ),
        "hawk_tm.sync();",
    ]));
    Ok(lines)
}

/// The call of an ordered fold: the team of key `hawk_lane` adds the key's contributions to each of
/// the `n` slots in lane order.
fn ordered_fold(step: &Step, banded: bool) -> Result<Vec<String>, Refusal> {
    let n = step.n;
    if banded {
        return Ok(band::ordered_fold(
            n,
            &scratch(step.plane("v")?),
            &scratch(step.plane("i")?),
            &scratch(step.plane("out")?),
        ));
    }
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
fn chunk_count(step: &Step) -> Result<Vec<String>, Refusal> {
    if step.max_chunks == 0 {
        return Err(unknown("an ordered scatter step without max_chunks"));
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
fn scatter_zero(step: &Step, banded: bool) -> Result<Vec<String>, Refusal> {
    let mut lines = chunk_count(step)?;
    let (pointer, zero) = if banded {
        band::scatter_zero(&scratch(step.plane("part")?))
    } else {
        (
            format!("Real* const hawk_p = {}.data() + static_cast<std::size_t>(hawk_lane) * hawk_chunks;", scratch(step.plane("part")?)),
            "Real(0)".to_string(),
        )
    };
    lines.extend(strs(&[&pointer,
        &format!("std::int32_t* const hawk_m = const_cast<std::int32_t*>({}.data());", scratch(step.plane("status")?)),
        "if (hawk_tm.rank == 0) {", &format!("    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_p[hawk_b] = {zero};"),
        "    if (hawk_lane == 0) hawk_m[0] = static_cast<std::int32_t>(hawk_cells);", "}", "hawk_tm.sync();"]));
    Ok(lines)
}

/// An ordered scatter's middle stage: block `hawk_lane` adds the slots of its lane chunk, serially,
/// into its own partial sums.
fn scatter_chunks(step: &Step, device: bool, banded: bool) -> Result<Vec<String>, Refusal> {
    let n = step.n;
    let part = if banded {
        band::scatter_part(&scratch(step.plane("part")?))
    } else {
        format!(
            "Real* const hawk_p = {}.data();",
            scratch(step.plane("part")?)
        )
    };
    let mut lines = strs(&["const std::size_t hawk_chunks = static_cast<std::size_t>(count);",
        "const std::size_t hawk_span = (static_cast<std::size_t>(nSamples) + hawk_chunks - 1) / hawk_chunks;",
        "const std::size_t hawk_lo = static_cast<std::size_t>(hawk_lane) * hawk_span;",
        "const std::size_t hawk_hi = hawk_lo + hawk_span < static_cast<std::size_t>(nSamples) ? hawk_lo + hawk_span : static_cast<std::size_t>(nSamples);",
        &format!("std::int32_t* const hawk_k = const_cast<std::int32_t*>({}.data());", scratch(step.plane("k")?)),
        &format!("const Real* const hawk_v = {}.data();", scratch(step.plane("v")?)),
        &part,
        &format!("std::int32_t* const hawk_m = const_cast<std::int32_t*>({}.data());", scratch(step.plane("status")?)),
        "const std::int64_t hawk_cells = hawk_m[0];"]);
    lines.extend(if device {
        device_walk(n, banded)
    } else {
        serial_walk(n, banded)
    });
    Ok(lines)
}

/// The slots of a chunk added serially by one thread of the team, in order.
fn serial_walk(n: i64, banded: bool) -> Vec<String> {
    let add = if banded {
        band::slot_add()
    } else {
        "            hawk_p[static_cast<std::size_t>(hawk_key) * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_v[hawk_r];".to_string()
    };
    strs(&[
        "if (hawk_tm.rank == 0)",
        &format!(
            "    for (std::size_t hawk_r = hawk_lo * {n}; hawk_r < hawk_hi * {n}; ++hawk_r) {{"
        ),
        "        const std::int32_t hawk_key = hawk_k[hawk_r];",
        "        if (hawk_key == -1) continue;",
        "        if (hawk_key >= 0 && hawk_key < hawk_cells)",
        &add,
        "        else",
        "            hawk_m[1 + hawk_lane] = 1;",
        "        hawk_k[hawk_r] = -1;",
        "    }",
        "hawk_tm.sync();",
    ])
}

/// The slots of a chunk on the device: up to 1024 cells, a warp takes 32 rows at a time into a shared bin, each thread owning the
/// cells of its residue modulo 32 and adding them in row order, so every cell's sum has the order of the serial walk; a larger
/// target walks serially.
fn device_walk(n: i64, banded: bool) -> Vec<String> {
    let forms = banded.then(band::bin_forms);
    let decl = forms
        .as_ref()
        .map_or("__shared__ Real hawk_bin[1024];".to_string(), |b| {
            b.decl.clone()
        });
    let zero = forms
        .as_ref()
        .map_or("Real(0)".to_string(), |b| b.zero.clone());
    let shuffled = forms.as_ref().map_or(
        vec![
            "                const Real hawk_x = __shfl_sync(0xffffffffu, hawk_mv, hawk_src);"
                .to_string(),
        ],
        |b| b.shuffled.clone(),
    );
    let add = forms.as_ref().map_or(
        "                    hawk_bin[hawk_key] += hawk_x;".to_string(),
        |b| b.add.clone(),
    );
    let combine = forms.as_ref().map_or(
        "        hawk_p[hawk_j * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_bin[hawk_j];".to_string(),
        |b| b.combine.clone(),
    );
    let mut lines = strs(&[
        &decl,
        "if (hawk_cells <= 1024) {",
        &format!("    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); hawk_j += static_cast<std::size_t>(hawk_tm.size)) hawk_bin[hawk_j] = {zero};"),
        "    hawk_tm.sync();",
        "    if (hawk_tm.rank < 32) {",
        "        const unsigned hawk_ln = static_cast<unsigned>(hawk_tm.rank);",
        &format!("        const std::size_t hawk_end = hawk_hi * {n};"),
        &format!("        for (std::size_t hawk_b = hawk_lo * {n}; hawk_b < hawk_end; hawk_b += 32) {{"),
        "            const std::size_t hawk_r = hawk_b + hawk_ln;",
        "            const bool hawk_in = hawk_r < hawk_end;",
        "            const std::int32_t hawk_mk = hawk_in ? hawk_k[hawk_r] : -1;",
        "            const Real hawk_mv = hawk_in ? hawk_v[hawk_r] : Real{};",
        "            unsigned hawk_mine = 0u, hawk_bad = 0u;",
        "            for (unsigned hawk_i = 0; hawk_i < 32; ++hawk_i) {",
        "                const std::int32_t hawk_key = __shfl_sync(0xffffffffu, hawk_mk, hawk_i);",
        "                if (hawk_key == -1) continue;",
        "                if (hawk_key >= 0 && hawk_key < hawk_cells) {",
        "                    if ((static_cast<unsigned>(hawk_key) & 31u) == hawk_ln) hawk_mine |= 1u << hawk_i;",
        "                } else hawk_bad = 1u;",
        "            }",
        "            if (hawk_bad && hawk_ln == 0) hawk_m[1 + hawk_lane] = 1;",
        "            while (__any_sync(0xffffffffu, hawk_mine != 0u)) {",
        "                const unsigned hawk_src = hawk_mine != 0u ? static_cast<unsigned>(__ffs(hawk_mine) - 1) : 0u;",
    ]);
    lines.extend(shuffled);
    lines.extend(strs(&[
        "                const std::int32_t hawk_key = __shfl_sync(0xffffffffu, hawk_mk, hawk_src);",
        "                if (hawk_mine != 0u) {",
        &add,
        "                    hawk_mine &= hawk_mine - 1u;",
        "                }",
        "            }",
        "            if (hawk_in && hawk_mk != -1) hawk_k[hawk_r] = -1;",
        "        }",
        "    }",
        "    hawk_tm.sync();",
        "    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); hawk_j += static_cast<std::size_t>(hawk_tm.size))",
        &combine,
        "} else {",
    ]));
    lines.extend(serial_walk(n, banded));
    lines.push("}".to_string());
    lines
}

/// An ordered scatter's last stage: one thread adds the cell's partial sums in chunk order and the
/// total to the cell.
fn scatter_cells(step: &Step, banded: bool) -> Result<Vec<String>, Refusal> {
    let mut lines = chunk_count(step)?;
    if banded {
        return Ok(band::scatter_cells(
            lines,
            &scratch(step.plane("part")?),
            &scratch(step.plane("out")?),
        ));
    }
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
    device: bool,
    banded: bool,
) -> Result<Vec<String>, Refusal> {
    match step.op.as_str() {
        "ordered_fold" => return ordered_fold(step, banded),
        "scatter_zero" => return scatter_zero(step, banded),
        "scatter_chunks" => return scatter_chunks(step, device, banded),
        "scatter_cells" => return scatter_cells(step, banded),
        // under an index map the partial blocks are the sums' own: this step writes nothing
        "block_part" if active.is_some() => return Ok(vec![]),
        "block_part" => return block_part(step, per_lane, reads, device, banded),
        "block_sum" => return block_sum(step, per_lane, active, device, banded),
        "eigh_warp" => return Err(nib("a team step eigh_warp")),
        _ => {}
    }
    if banded {
        return Err(nib(format!(
            "team step {} in a banded scalar mode",
            step.op
        )));
    }
    let mut lines = vec![];
    for (role, name) in &step.planes {
        let width = per_lane
            .iter()
            .find(|x| &x.0 == name)
            .map(|x| x.1)
            .ok_or_else(|| unknown("a team plane without width"))?;
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
/// host row without an index map; a banded stage is written on the device, for the ordered sums and scatters, and an index-set build is refused.
pub fn stage_body(ctx: *mut RxCtx, unit: &Unit, stage: &TeamStage) -> Result<Rendered, Refusal> {
    let active = match &stage.active {
        None => None,
        Some(a) => {
            if a.map_bits != MapBits::B32 {
                return Err(nib("an index map that is not 32 bits wide"));
            }
            match &a.map_binding {
                Some(map) => Some((map.clone(), a.count_binding.clone())),
                None => return Err(unknown("an index map record")),
            }
        }
    };
    // only the device writes the banded forms of the ordered sums and scatters
    let banded = unit.scalar == Scalar::Banded;
    if banded && unit.target != Target::Device {
        return Err(nib("a team stage in a banded scalar mode on the host"));
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
            step_lines(
                &step,
                &per_lane,
                &reads,
                active.as_ref(),
                unit.target == Target::Device,
                banded,
            )?
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

#[cfg(test)]
mod tests {
    //! The groups of a block sum are read from the partials plane's width, not baked in.
    use super::*;

    /// A step that holds a partials plane named `gp`.
    fn step() -> Step {
        Step {
            op: "block_part".into(),
            n: 1,
            k: 1,
            m: 1,
            max_chunks: 0,
            planes: vec![("gp".into(), "parts".into())],
        }
    }

    #[test]
    fn the_groups_are_the_width_of_the_partials_plane() {
        let planes = |width| vec![("parts".to_string(), width)];
        assert_eq!(groups_of(&step(), &planes(64)), Ok(64));
        assert_eq!(groups_of(&step(), &planes(32)), Ok(32));
        let why = groups_of(&step(), &planes(48)).unwrap_err();
        assert!(why.reason.contains("48 groups"), "{}", why.reason);
        assert!(groups_of(&step(), &[]).is_err());
    }
}
