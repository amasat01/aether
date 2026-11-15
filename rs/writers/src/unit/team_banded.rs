//! The banded forms of the ordered team stages: every partial sum is a `BandPartial` folded with
//! `BandAccum`, in the order of the float stages, and a stored cell is encoded once, at the end.
//! Each function writes the lines a step writes in the float form, over the bindings its caller names.

/// The accumulator type.
const ACC: &str = "aether::banded::BandAccum";
/// The 16-byte partial an accumulator is saved as.
const PART: &str = "aether::banded::BandPartial";
/// The band value type.
const BAND: &str = "aether::banded::Band";
/// Unrolls a fold of banded rows by the accumulator's renormalise period.
const UNROLL: &str = "#pragma unroll aether::banded::BandAccum::period";
/// Lanes one block of a banded plain sum folds, one accumulator each over its whole run of rows.
const BAND_LANES: usize = 128;

/// The constant one an ordered plain sum multiplies by.
pub(super) const ONE: &str = "Real(aether::banded::Band(1.0f, 0.0f, 0.0f))";

/// A `BandPartial` pointer `offset` elements into the scratch plane bound as `binding`.
pub(super) fn part_ptr(binding: &str, offset: &str) -> String {
    format!("(reinterpret_cast<{PART}*>({binding}.data()) + {offset})")
}

/// `dst` (a partial) becomes the combined partial of `dst` and `src`.
fn combine(dst: &str, src: &str) -> String {
    format!("{{ {ACC} hawk_a = {ACC}::fromPartial({dst}); hawk_a.combine({ACC}::fromPartial({src})); {dst} = hawk_a.partial(); }}")
}

/// The halving tree over `hawk_sh`, `width` partials, ending with the barrier.
fn halving_tree(width: usize) -> Vec<String> {
    vec![
        format!("for (std::size_t hawk_w = {}; hawk_w > 0; hawk_w >>= 1) {{", width / 2),
        "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; hawk_i += static_cast<std::size_t>(hawk_tm.size))".to_string(),
        format!("        {}", combine("hawk_sh[hawk_i]", "hawk_sh[hawk_i + hawk_w]")),
        "    hawk_tm.sync();".to_string(),
        "}".to_string(),
    ]
}

/// The last stage of a plain ordered sum: one block folds the `groups` partials of key `hawk_key`
/// and stores the encoded sum in `out`.
pub(super) fn block_groups(gp: &str, out: &str, groups: usize) -> Vec<String> {
    let mut lines = vec![
        format!("__shared__ {PART} hawk_sh[{groups}];"),
        format!("for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < {groups}; hawk_i += static_cast<std::size_t>(hawk_tm.size))"),
        format!("    hawk_sh[hawk_i] = {}[hawk_i];", part_ptr(gp, &format!("hawk_key * {groups}"))),
        "hawk_tm.sync();".to_string(),
    ];
    lines.extend(halving_tree(groups));
    lines.push(format!("if (hawk_tm.rank == 0) {out}.data()[hawk_key] = {ACC}::fromPartial(hawk_sh[0]).toBandedReal();"));
    lines.push("hawk_tm.sync();".to_string());
    lines
}

/// The first stage of a plain ordered sum: block `hawk_lane` folds its accumulators over its run of
/// rows (`product`: the term is the product of two factors) and stores its partial in `gp`.
pub(super) fn block_fold(gp: &str, product: bool, groups: usize) -> Vec<String> {
    let term = if product {
        format!("{BAND}({BAND}(hawk_u[hawk_r]) * {BAND}(hawk_v[hawk_r]))")
    } else {
        format!("{BAND}({BAND}(hawk_u[hawk_r]))")
    };
    let mut lines = vec![
        format!("__shared__ {PART} hawk_sh[{BAND_LANES}];"),
        format!("for (std::size_t hawk_t = static_cast<std::size_t>(hawk_tm.rank); hawk_t < {BAND_LANES}; hawk_t += static_cast<std::size_t>(hawk_tm.size)) {{"),
        format!("    {ACC} hawk_s;"),
        format!("    {UNROLL}"),
        format!("    for (std::size_t hawk_r = hawk_grp * {BAND_LANES} + hawk_t; hawk_r < hawk_rows; hawk_r += {})", BAND_LANES * groups),
        format!("        hawk_s.addTerm({term});"),
        "    hawk_sh[hawk_t] = hawk_s.partial();".to_string(),
        "}".to_string(),
        "hawk_tm.sync();".to_string(),
    ];
    lines.extend(halving_tree(BAND_LANES));
    lines.push(format!(
        "if (hawk_tm.rank == 0) {}[hawk_grp] = hawk_sh[0];",
        part_ptr(gp, &format!("hawk_key * {groups}"))
    ));
    lines.push("hawk_tm.sync();".to_string());
    lines
}

/// The call of an ordered fold: the team of key `hawk_lane` adds the key's contributions to each of
/// the `n` slots, over the bindings `v` (values), `cell` (slot cells) and `out`.
pub(super) fn ordered_fold(n: i64, v: &str, cell: &str, out: &str) -> Vec<String> {
    vec![
        "const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);".to_string(),
        "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / static_cast<std::size_t>(count);".to_string(),
        format!("const Real* const hawk_v = {v}.data();"),
        format!("const Real* const hawk_c = {cell}.data();"),
        format!("Real* const hawk_out = {out}.data();"),
        format!("for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {n}; hawk_e += static_cast<std::size_t>(hawk_tm.size)) {{"),
        format!("    {ACC} hawk_s;"),
        format!("    {UNROLL}"),
        "    for (std::size_t hawk_r = 0; hawk_r < hawk_rows; ++hawk_r)".to_string(),
        format!("        hawk_s.addTerm({BAND}(hawk_v[(hawk_key * hawk_rows + hawk_r) * {n} + hawk_e]));"),
        format!("    hawk_out[static_cast<std::size_t>(hawk_c[hawk_key * {n} + hawk_e])] = hawk_s.toBandedReal();"),
        "}".to_string(),
        "hawk_tm.sync();".to_string(),
    ]
}

/// The partials pointer of an ordered scatter's first stage, and the zero a partial starts at.
pub(super) fn scatter_zero(part: &str) -> (String, String) {
    (
        format!(
            "{PART}* const hawk_p = {};",
            part_ptr(part, "static_cast<std::size_t>(hawk_lane) * hawk_chunks")
        ),
        format!("{PART}{{}}"),
    )
}

/// The partials pointer of an ordered scatter's chunk stage.
pub(super) fn scatter_part(part: &str) -> String {
    format!("{PART}* const hawk_p = {};", part_ptr(part, "0"))
}

/// An ordered scatter's last stage: one thread adds the cell's partials in chunk order and the total
/// to the cell (`chunks`: the lines that count the chunks).
pub(super) fn scatter_cells(chunks: Vec<String>, part: &str, out: &str) -> Vec<String> {
    let mut lines = chunks;
    lines.extend([
        format!("const {PART}* const hawk_p = {};", part_ptr(part, "static_cast<std::size_t>(hawk_lane) * hawk_chunks")),
        format!("Real* const hawk_out = {out}.data();"),
        "if (hawk_tm.rank == 0) {".to_string(),
        format!("    {ACC} hawk_s;"),
        format!("    hawk_s.addTerm({BAND}(hawk_out[hawk_lane]));"),
        format!("    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_s.combine({ACC}::fromPartial(hawk_p[hawk_b]));"),
        "    hawk_out[hawk_lane] = hawk_s.toBandedReal();".to_string(),
        "}".to_string(),
        "hawk_tm.sync();".to_string(),
    ]);
    lines
}

/// One slot added into its chunk's partial, in the serial walk.
pub(super) fn slot_add() -> String {
    format!("            {{ {PART}& hawk_q = hawk_p[static_cast<std::size_t>(hawk_key) * hawk_chunks + static_cast<std::size_t>(hawk_lane)]; {ACC} hawk_a = {ACC}::fromPartial(hawk_q); hawk_a.addTerm({BAND}(hawk_v[hawk_r])); hawk_q = hawk_a.partial(); }}")
}

/// The device walk's pieces that differ in the banded form: the shared bin's declaration, the zero
/// it starts at, the 64-bit shuffle of a cell value, its add and its combine into the chunk's partial.
pub(super) struct BinForms {
    /// The shared bin's declaration.
    pub decl: String,
    /// The value a bin cell starts at.
    pub zero: String,
    /// The lines that bring the shuffled cell value `hawk_x` in.
    pub shuffled: Vec<String>,
    /// The add of `hawk_x` into its bin cell.
    pub add: String,
    /// The line that combines a bin cell into the chunk's partial.
    pub combine: String,
}

/// The banded pieces of the device walk.
pub(super) fn bin_forms() -> BinForms {
    BinForms {
        decl: format!("__shared__ {ACC} hawk_bin[1024];"),
        zero: format!("{ACC}{{}}"),
        shuffled: [
            "unsigned long long hawk_xb;",
            "__builtin_memcpy(&hawk_xb, &hawk_mv, sizeof hawk_xb);",
            "hawk_xb = __shfl_sync(0xffffffffu, hawk_xb, hawk_src);",
            "Real hawk_x;",
            "__builtin_memcpy(&hawk_x, &hawk_xb, sizeof hawk_x);",
        ]
        .map(|l| format!("                {l}"))
        .to_vec(),
        add: format!("                    hawk_bin[hawk_key].addTerm({BAND}(hawk_x));"),
        combine: format!("        {{ {PART}& hawk_q = hawk_p[hawk_j * hawk_chunks + static_cast<std::size_t>(hawk_lane)]; {ACC} hawk_a = {ACC}::fromPartial(hawk_q); hawk_a.combine(hawk_bin[hawk_j]); hawk_q = hawk_a.partial(); }}"),
    }
}
