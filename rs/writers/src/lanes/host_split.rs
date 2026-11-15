//! The host's lane split of a finishing kernel: the tile's samples run their heads, one chunk
//! of lanes runs the trips of the
//! live samples (a finished lane takes the tile's next live sample), then every sample runs its
//! tail and finishes into the
//! tile's guard lanes. Also the tile rule (the samples one tile advances together) and the
//! entry that runs a kernel to the end.
use super::host_interchange::{
    independent_loop, lane_elem, lane_type, width, Declines, LANE_PAD, TILE_LOOP_HEAD,
};
use super::host_text::{
    COPY_IN_NO_UNROLL, HOST_MASK_HELPERS, INDEPENDENT_ITERATIONS, INTERCHANGE_HELPERS, RUN_ENTRY,
    TRIP_ITERATIONS,
};
use crate::body::Ty;
use crate::contract::types::Unit;
use crate::ir::door::RxCtx;
use crate::lanes::cut::Carry;
use crate::lanes::split::LaneSplit;
use crate::refuse::nib;

/// The share of a cache level a tile's lane arrays may fill.
const CACHE_SHARE: f64 = 0.5;
/// The samples of one lane block.
const GUARD_BLOCK: i64 = 64;
/// A step of at most this many operations is short.
const SHORT_STEP_OPS: usize = 256;
/// The caches a build that states none is assumed to have: 32 KiB of L1d and 1 MiB of L2.
const FALLBACK_CACHES: (i64, i64, &str) = (32 * 1024, 1024 * 1024, "fallback");

/// What the tile rule decided and every input it decided from.
pub struct TileRule {
    /// The number of samples in a tile.
    pub tile: i64,
    /// The bytes one lane of the split's rows takes.
    lane_bytes: i64,
    /// The operations of one trip of the loop.
    op_count: usize,
    /// Whether the kernel counts finished samples.
    finishes: bool,
    /// The loop's static trip bound.
    stop: i64,
    /// The first-level data cache size the rule assumed.
    l1d: i64,
    /// The second-level cache size the rule assumed.
    l2: i64,
    /// Where the cache sizes came from (the build or the fallback).
    source: String,
}

/// The interchanged loop's tile size, from the lane arrays' bytes per sample and the machine's
/// cache sizes.
pub fn tile_rule(
    lane_bytes: i64,
    op_count: usize,
    stop: i64,
    caches: Option<(i64, i64, String)>,
    tile_bounds: (i64, i64),
) -> TileRule {
    let (l1d, l2, source) = caches.unwrap_or((
        FALLBACK_CACHES.0,
        FALLBACK_CACHES.1,
        FALLBACK_CACHES.2.to_string(),
    ));
    let lane_bytes = lane_bytes.max(1);
    let l1 = (l1d as f64 * CACHE_SHARE) as i64;
    let budget = if lane_bytes * GUARD_BLOCK <= l1 {
        l1
    } else {
        (l2 as f64 * CACHE_SHARE) as i64
    };
    let tile = budget / lane_bytes;
    let tile = (tile - tile % GUARD_BLOCK)
        .max(tile_bounds.0)
        .min(tile_bounds.1);
    TileRule {
        tile,
        lane_bytes,
        op_count,
        finishes: true,
        stop,
        l1d,
        l2,
        source,
    }
}

impl TileRule {
    /// The rule's record as one C++ comment line.
    pub fn comment(&self) -> String {
        let level = if self.lane_bytes * GUARD_BLOCK <= (self.l1d as f64 * CACHE_SHARE) as i64 {
            "L1d"
        } else {
            "L2"
        };
        format!("// hawk host tile rule: tile={} k={} level={level} (lane_bytes={} op_count={} finishes={} stop={} l1d={} l2={} caches={})",
                self.tile, self.stop, self.lane_bytes, self.op_count, self.finishes as i32, self.stop, self.l1d, self.l2, self.source)
    }
}

/// The bytes one element of an IR dtype stores as: a float is the compiled `Real`.
pub fn bytes_of(dtype: &str, real_bytes: i64) -> i64 {
    match dtype {
        "f64" | "f32" => real_bytes,
        "i32" | "i64" => 8,
        "bool" => 1,
        _ => 4,
    }
}

/// The bytes of one sample's share of the tile's lane arrays.
pub fn lane_bytes(split: &LaneSplit, guards: usize, real_bytes: i64) -> i64 {
    let mut total = real_bytes * (1 + guards as i64) + if split.count.is_some() { 8 } else { 0 };
    for ty in split
        .carries
        .iter()
        .map(|c| &c.ty)
        .chain(split.live_ins.iter().map(|l| &l.1))
    {
        total += width(ty) as i64 * bytes_of(&ty.dtype, real_bytes);
    }
    total
}

/// The vector registers of lanes one chunk advances together.
fn chunk_vectors(op_count: usize) -> usize {
    if op_count <= SHORT_STEP_OPS {
        2
    } else {
        1
    }
}

/// The tile's live lanes first, in tile order, so the trips run only the live ones; sorted by
/// the exit-bound carry when known.
fn compact_live(split: &LaneSplit) -> Vec<String> {
    let mut lines: Vec<String> = [
        "    alignas(64) aether::offset_t hawk_perm[hawk_tile];",
        "    aether::offset_t hawk_nlive = 0;",
        "    for (aether::offset_t hawk_t = 0; hawk_t < hawk_tn; ++hawk_t)",
        "        if (hawk_on[hawk_t] != 0.0) hawk_perm[hawk_nlive++] = hawk_t;",
        "    for (aether::offset_t hawk_t = 0, hawk_o = hawk_nlive; hawk_t < hawk_tn; ++hawk_t)",
        "        if (hawk_on[hawk_t] == 0.0) hawk_perm[hawk_o++] = hawk_t;",
    ]
    .iter()
    .map(|l| l.to_string())
    .collect();
    if let Some((carry, descending)) = split.exit_order {
        lines.extend([
            "#if defined(__aarch64__)".to_string(),
            "    std::sort(hawk_perm, hawk_perm + hawk_nlive,".into(),
            "              [&](aether::offset_t a, aether::offset_t b) {".into(),
            format!("                  const auto x = hawk_lc{carry}[a], y = hawk_lc{carry}[b];"),
            "                  if (x != x) return false;".into(),
            "                  if (y != y) return true;".into(),
            format!(
                "                  return x {} y; }});",
                if descending { '<' } else { '>' }
            ),
            "#endif".into(),
        ]);
    }
    lines
}

/// The lines declaring `hawk_t`, the tile sample the lane `hawk_l` holds.
fn chunk_sample(clamped: bool) -> Vec<String> {
    if clamped {
        vec![
            "        const aether::offset_t hawk_t =".into(),
            "            hawk_rt[hawk_l] < hawk_tn ? hawk_rt[hawk_l] : hawk_tn - 1;".into(),
        ]
    } else {
        vec!["        const aether::offset_t hawk_t = hawk_rt[hawk_l];".into()]
    }
}

/// One lane row: its array, the name it holds, its type.
type Row<'a> = (String, &'a str, &'a Ty);

/// The head of the loop over the samples of a tile.
const SAMPLE_LOOP: &str =
    "    for (aether::offset_t hawk_j = hawk_tlo; hawk_j < hawk_thi; ++hawk_j) {";

/// The lines that open a sample's body: the index of the sample in its tile and as a sample index.
fn sample_lines() -> Vec<String> {
    [
        "        const aether::offset_t hawk_t = hawk_j - hawk_tlo;",
        "        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);",
        "        const aether::SampleIndex i =",
        "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));",
        "        (void)hawk_t;",
        "        (void)hawk_i;",
        "        (void)nSamples;",
    ]
    .iter()
    .map(|l| l.to_string())
    .collect()
}

/// The lane rows of a split: its carries, then its live values.
fn split_rows(split: &LaneSplit) -> Vec<Row<'_>> {
    let mut rows: Vec<Row> = vec![];
    for (j, Carry { name, ty, .. }) in split.carries.iter().enumerate() {
        rows.push((format!("hawk_lc{j}"), name, ty))
    }
    for (m, (name, ty)) in split.live_ins.iter().enumerate() {
        rows.push((format!("hawk_li{m}"), name, ty))
    }
    rows
}

/// The declarations ahead of the tile loop and the tile's masks, finish and tally bindings.
fn tile_header(
    split: &LaneSplit,
    guards: &[String],
    finish: &(String, String, String),
    tile: i64,
    rows: &[Row],
    on_k: usize,
) -> Result<Vec<String>, Declines> {
    let (mask, twin, counter) = finish;
    let mut lines: Vec<String> = vec![
        format!("    constexpr aether::offset_t hawk_tile = {tile};"),
        "    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);".into(),
        "    const aether::offset_t hawk_hi =".into(),
        "        static_cast<aether::offset_t>(base + count);".into(),
    ];
    for k in 0..guards.len() {
        lines.push(format!(
            "    alignas(64) HAWK_LANE_T hawk_tg{k}[hawk_tile];"
        ))
    }
    lines.push("    alignas(64) HAWK_LANE_T hawk_on[hawk_tile] = {};".into());
    if split.count.is_some() {
        lines.push("    alignas(64) Int hawk_cnt[hawk_tile] = {};".into())
    }
    for (arr, _, ty) in rows {
        lines.push(format!(
            "    alignas(64) {} {arr}[{} * hawk_tile] = {{}};",
            lane_elem(ty)?,
            width(ty)
        ))
    }
    lines.extend(TILE_LOOP_HEAD.iter().map(|l| l.to_string()));
    for (k, g) in guards.iter().enumerate() {
        lines.push(format!("    const hawk_abi::HostLaneMask hawk_tlane{k} =\n        {g}.lanes(hawk_tg{k}, hawk_tlo, hawk_thi);"));
    }
    lines.extend([
        format!("    const hawk_abi::HostMask hawk_finish_mask = {mask};"),
        format!("    const auto& hawk_finish_count = {counter};"),
        "    HAWK_LANE_T hawk_finished = 0.0;".into(),
    ]);
    for (k, g) in guards.iter().enumerate() {
        lines.push(format!(
            "    const hawk_abi::HostLaneMask {g} = hawk_tlane{k};"
        ))
    }
    lines.extend([
        format!("    const hawk_abi::HostLaneFinish {twin}{{hawk_tg{on_k}, hawk_tlo}};"),
        format!("    const hawk_abi::HostLaneTally {counter}{{&hawk_finished}};"),
        format!("    (void){twin};"),
        format!("    (void){counter};"),
    ]);
    Ok(lines)
}

/// The statements that fill a sample's lane at the head: the carries' initial values, the live values and the trip count.
fn lane_puts(split: &LaneSplit, rows: &[Row]) -> Result<Vec<String>, Declines> {
    let ncar = split.carries.len();
    let mut puts: Vec<String> = vec![];
    for ((arr, _, ty), Carry { init, .. }) in rows.iter().zip(&split.carries) {
        let lt = lane_type(ty)?;
        let value = if ty.shape.is_empty() {
            format!("static_cast<{lt}>({init})")
        } else {
            format!("{lt}({init})")
        };
        puts.push(format!(
            "hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {value});"
        ));
    }
    for (arr, name, ty) in rows.iter().skip(ncar) {
        let lt = lane_type(ty)?;
        if !ty.shape.is_empty() {
            puts.push(format!(
                "hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {lt}({name}));"
            ));
        } else {
            let same = format!("std::is_same_v<std::remove_cv_t<decltype({name})>, {lt}>");
            puts.push(format!(
                "static_assert({same}, \"hawk: a lane row holds its value's own type\");"
            ));
            puts.push(format!(
                "hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {name});"
            ));
        }
    }
    match &split.count {
        Some(c) => {
            puts.push(format!("hawk_cnt[hawk_t] = {c};"));
            puts.push(format!("hawk_go = {c} > 0 ? 1.0 : 0.0;"))
        }
        None => puts.push("hawk_go = 1.0;".into()),
    }
    Ok(puts)
}

/// (A) the head of every sample of the tile: lanes filled, live flag set.
fn sample_heads(split: &LaneSplit, puts: &[String]) -> Vec<String> {
    let pad = LANE_PAD;
    let mut lines: Vec<String> = vec![];
    independent_loop(&mut lines, SAMPLE_LOOP);
    lines.extend(sample_lines());
    lines.push("        HAWK_LANE_T hawk_go = 0.0;".into());
    if let Some(g) = &split.guard {
        lines.push(format!("        if (!{g}) {{"))
    }
    if !split.pre.is_empty() {
        lines.push(split.pre.text())
    }
    for p in puts {
        lines.push(format!("{pad}{p}"))
    }
    if split.guard.is_some() {
        lines.push("        }".into())
    }
    lines.extend([
        "        hawk_on[hawk_t] = hawk_go;".to_string(),
        "    }".into(),
    ]);
    lines
}

/// (C) the tail of every sample: commits and the finish, from the lanes.
fn sample_tails(split: &LaneSplit, rows: &[Row], on_k: usize) -> Result<Vec<String>, Declines> {
    let (pad, ncar) = (LANE_PAD, split.carries.len());
    let mut lines: Vec<String> = vec![];
    independent_loop(&mut lines, SAMPLE_LOOP);
    lines.extend(sample_lines());
    if let Some(g) = &split.guard {
        lines.push(format!("        if (!{g}) {{"))
    }
    for (arr, name, ty) in rows.iter().take(ncar) {
        let lt = lane_type(ty)?;
        lines.push(format!(
            "{pad}const {lt} {name} = hawk_abi::lane_get<{lt}, hawk_tile>({arr}, hawk_t);"
        ));
    }
    if !split.post.is_empty() {
        lines.push(split.post.text())
    }
    if split.guard.is_some() {
        lines.push("        }".into())
    }
    lines.extend([
        "    }".to_string(),
        format!("    hawk_abi::finish_lanes(hawk_finish_mask, hawk_tg{on_k}, hawk_tlo, hawk_thi, hawk_finished, hawk_finish_count);"),
        "    }".into(),
    ]);
    Ok(lines)
}

/// The interchanged loop over `[base, base + count)` of a finishing kernel, per tile of `tile`
/// samples.
fn interchange_text(
    split: &LaneSplit,
    guards: &[String],
    finish: &(String, String, String),
    tile: i64,
) -> Result<String, Declines> {
    let on_k = guards.iter().position(|g| g == &finish.0).ok_or(Declines)?;
    let rows = split_rows(split);
    let mut lines = tile_header(split, guards, finish, tile, &rows, on_k)?;
    let puts = lane_puts(split, &rows)?;
    lines.extend(sample_heads(split, &puts));
    lines.extend(trips(split, &rows, &compact_live(split), LANE_PAD)?);
    lines.extend(sample_tails(split, &rows, on_k)?);
    Ok(lines.join("\n"))
}

/// One lane row as a chunk holds it: the chunk's own array, the tile's array it is copied from and to, its type and the name
/// of the value.
struct ChunkRow<'a> {
    /// The chunk's own array.
    mine: String,
    /// The tile's array.
    arr: String,
    /// The value's type.
    ty: &'a Ty,
    /// The value's name.
    name: &'a str,
}

/// The lines a chunk repeats: the loads of a lane's values, the trip count, the condition to go on and the copies between the
/// tile's rows and the chunk's.
struct ChunkLines {
    /// The loads of a lane's values at a trip.
    trip_loads: Vec<String>,
    /// The trip count of a lane.
    lane_count: String,
    /// The condition under which a lane goes on.
    more: String,
    /// The copy of a sample's values into a chunk lane.
    copy_in: Vec<String>,
    /// The copy of a finished sample's carries back to the tile.
    copy_out: Vec<String>,
}

/// The chunk's own array for each lane row: `hawk_rc<i>` for a carry, `hawk_ri<i>` for a live value.
fn chunk_rows<'a>(split: &LaneSplit, rows: &[Row<'a>]) -> Vec<ChunkRow<'a>> {
    let ncar = split.carries.len();
    let mut mine_rows: Vec<ChunkRow> = vec![];
    for (i, (arr, name, ty)) in rows.iter().enumerate() {
        let mine = if i < ncar {
            format!("hawk_rc{i}")
        } else {
            format!("hawk_ri{}", i - ncar)
        };
        mine_rows.push(ChunkRow {
            mine,
            arr: arr.clone(),
            ty,
            name,
        });
    }
    mine_rows
}

/// The chunk's rows and the lines the phases of a chunk share.
fn chunk_lines<'a>(
    split: &LaneSplit,
    rows: &[Row<'a>],
) -> Result<(Vec<ChunkRow<'a>>, ChunkLines), Declines> {
    let ncar = split.carries.len();
    let mine_rows = chunk_rows(split, rows);
    let mut trip_loads = vec![];
    for ChunkRow { mine, ty, name, .. } in &mine_rows {
        let lt = lane_type(ty)?;
        trip_loads.push(format!(
            "        const {lt} {name} = hawk_abi::lane_get<{lt}, hawk_w>({mine}, hawk_l);"
        ));
    }
    let lane_count = match &split.count {
        Some(_) => "hawk_cnt[hawk_t]".to_string(),
        None => format!("static_cast<Int>({})", split.stop),
    };
    let more = format!("!{} && hawk_trip + 1 < hawk_rcnt[hawk_l]", split.exit);
    let mut copy_in = vec![];
    for ChunkRow { mine, arr, ty, .. } in &mine_rows {
        copy_in.push(format!(
            "        for (aether::offset_t hawk_c = 0; hawk_c < {}; ++hawk_c)",
            width(ty)
        ));
        copy_in.push(format!(
            "            {mine}[hawk_c * hawk_w + hawk_l] = {arr}[hawk_c * hawk_tile + hawk_t];"
        ));
    }
    let mut copy_out = vec![];
    for ChunkRow { mine, arr, ty, .. } in mine_rows.iter().take(ncar) {
        copy_out.push(format!(
            "        for (aether::offset_t hawk_c = 0; hawk_c < {}; ++hawk_c)",
            width(ty)
        ));
        copy_out.push(format!(
            "            {arr}[hawk_c * hawk_tile + hawk_t] = {mine}[hawk_c * hawk_w + hawk_l];"
        ));
    }
    Ok((
        mine_rows,
        ChunkLines {
            trip_loads,
            lane_count,
            more,
            copy_in,
            copy_out,
        },
    ))
}

/// The extra condition a lane needs to be live at its start when the trip bound is static (none with a runtime count).
fn static_bound_check(split: &LaneSplit) -> String {
    if split.count.is_some() {
        String::new()
    } else {
        format!(" && 0 < {}", split.stop)
    }
}

/// The chunk's declarations and the copy of the first live samples into its lanes.
fn chunk_open(
    split: &LaneSplit,
    mine_rows: &[ChunkRow],
    compact: &[String],
    cl: &ChunkLines,
) -> Result<Vec<String>, Declines> {
    let ChunkLines {
        lane_count,
        copy_in,
        ..
    } = cl;
    let mut lines: Vec<String> = vec![
        format!("    constexpr aether::offset_t hawk_w = hawk_abi::kHostChunk * hawk_abi::kHostChunkVectors * {};", chunk_vectors(split.op_count)),
        "    static_assert(hawk_tile % hawk_w == 0,".into(),
        "                  \"hawk: a tile holds whole chunks\");".into(),
    ];
    lines.extend(compact.iter().cloned());
    lines.extend([
        "    for (aether::offset_t hawk_cb = 0; hawk_cb < 1; ++hawk_cb) {".to_string(),
        "    if (hawk_nlive == 0) continue;".into(),
    ]);
    for ChunkRow { mine, ty, .. } in mine_rows {
        lines.push(format!(
            "    alignas(64) {} {mine}[{} * hawk_w];",
            lane_elem(ty)?,
            width(ty)
        ))
    }
    lines.extend([
        "    alignas(64) HAWK_LANE_T hawk_ron[hawk_w];".to_string(),
        "    alignas(64) Int hawk_rcnt[hawk_w];".into(),
        "    alignas(64) aether::offset_t hawk_rt[hawk_w];".into(),
        "    alignas(64) Int hawk_rk0[hawk_w];".into(),
        INDEPENDENT_ITERATIONS.to_string(),
        COPY_IN_NO_UNROLL.to_string(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {".into(),
        "        const aether::offset_t hawk_t = hawk_l < hawk_tn ? hawk_perm[hawk_l] : hawk_l;"
            .into(),
    ]);
    lines.extend(copy_in.iter().cloned());
    lines.extend([
        "        hawk_rt[hawk_l] = hawk_t;".to_string(),
        "        hawk_rk0[hawk_l] = 0;".into(),
        format!("        hawk_rcnt[hawk_l] = {lane_count};"),
        format!(
            "        const bool hawk_in = hawk_t < hawk_tn && hawk_on[hawk_t] != 0.0{};",
            static_bound_check(split)
        ),
        "        hawk_ron[hawk_l] = hawk_in ? 1.0 : 0.0;".into(),
        "    }".into(),
        "    aether::offset_t hawk_next = hawk_nlive < hawk_w ? hawk_nlive : hawk_w;".into(),
        "    for (Int hawk_trip = 0;; ++hawk_trip) {".into(),
        "    std::int64_t hawk_live = 0;".into(),
        TRIP_ITERATIONS.to_string(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {".into(),
    ]);
    Ok(lines)
}

/// One trip of every lane of the chunk: the step for the live lanes and the carries put back.
fn chunk_trip(
    split: &LaneSplit,
    mine_rows: &[ChunkRow],
    cl: &ChunkLines,
    pad: &str,
    lines: &mut Vec<String>,
) {
    let ChunkLines {
        trip_loads, more, ..
    } = cl;
    lines.extend(chunk_sample(true));
    lines.extend([
        "        const aether::SampleIndex i =".to_string(),
        "            aether::SampleIndex::make(".into(),
        "                static_cast<std::size_t>(hawk_tlo + hawk_t));".into(),
        "        (void)i;".into(),
        "        const bool hawk_lane_on = hawk_ron[hawk_l] != 0.0;".into(),
        format!(
            "        const Int {} = hawk_trip - hawk_rk0[hawk_l];",
            split.index
        ),
        format!("        (void){};", split.index),
    ]);
    if split.speculative {
        lines.extend(trip_loads.iter().cloned());
        lines.push(split.step.text());
        for (ChunkRow { mine, .. }, Carry { next: nxt, .. }) in mine_rows.iter().zip(&split.carries)
        {
            lines.push(format!(
                "        hawk_abi::lane_keep<hawk_w>({mine}, hawk_l, hawk_lane_on, {nxt});"
            ));
        }
        lines.push(format!(
            "        const bool hawk_more = hawk_lane_on && {more};"
        ));
    } else {
        lines.extend([
            "        bool hawk_more = false;".to_string(),
            "        if (hawk_lane_on) {".into(),
        ]);
        lines.extend(trip_loads.iter().map(|l| format!("    {l}")));
        lines.push(split.step.text());
        for (ChunkRow { mine, .. }, Carry { next: nxt, .. }) in mine_rows.iter().zip(&split.carries)
        {
            lines.push(format!(
                "{pad}hawk_abi::lane_put<hawk_w>({mine}, hawk_l, {nxt});"
            ));
        }
        lines.extend([format!("{pad}hawk_more = {more};"), "        }".into()]);
    }
}

/// After a trip: the live flags, the refill of finished lanes from the tile's next live samples and the write-back.
fn chunk_refill(cl: &ChunkLines, lines: &mut Vec<String>) {
    let ChunkLines {
        lane_count,
        copy_in,
        copy_out,
        ..
    } = cl;
    lines.extend([
        "        hawk_ron[hawk_l] = hawk_more ? 1.0 : 0.0;".to_string(),
        "    }".into(),
        "    std::int64_t hawk_idle = 0;".into(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {".into(),
        "        hawk_live |= static_cast<std::int64_t>(hawk_ron[hawk_l] != 0.0);".into(),
        "        hawk_idle |= static_cast<std::int64_t>(hawk_ron[hawk_l] == 0.0);".into(),
        "    }".into(),
        "    if (hawk_idle != 0 && hawk_next < hawk_nlive) {".into(),
        COPY_IN_NO_UNROLL.to_string(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w && hawk_next < hawk_nlive; ++hawk_l) {".into(),
        "        if (hawk_ron[hawk_l] != 0.0) continue;".into(),
        "        {".into(),
    ]);
    for l in chunk_sample(false).iter().chain(copy_out.iter()) {
        lines.push(format!("    {l}"))
    }
    lines.extend([
        "        }".to_string(),
        "        const aether::offset_t hawk_t = hawk_perm[hawk_next++];".into(),
    ]);
    lines.extend(copy_in.iter().cloned());
    lines.extend([
        "        hawk_rt[hawk_l] = hawk_t;".to_string(),
        "        hawk_rk0[hawk_l] = hawk_trip + 1;".into(),
        format!("        hawk_rcnt[hawk_l] = hawk_trip + 1 + {lane_count};"),
        "        hawk_ron[hawk_l] = 1.0;".into(),
        "    }".into(),
        "    hawk_live = 1;".into(),
        "    }".into(),
        "    if (hawk_live == 0) break;".into(),
        "    }".into(),
        INDEPENDENT_ITERATIONS.to_string(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {".into(),
    ]);
    lines.extend(chunk_sample(false));
    lines.extend(copy_out.iter().cloned());
}

/// (B) the trips, on one chunk of lanes: it takes the tile's first live samples, runs every
/// trip with the lanes innermost,
/// and between trips a finished lane writes its sample back and takes the next live one.
fn trips(
    split: &LaneSplit,
    rows: &[Row],
    compact: &[String],
    pad: &str,
) -> Result<Vec<String>, Declines> {
    let (mine_rows, cl) = chunk_lines(split, rows)?;
    let mut lines = chunk_open(split, &mine_rows, compact, &cl)?;
    chunk_trip(split, &mine_rows, &cl, pad, &mut lines);
    chunk_refill(&cl, &mut lines);
    lines.extend(["    }".to_string(), "    }".into()]);
    Ok(lines)
}

/// The tile export of the unit, and the entry that runs the kernel to completion when it has
/// the steps word.
pub fn export(name: &str, rule: &TileRule, run: Option<String>) -> String {
    let base = format!(
        "extern \"C\" const std::int64_t {name}_host_tile = HAWK_HOST_INTERCHANGE ? {} : 0;",
        rule.tile
    );
    match run {
        Some(r) => format!("{base}\n{r}"),
        None => base,
    }
}

/// A finishing kernel's interchanged unit pieces: the loop text, the prelude and the exports it
/// adds.
pub struct SplitUnit {
    /// The needs recorded while the loop text's body was written.
    pub needs: crate::body::frame::needs::Needs,
    /// The text of the interchanged loop.
    pub text: String,
    /// The prelude the loop text needs.
    pub prelude: String,
    /// The tile export and the entry that runs the kernel to completion.
    pub export: String,
}

/// What the lane split of a finishing kernel is asked for: the unit's name, its mask guards, the finish bindings (mask, twin,
/// counter), the bytes of a real, the plane slots and the finish plane's name.
pub struct SplitRequest<'a> {
    /// The unit's name.
    pub name: &'a str,
    /// The mask guards of the kernel.
    pub guards: &'a [String],
    /// The mask, its lane twin and the counter the finish uses.
    pub finish: &'a (String, String, String),
    /// The bytes of one real.
    pub real_bytes: i64,
    /// The role and name of each plane slot.
    pub slots: &'a [(String, String)],
    /// The name of the finish plane.
    pub finish_name: &'a str,
}

/// The lane split of a finishing kernel as a unit's pieces, or `None` when the body is not of
/// that shape (the unit then
/// takes the interchanged loops, if it has any).
pub fn split_unit(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    request: &SplitRequest,
) -> Result<Option<SplitUnit>, String> {
    let SplitRequest {
        name,
        guards,
        finish,
        real_bytes,
        slots,
        finish_name,
    } = *request;
    let Some((split, needs)) = crate::lanes::lane_split(ctx, walk, unit, "        ")? else {
        return Ok(None);
    };
    let caches = unit
        .machine
        .caches
        .as_ref()
        .map(|c| (c.l1d_bytes, c.l2_bytes, c.source.clone()));
    let bounds = unit
        .build
        .lanes
        .tile
        .ok_or_else(|| nib("a spec without the tile bounds"))?;
    let rule = tile_rule(
        lane_bytes(&split, guards.len(), real_bytes),
        split.op_count,
        split.stop,
        caches,
        bounds,
    );
    let text = interchange_text(&split, guards, finish, rule.tile)
        .map_err(|_| nib("a lane row of an unspelled type"))?;
    let masks = HOST_MASK_HELPERS.replacen(
        "#if HAWK_HOST_LANE_GUARDS\nstruct HostLaneFinishRef",
        "#if HAWK_HOST_LANE_GUARDS || HAWK_HOST_INTERCHANGE\nstruct HostLaneFinishRef",
        1,
    );
    let prelude = format!("#ifndef HAWK_HOST_INTERCHANGE\n#define HAWK_HOST_INTERCHANGE 1\n#endif\n{}\n{masks}\n{INTERCHANGE_HELPERS}", rule.comment());
    Ok(Some(SplitUnit {
        needs,
        text,
        prelude,
        export: export(name, &rule, run_entry(name, slots, finish_name, split.stop)),
    }))
}

/// The entry that runs a kernel to completion in rounds of the steps word, when the kernel
/// carries that word and a
/// finished mask; `None` otherwise.
fn run_entry(
    name: &str,
    slots: &[(String, String)],
    finish_name: &str,
    stop: i64,
) -> Option<String> {
    let at = |role: &str, slot: &str| slots.iter().position(|s| s.0 == role && s.1 == slot);
    let (word, done) = (at("lookup", "fused_steps")?, at("terminated", finish_name)?);
    Some(
        RUN_ENTRY
            .replace("@NAME@", name)
            .replace("@STOP@", &stop.to_string())
            .replace("@LEN@", &slots.len().to_string())
            .replace("@WORD@", &word.to_string())
            .replace("@DONE@", &done.to_string()),
    )
}
