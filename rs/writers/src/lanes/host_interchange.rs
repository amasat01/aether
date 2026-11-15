//! The host's loop interchange: a body cut at its interchanged loops (the core's lane segments)
//! becomes one loop over tiles of
//! samples, each cut loop running its trips outside and the tile's lanes inside. Every value a
//! later text reads lives in a lane
//! row `hawk_r_<name>[c * tile + lane]`.
use super::host_text::{
    INDEPENDENT_ITERATIONS, INTERCHANGE_HELPERS, LANE_ROW_HELPER, LOOP_HELPERS, LOOP_PRELUDE,
};
use crate::body::frame::namer::NameKind;
use crate::body::Ty;
use crate::lanes::cut::{Carry, CutLoop, LaneSegments};
use crate::refuse::{nib, Refusal};
use crate::spell::carry_type;
use std::collections::{BTreeMap, BTreeSet};

/// The head of the loop over tiles of samples.
pub(super) const TILE_LOOP_HEAD: [&str; 5] = [
    "    for (aether::offset_t hawk_tlo = hawk_lo; hawk_tlo < hawk_hi;",
    "         hawk_tlo += hawk_tile) {",
    "    const aether::offset_t hawk_thi =",
    "        hawk_hi - hawk_tlo < hawk_tile ? hawk_hi : hawk_tlo + hawk_tile;",
    "    const aether::offset_t hawk_tn = hawk_thi - hawk_tlo;",
];

/// Where a lane row is written: by a segment, or by a loop (its carried value).
#[derive(Clone, Copy, PartialEq)]
enum Origin {
    /// Written by the segment with this number.
    Segment(usize),
    /// Written by the loop with this number (its carried value).
    Loop(usize),
}

/// The prelude text an interchanged unit carries: the macro, the lane helpers and the tile size.
pub fn prelude(masks: &str, rows: bool) -> String {
    let helper = if rows { LANE_ROW_HELPER } else { "" };
    format!("{LOOP_PRELUDE}{helper}{masks}\n{INTERCHANGE_HELPERS}\n{LOOP_HELPERS}")
}

/// How many scalars a value of the type holds.
pub(super) fn width(ty: &Ty) -> usize {
    if ty.shape.is_empty() {
        1
    } else {
        ty.shape.iter().product()
    }
}
/// Whether the type is a scalar real.
fn is_real(ty: &Ty) -> bool {
    ty.shape.is_empty() && matches!(ty.dtype.as_str(), "f64" | "f32")
}

/// The element a lane row of a value is declared with.
pub(super) fn lane_elem(ty: &Ty) -> Result<String, Declines> {
    if is_real(ty) {
        Ok("Real".into())
    } else {
        carry_type(&ty.dtype, &[]).map_err(|_| Declines)
    }
}

/// The value a lane row holds back as: the carry's own spelling.
pub(super) fn lane_type(ty: &Ty) -> Result<String, Declines> {
    if is_real(ty) {
        Ok("Real".into())
    } else {
        carry_type(&ty.dtype, &ty.shape).map_err(|_| Declines)
    }
}

/// The statements that store a name (or the text of a carry's initial value) into lane `hawk_t`
/// of a row.
fn row_put(row: &str, name: &str, ty: &Ty, value: Option<&str>) -> Result<Vec<String>, Declines> {
    let lane = lane_type(ty)?;
    if let Some(v) = value {
        let cast = if ty.shape.is_empty() {
            format!("static_cast<{lane}>({v})")
        } else {
            format!("{lane}({v})")
        };
        return Ok(vec![format!(
            "hawk_abi::lane_put<hawk_tile>({row}, hawk_t, {cast});"
        )]);
    }
    if !ty.shape.is_empty() {
        return Ok(vec![format!(
            "hawk_abi::lane_put<hawk_tile>({row}, hawk_t, {lane}({name}));"
        )]);
    }
    let same = format!("std::is_same_v<std::remove_cv_t<decltype({name})>, {lane}>");
    Ok(vec![
        format!("static_assert({same}, \"hawk: a lane row holds its value's own type\");"),
        format!("hawk_abi::lane_put<hawk_tile>({row}, hawk_t, {name});"),
    ])
}

/// The rows of the interchanged text: each name a later text reads, with its type and where it
/// is written.
struct Rows {
    /// Each row's value type and where it is written, by name.
    map: BTreeMap<String, (Ty, Origin)>,
}

impl Rows {
    /// The lines that load the named rows into locals for the lane `lane`.
    fn load_lines(&self, names: &[&String], lane: &str) -> Result<Vec<String>, Declines> {
        let mut sorted: Vec<&&String> = names.iter().collect();
        sorted.sort();
        sorted
            .iter()
            .map(|x| {
                let lt = lane_type(&self.map[**x].0)?;
                Ok(format!(
                    "const {lt} {x} = hawk_abi::lane_get<{lt}, hawk_tile>(hawk_r_{x}, {lane});"
                ))
            })
            .collect()
    }
}

/// Why a lane text is not written: a lane row has no spelling, or a mask names no lane. The caller then refuses or falls back.
#[derive(Debug)]
pub(super) struct Declines;

/// The indent of the statements inside a lane's guarded block.
pub(super) const LANE_PAD: &str = "            ";

/// Opens a loop whose iterations are independent: the pragma line, then the loop head.
pub(super) fn independent_loop(lines: &mut Vec<String>, head: &str) {
    lines.extend([INDEPENDENT_ITERATIONS.to_string(), head.to_string()]);
}

/// The refusal for a lane row whose element has no spelling.
fn unspelled(_: Declines) -> Refusal {
    nib("a lane row of an unspelled type")
}

/// The text of the interchanged loops over `[base, base + count)`, per tile of samples. Refused when a value crossing a cut
/// has no recorded type or a lane row has no spelling: the old rule for keeping the per-sample loop there is not written.
pub fn loop_interchange_text(
    segs: &LaneSegments,
    needs: &mut crate::body::frame::needs::Needs,
) -> Result<String, Refusal> {
    // the lane proxies read their rows through the tile-lane helper
    if !segs.proxies.is_empty() {
        needs.record(crate::body::frame::needs::Need::TileLane)
    }
    build(segs)
}

/// What the cut body says about its names: those each segment declares, the plane views, and the names each segment and each
/// cut step read.
struct CutNames {
    /// The names each segment declares.
    declared: Vec<BTreeSet<String>>,
    /// The plane views, by name, with the line that binds each.
    views: BTreeMap<String, String>,
    /// The names each segment reads, with those its loop's carries and bounds read.
    seg_words: Vec<BTreeSet<String>>,
    /// The names each cut loop's step reads.
    step_words: Vec<BTreeSet<String>>,
}

/// Reads the names of the segments, the views and the steps from the buffers' records.
fn cut_names(segs: &LaneSegments) -> CutNames {
    let declared: Vec<BTreeSet<String>> = segs.segments.iter().map(|b| b.declared_all()).collect();
    let mut views: BTreeMap<String, String> = BTreeMap::new();
    for segment in &segs.segments {
        for decl in &segment.decls {
            if decl.name.kind == NameKind::LeafView {
                views.insert(
                    decl.name.text.clone(),
                    segment.lines[decl.line].trim().to_string(),
                );
            }
        }
    }
    let mut known: BTreeSet<String> = declared.iter().flatten().cloned().collect();
    known.extend(views.keys().cloned());
    for lp in &segs.loops {
        known.extend(lp.carries.iter().map(|c| c.name.clone()));
    }
    let mut seg_words: Vec<BTreeSet<String>> =
        segs.segments.iter().map(|b| b.reads_of(&known)).collect();
    let step_words: Vec<BTreeSet<String>> =
        segs.loops.iter().map(|l| l.step.reads_of(&known)).collect();
    for (k, lp) in segs.loops.iter().enumerate() {
        let recorded: BTreeSet<String> = lp
            .carries
            .iter()
            .flat_map(|c| c.init_reads.iter())
            .chain(lp.bounds_reads.iter())
            .filter(|name| known.contains(*name))
            .cloned()
            .collect();
        seg_words[k].extend(recorded);
    }
    CutNames {
        declared,
        views,
        seg_words,
        step_words,
    }
}

/// The values that cross a cut (a later segment or step reads what an earlier segment declares) and the carries, each in the
/// lane row it lives in; refused when a crossing value has no recorded type.
fn crossing_rows(segs: &LaneSegments, names: &CutNames) -> Result<Rows, Refusal> {
    let n = segs.loops.len();
    let type_of = |name: &str| segs.types.iter().find(|t| t.0 == name).map(|t| t.1.clone());
    let mut rows = Rows {
        map: BTreeMap::new(),
    };
    for s in 0..=n {
        let mut later: BTreeSet<&String> = BTreeSet::new();
        for w in names.seg_words.iter().skip(s + 1) {
            later.extend(w)
        }
        for w in names.step_words.iter().skip(s) {
            later.extend(w)
        }
        for name in names.declared[s]
            .iter()
            .filter(|x| later.contains(x) && !names.views.contains_key(*x))
        {
            let Some(ty) = type_of(name) else {
                return Err(nib("a value crossing a cut with no recorded type"));
            };
            rows.map.insert(name.clone(), (ty, Origin::Segment(s)));
        }
    }
    for (k, lp) in segs.loops.iter().enumerate() {
        for Carry { name, ty, .. } in &lp.carries {
            rows.map.insert(name.clone(), (ty.clone(), Origin::Loop(k)));
        }
    }
    Ok(rows)
}

/// The declarations ahead of the tile loop: the tile bounds, the lane rows, the plane rows and the runtime bounds' arrays.
fn tile_decls(segs: &LaneSegments, rows: &Rows) -> Result<Vec<String>, Refusal> {
    let mut lines: Vec<String> = vec![
        "    constexpr aether::offset_t hawk_tile = hawk_abi::kHostLoopTile;".into(),
        "    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);".into(),
        "    const aether::offset_t hawk_hi =".into(),
        "        static_cast<aether::offset_t>(base + count);".into(),
        "    alignas(64) HAWK_LANE_T hawk_on[hawk_tile] = {};".into(),
        "    alignas(64) HAWK_LANE_T hawk_ron[hawk_tile] = {};".into(),
    ];
    for (x, (t, _)) in &rows.map {
        lines.push(format!(
            "    alignas(64) {} hawk_r_{x}[{} * hawk_tile] = {{}};",
            lane_elem(t).map_err(unspelled)?,
            width(t)
        ))
    }
    let mut planes: Vec<(&String, usize)> = segs.proxies.iter().map(|x| (&x.1, x.2)).collect();
    planes.sort();
    planes.dedup();
    for (plane, wide) in planes {
        lines.push(format!(
            "    alignas(64) Real hawk_row_{plane}[{wide} * hawk_tile];"
        ))
    }
    for (k, lp) in segs.loops.iter().enumerate() {
        if let Some(b) = &lp.bounds {
            lines.push(format!(
                "    alignas(64) {} hawk_b{k}[hawk_tile] = {{}};",
                b.0
            ));
            lines.push(format!(
                "    alignas(64) {} hawk_e{k}[hawk_tile] = {{}};",
                b.0
            ));
        }
    }
    lines.extend(TILE_LOOP_HEAD.iter().map(|l| l.to_string()));
    Ok(lines)
}

/// The loop over the samples of a tile that runs segment `s`: its prologue, the rows it reads, the segment, the rows it
/// writes and the initial values of the loop that follows it.
fn segment_loop(
    segs: &LaneSegments,
    names: &CutNames,
    rows: &Rows,
    proxy_lines: &[String],
    s: usize,
    lines: &mut Vec<String>,
) -> Result<(), Refusal> {
    segment_prologue(segs, names, proxy_lines, s, lines);
    segment_body(segs, names, rows, s, lines)
}

/// The head of a segment's loop: the sample indexes, the proxies, the views it uses and the guard.
fn segment_prologue(
    segs: &LaneSegments,
    names: &CutNames,
    proxy_lines: &[String],
    s: usize,
    lines: &mut Vec<String>,
) {
    let (declared, views, seg_words) = (&names.declared, &names.views, &names.seg_words);
    independent_loop(
        lines,
        "    for (aether::offset_t hawk_j = hawk_tlo; hawk_j < hawk_thi; ++hawk_j) {",
    );
    lines.extend([
        "        const aether::offset_t hawk_t = hawk_j - hawk_tlo;".into(),
        "        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);".into(),
        "        const aether::SampleIndex i =".into(),
        "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));".into(),
        "        (void)hawk_t;".into(),
        "        (void)hawk_i;".into(),
        "        (void)nSamples;".into(),
    ]);
    lines.extend(proxy_lines.iter().cloned());
    for (x, line) in views {
        if seg_words[s].contains(x) && !declared[s].contains(x) {
            lines.push(format!("        {line}"))
        }
    }
    if s == 0 {
        lines.push("        HAWK_LANE_T hawk_go = 0.0;".into())
    }
    if let Some(g) = &segs.guard {
        lines.push(format!("        if (!{g}) {{"))
    }
}

/// A segment's rows read, its text, the rows it writes and the initial values of the loop after it.
fn segment_body(
    segs: &LaneSegments,
    names: &CutNames,
    rows: &Rows,
    s: usize,
    lines: &mut Vec<String>,
) -> Result<(), Refusal> {
    let n = segs.loops.len();
    let seg_words = &names.seg_words;
    let pad = LANE_PAD;
    let readable: Vec<&String> = rows
        .map
        .iter()
        .filter(|(x, (_, o))| {
            seg_words[s].contains(*x) && origin_index(*o) < s && *o != Origin::Segment(s)
        })
        .map(|(x, _)| x)
        .collect();
    for l in rows.load_lines(&readable, "hawk_t").map_err(unspelled)? {
        lines.push(format!("{pad}{l}"))
    }
    if !segs.segments[s].is_empty() {
        lines.push(segs.segments[s].text())
    }
    for (x, (t, o)) in &rows.map {
        if *o == Origin::Segment(s) {
            for y in row_put(&format!("hawk_r_{x}"), x, t, None).map_err(unspelled)? {
                lines.push(format!("{pad}{y}"))
            }
        }
    }
    if s < n {
        for Carry {
            name: x,
            ty: t,
            init,
            ..
        } in &segs.loops[s].carries
        {
            for y in row_put(&format!("hawk_r_{x}"), x, t, Some(init)).map_err(unspelled)? {
                lines.push(format!("{pad}{y}"))
            }
        }
        if let Some(b) = &segs.loops[s].bounds {
            lines.push(format!("{pad}hawk_b{s}[hawk_t] = {};", b.1));
            lines.push(format!("{pad}hawk_e{s}[hawk_t] = {};", b.3));
        }
    }
    if s == 0 {
        lines.push(format!("{pad}hawk_go = 1.0;"))
    }
    if segs.guard.is_some() {
        lines.push("        }".into())
    }
    if s == 0 {
        lines.push("        hawk_on[hawk_t] = hawk_go;".into())
    }
    lines.push("    }".into());
    Ok(())
}

/// The text of the interchanged loops.
fn build(segs: &LaneSegments) -> Result<String, Refusal> {
    let n = segs.loops.len();
    let names = cut_names(segs);
    let rows = crossing_rows(segs, &names)?;
    let mut lines = tile_decls(segs, &rows)?;
    let mut proxies = segs.proxies.clone();
    proxies.sort();
    let proxy_lines: Vec<String> = proxies.iter().map(|p| proxy_line(p, "hawk_t")).collect();
    for s in 0..=n {
        segment_loop(segs, &names, &rows, &proxy_lines, s, &mut lines)?;
        if s == n {
            break;
        }
        let step_views: Vec<String> = names
            .views
            .iter()
            .filter(|(x, _)| names.step_words[s].contains(*x))
            .map(|(_, l)| format!("        {l}"))
            .collect();
        lines.extend(
            trips(
                &segs.loops[s],
                s,
                &rows,
                &names.step_words[s],
                &proxies,
                &step_views,
            )
            .map_err(unspelled)?,
        );
    }
    lines.push("    }".into());
    Ok(lines.join("\n"))
}

/// The number of the segment or loop that writes a row.
fn origin_index(o: Origin) -> usize {
    match o {
        Origin::Segment(s) | Origin::Loop(s) => s,
    }
}

/// The line that binds a placed plane's proxy to the tile's row for the lane named `lane`.
fn proxy_line((proxy, plane, wide): &(String, String, usize), lane: &str) -> String {
    format!(
        "        const auto {proxy} = hawk_abi::tile_lane<Real, {wide}, hawk_tile>(hawk_row_{plane}, {lane});"
    )
}

/// How a cut loop's lanes count their trips: the index type, the extra lane conditions of a runtime bound and the lane's
/// index at the current trip.
struct TripBounds {
    /// The index type.
    itype: String,
    /// The condition a lane adds to be in the loop at its start.
    runs: String,
    /// The condition a lane adds to go on after a trip.
    more: String,
    /// The lane's index value at the current trip.
    index: String,
}

/// The index type and the lane conditions of a cut loop, from its runtime bounds or its static start and step.
fn trip_bounds(lp: &CutLoop, k: usize) -> TripBounds {
    let (itype, runs, more, index) = match &lp.bounds {
        Some((itype, _, cmp, _)) => (
            itype.clone(),
            format!(" && hawk_b{k}[hawk_l] {cmp} hawk_e{k}[hawk_l]"),
            format!(" && {} + {} {cmp} hawk_e{k}[hawk_l]", lp.index, lp.by),
            format!(
                "hawk_b{k}[hawk_l] + static_cast<{itype}>(hawk_trip) * {}",
                lp.by
            ),
        ),
        None => (
            "Int".to_string(),
            String::new(),
            String::new(),
            if (lp.start, lp.by) == (0, 1) {
                "hawk_trip".to_string()
            } else {
                format!(
                    "static_cast<Int>({}) + hawk_trip * static_cast<Int>({})",
                    lp.start, lp.by
                )
            },
        ),
    };
    TripBounds {
        itype,
        runs,
        more,
        index,
    }
}

/// The lanes that are in the loop at its start and the head of the trip loop.
fn trip_loop_open(lp: &CutLoop, bounds: &TripBounds) -> Vec<String> {
    let counted = lp.bounds.is_some();
    let runs = &bounds.runs;
    vec![
        "    {".into(),
        "    std::int64_t hawk_any = 0;".into(),
        INDEPENDENT_ITERATIONS.to_string(),
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_tile; ++hawk_l) {".into(),
        format!("        const bool hawk_in = hawk_l < hawk_tn && hawk_on[hawk_l] != 0.0{runs};"),
        "        hawk_ron[hawk_l] = hawk_in ? 1.0 : 0.0;".into(),
        "        hawk_any |= hawk_in ? std::int64_t{1} : std::int64_t{0};".into(),
        "    }".into(),
        if !counted {
            format!(
                "    for (Int hawk_trip = 0; hawk_any != 0 && hawk_trip < {}; ++hawk_trip) {{",
                lp.trips
            )
        } else {
            "    for (Int hawk_trip = 0; hawk_any != 0; ++hawk_trip) {".into()
        },
    ]
}

/// The head of the lane loop inside a trip: the sample index, the proxies and views, the lane's own flag and index.
fn lane_prologue(
    lp: &CutLoop,
    bounds: &TripBounds,
    proxies: &[(String, String, usize)],
    views: &[String],
    lines: &mut Vec<String>,
) {
    let (itype, index) = (&bounds.itype, &bounds.index);
    independent_loop(
        lines,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_tile; ++hawk_l) {",
    );
    lines.extend([
        "        const aether::offset_t hawk_t = hawk_l < hawk_tn ? hawk_l : hawk_tn - 1;".into(),
        "        const aether::SampleIndex i =".into(),
        "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_tlo + hawk_t));"
            .into(),
        "        (void)i;".into(),
    ]);
    for proxy in proxies {
        lines.push(proxy_line(proxy, "hawk_l"))
    }
    lines.extend(views.iter().cloned());
    lines.extend([
        "        const bool hawk_lane_on = hawk_ron[hawk_l] != 0.0;".to_string(),
        format!("        const {itype} {} = {index};", lp.index),
        format!("        (void){};", lp.index),
    ]);
}

/// One lane's trip: its rows loaded, the step, the carries put back and whether the lane goes on. A speculative step runs
/// for every lane and keeps the result of the live ones; any other runs only under the lane's flag.
fn lane_step(
    lp: &CutLoop,
    rows: &Rows,
    reads: &[&String],
    bounds: &TripBounds,
    stops: bool,
    lines: &mut Vec<String>,
) -> Result<(), Declines> {
    let (pad, more) = (LANE_PAD, &bounds.more);
    if lp.speculative {
        for l in rows.load_lines(reads, "hawk_l")? {
            lines.push(format!("        {l}"))
        }
        lines.push(lp.step.text());
        for Carry {
            name: x, next: nxt, ..
        } in &lp.carries
        {
            lines.push(format!(
                "        hawk_abi::lane_keep<hawk_tile>(hawk_r_{x}, hawk_l, hawk_lane_on, {nxt});"
            ))
        }
        if stops {
            lines.push(format!(
                "        const bool hawk_more = hawk_lane_on && !{}{more};",
                lp.exit
            ))
        }
    } else {
        if stops {
            lines.push("        bool hawk_more = false;".into())
        }
        lines.push("        if (hawk_lane_on) {".into());
        for l in rows.load_lines(reads, "hawk_l")? {
            lines.push(format!("{pad}{l}"))
        }
        lines.push(lp.step.text());
        for Carry {
            name: x, next: nxt, ..
        } in &lp.carries
        {
            lines.push(format!(
                "{pad}hawk_abi::lane_put<hawk_tile>(hawk_r_{x}, hawk_l, {nxt});"
            ))
        }
        if stops {
            lines.push(format!("{pad}hawk_more = !{}{more};", lp.exit))
        }
        lines.push("        }".into());
    }
    Ok(())
}

/// The trips of one interchanged loop: trips outside, the tile's lanes inside.
fn trips(
    lp: &CutLoop,
    k: usize,
    rows: &Rows,
    step_words: &BTreeSet<String>,
    proxies: &[(String, String, usize)],
    views: &[String],
) -> Result<Vec<String>, Declines> {
    let stops = lp.exit != "false" || lp.bounds.is_some();
    let bounds = trip_bounds(lp, k);
    let reads: Vec<&String> = rows
        .map
        .iter()
        .filter(|(x, (_, o))| step_words.contains(*x) && origin_index(*o) <= k)
        .map(|(x, _)| x)
        .collect();
    let mut lines = trip_loop_open(lp, &bounds);
    if stops {
        lines.push("    std::int64_t hawk_live = 0;".into())
    }
    lane_prologue(lp, &bounds, proxies, views, &mut lines);
    lane_step(lp, rows, &reads, &bounds, stops, &mut lines)?;
    if stops {
        lines.push("        hawk_ron[hawk_l] = hawk_more ? 1.0 : 0.0;".into())
    }
    lines.push("    }".into());
    if stops {
        lines.extend([
            "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_tile; ++hawk_l)".into(),
            "        hawk_live |= static_cast<std::int64_t>(hawk_ron[hawk_l] != 0.0);".into(),
            "    if (hawk_live == 0) break;".into(),
        ]);
    }
    lines.extend(["    }".to_string(), "    }".to_string()]);
    Ok(lines)
}
