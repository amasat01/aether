//! The spelling of what only a group has: the lane's place and size, and the scratch arrays with their elements. A scratch
//! plane is local to one group; a value of rank above zero is held one array per component (structure of arrays), so the
//! loop over the lanes of a phase reads and writes unit-stride arrays.
use super::{binding_name, edged, element, ext, guarded};
use crate::refuse::{nib, Refusal};

/// The lane's place in its group, as an index value.
pub const GROUP_RANK: &str = "static_cast<Int>(hawk_rank)";

/// The number of lanes of the lane's group, as an index value.
pub const GROUP_SIZE: &str = "static_cast<Int>(hawk_size)";

/// The extent a scratch array is indexed against: the lanes of this group.
pub const GROUP_EXTENT: &str = "static_cast<std::size_t>(hawk_size)";

/// The name of the pointer a group hoists for the plane bound as `binding`, and the lane read or written through it: the plane's data
/// as a typed `__restrict` pointer, so the compiler sees an affine base in the phase loop and no alias with the scratch arrays.
pub fn lane_pointer(binding: &str) -> String {
    format!("hawk_p_{binding}")
}

/// The name of the component pitch a group hoists beside the pointer of a vector or matrix input plane: the distance, in
/// elements, between the same lane's neighbouring components in the plane's structure-of-arrays storage.
pub fn lane_pitch(binding: &str) -> String {
    format!("hawk_pitch_{binding}")
}

/// The declarations of the pointer of `lane_pointer` over the view bound as `binding`, and for a value of rank above zero (`rank`)
/// the pitch of `lane_pitch`, which is the view's last extent. The pointer is to const for a plane the unit reads and to
/// writable memory for one it `written`. The promise `__restrict` makes is stated by the callers' rules (`unit::group::input_planes`
/// and `unit::group::output_planes`): nothing else of the unit touches the plane's memory through another name in the loops.
pub fn lane_pointer_declarations(binding: &str, rank: usize, written: bool) -> Vec<String> {
    let constness = if written { "" } else { "const " };
    let mut lines = vec![format!(
        "{constness}auto* __restrict {} = {binding}.data();",
        lane_pointer(binding)
    )];
    if rank > 0 {
        lines.push(format!(
            "const std::int64_t {} = static_cast<std::int64_t>({binding}.extent({rank}));",
            lane_pitch(binding)
        ));
    }
    lines
}

/// Component `k` (row-major) of the lane `hawk_i` of the vector or matrix plane bound as `binding`, through its hoisted pointer and
/// pitch: `k` pitches after the lane's own place.
pub fn lane_component(binding: &str, k: usize) -> String {
    format!(
        "{}[{k} * {} + hawk_i]",
        lane_pointer(binding),
        lane_pitch(binding)
    )
}

/// The row of a vector or matrix input plane at the lane `hawk_i`, as an item of its components, read through the hoisted pointer
/// and pitch: component `k` (row-major) of the lane is `k` pitches after the lane's own place.
pub fn lane_row(binding: &str, value: (&str, &[usize])) -> Result<String, Refusal> {
    let (dtype, shape) = value;
    let cells: Vec<String> = (0..shape.iter().product::<usize>())
        .map(|k| lane_component(binding, k))
        .collect();
    Ok(format!(
        "aether::Item<{}, {}>{{{}}}",
        element(dtype)?,
        ext(shape),
        cells.join(", ")
    ))
}

/// The opening of the function that is the body of one group of a `count` group: its base lane and its size are its arguments.
/// The text holds the phases once and calls the function twice (`group_calls`), so the function is forced inline (the attribute
/// after the parameters is the spelling clang and gcc both accept on a lambda): each call site is then compiled with its own size.
pub fn group_function_open() -> String {
    "    const auto hawk_group = [&](const std::int64_t hawk_base, const std::int64_t hawk_size) __attribute__((always_inline)) {"
        .to_string()
}

/// The calls of `hawk_group` for the groups `[base, base + count)` of `n` lanes, after the function's definition: the whole groups
/// get the literal `n` as their size, so every phase loop has a known trip count, and only the last group of the launch (the one
/// with fewer than `n` lanes left, if there is one) gets its size at run time. `hawk_split` is the first group that is not whole,
/// held to the call's range, so a launch tiled over several calls is right whether or not a call holds the short group.
pub fn group_calls(n: i64) -> Vec<String> {
    vec![
        format!("    const std::int64_t hawk_whole = nSamples / {n};"),
        "    const std::int64_t hawk_end = base + count;".to_string(),
        "    const std::int64_t hawk_split = hawk_whole < base ? base : hawk_whole < hawk_end ? hawk_whole : hawk_end;".to_string(),
        format!("    for (std::int64_t hawk_g = base; hawk_g < hawk_split; ++hawk_g) hawk_group(hawk_g * {n}, {n});"),
        format!("    for (std::int64_t hawk_g = hawk_split; hawk_g < hawk_end; ++hawk_g) hawk_group(hawk_g * {n}, nSamples - hawk_g * {n});"),
    ]
}

/// The array that holds component `component` of the scratch plane `name`; a scalar plane has the one array of its name.
pub fn scratch_array(name: &str, component: Option<usize>) -> String {
    match component {
        None => format!("scr_{name}"),
        Some(k) => format!("scr_{name}_{k}"),
    }
}

/// The C++ element type of a scratch scalar.
pub(crate) fn scratch_type(dtype: &str) -> Result<&'static str, Refusal> {
    match dtype {
        "f64" | "f32" => Ok("Real"),
        "i32" => Ok("std::int32_t"),
        "i64" => Ok("std::int64_t"),
        "bool" => Ok("bool"),
        other => Err(nib(format!("a scratch of element type {other}"))),
    }
}

/// The declarations of the arrays of the scratch plane `name` for a group of at most `bound` lanes: one per component of a value.
pub fn scratch_declare(
    name: &str,
    dtype: &str,
    shape: &[usize],
    bound: i64,
) -> Result<Vec<String>, Refusal> {
    let element = scratch_type(dtype)?;
    let components: usize = shape.iter().product();
    Ok((0..components)
        .map(|k| {
            let array = scratch_array(name, (!shape.is_empty()).then_some(k));
            format!("{element} {array}[{bound}];")
        })
        .collect())
}

/// The name of the slot the runtime makes for the scratch plane `name` when the plane is held outside the unit's text.
pub fn scratch_slot(name: &str) -> String {
    format!("group_scratch_{name}")
}

/// The declarations of the arrays of the scratch plane `name` held by the runtime, for a group of at most `bound` lanes: one
/// per component of a value, each the same name and element type as the local array of `scratch_declare`, so the phases address
/// both alike. The plane holds the arrays of group `group` (an index expression) one after another, `bound` elements each,
/// `bound` times the components apart from the next group's. `__restrict` states what the arrays are: disjoint ranges, so the
/// compiler vectorises the phase loops as it does over local arrays. The plane's view has the walk's element type (`Int` for a
/// whole number) and the plane its own width, so the pointer is taken from the view's address as the scratch's element type.
pub fn scratch_slice(
    name: &str,
    dtype: &str,
    shape: &[usize],
    (bound, group): (i64, &str),
) -> Result<Vec<String>, Refusal> {
    let element = scratch_type(dtype)?;
    let components: usize = shape.iter().product();
    let plane = binding_name("lookup", 0, &scratch_slot(name))?;
    let stride = components as i64 * bound;
    Ok((0..components)
        .map(|k| {
            let array = scratch_array(name, (!shape.is_empty()).then_some(k));
            let first = k as i64 * bound;
            format!(
                "{element}* __restrict {array} = reinterpret_cast<{element}*>({plane}.data()) + {group} * {stride} + {first};"
            )
        })
        .collect())
}

/// The cell of component `component` of the scratch plane `name` at the lane `index`.
pub fn scratch_cell(name: &str, component: Option<usize>, index: &str) -> String {
    format!("{}[{index}]", scratch_array(name, component))
}

/// The slots of the runtime's in-leaf reduction tree (`eagle/reduce/canonical.h`): the terms of a lane count above this fold into
/// slot `lane % SLOTS` in lane order before the halving tree runs.
pub const TREE_SLOTS: i64 = 256;

/// The most lanes one in-leaf tree holds (`slots * terms` of the runtime): a group sum over more lanes would continue with the
/// runtime's pairwise levels, which the host text does not write.
pub const TREE_LANES: i64 = 4096;

/// How the terms of a group-wide sum are added.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Association {
    /// The runtime's fixed tree: the two halves of the slots are added, the width halving down to one.
    Tree,
    /// One by one in rank order.
    Lane,
}

/// The array that holds the operand of the group-wide reduce or scan node `node`, one element per lane of the group.
pub fn work_array(node: u32) -> String {
    format!("hawk_work_{node}")
}

/// The name the value of the group-wide reduce `node` is read under, after the reduction.
pub fn reduce_value(node: u32) -> String {
    format!("hawk_total_{node}")
}

/// The name the value of the group-wide scan `node` is read under: the lane's own prefix, in the lane's loop.
pub fn scan_value(node: u32) -> String {
    format!("{}[hawk_rank]", work_array(node))
}

/// The declaration of the work array of `node` for a group of at most `bound` lanes.
pub fn work_declare(node: u32, dtype: &str, bound: i64) -> Result<String, Refusal> {
    Ok(format!(
        "{} {}[{bound}];",
        scratch_type(dtype)?,
        work_array(node)
    ))
}

/// The store of a lane's operand into its place of the work array of `node`.
pub fn work_store(node: u32, value: &str) -> String {
    format!("{}[hawk_rank] = {value};", work_array(node))
}

/// `a` and `b` combined by the reduce `op` over values of `dtype`. A real minimum or maximum ignores a NaN term and is NaN only
/// when every term is (the rule of `numpy.nanmin`, which is what the library's other minima over a column follow).
fn fold(op: &str, dtype: &str, a: &str, b: &str) -> Result<String, Refusal> {
    let real = matches!(dtype, "f64" | "f32");
    Ok(match (op, real) {
        ("sum", _) => format!("{a} + {b}"),
        ("min", true) => format!("aether::math::fmin({a}, {b})"),
        ("max", true) => format!("aether::math::fmax({a}, {b})"),
        ("min", false) => format!("{a} < {b} ? {a} : {b}"),
        ("max", false) => format!("{a} > {b} ? {a} : {b}"),
        ("any", _) => format!("{a} || {b}"),
        ("all", _) => format!("{a} && {b}"),
        _ => return Err(nib(format!("a group reduce {op}"))),
    })
}

/// The statements that reduce the work array of `node` to its first element over the `hawk_size` lanes of the group (a group of at
/// most `bound` lanes) and bind that element as `reduce_value`. `Association::Tree` is the association of the runtime's in-leaf
/// tree: lane `l` of a count above `TREE_SLOTS` folds into slot `l % TREE_SLOTS` in lane order, then for the width halving from
/// half the slots to one, slot `t < width` takes slot `t + width` when that one is present. A width is a literal, so the compiler
/// sees a fixed distance between the two halves and vectorises the loop.
pub fn reduce_lines(
    node: u32,
    (op, dtype): (&str, &str),
    association: Association,
    bound: i64,
) -> Result<Vec<String>, Refusal> {
    let work = work_array(node);
    let held = scratch_type(dtype)?;
    let mut steps = vec![];
    match association {
        Association::Lane => {
            let step = fold(op, dtype, "hawk_acc", &format!("{work}[hawk_k]"))?;
            steps.push(format!("{held} hawk_acc = {work}[0];"));
            steps.push(format!(
                "for (std::int64_t hawk_k = 1; hawk_k < hawk_size; ++hawk_k) hawk_acc = {step};"
            ));
            steps.push(format!("{work}[0] = hawk_acc;"));
        }
        Association::Tree => {
            if bound > TREE_SLOTS {
                let step = fold(
                    op,
                    dtype,
                    &format!("{work}[hawk_t]"),
                    &format!("{work}[hawk_k + hawk_t]"),
                )?;
                steps.push(format!("for (std::int64_t hawk_k = {TREE_SLOTS}; hawk_k < hawk_size; hawk_k += {TREE_SLOTS}) {{"));
                steps.push(format!("    const std::int64_t hawk_m = hawk_size - hawk_k < {TREE_SLOTS} ? hawk_size - hawk_k : {TREE_SLOTS};"));
                steps.push(format!(
                    "    for (std::int64_t hawk_t = 0; hawk_t < hawk_m; ++hawk_t) {work}[hawk_t] = {step};"
                ));
                steps.push("}".to_string());
            }
            let mut width = TREE_SLOTS / 2;
            while width >= 1 {
                if width < bound {
                    let step = fold(
                        op,
                        dtype,
                        &format!("{work}[hawk_t]"),
                        &format!("{work}[hawk_t + {width}]"),
                    )?;
                    steps.push(format!("const std::int64_t hawk_m{width} = hawk_size - {width} < {width} ? hawk_size - {width} : {width};"));
                    steps.push(format!(
                        "for (std::int64_t hawk_t = 0; hawk_t < hawk_m{width}; ++hawk_t) {work}[hawk_t] = {step};"
                    ));
                }
                width /= 2;
            }
        }
    }
    // the steps name their own counters: one block each, so several reduces of a group do not share a scope
    let mut lines = vec!["{".to_string()];
    lines.extend(steps.into_iter().map(|step| format!("    {step}")));
    lines.push("}".to_string());
    lines.push(format!("const {held} {} = {work}[0];", reduce_value(node)));
    Ok(lines)
}

/// The statements that replace the work array of `node` by its inclusive running sum in rank order: lane `l` then holds the
/// first `l + 1` operands added one by one, as `numpy.cumsum` adds them.
pub fn scan_lines(node: u32) -> Vec<String> {
    let work = work_array(node);
    vec![format!(
        "for (std::int64_t hawk_k = 1; hawk_k < hawk_size; ++hawk_k) {work}[hawk_k] = {work}[hawk_k - 1] + {work}[hawk_k];"
    )]
}

/// How a scratch index is held to the plane.
#[derive(Clone, Copy)]
pub enum Fold<'a> {
    /// The index is in the plane by construction (an interior lane): it is used as it is.
    Plain,
    /// The index may be outside the plane and the edge rule (`clamp`, `wrap` or `zero`) says what it means.
    Edge(&'a str),
}

/// The element of the scratch plane `name` at `index` held to the plane by `fold`, against `extent` lanes: a scalar is the cell,
/// a value of rank above zero an item of its components' cells; a `zero` edge reads zero outside the plane.
pub fn scratch_read(
    name: &str,
    value: (&str, &[usize]),
    index: &str,
    fold: Fold,
    extent: &str,
) -> Result<String, Refusal> {
    let (dtype, shape) = value;
    let (position, test) = match fold {
        Fold::Plain => (index.to_string(), None),
        Fold::Edge(rule) => edged(rule, index, extent)?,
    };
    let cells: Vec<String> = (0..shape.iter().product::<usize>())
        .map(|k| {
            let array = scratch_array(name, (!shape.is_empty()).then_some(k));
            format!("{array}[{position}]")
        })
        .collect();
    let read = if shape.is_empty() {
        cells.into_iter().next().unwrap_or_default()
    } else {
        format!(
            "aether::Item<{}, {}>{{{}}}",
            element(dtype)?,
            ext(shape),
            cells.join(", ")
        )
    };
    guarded(read, test, dtype, shape)
}

#[cfg(test)]
mod tests {
    //! The two realisations of a scratch plane: the array of the text, and the slice of the plane the runtime holds.
    use super::*;

    /// Oracle: the declarations written by hand. A scalar plane is one array of `bound` elements named for the plane; a vector of
    /// three is three arrays, one per component, so both realisations are addressed by the same `scr_<name>[_k][lane]`.
    #[test]
    fn a_local_scratch_is_one_array_per_component_of_the_bound() {
        assert_eq!(
            scratch_declare("t", "f64", &[], 8).unwrap(),
            vec!["Real scr_t[8];"]
        );
        assert_eq!(
            scratch_declare("v", "i32", &[3], 8).unwrap(),
            vec![
                "std::int32_t scr_v_0[8];",
                "std::int32_t scr_v_1[8];",
                "std::int32_t scr_v_2[8];"
            ]
        );
    }

    /// Oracle: the offsets by hand. Group `g` of a plane of 3 components over groups of 8 lanes starts at `g * 24`, and component
    /// `k` of it `k * 8` further: the arrays are the local ones, laid in the plane.
    #[test]
    fn a_planed_scratch_is_the_same_arrays_at_the_slice_of_its_group() {
        assert_eq!(
            scratch_slice("t", "f64", &[], (8, "hawk_g")).unwrap(),
            vec!["Real* __restrict scr_t = reinterpret_cast<Real*>(lut_group_scratch_t.data()) + hawk_g * 8 + 0;"]
        );
        let planed = scratch_slice("v", "i32", &[3], (8, "(hawk_base / 8)")).unwrap();
        assert_eq!(planed.len(), 3);
        assert_eq!(
            planed[2],
            "std::int32_t* __restrict scr_v_2 = reinterpret_cast<std::int32_t*>(lut_group_scratch_v.data()) + (hawk_base / 8) * 24 + 16;"
        );
    }

    /// Oracle: the declared word. A scratch of a type no scratch holds is refused by name in both realisations.
    #[test]
    fn a_scratch_of_an_unwritten_type_is_refused_in_both_realisations() {
        assert!(scratch_declare("t", "u8", &[], 8).is_err());
        assert!(scratch_slice("t", "u8", &[], (8, "hawk_g")).is_err());
    }

    /// Oracle: the lines written by hand from the in-leaf tree's definition. A group of at most 8 lanes has the widths 4, 2 and 1
    /// (the widths below the bound), each a literal distance between the halves and each limited to the slots whose partner is
    /// present; there is no slot walk below 257 lanes.
    #[test]
    fn a_tree_sum_of_eight_lanes_adds_the_two_halves_at_the_widths_four_two_and_one() {
        let lines = reduce_lines(5, ("sum", "f64"), Association::Tree, 8).unwrap();
        assert_eq!(
            lines,
            vec![
                "{",
                "    const std::int64_t hawk_m4 = hawk_size - 4 < 4 ? hawk_size - 4 : 4;",
                "    for (std::int64_t hawk_t = 0; hawk_t < hawk_m4; ++hawk_t) hawk_work_5[hawk_t] = hawk_work_5[hawk_t] + hawk_work_5[hawk_t + 4];",
                "    const std::int64_t hawk_m2 = hawk_size - 2 < 2 ? hawk_size - 2 : 2;",
                "    for (std::int64_t hawk_t = 0; hawk_t < hawk_m2; ++hawk_t) hawk_work_5[hawk_t] = hawk_work_5[hawk_t] + hawk_work_5[hawk_t + 2];",
                "    const std::int64_t hawk_m1 = hawk_size - 1 < 1 ? hawk_size - 1 : 1;",
                "    for (std::int64_t hawk_t = 0; hawk_t < hawk_m1; ++hawk_t) hawk_work_5[hawk_t] = hawk_work_5[hawk_t] + hawk_work_5[hawk_t + 1];",
                "}",
                "const Real hawk_total_5 = hawk_work_5[0];",
            ]
        );
    }

    /// Oracle: a group of one lane has no partner at any width, so its sum is its first element.
    #[test]
    fn a_tree_sum_of_one_lane_is_its_first_element() {
        let lines = reduce_lines(2, ("sum", "f64"), Association::Tree, 1).unwrap();
        assert_eq!(
            lines,
            vec!["{", "}", "const Real hawk_total_2 = hawk_work_2[0];"]
        );
    }

    /// Oracle: lanes above the slot count fold into slot `lane % 256` in lane order, in blocks of 256, before the tree starts at
    /// width 128.
    #[test]
    fn a_tree_sum_above_the_slot_count_folds_the_extra_lanes_into_the_slots_first() {
        let lines = reduce_lines(1, ("sum", "f32"), Association::Tree, 1000).unwrap();
        assert_eq!(
            lines[1],
            "    for (std::int64_t hawk_k = 256; hawk_k < hawk_size; hawk_k += 256) {"
        );
        assert!(lines[3]
            .contains("hawk_work_1[hawk_t] = hawk_work_1[hawk_t] + hawk_work_1[hawk_k + hawk_t];"));
        assert!(lines.iter().any(|l| l.contains("hawk_t + 128];")));
    }

    /// Oracle: the serial order is one loop from the first lane to the last, whatever the bound.
    #[test]
    fn a_lane_order_sum_is_one_serial_loop_in_rank_order() {
        let lines = reduce_lines(3, ("sum", "f64"), Association::Lane, 8).unwrap();
        assert_eq!(
            lines,
            vec![
                "{",
                "    Real hawk_acc = hawk_work_3[0];",
                "    for (std::int64_t hawk_k = 1; hawk_k < hawk_size; ++hawk_k) hawk_acc = hawk_acc + hawk_work_3[hawk_k];",
                "    hawk_work_3[0] = hawk_acc;",
                "}",
                "const Real hawk_total_3 = hawk_work_3[0];",
            ]
        );
    }

    /// Oracle: a real minimum or maximum is the library's `fmin` / `fmax` (a NaN term ignored, as `numpy.nanmin` and `nanmax`), a whole
    /// number's is the comparison, and any and all are the logical operators on a bool.
    #[test]
    fn the_order_free_reduces_fold_by_their_own_operator() {
        let fold_of =
            |op, dtype| reduce_lines(7, (op, dtype), Association::Lane, 4).unwrap()[2].clone();
        assert!(fold_of("min", "f64")
            .contains("hawk_acc = aether::math::fmin(hawk_acc, hawk_work_7[hawk_k]);"));
        assert!(fold_of("max", "f32")
            .contains("hawk_acc = aether::math::fmax(hawk_acc, hawk_work_7[hawk_k]);"));
        assert!(fold_of("min", "i32").contains(
            "hawk_acc = hawk_acc < hawk_work_7[hawk_k] ? hawk_acc : hawk_work_7[hawk_k];"
        ));
        assert!(fold_of("any", "bool").contains("hawk_acc = hawk_acc || hawk_work_7[hawk_k];"));
        assert!(fold_of("all", "bool").contains("hawk_acc = hawk_acc && hawk_work_7[hawk_k];"));
    }

    /// Oracle: the scan adds each lane's operand to the running sum of the lane before, in rank order, and a lane reads its own
    /// prefix at its place.
    #[test]
    fn a_scan_is_the_serial_running_sum_read_at_the_lanes_place() {
        assert_eq!(
            scan_lines(4),
            vec!["for (std::int64_t hawk_k = 1; hawk_k < hawk_size; ++hawk_k) hawk_work_4[hawk_k] = hawk_work_4[hawk_k - 1] + hawk_work_4[hawk_k];"]
        );
        assert_eq!(scan_value(4), "hawk_work_4[hawk_rank]");
    }
}
