//! The spelling of what only a group has: the lane's place and size, and the scratch arrays with their elements. A scratch
//! plane is local to one group; a value of rank above zero is held one array per component (structure of arrays), so the
//! loop over the lanes of a phase reads and writes unit-stride arrays.
use super::{edged, element, ext, guarded};
use crate::refuse::{nib, Refusal};

/// The lane's place in its group, as an index value.
pub const GROUP_RANK: &str = "static_cast<Int>(hawk_rank)";

/// The number of lanes of the lane's group, as an index value.
pub const GROUP_SIZE: &str = "static_cast<Int>(hawk_size)";

/// The extent a scratch array is indexed against: the lanes of this group.
pub const GROUP_EXTENT: &str = "static_cast<std::size_t>(hawk_size)";

/// The name of the pointer a group hoists for the input plane bound as `binding`, and the lane read through it: the plane's data
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
/// the pitch of `lane_pitch`, which is the view's last extent. The promise `__restrict` makes holds because only an input plane
/// (a plane the unit never writes) gets one; a plane the body also writes is read through its view.
pub fn lane_pointer_declarations(binding: &str, rank: usize) -> Vec<String> {
    let mut lines = vec![format!(
        "const auto* __restrict {} = {binding}.data();",
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

/// The row of a vector or matrix input plane at the lane `hawk_i`, as an item of its components, read through the hoisted pointer
/// and pitch: component `k` (row-major) of the lane is `k` pitches after the lane's own place.
pub fn lane_row(binding: &str, value: (&str, &[usize])) -> Result<String, Refusal> {
    let (dtype, shape) = value;
    let (pointer, pitch) = (lane_pointer(binding), lane_pitch(binding));
    let cells: Vec<String> = (0..shape.iter().product::<usize>())
        .map(|k| format!("{pointer}[{k} * {pitch} + hawk_i]"))
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
fn scratch_type(dtype: &str) -> Result<&'static str, Refusal> {
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

/// The cell of component `component` of the scratch plane `name` at the lane `index`.
pub fn scratch_cell(name: &str, component: Option<usize>, index: &str) -> String {
    format!("{}[{index}]", scratch_array(name, component))
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
