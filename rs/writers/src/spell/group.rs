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
