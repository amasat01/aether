//! The sample-sum sinks of a unit and what both arms write for them: the row-store lambda the row
//! sum calls, the code of
//! the operation and the fill that sets a row to the operation's identity.
use super::{element, Walk};
use crate::contract::types::Active;
use crate::ir::door::{self, RxCtx};
use crate::refuse::{fault, Refusal};
use crate::spell::binding_name;

/// One sample-sum sink: its operation (`sum`, `max` or `min`), the plane it stores to and that
/// plane's element type.
pub struct SumSink {
    /// The reduction: `sum`, `max` or `min`.
    pub op: String,
    /// The role of the plane the sink writes.
    pub role: String,
    /// The name of the plane the sink writes.
    pub name: String,
    /// The element type of the plane.
    pub element: String,
}

/// The sample-sum sinks among a unit's sinks, in order.
pub fn sample_sum_nodes(ctx: *mut RxCtx, sinks: &[u32]) -> Result<Vec<u32>, Refusal> {
    let mut sums = vec![];
    for &sink in sinks {
        if door::class_is(ctx, sink, "SampleSum")? {
            sums.push(sink);
        }
    }
    Ok(sums)
}

/// The sample-sum sinks among `nodes`, read through the door; `walk` gives their planes' types.
pub fn sum_sinks(ctx: *mut RxCtx, nodes: &[u32], walk: &Walk) -> Result<Vec<SumSink>, Refusal> {
    let text =
        |node: u32, field: &str| door::field_text(ctx, node, field).map(Option::unwrap_or_default);
    let mut sinks = vec![];
    for node in nodes {
        let (op, role, name) = (
            text(*node, "op")?,
            text(*node, "role")?,
            text(*node, "name")?,
        );
        let slot = walk
            .slots
            .iter()
            .find(|slot| slot.name == name)
            .ok_or_else(|| fault("sample-sum sink without a slot"))?;
        sinks.push(SumSink {
            op,
            role,
            name,
            element: element(&slot.dtype)?,
        });
    }
    Ok(sinks)
}

#[derive(Debug, Clone, PartialEq)]
/// The row plane of one sample-sum sink: the cell it fills and the plane's position among the entry's
/// arguments.
pub struct RowPlane {
    /// The name of the cell the sink fills.
    pub cell: String,
    /// The position of the plane in the argument list.
    pub slot: usize,
}

/// The row planes of the sample-sum sinks among `sinks`, in order; `slots` are the entry's arguments.
pub fn row_planes(
    ctx: *mut RxCtx,
    sinks: &[u32],
    slots: &[crate::contract::types::Slot],
) -> Result<Vec<RowPlane>, Refusal> {
    let text =
        |node: u32, field: &str| door::field_text(ctx, node, field).map(Option::unwrap_or_default);
    sample_sum_nodes(ctx, sinks)?
        .into_iter()
        .map(|node| {
            let name = text(node, "name")?;
            let slot = slots
                .iter()
                .position(|slot| slot.name == name)
                .ok_or_else(|| fault("sample-sum sink without a slot"))?;
            Ok(RowPlane {
                cell: text(node, "cell")?,
                slot,
            })
        })
        .collect()
}

/// Whether the sink states the position of a minimum or maximum (`argmin`, `argmax`) rather than a value.
pub fn is_position(op: &str) -> bool {
    op.starts_with("arg")
}

/// The direction (`min` or `max`) of each column of a unit that asks for a position, or `None` when it asks for none. Such a unit
/// folds pairs (value, lane) in every column, so every column must be a minimum, a maximum or the position of one; a sum has no
/// lane to carry.
pub fn pair_directions(sinks: &[SumSink]) -> Result<Option<Vec<&'static str>>, Refusal> {
    if !sinks.iter().any(|sink| is_position(&sink.op)) {
        return Ok(None);
    }
    sinks
        .iter()
        .map(|sink| match sink.op.trim_start_matches("arg") {
            "min" => Ok("min"),
            "max" => Ok("max"),
            _ => Err(fault(
                "a position over samples shares its kernel only with minima, maxima and their positions",
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The refusal of a sum over samples under a lane set that runs its lanes in groups: the row folds count launch positions, and
/// a group's lanes are not one position each, so the cells would not be the sum of the live samples.
pub fn refuse_grouped_sum(name: &str, active: Option<&Active>) -> Result<(), Refusal> {
    if active.is_some_and(|set| set.group > 1) {
        return Err(fault(format!(
            "{name}: a sum over samples with a lane group above 1 is not written yet"
        )));
    }
    Ok(())
}

/// The code of the operation the row sum takes as a template argument.
pub fn op_code(op: &str) -> Result<i64, Refusal> {
    match op {
        "sum" => Ok(0),
        "max" => Ok(1),
        "min" => Ok(2),
        "argmin" | "argmax" => Err(fault(
            "a position over samples is written for the host only; this target's text has no row fold of pairs",
        )),
        other => Err(fault(format!("unknown sample-sum op {other:?}"))),
    }
}

/// The statement that sets `width` slots of `array` to the identity of `op`; nothing for a sum,
/// whose array is zeroed.
pub fn identity_fill(op: &str, width: i64, array: &str, element: &str) -> Result<String, Refusal> {
    if op == "sum" {
        return Ok(String::new());
    }
    let code = op_code(op)?;
    let identity = format!("hawk_abi::ssIdentity_<{code}, {element}>()");
    Ok(format!(
        "    for (int hawk_k = 0; hawk_k < {width}; ++hawk_k) {array}[hawk_k] = {identity};\n"
    ))
}

/// The entry-local `hawk_ss_store(column, sum, compensation)` the row sum calls: column `k` goes to
/// the `k`-th sink's
/// planes at row `hawk_row`; a banded (`cells`) mode converts the accumulators on the way out. With
/// `rows` above one, a column also names which of the folded rows it belongs to.
pub fn store_lambda(
    sinks: &[SumSink],
    rows: i64,
    cells_row: i64,
    cells: bool,
) -> Result<String, Refusal> {
    let (out, comp) = if cells {
        ("hawk_abi::ssOut_(hawk_s)", "hawk_abi::ssComp_(hawk_c)")
    } else {
        ("hawk_s", "hawk_c")
    };
    // the lanes of the winners of a unit that asks for a position are the second value of every column
    let paired = sinks.iter().any(|sink| is_position(&sink.op));
    let mut lines =
        vec!["    auto hawk_ss_store = [&](int hawk_col, auto hawk_s, auto hawk_c) {".to_string()];
    if rows > 1 {
        lines.push(format!(
            "        const int hawk_rr = hawk_col / {};",
            sinks.len()
        ));
        lines.push(format!(
            "        if (hawk_wbase + {cells_row} * hawk_rr >= count) return;"
        ));
        lines.push("        const long long hawk_row = hawk_row0 + hawk_rr;".to_string());
        lines.push(format!("        hawk_col %= {};", sinks.len()));
    }
    // A flat switch, not an if/else-if chain: with thousands of sinks a chain nests one level per
    // sink in the compiler's parser and exhausts it, while the cases stay flat.
    lines.push("        switch (hawk_col) {".to_string());
    for (k, sink) in sinks.iter().enumerate() {
        lines.push(format!("        case {k}: {{"));
        lines.push(format!(
            "            {}.data()[hawk_row] = {out};",
            binding_name(&sink.role, 0, &sink.name)?
        ));
        if sink.op == "sum" || paired {
            lines.push(format!(
                "            {}.data()[hawk_row] = {comp};",
                binding_name(&sink.role, 0, &format!("{}_comp", sink.name))?
            ));
        }
        lines.push("        } break;".to_string());
    }
    lines.push("        }".to_string());
    lines.push("    };".to_string());
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sink(op: &str, name: &str) -> SumSink {
        SumSink {
            op: op.into(),
            role: "out".into(),
            name: name.into(),
            element: "double".into(),
        }
    }

    #[test]
    fn the_store_is_a_switch_with_a_case_per_sink() {
        let sinks = [sink("sum", "a"), sink("max", "b")];
        let text = store_lambda(&sinks, 1, 0, false).unwrap();
        let (first, second) = (
            binding_name("out", 0, "a").unwrap(),
            binding_name("out", 0, "b").unwrap(),
        );
        let want = format!(
            "    auto hawk_ss_store = [&](int hawk_col, auto hawk_s, auto hawk_c) {{
        switch (hawk_col) {{
        case 0: {{
            {first}.data()[hawk_row] = hawk_s;
            {first}_comp.data()[hawk_row] = hawk_c;
        }} break;
        case 1: {{
            {second}.data()[hawk_row] = hawk_s;
        }} break;
        }}
    }};"
        );
        assert_eq!(text, want);
    }

    #[test]
    fn six_thousand_sinks_write_no_else_if_chain() {
        let sinks: Vec<SumSink> = (0..6000).map(|k| sink("sum", &format!("s{k}"))).collect();
        let text = store_lambda(&sinks, 1, 0, false).unwrap();
        assert!(!text.contains("else if (hawk_col"));
        assert_eq!(text.matches("        case ").count(), 6000);
    }

    #[test]
    fn a_position_shares_its_kernel_with_minima_and_maxima_but_not_sums() {
        let directions = pair_directions(&[sink("argmin", "a"), sink("max", "b")]).unwrap();
        assert_eq!(directions, Some(vec!["min", "max"]));
        assert_eq!(pair_directions(&[sink("min", "a")]).unwrap(), None);
        assert!(pair_directions(&[sink("argmax", "a"), sink("sum", "b")]).is_err());
    }

    #[test]
    fn a_sum_with_a_lane_group_above_one_is_refused_by_name() {
        let set = |group| Active {
            map_binding: Some("map".into()),
            count_binding: "count".into(),
            group,
            map_bits: crate::contract::types::MapBits::B32,
            lane_set: true,
        };
        let refusal = refuse_grouped_sum("k", Some(&set(4)));
        assert!(format!("{:?}", refusal.expect_err("a refusal"))
            .contains("k: a sum over samples with a lane group above 1 is not written yet"));
        assert!(refuse_grouped_sum("k", Some(&set(1))).is_ok());
        assert!(refuse_grouped_sum("k", None).is_ok());
    }
}
