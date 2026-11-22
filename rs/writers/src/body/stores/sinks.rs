//! Statements for sinks: own-column stores, sample sums and scattered accumulates.
use crate::body::frame::needs::Need;
use crate::body::values::text_field;
use crate::body::Class;
use crate::body::{CompensatedPlan, CompensatedTarget, Frame, Name, NameKind};
use crate::ir::door;
use crate::refuse::{fault, nib, Refusal};
use crate::spell::{binding_name, element_at, ZEROED_PLANE};

/// The text of a value stored into a plane element: a banded unit's plane element is a `Real`
/// built from the band.
fn plane_value(f: &Frame, text: &str) -> String {
    if f.banded() {
        format!("Real({text})")
    } else {
        text.to_string()
    }
}

/// One own-column store (early, right after the value, or at the end).
pub(crate) fn store(f: &mut Frame, sink: u32) -> Result<(), Refusal> {
    let p = f.at(sink)?;
    let v = f.opp(p)[0];
    let h = f.node(p).h;
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        0,
        &text_field(f.ctx(), h, "name")?,
    )?;
    if let Some(CompensatedPlan {
        plane,
        target: Some(CompensatedTarget {
            name,
            own_column: true,
        }),
    }) = f.compensated.clone()
    {
        if name == text_field(f.ctx(), h, "name")? {
            let comp = binding_name("mutable", 0, &plane)?;
            f.need(Need::StoreCompensated);
            f.emit(format!(
                "hawk_abi::store_compensated({target}, {comp}, i, {});",
                f.text(v)
            ));
            return Ok(());
        }
    }
    if *f.text(v) == format!("{target}[i]") {
        return Ok(());
    }
    // a value that lives in another plane is copied element by element
    if f.proxies.iter().any(|x| x.name == *f.text(v)) {
        f.need(Need::Rt);
        f.emit(format!("aether::rt::store({target}[i], {});", f.text(v)));
        return Ok(());
    }
    // a fresh item past the fold threshold is copied by a loop: an unrolled assignment holds a
    // register per element
    if crate::body::linalg::is_fresh(f, f.text(v))
        && f.node(v).numel() > f.unit.build.form.rolled_fold_min as usize
    {
        f.need(Need::Rt);
        f.emit(format!("aether::rt::store({target}[i], {});", f.text(v)))
    } else {
        let value = f.text(v).clone();
        f.emit(format!("{target}[i] = {};", plane_value(f, &value)))
    }
    Ok(())
}

/// An indexed store: the value into the plane element the index list names, under the write's edge rule (a scalar value
/// only). A `zero` edge drops a write outside the plane.
pub(crate) fn indexed_store(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    if !f.node(f.opp(p)[0]).ty.shape.is_empty() {
        return Err(nib("an indexed wide write of a rank > 0 value"));
    }
    let value = f.text(f.opp(p)[0]).clone();
    let indices: Vec<String> = f.opp(p)[1..].iter().map(|&q| f.text(q).clone()).collect();
    let h = f.node(p).h;
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        0,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let (element, test) = element_at(
        &target,
        &indices,
        &text_field(f.ctx(), h, "edge")?,
        door::axes_declared(f.ctx(), h)? > 0,
    )?;
    if element.contains("aether::edge::") {
        f.need(Need::Edge);
    }
    let store = format!("{element} = {};", plane_value(f, &value));
    f.emit(match test {
        Some(test) => format!("if ({test}) {store}"),
        None => store,
    });
    Ok(())
}

/// A thread box lane's own-column store, made only for a lane the thread owns.
pub(crate) fn box_store(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    let (value, lane) = (f.text(f.opp(p)[0]).clone(), f.text(f.opp(p)[1]).clone());
    let target = binding_name(
        &text_field(f.ctx(), f.node(p).h, "role")?,
        0,
        &text_field(f.ctx(), f.node(p).h, "name")?,
    )?;
    f.emit(format!("if ({lane} >= hawk_box_own) {target}[aether::SampleIndex::make(static_cast<std::size_t>({lane}))] = {value};"));
    Ok(())
}

/// The lane's slice of a wide value into the elements `j * 32 + lane` of the plane. A block that serves several lanes puts a
/// vector output through the shared tile its frame holds, which the block then copies out; otherwise the lane stores into the plane
/// itself (every lane holds a value of the output's own shape: lane 0 stores it).
pub(crate) fn warp_store(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    let value = f.opp(p)[0];
    let h = f.node(p).h;
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        0,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let (width, held) = (f.node(p).ty.shape.clone(), f.node(value).ty.shape.clone());
    let text = f.text(value).clone();
    let rows = f.unit.rows;
    let sliced = width.first().is_some_and(|&n| {
        n as i64
            >= f.unit
                .build
                .wide_card
                .as_ref()
                .and_then(|card| card.slice_min)
                .unwrap_or(i64::MAX)
    });
    if rows > 1 && sliced {
        let (n, tile) = (held[0], format!("hawk_stg_{target}"));
        f.emit("{".into());
        f.emit(format!("const aether::Item<Real, {n}> hawk_ws = {text};"));
        if n == width[0] {
            f.emit(format!("if (hawk_wl == 0) {{ for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) {tile}[hawk_lk * {} + hawk_wb] = hawk_ws.data()[hawk_lk]; }}", rows + 1));
        } else {
            f.emit("#pragma unroll".into());
            f.emit(format!("for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) {tile}[(hawk_lk * 32 + hawk_wl) * {} + hawk_wb] = hawk_ws.data()[hawk_lk];", rows + 1));
        }
        f.emit("if (hawk_wl == 0) hawk_wrote[hawk_wb] = 1;".into());
        f.emit("}".into());
        return Ok(());
    }
    if held == width {
        f.emit(format!(
            "if (hawk_wl == 0) {target}[i] = {};",
            plane_value(f, &text)
        ));
        return Ok(());
    }
    let n = held[0];
    f.emit("{".into());
    f.emit(format!("const aether::Item<Real, {n}> hawk_ws = {text};"));
    f.emit("#pragma unroll".into());
    f.emit(format!("for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) {target}[i].at(hawk_lk * 32 + hawk_wl) = hawk_ws.data()[hawk_lk];"));
    f.emit("}".into());
    Ok(())
}

/// The sample's term into its column of the row the wrapper sums.
pub(crate) fn sample_sum(f: &mut Frame, sink: u32, p: usize) -> Result<(), Refusal> {
    let mut column = 0;
    let mut seen = 0;
    for &q in &f.sinks().to_vec() {
        if f.node(f.at(q).unwrap_or(0)).class == Class::SampleSum {
            if q == sink {
                column = seen
            }
            seen += 1
        }
    }
    let value = f.text(f.opp(p)[0]).clone();
    if !f.banded() {
        f.emit(format!("hawk_ss[{column}] = {value};"));
        return Ok(());
    }
    // a banded sum adds the term to the column's accumulator; the other ops take the term as it is
    let term = format!("aether::banded::Band({value})");
    let sink_h = f.node(f.at(sink)?).h;
    if text_field(f.ctx(), sink_h, "op")? == "sum" {
        f.emit(format!("hawk_ss[{column}].addTerm({term});"));
    } else {
        f.emit(format!("hawk_ss[{column}] = {term};"));
    }
    Ok(())
}

/// A scattered accumulate: the target's atomic add, or once per run of lanes when the key moves in
/// runs.
pub(crate) fn accum(f: &mut Frame, sink: u32, _p: usize) -> Result<(), Refusal> {
    accum_of(f, sink)
}

/// The name the accumulate spells its form by: `or` for a bit-or, `compensated` or `plain` for an add by its summation, `auto`
/// otherwise. An add in lane order is lowered by the wings before the writer, so it is refused here.
fn accumulate_contract(f: &mut Frame, h: u32) -> Result<String, Refusal> {
    if text_field(f.ctx(), h, "order")? == "lane" {
        return Err(nib("accumulate under contract ordered"));
    }
    if text_field(f.ctx(), h, "op")? == "or" {
        return Ok("or".into());
    }
    text_field(f.ctx(), h, "summation")
}

/// A scattered accumulate by the sink's handle (a loop body's sink has no slot of its own).
pub(super) fn accum_of(f: &mut Frame, sink: u32) -> Result<(), Refusal> {
    let h = sink;
    let operands = crate::ir::door::operands(f.ctx(), h)?;
    let contract = accumulate_contract(f, h)?;
    if !crate::body::ty_of(f.ctx(), operands[0])?.shape.is_empty() {
        return Err(nib("a scattered accumulate of a rank > 0 value"));
    }
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        0,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let value = f.arg(operands[0])?;
    let mut comp = if crate::ir::door::compensates(f.ctx(), h)? {
        Some(binding_name(
            ZEROED_PLANE,
            0,
            &format!("{}_c", text_field(f.ctx(), h, "name")?),
        )?)
    } else {
        None
    };
    // under the compensated kind the kind's target commits against the kind's correction plane
    if let Some(CompensatedPlan {
        plane,
        target: Some(CompensatedTarget {
            name,
            own_column: false,
        }),
    }) = f.compensated.clone()
    {
        if name == text_field(f.ctx(), h, "name")? {
            comp = Some(binding_name(ZEROED_PLANE, 0, &plane)?)
        }
    }
    // One index under no edge rule or under `clamp` is the flat keyed form (runs, compensation) whatever the plane declares: a
    // view with one axis takes the key as its element, and a clamped key keeps its runs, being a monotone function of it. The
    // other edge rules drop or wrap a key, which a run cannot follow.
    let edge = text_field(f.ctx(), h, "edge")?;
    if operands.len() > 2 || !matches!(edge.as_str(), "unchecked" | "clamp") {
        return accum_at(f, h, &operands, &contract, &target, comp.as_deref(), &value);
    }
    let key = f.arg(operands[1])?;
    let position = if edge == "clamp" {
        f.need(Need::Edge);
        format!("aether::edge::clamp({key}, {target}.extent(0))")
    } else {
        key
    };
    let index = format!("aether::SampleIndex::make(static_cast<std::size_t>({position}))");
    if contract == "or" {
        f.need(Need::AccumOr);
        f.emit(format!("hawk_abi::accum_or({target}, {index}, {value});"));
        return Ok(());
    }
    let c = if comp.is_some() { "Compensated" } else { "" };
    let run = if f.unit.kind.lanes {
        None
    } else {
        let lane_settings = &f.unit.build.lanes;
        let (run_max, block_run) = lane_settings
            .run_max
            .zip(lane_settings.block_run)
            .ok_or_else(|| nib("a key run without the target's lane numbers"))?;
        crate::ir::analysis::key_run(f.ctx(), f.sinks(), sink, run_max, block_run)?
    };
    let line = match (run, &comp) {
        (None, None) => format!("hawk_abi::accum_add({target}, {index}, {value});"),
        (None, Some(cm)) => {
            f.need(Need::AccumCompensated);
            format!("hawk_abi::accum_add_compensated({target}, {cm}, {index}, {value});")
        }
        (Some((form, width, block)), _) => {
            let mut name = match form.as_str() {
                "proven" => format!("add{c}<{width}>"),
                "tested" => format!("addTested{c}"),
                "sorted" => format!("addSorted{c}"),
                fm => return Err(nib(format!("key run form {fm}"))),
            };
            let mut args: Vec<String> = vec![target.clone()];
            args.extend(comp.clone());
            args.push(index);
            args.push(value.clone());
            if form == "proven" {
                args.push("i".into())
            }
            if comp.is_some() && block && form == "proven" && f.block_fold {
                name = "addCompensatedBlock".into();
                args.push("aether::accum::run::blockFull(base, count)".into())
            }
            args.push(format!("hawk_run_{target}"));
            f.need(Need::AccumRun);
            f.need_run(&target);
            if name.starts_with("add") && comp.is_some() {
                f.need(Need::CompensatedRun)
            }
            format!("aether::accum::run::{name}({});", args.join(", "))
        }
    };
    f.emit(line);
    Ok(())
}

/// A scattered accumulate into a plane with axes: the term added at the element the index list names under the write's edge rule
/// (a `zero` edge drops a term outside the plane). A compensated accumulate adds the term's rounding error at the same element of
/// the companion plane `comp`, which has the axes of its target. A bit-or contract commits an or at the element.
fn accum_at(
    f: &mut Frame,
    h: u32,
    operands: &[u32],
    contract: &str,
    target: &str,
    comp: Option<&str>,
    value: &str,
) -> Result<(), Refusal> {
    let mut indices = vec![];
    for &operand in &operands[1..] {
        indices.push(f.arg(operand)?);
    }
    let edge = text_field(f.ctx(), h, "edge")?;
    let (element, test) = element_at(target, &indices, &edge, true)?;
    let add = match comp {
        _ if contract == "or" => {
            f.need(Need::AccumOrAt);
            format!("hawk_abi::accum_or_at({element}, {value});")
        }
        Some(comp) => {
            f.need(Need::AccumCompensatedAt);
            let (correction, _) = element_at(comp, &indices, &edge, true)?;
            format!("hawk_abi::accum_add_compensated_at({element}, {correction}, {value});")
        }
        None => {
            f.need(Need::AccumAt);
            format!("hawk_abi::accum_add_at({element}, {value});")
        }
    };
    if element.contains("aether::edge::") {
        f.need(Need::Edge);
    }
    f.emit(match test {
        Some(test) => format!("if ({test}) {add}"),
        None => add,
    });
    Ok(())
}

/// The finish epilogue, the last text of the guarded scope: an unfinished sample whose condition
/// holds sets its mask and adds one
/// to the reserved counter (FINISHED_PLANE = "finished_count", raptor wings cells.py).
pub(crate) fn finish(f: &mut Frame, sink: u32) -> Result<(), Refusal> {
    let p = f.at(sink)?;
    let name = text_field(f.ctx(), f.node(p).h, "name")?;
    let (mask, cond, counter) = (
        binding_name("terminated", 0, &name)?,
        f.text(f.opp(p)[0]).clone(),
        binding_name("lookup", 0, "finished_count")?,
    );
    f.emit(format!("if (!{mask}[i].eval() && {cond}) {{ {mask}_w[i] = true; hawk_abi::finish_count({counter}); }}"));
    f.mark_last(crate::body::frame::buffer::Mark::FinishCount);
    Ok(())
}

/// The flattened component indices of a shape, row-major.
fn flat_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![vec![]];
    for &extent in shape {
        out = out
            .iter()
            .flat_map(|at| {
                (0..extent).map(move |k| {
                    let mut v = at.clone();
                    v.push(k);
                    v
                })
            })
            .collect();
    }
    out
}

/// One tape row on the own column, under `if (on)`: component `row * width + c` of the sample's
/// plane gets the `c`-th
/// flattened component of the stored values, in order. Rendered where the sink stands.
pub(crate) fn tape_write(f: &mut Frame, p: usize) -> Result<(), Refusal> {
    let h = f.node(p).h;
    tape_write_of(f, h)
}

/// A tape row write by the sink's handle (a loop body's sink has no slot of its own): the row, the
/// switch, then the values.
pub(super) fn tape_write_of(f: &mut Frame, h: u32) -> Result<(), Refusal> {
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        0,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let width = door::field_int(f.ctx(), h, "width")?
        .ok_or_else(|| fault("a tape write holds no width"))?;
    let operands = crate::ir::door::operands(f.ctx(), h)?;
    let (row, on) = (f.arg(operands[0])?, f.arg(operands[1])?);
    let mut lines = vec![];
    let mut offset = 0;
    for &value in &operands[2..] {
        let text = f.arg(value)?;
        let shape = crate::body::ty_of(f.ctx(), value)?.shape;
        let parts: Vec<String> = if shape.is_empty() {
            vec![text]
        } else {
            flat_indices(&shape)
                .iter()
                .map(|at| {
                    format!(
                        "{text}.eval<{}>(i)",
                        at.iter()
                            .map(|k| k.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect()
        };
        for part in parts {
            lines.push(format!("{target}(static_cast<std::size_t>(hawk_tape_row * {width} + {offset}), i.global()) = {part};"));
            offset += 1;
        }
    }
    f.emit(format!("if ({on} != 0) {{"));
    f.buf.indent.push_str("    ");
    f.declare_as(
        Name::new("hawk_tape_row".to_string(), NameKind::Local),
        "std::int64_t".to_string(),
        Some(&row),
    );
    for line in lines {
        f.emit(line)
    }
    f.buf.indent.truncate(f.buf.indent.len() - 4);
    f.emit("}".into());
    Ok(())
}
