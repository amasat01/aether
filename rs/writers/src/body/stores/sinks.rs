//! Statements for sinks: own-column stores, sample sums and scattered accumulates.
use crate::body::frame::needs::Need;
use crate::body::values::text_field;
use crate::body::Class;
use crate::body::{CompensatedPlan, CompensatedTarget, Frame, Name, NameKind};
use crate::ir::door;
use crate::refuse::nib;
use crate::spell::binding_name;

/// One own-column store (early, right after the value, or at the end).
pub(crate) fn store(f: &mut Frame, sink: u32) -> Result<(), String> {
    let p = f.at(sink)?;
    let v = f.opp(p)[0];
    let h = f.node(p).h;
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
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
            let comp = binding_name("mutable", &plane)?;
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
        f.emit(format!("{target}[i] = {};", f.text(v)))
    }
    Ok(())
}

/// A thread box lane's own-column store, made only for a lane the thread owns.
pub(crate) fn box_store(f: &mut Frame, p: usize) -> Result<(), String> {
    let (value, lane) = (f.text(f.opp(p)[0]).clone(), f.text(f.opp(p)[1]).clone());
    let target = binding_name(
        &text_field(f.ctx(), f.node(p).h, "role")?,
        &text_field(f.ctx(), f.node(p).h, "name")?,
    )?;
    f.emit(format!("if ({lane} >= hawk_box_own) {target}[aether::SampleIndex::make(static_cast<std::size_t>({lane}))] = {value};"));
    Ok(())
}

/// The sample's term into its column of the row the wrapper sums.
pub(crate) fn sample_sum(f: &mut Frame, sink: u32, p: usize) -> Result<(), String> {
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
    f.emit(format!("hawk_ss[{column}] = {};", f.text(f.opp(p)[0])));
    Ok(())
}

/// A scattered accumulate: the target's atomic add, or once per run of lanes when the key moves in
/// runs.
pub(crate) fn accum(f: &mut Frame, sink: u32, _p: usize) -> Result<(), String> {
    accum_of(f, sink)
}

/// A scattered accumulate by the sink's handle (a loop body's sink has no slot of its own).
pub(super) fn accum_of(f: &mut Frame, sink: u32) -> Result<(), String> {
    let h = sink;
    let operands = crate::ir::door::operands(f.ctx(), h)?;
    let contract = text_field(f.ctx(), h, "contract")?;
    if !matches!(
        contract.as_str(),
        "auto" | "plain" | "or" | "compensated" | "keyed_auto"
    ) {
        return Err(nib(format!("accumulate under contract {contract}")));
    }
    if !crate::body::ty_of(f.ctx(), operands[0])?.shape.is_empty() {
        return Err(nib("a scattered accumulate of a rank > 0 value"));
    }
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let (value, index) = (
        f.arg(operands[0])?,
        format!(
            "aether::SampleIndex::make(static_cast<std::size_t>({}))",
            f.arg(operands[1])?
        ),
    );
    if contract == "or" {
        f.need(Need::AccumOr);
        f.emit(format!("hawk_abi::accum_or({target}, {index}, {value});"));
        return Ok(());
    }
    let mut comp = if crate::ir::door::compensates(&contract)? {
        Some(binding_name(
            "accum_out",
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
            comp = Some(binding_name("accum_out", &plane)?)
        }
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

/// The finish epilogue, the last text of the guarded scope: an unfinished sample whose condition
/// holds sets its mask and adds one
/// to the reserved counter (FINISHED_PLANE = "finished_count", raptor wings cells.py).
pub(crate) fn finish(f: &mut Frame, sink: u32) -> Result<(), String> {
    let p = f.at(sink)?;
    let name = text_field(f.ctx(), f.node(p).h, "name")?;
    let (mask, cond, counter) = (
        binding_name("terminated", &name)?,
        f.text(f.opp(p)[0]).clone(),
        binding_name("lookup", "finished_count")?,
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
pub(crate) fn tape_write(f: &mut Frame, p: usize) -> Result<(), String> {
    let h = f.node(p).h;
    tape_write_of(f, h)
}

/// A tape row write by the sink's handle (a loop body's sink has no slot of its own): the row, the
/// switch, then the values.
pub(super) fn tape_write_of(f: &mut Frame, h: u32) -> Result<(), String> {
    let target = binding_name(
        &text_field(f.ctx(), h, "role")?,
        &text_field(f.ctx(), h, "name")?,
    )?;
    let width = door::field_int(f.ctx(), h, "width")?.ok_or("a tape write holds no width")?;
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
