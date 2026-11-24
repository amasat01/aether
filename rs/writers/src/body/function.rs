//! Functions: a traced function is written once per unit as a `static inline` definition, and a call writes the function's
//! name over the texts of its arguments. The body is written by the same frame as a kernel's: it is a walk over one result
//! sink, whose value ends the body as `return <value>;`, and a parameter reads as its own name.
use super::{Frame, Out};
use crate::body::frame::needs::Needs;
use crate::contract::types::{FunctionUnit, Target, Unit};
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, not_written, Refusal};
use crate::spell;

/// The name a function's definition and calls carry in the text: the function's name, made unique among functions of one
/// name by the start of its digest.
fn symbol(name: &str, digest: &str) -> String {
    format!("hawk_fn_{name}_{}", &digest[..digest.len().min(8)])
}

/// The `(name, digest)` a call node names its function by.
fn callee(ctx: *mut RxCtx, call: u32) -> Result<(String, String), Refusal> {
    match door::field_names(ctx, call, "function")?.as_deref() {
        Some([name, digest]) => Ok((name.clone(), digest.clone())),
        other => Err(nib(format!("a call that names its function as {other:?}"))),
    }
}

/// The text of a parameter: its name.
pub(super) fn param(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    Ok(Out::Value(
        super::values::text_field(f.ctx(), f.node(p).h, "name")?,
        None,
    ))
}

/// The text of a call in a kernel body: the function's symbol over the argument texts.
pub(super) fn call(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    if f.unit.target == Target::Device {
        return Err(not_written(format!(
            "{:?}: a call of a function is not written on the device yet",
            f.unit.name
        )));
    }
    let (name, digest) = callee(f.ctx(), f.node(p).h)?;
    let args: Vec<String> = f.opp(p).iter().map(|&q| f.text(q).clone()).collect();
    Ok(Out::Value(
        format!("{}({})", symbol(&name, &digest), args.join(", ")),
        None,
    ))
}

/// The definitions of the unit's functions, each after the functions it calls, with the helpers their text calls for.
pub(crate) fn definitions(ctx: *mut RxCtx, unit: &Unit) -> Result<(String, Needs), Refusal> {
    let mut needs = Needs::default();
    let mut text = String::new();
    if unit.target == Target::Device && unit.functions.is_some() {
        return Err(not_written(format!(
            "{:?}: a function is not written on the device yet",
            unit.name
        )));
    }
    for function in unit.functions.iter().flatten() {
        text += &definition(ctx, unit, function, &mut needs)?;
    }
    Ok((text, needs))
}

/// The C++ type a parameter or a result of `dtype` and `shape` is spelled with: a scalar by value, anything else as the item
/// the frame holds it in, by constant reference when it is a parameter.
fn type_text(dtype: &str, shape: &[i64], parameter: bool) -> Result<String, Refusal> {
    let extents: Vec<usize> = shape.iter().map(|&e| e as usize).collect();
    let held = spell::carry_type(dtype, &extents)?;
    Ok(match (shape.is_empty(), parameter) {
        (true, true) => format!("const {held}"),
        (false, true) => format!("const {held}&"),
        (_, false) => held,
    })
}

/// One function's definition: its signature, and the body the frame writes over the function's own walk.
fn definition(
    ctx: *mut RxCtx,
    unit: &Unit,
    function: &FunctionUnit,
    needs: &mut Needs,
) -> Result<String, Refusal> {
    let params = function
        .params
        .iter()
        .map(|p| {
            Ok(format!(
                "{} {}",
                type_text(&p.dtype, &p.shape, true)?,
                p.name
            ))
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let mut body_unit = unit.clone();
    body_unit.walk = Some(function.slots.clone());
    body_unit.sinks = vec![function.sink];
    body_unit.functions = None;
    let rendered =
        super::render_function_body(ctx, function.walk, &body_unit, "    ", function.sink)?;
    needs.merge(&rendered.needs);
    Ok(format!(
        "static inline {} {}({})\n{{\n{}\n}}\n\n",
        type_text(&function.result, &function.result_shape, false)?,
        symbol(&function.name, &function.digest),
        params.join(", "),
        rendered.text
    ))
}
