//! Functions: a function is written once per unit as a definition (`static inline` on the host, force-inlined `__device__` on the
//! device, where a call frame would cost registers), and a call writes the function's name over the texts of its arguments. A traced body is written by the same frame as a kernel's: it is a walk over one result
//! sink, whose value ends the body as `return <value>;`, and a parameter reads as its own name. A native body is the author's
//! text for the target, placed inside the definition: it sees the parameters the signature declares and nothing else.
use super::{Frame, Out};
use crate::body::frame::needs::Needs;
use crate::contract::types::{FunctionBody, FunctionUnit, Target, Unit};
use crate::ir::door::{self, RxCtx};
use crate::refuse::{nib, Refusal};
use crate::spell;
use crate::unit::device::prologue::QUALIFIER as DEVICE_QUALIFIER;

/// The line number the directive that ends a native text carries until the unit is assembled; [`resume_lines`] puts the real one.
const RESUME_LINE: &str = "HAWK_RESUME_LINE";

/// The unit text with the line directive after each native text numbered: the line after the directive is the line the unit's own
/// numbering gives it.
pub(crate) fn resume_lines(unit: &str) -> String {
    unit.split('\n')
        .enumerate()
        .map(|(at, line)| match line.strip_prefix("#line ") {
            Some(rest) if rest.starts_with(RESUME_LINE) => {
                line.replacen(RESUME_LINE, &(at + 2).to_string(), 1)
            }
            _ => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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
    let body = match &function.body {
        FunctionBody::Traced(traced) => {
            let mut body_unit = unit.clone();
            body_unit.walk = Some(traced.slots.clone());
            body_unit.sinks = vec![traced.sink];
            body_unit.functions = None;
            let rendered =
                super::render_function_body(ctx, traced.walk, &body_unit, "    ", traced.sink)?;
            needs.merge(&rendered.needs);
            rendered.text
        }
        // the text goes in byte for byte between two line directives: a compiler then reads the author's own line numbers under
        // the function's name, and the unit's numbering resumes where the text ends (the unit's assembly fills `RESUME_LINE`)
        FunctionBody::Native(text) => {
            format!(
                "#line 1 \"native function {}\"\n{text}\n#line {RESUME_LINE} __BASE_FILE__",
                function.name
            )
        }
    };
    let qualifier = match unit.target {
        Target::Device => DEVICE_QUALIFIER,
        _ => "static inline",
    };
    Ok(format!(
        "{qualifier} {} {}({})\n{{\n{}\n}}\n\n",
        type_text(&function.result, &function.result_shape, false)?,
        symbol(&function.name, &function.digest),
        params.join(", "),
        body
    ))
}
