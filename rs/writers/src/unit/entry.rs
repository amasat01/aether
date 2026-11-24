//! The one entry of the crate: the source text of a host or device unit from a walk of the IR core
//! and the build contract.
//! The entry reads the contract once into a typed `Unit`, owns what one render adds to it (the body
//! and whether it draws
//! random numbers), and hands the recorded contract on to the frame unchanged.
use crate::body::frame::needs::{Need, Needs, Rendered};
use crate::contract::types::{BodySource, Form, Target, Unit};
use crate::contract::Contract;
use crate::ir::door::RxCtx;
use crate::refuse::{fault, unknown, Refusal};
use crate::unit::{CompanionFact, Facts, Meta, StepFacts};

/// The result of a render: the text, the body inside it, the entry name and the rest (target, mode,
/// vectorised, box) as JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct HostSource {
    /// The unit's whole translation-unit text.
    pub text: String,
    /// The body inside the text.
    pub body: String,
    /// The entry function's name.
    pub entry: String,
    /// The unit's facts: target, mode, vectorised and the thread box (`Meta::json` writes them as JSON).
    pub meta: Meta,
    /// What the writer decided about the unit that its caller states.
    pub facts: Facts,
    /// The shadow check of the lane text's recorded needs: none without a lane text, else what
    /// differs from its text.
    #[cfg(feature = "shadow")]
    pub needs_diffs: Option<Vec<String>>,
}

/// What one render owns besides the unit: the body it wrote and whether that body draws random
/// numbers.
pub struct Render<'a> {
    /// The typed description of the unit.
    pub unit: &'a Unit,
    /// The body inside the unit's text.
    pub body: String,
    /// Whether the body draws random numbers.
    pub needs_random: bool,
    /// What the body needs from the unit: recorded while the walk wrote it, else read from its text
    /// (a body the caller gave).
    pub needs: Needs,
}

/// A request read once: the typed unit and the body a caller supplied, if any.
struct Prepared {
    /// The walk handle the request loaded.
    walk: u32,
    /// The typed unit of the request.
    unit: Unit,
    /// The body a caller supplied and whether it draws random numbers, if any.
    supplied: Option<(String, bool)>,
}

/// The body of the unit and whether it draws random numbers: the caller's when it gave one, else
/// written here
/// (a team stage's body is the stage's, not the walk's).
/// The needs recorded while the walk's own render wrote it come along; a body the caller gave, and
/// one composed from several
/// renders, has none recorded.
fn body_of(ctx: *mut RxCtx, prepared: &Prepared) -> Result<(String, bool, Option<Needs>), Refusal> {
    if let Some(supplied) = &prepared.supplied {
        return Ok((supplied.0.clone(), supplied.1, None));
    }
    let (walk, unit) = (prepared.walk, &prepared.unit);
    let rendered = if let Some(group) = crate::unit::group::of(ctx, walk, unit)? {
        crate::unit::group::group_body(ctx, walk, unit, group)?
    } else if let Form::TeamStage(stage) = &unit.form {
        crate::unit::team_stage::stage_body(ctx, unit, stage)?
    } else if let Form::Fused(_) = &unit.form {
        crate::unit::fused::fused_body(ctx, walk, unit, "        ")?
    } else if let Some(boxed) = crate::unit::boxed::boxed_body(ctx, walk, unit)? {
        boxed
    } else if let Some(parts) = crate::unit::two_path::two_path_body(ctx, walk, unit)? {
        Rendered {
            text: parts.joined(),
            needs: parts.needs,
        }
    } else {
        crate::body::render_body_with_needs(ctx, walk, unit, "        ")?
    };
    let random = rendered.needs.has(Need::Random);
    Ok((rendered.text, random, Some(rendered.needs)))
}

/// The target a binding names in text: `cuda` is the device, anything else the host.
fn target_named(text: &str) -> Target {
    if text == "cuda" {
        Target::Device
    } else {
        Target::Host
    }
}

/// The unit a request describes: its typed JSON (it carries `"v"`), or, in a build with the
/// `old-contract` feature, the old shape.
fn read_unit(json: &str, target: Target) -> Result<Unit, Refusal> {
    let parsed = Contract::parse(json)?;
    if parsed.get("v").is_none() {
        return old_shape(&parsed, target);
    }
    let unit = Unit::from_value(&parsed.0)?;
    if unit.target != target {
        return Err(unknown("a unit for the other target"));
    }
    Ok(unit)
}

/// The unit an old-shape request describes.
#[cfg(feature = "old-contract")]
fn old_shape(contract: &Contract, target: Target) -> Result<Unit, Refusal> {
    crate::contract::old::typed_of(contract, target)
}

/// Without the `old-contract` feature the old shape is refused by name.
#[cfg(not(feature = "old-contract"))]
fn old_shape(_contract: &Contract, _target: Target) -> Result<Unit, Refusal> {
    Err(unknown(
        "an old-shape contract (a typed contract carries \"v\")",
    ))
}

/// Reads the request once and takes the unit with its least-squares nodes lowered first, as the old
/// flow does before it
/// renders: the walk taken again over the lowered sinks (with the unit's declared slots) and the
/// unit and the contract
/// naming them. Nothing changes without such a node.
fn lowered(
    ctx: *mut RxCtx,
    walk: u32,
    contract_json: &str,
    target: Target,
) -> Result<Prepared, Refusal> {
    let mut unit = read_unit(contract_json, target)?;
    let supplied = match &unit.body {
        BodySource::Walk => None,
        BodySource::Given(given) => Some((given.text.clone(), given.draws_random)),
    };
    let unchanged = |walk, unit| Prepared {
        walk,
        unit,
        supplied: supplied.clone(),
    };
    if walk == u32::MAX {
        return Ok(unchanged(walk, unit));
    }
    let max_dim = unit.build.linalg_max_dim;
    let Some(new_sinks) = crate::ir::analysis::lowered_sinks(ctx, &unit.sinks, max_dim)? else {
        return Ok(unchanged(walk, unit));
    };
    let walk = crate::ir::door::canonical_over(ctx, &new_sinks, walk)?;
    unit.sinks = new_sinks;
    Ok(unchanged(walk, unit))
}

/// The body the entry returns for a unit (boxed, two-path, team stage and plain alike), without the
/// unit's text around it.
pub fn entry_body(
    ctx: *mut RxCtx,
    walk: u32,
    contract_json: &str,
    target: &str,
) -> Result<String, Refusal> {
    let prepared = lowered(ctx, walk, contract_json, target_named(target))?;
    Ok(body_of(ctx, &prepared)?.0)
}

/// The text of a unit from what `lowered` read, for the arm `render` writes.
fn render_with(
    ctx: *mut RxCtx,
    prepared: Prepared,
    render: fn(*mut RxCtx, u32, &Render) -> Result<HostSource, Refusal>,
) -> Result<HostSource, Refusal> {
    let (body, needs_random, recorded) = body_of(ctx, &prepared)?;
    let needs = recorded.unwrap_or_else(|| Needs::scan(&body));
    let walk = prepared.walk;
    let render_input = Render {
        unit: &prepared.unit,
        body,
        needs_random,
        needs,
    };
    let mut source = render(ctx, walk, &render_input)?;
    source.facts = facts_of(ctx, walk, &prepared.unit, &source)?;
    Ok(source)
}

/// What the writer decided about a rendered unit: the exported symbol (a host entry carries the host
/// writer's suffix, a device kernel is its name), the row planes of its sample sums and its
/// companion planes.
fn facts_of(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    source: &HostSource,
) -> Result<Facts, Refusal> {
    let host = source.meta.target == Target::Host;
    let slots = &unit
        .walk
        .as_ref()
        .ok_or_else(|| fault("a unit without a walk"))?
        .slots;
    Ok(Facts {
        symbol: if host {
            crate::unit::host::render::exported(&source.entry)
        } else {
            source.entry.clone()
        },
        row_planes: crate::unit::sums::row_planes(ctx, &unit.sinks, slots)?,
        companions: crate::ir::analysis::companions(ctx, walk)?
            .into_iter()
            .map(|(plane, target)| CompanionFact {
                fold_entry: if host {
                    crate::unit::host::render::fold_entry(&source.entry, &target)
                } else {
                    crate::unit::device::render::fold_entry(&source.entry, &target)
                },
                plane,
                target,
            })
            .collect(),
        step: step_of(ctx, walk, unit)?,
        mix_planes: mix_planes(ctx, walk)?,
        // the device render knows whether it cut the fast entries and names them
        persist_function: source.facts.persist_function.clone(),
        range_function: source.facts.range_function.clone(),
    })
}

/// The branch-mix plane of each auto dispatch group of the walk, by the group's name (`dispatch<n>`, n its place among the walk's
/// dispatches).
fn mix_planes(ctx: *mut RxCtx, walk: u32) -> Result<Vec<(String, String)>, Refusal> {
    let mut out = vec![];
    for (index, &node) in crate::ir::door::dispatch_nodes(ctx, walk)?
        .iter()
        .enumerate()
    {
        if crate::ir::door::is_a(ctx, node, "auto_dispatch")? {
            let group = format!("dispatch{index}");
            out.push((group.clone(), crate::ir::runtime_names::mix_plane(&group)));
        }
    }
    Ok(out)
}

/// The step of a finishing unit, from the lane split of its body; none when the unit does not
/// finish or its body is not that shape.
fn step_of(ctx: *mut RxCtx, walk: u32, unit: &Unit) -> Result<Option<StepFacts>, Refusal> {
    let finishes = unit.walk.as_ref().is_some_and(|w| w.finish.is_some());
    if !finishes {
        return Ok(None);
    }
    Ok(
        crate::lanes::lane_split(ctx, walk, unit, "        ")?.map(|(split, _)| StepFacts {
            ops: split.op_count as i64,
            entry_counts: split.guard.is_some(),
        }),
    )
}

/// The host unit of a walk and a build contract.
pub fn host_render(ctx: *mut RxCtx, walk: u32, contract_json: &str) -> Result<HostSource, Refusal> {
    let prepared = lowered(ctx, walk, contract_json, Target::Host)?;
    render_with(ctx, prepared, crate::unit::host::render::render)
}

/// The device unit of a walk and a build contract.
pub fn cuda_render(ctx: *mut RxCtx, walk: u32, contract_json: &str) -> Result<HostSource, Refusal> {
    let prepared = lowered(ctx, walk, contract_json, Target::Device)?;
    render_with(ctx, prepared, crate::unit::device::render::render)
}

/// The refusal the typed description gives for a request, none when it converts: what the replay
/// tool counts. `target` is `cuda` for a device unit and anything else for a host unit.
#[cfg(feature = "old-contract")]
pub fn typed_refusal(contract_json: &str, target: &str) -> Result<Option<String>, String> {
    let recorded = Contract::parse(contract_json).map_err(|refusal| refusal.reason)?;
    Ok(
        crate::contract::old::typed_of(&recorded, target_named(target))
            .err()
            .map(|refusal| refusal.reason),
    )
}

/// The shadow check of the recorded needs: for a body the walk's own render wrote, the differences
/// between what was recorded while
/// writing it and what its text shows by substring (empty when they agree); none for a body that
/// has no records.
#[cfg(feature = "shadow")]
pub fn needs_check(
    ctx: *mut RxCtx,
    walk: u32,
    contract_json: &str,
    target: &str,
) -> Result<Option<Vec<String>>, Refusal> {
    let target = if target == "device" {
        Target::Device
    } else {
        Target::Host
    };
    let prepared = lowered(ctx, walk, contract_json, target)?;
    let (body, _, recorded) = body_of(ctx, &prepared)?;
    Ok(recorded.map(|recorded| {
        let shown = Needs::scan(&body);
        let mut out: Vec<String> = vec![];
        for need in recorded.set.symmetric_difference(&shown.set) {
            let side = if recorded.has(*need) {
                "recorded, not in the text"
            } else {
                "in the text, not recorded"
            };
            out.push(format!("{need:?}: {side}"))
        }
        let states =
            |needs: &Needs| -> Vec<String> { needs.runs.iter().map(|r| r.state.clone()).collect() };
        if states(&recorded) != states(&shown) {
            out.push(format!(
                "runs: recorded {:?}, text {:?}",
                states(&recorded),
                states(&shown)
            ))
        }
        out
    }))
}

/// The refusal the typed wire gives for a request, none when the typed unit survives its own JSON
/// text unchanged: what the replay tool counts as a round trip.
#[cfg(feature = "old-contract")]
pub fn wire_refusal(contract_json: &str, target: &str) -> Result<Option<String>, String> {
    let recorded = Contract::parse(contract_json).map_err(|refusal| refusal.reason)?;
    let unit = crate::contract::old::typed_of(&recorded, target_named(target))
        .map_err(|refusal| refusal.reason)?;
    Ok(match Unit::from_json(&unit.to_json()) {
        Ok(again) if again == unit => None,
        Ok(_) => Some("the unit changed on the wire".to_string()),
        Err(why) => Some(why.reason),
    })
}
