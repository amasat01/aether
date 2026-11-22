//! The one place that knows the recorded shape of a render request: tagged records, module
//! references, the two call forms, wrapped frozen maps. `typed_of` turns it into a `Unit`; a value
//! it does not know is refused by name, never given a default.
use super::types::*;
use super::{Contract, Json};
use crate::refuse::{bad, unknown, Refusal};

/// The request as recorded, in either call form.
struct Old<'a> {
    /// The recorded request.
    top: &'a Json,
    /// The keyword arguments of the source form, none for the backend form.
    kwargs: Option<&'a Json>,
}

/// The value unless it is absent or null.
fn live(j: Option<&Json>) -> Option<&Json> {
    match j {
        None | Some(Json::Null) => None,
        other => other,
    }
}

/// The wrapped value of a frozen map, or the value itself when it is not wrapped.
fn of(j: &Json) -> &Json {
    j.get("of").unwrap_or(j)
}

/// The text of a string value, empty when it is not one.
fn text(j: Option<&Json>) -> &str {
    j.and_then(Json::str).unwrap_or("")
}

/// The integer under `key`.
fn int_at(j: &Json, key: &str) -> Option<i64> {
    j.get(key).and_then(Json::int)
}

/// The node handles of a list of node references, in order (a wrapped list is unwrapped).
fn nodes(list: Option<&Json>) -> Vec<u32> {
    let items = list.map_or(&[][..], |l| of(l).arr());
    let handle = |node: &Json| node.get("$node").and_then(Json::int);
    items
        .iter()
        .filter_map(handle)
        .filter(|n| *n >= 0)
        .map(|n| n as u32)
        .collect()
}

/// The first number the key holds anywhere in a recorded value.
fn find_number(j: &Json, key: &str) -> Option<i64> {
    match j {
        Json::Obj(items) => items.iter().find_map(|(k, v)| {
            if k == key {
                v.int()
            } else {
                find_number(v, key)
            }
        }),
        Json::Arr(items) => items.iter().find_map(|v| find_number(v, key)),
        _ => None,
    }
}

/// The typed description of a recorded request, or the refusal that names what is not known.
pub fn typed_of(contract: &Contract, target: Target) -> Result<Unit, Refusal> {
    let top = &contract.0;
    let source = live(top.get("args")).is_some();
    let old = Old {
        top,
        kwargs: if source { top.get("kwargs") } else { None },
    };
    let kernel = if source { None } else { top.get("kernel") };
    if !source && kernel.is_none() {
        return Err(unknown("an unknown call form"));
    }
    let form = old.form(kernel)?;
    let active = old.active(kernel)?;
    let spec = old.spec().ok_or_else(|| bad("the call carries no spec"))?;
    Ok(Unit {
        name: old.name(),
        target,
        scalar: old.scalar()?,
        kind: kind_of(old.arg("kind"))?,
        sinks: old.sinks(kernel),
        walk: old.walk(kernel)?,
        body: given_body(top).map_or(BodySource::Walk, BodySource::Given),
        active,
        segment: old.segment(),
        thread_box: if source { None } else { old.thread_box() },
        one_step: old.one_step()?,
        rows: old.arg("rows").and_then(Json::int).unwrap_or(DEFAULT_ROWS),
        team: old.team(kernel, spec)?,
        build: build_of(spec, &target, source)?,
        machine: machine_of(spec, top.get("facts"))?,
        exports: old.exports()?,
        literal_table: old.literal_table(),
        form,
    })
}

impl<'a> Old<'a> {
    /// The named argument of the call, absent when left out or null.
    fn arg(&self, key: &str) -> Option<&'a Json> {
        match self.kwargs {
            Some(kwargs) => live(kwargs.get(key)),
            None => live(self.top.get(key)),
        }
    }

    /// The build spec the call carries.
    fn spec(&self) -> Option<&'a Json> {
        self.arg("spec").or_else(|| self.top.get("spec"))
    }

    /// The unit's name: the first positional argument of the source form.
    fn name(&self) -> String {
        match self.kwargs {
            Some(_) => text(self.top.get("args").and_then(|a| a.arr().first())),
            None => text(self.top.get("name")),
        }
        .to_string()
    }

    /// The scalar mode; an omitted mode is `Float64`.
    fn scalar(&self) -> Result<Scalar, Refusal> {
        let mode = match self.kwargs {
            Some(kwargs) => live(kwargs.get("mode")),
            None => live(self.top.get("smode")),
        };
        let Some(mode) = mode else {
            return Ok(Scalar::Float64);
        };
        match text(mode.get("$ref")).rsplit(':').next().unwrap_or("") {
            "FLOAT64" => Ok(Scalar::Float64),
            "FLOAT32" => Ok(Scalar::Float32),
            "BANDED" => Ok(Scalar::Banded),
            other => Err(unknown(format!("scalar mode {other:?}"))),
        }
    }

    /// The node handles the unit writes: the kernel's sinks, or the second positional argument.
    fn sinks(&self, kernel: Option<&Json>) -> Vec<u32> {
        match kernel {
            Some(kernel) => nodes(kernel.get("sinks")),
            None => nodes(self.top.get("args").and_then(|a| a.arr().get(1))),
        }
    }

    /// The unit's own arguments, when the record carries a walk.
    fn walk(&self, kernel: Option<&Json>) -> Result<Option<Walk>, Refusal> {
        let walk = match kernel {
            Some(kernel) => kernel.get("walk"),
            None => self.top.get("args").and_then(|a| a.arr().get(2)),
        };
        walk.map(walk_of).transpose()
    }

    /// What the body is, from the kernel record and the call.
    fn form(&self, kernel: Option<&Json>) -> Result<Form, Refusal> {
        let Some(kernel) = kernel else {
            return Ok(Form::Plain);
        };
        let record = text(kernel.get("$type"));
        let team = live(kernel.get("team"));
        let active = live(self.top.get("active")).is_some();
        match (record, team) {
            ("TeamStage", Some(team)) => {
                Ok(Form::TeamStage(stage_of(team, self.active_record()?)?))
            }
            ("Kernel", Some(team)) if active => {
                Ok(Form::TeamStage(stage_of(team, self.active_record()?)?))
            }
            ("FusedKernel", _) => fused_of(kernel),
            ("WarpStage", _) => Ok(Form::WarpStage),
            ("Kernel" | "TeamStage" | "Chain" | "SegmentUnit" | "ClassifyUnit", _) => {
                Ok(Form::Plain)
            }
            (other, _) => Err(unknown(format!("kernel record {other}"))),
        }
    }

    /// The active set the launch takes. A launch over keys, chunks or targets takes none; a team
    /// stage
    /// over one of them still reads its own map (see `TeamStage::active`).
    fn active(&self, kernel: Option<&Json>) -> Result<Option<Active>, Refusal> {
        let grid = text(
            kernel
                .and_then(|k| k.get("team"))
                .and_then(|t| t.get("grid")),
        );
        let ignored = matches!(grid, "keys" | "chunks" | "targets");
        if ignored {
            return Ok(None);
        }
        self.active_record()
    }

    /// The active-set record of the call, whatever the launch shape.
    fn active_record(&self) -> Result<Option<Active>, Refusal> {
        let Some(spec) = self.arg("active") else {
            return Ok(None);
        };
        let map_bits = match int_at(spec, "map_bits") {
            None | Some(32) => MapBits::B32,
            Some(64) => MapBits::B64,
            Some(other) => return Err(unknown(format!("an index map {other} bits wide"))),
        };
        let Some(count_binding) = spec.get("count_binding").and_then(Json::str) else {
            return Err(unknown("an index map record"));
        };
        Ok(Some(Active {
            map_binding: live(spec.get("map_binding"))
                .and_then(Json::str)
                .map(String::from),
            count_binding: count_binding.to_string(),
            group: int_at(spec, "group").unwrap_or(DEFAULT_GROUP),
            map_bits,
            lane_set: matches!(spec.get("lane_set"), Some(Json::Bool(true))),
        }))
    }

    /// The run segment, when the call has one.
    fn segment(&self) -> Option<Segment> {
        let segment = self.arg("segment")?;
        Some(Segment {
            offsets_binding: text(segment.get("offsets_binding")).to_string(),
            j: int_at(segment, "j").unwrap_or(DEFAULT_RUN_START),
        })
    }

    /// The thread box, when the call has one.
    fn thread_box(&self) -> Option<ThreadBox> {
        let record = self.arg("box")?;
        Some(ThreadBox {
            axis: int_at(record, "axis"),
            t: int_at(record, "T").unwrap_or(DEFAULT_THREADS),
            stride: int_at(record, "stride").unwrap_or(DEFAULT_STRIDE),
        })
    }

    /// The single-step variant, when the call has one.
    fn one_step(&self) -> Result<Option<OneStep>, Refusal> {
        let Some(step) = self.arg("one_step") else {
            return Ok(None);
        };
        Ok(Some(OneStep {
            sinks: nodes(step.get("sinks")),
            walk: step.get("walk").map(walk_of).transpose()?,
        }))
    }

    /// The team of the unit: the kernel's team record, or the source form's thread count.
    fn team(&self, kernel: Option<&Json>, spec: &Json) -> Result<Option<TeamShape>, Refusal> {
        let settings_warps = spec.get("settings").and_then(|s| int_at(s, "warps"));
        if self.kwargs.is_some() {
            let Some(given) = self.arg("team") else {
                return Ok(None);
            };
            let threads = given
                .int()
                .ok_or_else(|| unknown("a team of another form"))?;
            return Ok(Some(TeamShape {
                threads,
                warps: settings_warps.unwrap_or(DEFAULT_WARPS),
            }));
        }
        let Some(team) = live(kernel.and_then(|k| k.get("team"))) else {
            return Ok(None);
        };
        Ok(Some(TeamShape {
            threads: int_at(team, "threads").unwrap_or(DEFAULT_THREADS),
            warps: int_at(team, "warps").unwrap_or(DEFAULT_WARPS),
        }))
    }

    /// Whether the build asks for the table of literals after the device text.
    fn literal_table(&self) -> bool {
        let options = self.top.get("options");
        let asked = options.and_then(|o| o.get("cuda.literal_table"));
        matches!(asked, Some(Json::Bool(true)))
    }

    /// The export block: finished text in the source form, layout sizes in the backend form.
    fn exports(&self) -> Result<Exports, Refusal> {
        if let Some(kwargs) = self.kwargs {
            return Ok(Exports::Text(text(kwargs.get("exports")).to_string()));
        }
        let Some(sizes) = live(
            self.top
                .get("options")
                .and_then(|o| o.get("layout_sizes_override")),
        ) else {
            return Ok(Exports::Derived);
        };
        let mut values = [0u64; 5];
        if sizes.arr().len() != values.len() {
            return Err(bad("layout_sizes_override= takes 5 sizes"));
        }
        for (slot, size) in values.iter_mut().zip(sizes.arr()) {
            let size = size.int().and_then(|n| u64::try_from(n).ok());
            *slot = size.ok_or_else(|| bad("layout size"))?;
        }
        Ok(Exports::Sizes(values))
    }
}

/// The module reference a recorded value names, if it is one.
fn reference(j: Option<&Json>) -> Option<&str> {
    j.and_then(|j| j.get("$ref")).and_then(Json::str)
}

/// The kind of unit from its record or its default reference.
fn kind_of(kind: Option<&Json>) -> Result<Kind, Refusal> {
    let Some(kind) = kind else {
        return Err(unknown("kernel kind (absent)"));
    };
    if let Some(path) = reference(Some(kind)) {
        return match path {
            "hawk.ext:DEFAULT_KIND" => Ok(Kind {
                guard: Guard::Default,
                lanes: false,
                sink: SinkPolicy::Plain,
            }),
            other => Err(unknown(format!("kernel kind {other}"))),
        };
    }
    if text(kind.get("$type")) != "Kind" {
        return Err(unknown(format!("kernel kind {}", text(kind.get("$type")))));
    }
    Ok(Kind {
        guard: guard_of(kind.get("guard"))?,
        lanes: live(kind.get("lanes")).is_some(),
        sink: sink_of(kind.get("sink"), kind.get("output"))?,
    })
}

/// The sink policy of a kind record.
fn sink_of(sink: Option<&Json>, output: Option<&Json>) -> Result<SinkPolicy, Refusal> {
    let other = || unknown("kind with a sink policy of another form");
    let sink = sink.ok_or_else(other)?;
    if sink.get("$ref").and_then(Json::str) == Some("hawk.ext:PLAIN") {
        return Ok(SinkPolicy::Plain);
    }
    let compensated =
        text(sink.get("$type")) == "SinkPolicy" && text(sink.get("id")) == "compensated";
    match sink.get("compensation").and_then(Json::str) {
        Some(plane) if compensated => Ok(compensated_sink(plane, output)),
        _ => Err(other()),
    }
}

/// The compensated sink policy with the kind's declared output (the record may wrap it in `v`).
fn compensated_sink(plane: &str, output: Option<&Json>) -> SinkPolicy {
    let output = output.map(|o| o.get("v").unwrap_or(o));
    let names = output.and_then(|o| o.get("names"));
    let listed = names.map_or(&[][..], |n| n.get("t").unwrap_or(n).arr());
    SinkPolicy::Compensated {
        plane: plane.to_string(),
        output_slot: output
            .and_then(|o| o.get("slot"))
            .and_then(Json::str)
            .map(String::from),
        output_names: listed
            .iter()
            .filter_map(|x| x.str().map(String::from))
            .collect(),
    }
}

/// The guard of a kind record: a named constant or a record that lists masks.
fn guard_of(guard: Option<&Json>) -> Result<Guard, Refusal> {
    let other = || unknown("kind with a guard of another form");
    let guard = guard.ok_or_else(other)?;
    match guard.get("$ref").and_then(Json::str) {
        Some("hawk.ext:DEFAULT_GUARD") => return Ok(Guard::Default),
        Some("hawk.ext:DATA_ONLY") => return Ok(Guard::DataOnly),
        Some(_) => return Err(other()),
        None => {}
    }
    if text(guard.get("$type")) != "Guard" {
        return Err(other());
    }
    let masks: Vec<String> = guard
        .get("masks")
        .map_or(&[][..], |m| m.arr())
        .iter()
        .filter_map(|mask| mask.str().map(String::from))
        .collect();
    if !masks.is_empty() {
        return Ok(Guard::Masks(masks));
    }
    Ok(Guard::Masks(
        guard
            .get("mask")
            .and_then(Json::str)
            .map(|m| vec![m.to_string()])
            .unwrap_or_default(),
    ))
}

/// The members of a fused kernel record.
fn fused_of(kernel: &Json) -> Result<Form, Refusal> {
    let mut lanes = vec![];
    for member in kernel.get("members").map_or(&[][..], |m| m.arr()) {
        let name = member.get("name").and_then(Json::str);
        if !text(member.get("prologue")).is_empty() {
            return Err(unknown("a fused lane with a prologue text"));
        }
        lanes.push(FusedLane {
            name: name
                .ok_or_else(|| unknown("a fused lane without a name"))?
                .to_string(),
            sinks: nodes(member.get("sinks")),
        });
    }
    Ok(Form::Fused(lanes))
}

/// The team stage a team record describes.
fn stage_of(team: &Json, active: Option<Active>) -> Result<TeamStage, Refusal> {
    let grid = match text(team.get("grid")) {
        "lanes" => Grid::Lanes,
        "keys" => Grid::Keys,
        "chunks" => Grid::Chunks,
        "targets" => Grid::Targets,
        other => return Err(unknown(format!("team grid {other:?}"))),
    };
    let mut planes = vec![];
    for plane in team.get("planes").map_or(&[][..], |p| p.arr()) {
        match (
            plane.get("name").and_then(Json::str),
            int_at(plane, "per_lane"),
        ) {
            (Some(name), Some(width)) => planes.push((name.to_string(), width)),
            _ => return Err(unknown("a team plane record")),
        }
    }
    let mut reads = vec![];
    for leaf in team.get("reads").map_or(&[][..], |r| r.arr()) {
        match int_at(leaf, "$node") {
            Some(node) if node >= 0 => reads.push(node as u32),
            _ => return Err(unknown("a team read record")),
        }
    }
    let steps = team.get("steps").map_or(&[][..], |s| s.arr());
    let steps = steps.iter().map(step_of).collect::<Result<_, _>>()?;
    Ok(TeamStage {
        grid,
        planes,
        reads,
        steps,
        active,
    })
}

/// One step record of a team stage.
fn step_of(step: &Json) -> Result<Step, Refusal> {
    let number =
        |key: &str| int_at(step, key).ok_or_else(|| unknown(format!("a team step without {key}")));
    let mut planes = vec![];
    for pair in step.get("planes").map_or(&[][..], |p| p.arr()) {
        match (
            pair.arr().first().and_then(Json::str),
            pair.arr().get(1).and_then(Json::str),
        ) {
            (Some(role), Some(name)) => planes.push((role.to_string(), name.to_string())),
            _ => return Err(unknown("a team step plane")),
        }
    }
    Ok(Step {
        op: step_op(
            step.get("op")
                .and_then(Json::str)
                .ok_or_else(|| unknown("a team step without op"))?,
        )?,
        n: number("n")?,
        k: number("k")?,
        m: number("m")?,
        max_chunks: number("max_chunks")?,
        planes,
    })
}

/// The operation a team step names.
fn step_op(name: &str) -> Result<StepOp, Refusal> {
    StepOp::named(name).ok_or_else(|| unknown(format!("team step {name}")))
}

/// The `(role, name)` pair a slot key records.
fn pair_key(key: &Json) -> (&str, &str) {
    (text(key.arr().first()), text(key.arr().get(1)))
}

/// The entries of a map recorded as a list of pairs.
fn map_pairs(map: Option<&Json>) -> Vec<(&Json, &Json)> {
    let pairs = map.and_then(|m| m.get("$map")).map_or(&[][..], |p| p.arr());
    pairs
        .iter()
        .filter_map(|e| Some((e.arr().first()?, e.arr().get(1)?)))
        .collect()
}

/// The value a pair map holds under the slot `(role, name)`.
fn lookup<'a>(map: &[(&'a Json, &'a Json)], role: &str, name: &str) -> Option<&'a Json> {
    let found = map.iter().find(|entry| pair_key(entry.0) == (role, name));
    found.map(|entry| entry.1)
}

/// The walk a walk record describes.
fn walk_of(walk: &Json) -> Result<Walk, Refusal> {
    let types = map_pairs(walk.get("slot_types"));
    let indices = map_pairs(walk.get("slot_of"));
    let spec = walk
        .get("arg_spec")
        .ok_or_else(|| bad("contract: walk without arg_spec"))?;
    let mut slots = vec![];
    for key in spec.arr() {
        let (role, name) = pair_key(key);
        let ty = lookup(&types, role, name).ok_or_else(|| bad("contract: slot without a type"))?;
        let index = lookup(&indices, role, name)
            .and_then(Json::int)
            .ok_or_else(|| bad("contract: slot without an index"))?;
        slots.push(Slot {
            role: role_of(role)?,
            name: name.to_string(),
            dtype: dtype_of(text(ty.get("dtype")))?,
            shape: ty
                .get("shape")
                .map_or(&[][..], |s| s.arr())
                .iter()
                .filter_map(Json::int)
                .collect(),
            index,
            axes: None,
        });
    }
    let access = walk.get("access").and_then(|a| a.get("cls"));
    Ok(Walk {
        access: access_of(text(access))?,
        finish: live(walk.get("finish"))
            .and_then(Json::str)
            .map(String::from),
        slots,
    })
}

/// The role a slot records.
fn role_of(name: &str) -> Result<Role, Refusal> {
    Ok(match name {
        "mutable" => Role::Mutable,
        "plane_in" => Role::PlaneIn,
        "lookup" => Role::Lookup,
        "wide_out" => Role::WideOut,
        "uniform" => Role::Uniform,
        "accum_out" => Role::AccumOut,
        "terminated" => Role::Terminated,
        "wide_in" => Role::WideIn,
        "nsamples" => Role::Nsamples,
        "out" => Role::Out,
        other => return Err(unknown(format!("slot role {other:?}"))),
    })
}

/// The element type a slot records.
fn dtype_of(name: &str) -> Result<Dtype, Refusal> {
    Ok(match name {
        "f64" => Dtype::F64,
        "f32" => Dtype::F32,
        "i32" => Dtype::I32,
        "i64" => Dtype::I64,
        "int32" => Dtype::Int32,
        "uint32" => Dtype::Uint32,
        "bool" => Dtype::Bool,
        other => return Err(unknown(format!("slot dtype {other:?}"))),
    })
}

/// The access class a walk records.
fn access_of(name: &str) -> Result<Access, Refusal> {
    Ok(match name {
        "sample_local" => Access::SampleLocal,
        "cross_sample_read" => Access::CrossSampleRead,
        "cross_sample_write" => Access::CrossSampleWrite,
        other => return Err(unknown(format!("access class {other:?}"))),
    })
}

/// The wing a setting is read from: the host wing, or the device wing when the host wing has no
/// form section and the build is for the device.
fn primary_wing<'a>(spec: &'a Json, target: &Target) -> Option<&'a Json> {
    let host = spec.get("host").map(of);
    let has_form = host.is_some_and(|wing| wing.get("form").is_some());
    match (target, live(spec.get("device"))) {
        (Target::Device, Some(device)) if !has_form => Some(of(device)),
        _ => host,
    }
}

/// The number a wing holds under `group.key`.
fn setting(wing: Option<&Json>, group: &str, key: &str) -> Option<i64> {
    int_at(of(wing?.get(group)?), key)
}

/// The number under `by_target.<name>.<section>.<key>` (wrapped maps at every step).
fn by_target(spec: &Json, name: &str, section: &str, key: &str) -> Option<i64> {
    let wing = of(spec.get("by_target")?.get(name)?);
    int_at(of(wing.get(section)?), key)
}

/// The build's settings and the numbers of the target.
fn build_of(spec: &Json, target: &Target, source: bool) -> Result<Build, Refusal> {
    let wing = primary_wing(spec, target);
    let settings = spec.get("settings");
    let required =
        |key: &str| setting(wing, "form", key).ok_or_else(|| bad(format!("spec: form.{key}")));
    let fold_form = match settings.and_then(|s| s.get("form")) {
        None => FoldForm::default(),
        Some(form) => match text(Some(form)) {
            "auto" => FoldForm::Auto,
            "static" => FoldForm::Static,
            "rolled" => FoldForm::Rolled,
            other => return Err(unknown(format!("fold form {other:?}"))),
        },
    };
    let launch = live(settings.and_then(|s| s.get("launch_bounds")));
    let name = if *target == Target::Device {
        "cuda"
    } else {
        "host"
    };
    let bounds = launch.and_then(|b| Some((b.arr().first()?.int()?, b.arr().get(1)?.int()?)));
    Ok(Build {
        form: FormSettings {
            lazy_leaf_width: required("lazy_leaf_width")?,
            rolled_fold_min: required("rolled_fold_min")?,
            lean_width: setting(wing, "form", "lean_width"),
            uniform_run: find_number(spec, "uniform_run"),
        },
        fold_form,
        lanes: lanes_of(spec, wing),
        linalg_max_dim: spec
            .get(if *target == Target::Device {
                "device"
            } else {
                "host"
            })
            .and_then(|w| setting(Some(of(w)), "linalg", "max_dim")),
        cells_row: by_target(spec, name, "cells", "row").unwrap_or(DEFAULT_CELLS_ROW),
        blend: if source {
            matches!(
                settings.and_then(|s| s.get("blend")),
                Some(Json::Bool(true))
            )
        } else {
            *target == Target::Host
        },
        warps: settings
            .and_then(|s| int_at(s, "warps"))
            .unwrap_or(DEFAULT_WARPS),
        launch_bounds: bounds,
        fold_k: live(settings.and_then(|s| s.get("fold_k")))
            .and_then(Json::int)
            .filter(|k| *k != 0),
        fold_limits: if *target == Target::Device {
            fold_limits(spec)
        } else {
            None
        },
        wide_card: if *target == Target::Device {
            wide_card(spec)
        } else {
            None
        },
    })
}

/// The numbers for runs of lanes.
fn lanes_of(spec: &Json, wing: Option<&Json>) -> LaneSettings {
    let tile = |key: &str| by_target(spec, "host", "lanes", key);
    LaneSettings {
        run_max: setting(wing, "lanes", "run_max"),
        block_run: setting(wing, "lanes", "block_run"),
        tile: tile("tile_min").zip(tile("tile_max")),
    }
}

/// The device's folding limits, when the build states all four.
fn fold_limits(spec: &Json) -> Option<FoldLimits> {
    let field = |key: &str| by_target(spec, "cuda", "fold", key);
    Some(FoldLimits {
        k_max: field("k_max")?,
        regs_base: field("regs_base")?,
        regs_column: field("regs_column")?,
        residency_regs: field("residency_regs")?,
    })
}

/// The device's wide-vector rule, when the build has a wide card.
fn wide_card(spec: &Json) -> Option<WideCard> {
    let device = of(spec.get("device")?);
    let wide = of(of(device.get("cards")?).get("wide")?);
    let rows = wide.get("rows").map_or(&[][..], |r| of(r).arr());
    let forced = spec
        .get("wide_override")
        .map(of)
        .is_some_and(|o| !o.arr().is_empty());
    Some(WideCard {
        row_scalars: rows
            .iter()
            .map(|r| text(of(r).get("dtype")).to_string())
            .collect(),
        slice_min: wide
            .get("defaults")
            .and_then(|d| int_at(of(d), "SLICE_MIN")),
        forced,
    })
}

/// The machine facts of the spec, and the recorded answers to the select-weight question.
fn machine_of(spec: &Json, answers: Option<&Json>) -> Result<Machine, Refusal> {
    let machine = spec
        .get("machine")
        .ok_or_else(|| unknown("a spec without machine facts"))?;
    let isa = match text(machine.get("isa")) {
        "" => None,
        "arm64" => Some(Isa::Arm64),
        "x86_64" => Some(Isa::X86_64),
        other => return Err(unknown(format!("machine isa {other:?}"))),
    };
    let facts = machine.get("facts");
    let profile = text(facts.and_then(|f| f.get("host_profile")));
    let exact_list = facts
        .and_then(|f| f.get("exact_host_profiles"))
        .map_or(&[][..], |a| a.arr());
    let listed = exact_list.iter().any(|x| x.str() == Some(profile));
    let arm64_gcc = matches!(
        facts.and_then(|f| f.get("arm64_gcc")),
        Some(Json::Bool(true))
    );
    let vector_isa = text(facts.and_then(|f| f.get("vector_isa")));
    Ok(Machine {
        isa,
        vector_isa: Some(vector_isa.to_string()).filter(|name| !name.is_empty()),
        exact_libm: listed || isa != Some(Isa::X86_64),
        routed_pow: !listed && arm64_gcc,
        caches: caches_of(facts),
        select_weight: SelectWeight {
            float64: select_weight(answers, "float64"),
            float32: select_weight(answers, "float32"),
        },
    })
}

/// The cache sizes, when the machine reports both.
fn caches_of(facts: Option<&Json>) -> Option<Caches> {
    let size = |key: &str| facts.and_then(|f| int_at(f, key));
    match (size("l1d_cache_bytes"), size("l2_cache_bytes")) {
        (Some(l1d_bytes), Some(l2_bytes)) if l1d_bytes > 0 && l2_bytes > 0 => Some(Caches {
            l1d_bytes,
            l2_bytes,
            source: text(facts.and_then(|f| f.get("cache_source"))).to_string(),
        }),
        _ => None,
    }
}

/// The recorded answer to the select-weight question for a scalar type: each entry is the call's
/// arguments (the type first, as `{"l": [{"s": type}]}`) and its answer (`{"f": number text}`).
fn select_weight(answers: Option<&Json>, scalar: &str) -> Option<f64> {
    let entries = answers
        .and_then(|a| a.get("select_weight"))
        .map_or(&[][..], |a| a.arr());
    entries.iter().find_map(|entry| {
        let [args, answer] = entry.arr() else {
            return None;
        };
        let first = args.get("l")?.arr().first()?.get("s")?.str()?;
        if first != scalar {
            return None;
        }
        answer.get("f")?.str()?.parse::<f64>().ok()
    })
}

/// The body a caller put in the recorded request under `$body`, with whether it draws random
/// numbers.
fn given_body(top: &Json) -> Option<GivenBody> {
    let text = top.get("$body").and_then(Json::str)?.to_string();
    let draws_random = top.get("$needs_random") == Some(&Json::Bool(true));
    Some(GivenBody { text, draws_random })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WALK: &str = r#"{"arg_spec":[["plane_in","x"]],
        "slot_types":{"$map":[[["plane_in","x"],{"dtype":"f64","shape":[]}]]},
        "slot_of":{"$map":[[["plane_in","x"],0]]},
        "access":{"cls":"sample_local"},"finish":null}"#;
    const SPEC: &str = r#"{"host":{"of":{"form":{"of":{"lazy_leaf_width":8,"rolled_fold_min":4}}}},
        "machine":{"isa":"arm64","facts":{"host_profile":"native","exact_host_profiles":[],
        "arm64_gcc":false}},"settings":{"form":"auto"},"by_target":{}}"#;
    const KIND: &str = r#"{"$ref":"hawk.ext:DEFAULT_KIND"}"#;
    const MODE: &str = r#"{"$ref":"wrapper:FLOAT64"}"#;

    /// The typed unit of a recorded request given as text.
    fn typed(json: &str) -> Result<Unit, Refusal> {
        typed_of(
            &Contract::parse(json).expect("the test contract parses"),
            Target::Host,
        )
    }

    /// A backend-form request with the given kernel record, kind and mode.
    fn backend(kernel: &str, kind: &str, mode: &str) -> String {
        format!(
            r#"{{"name":"u","kernel":{kernel},"kind":{kind},"smode":{mode},"spec":{SPEC},
            "rows":1,"options":{{"layout_sizes_override":null}}}}"#
        )
    }

    #[test]
    fn a_plain_kernel_converts() {
        let kernel = format!(r#"{{"$type":"Kernel","sinks":[{{"$node":3}}],"walk":{WALK}}}"#);
        let unit = typed(&backend(&kernel, KIND, MODE)).expect("converts");
        assert_eq!(unit.name, "u");
        assert_eq!(
            (unit.scalar, unit.form.clone()),
            (Scalar::Float64, Form::Plain)
        );
        assert_eq!(
            unit.kind,
            Kind {
                guard: Guard::Default,
                lanes: false,
                sink: SinkPolicy::Plain
            }
        );
        assert_eq!(unit.sinks, vec![3]);
        assert_eq!(
            unit.walk.map(|w| (w.access, w.slots.len())),
            Some((Access::SampleLocal, 1))
        );
        assert_eq!(unit.build.form.lazy_leaf_width, 8);
        assert_eq!(
            (unit.machine.isa, unit.machine.exact_libm),
            (Some(Isa::Arm64), true)
        );
        assert_eq!((unit.rows, unit.exports), (1, Exports::Derived));
    }

    #[test]
    fn a_team_stage_converts() {
        let kernel = r#"{"$type":"TeamStage","sinks":[],"team":{"grid":"lanes","threads":4,
            "warps":1,"planes":[{"name":"tm0_a","per_lane":9}],"reads":[{"$node":2}],
            "steps":[{"op":"lu","n":3,"k":1,"m":0,"max_chunks":0,"planes":[["a","tm0_a"]]}]}}"#;
        let unit = typed(&backend(kernel, KIND, MODE)).expect("converts");
        let Form::TeamStage(stage) = unit.form else {
            panic!("not a team stage")
        };
        assert_eq!((stage.grid, stage.reads), (Grid::Lanes, vec![2]));
        assert_eq!(stage.planes, vec![("tm0_a".to_string(), 9)]);
        assert_eq!((stage.steps[0].op, stage.steps[0].n), (StepOp::Lu, 3));
        assert_eq!(
            unit.team,
            Some(TeamShape {
                threads: 4,
                warps: 1
            })
        );
        assert!(unit.walk.is_none());
    }

    #[test]
    fn a_source_form_call_converts() {
        let json = format!(
            r#"{{"args":["v",[{{"$node":1}}],{WALK}],"kwargs":{{"kind":{KIND},"spec":{SPEC},
            "mode":{{"$ref":"wrapper:FLOAT32"}},"exports":"// sizes"}}}}"#
        );
        let unit = typed(&json).expect("converts");
        assert_eq!(
            (unit.name.as_str(), unit.scalar, unit.sinks),
            ("v", Scalar::Float32, vec![1])
        );
        assert_eq!(unit.exports, Exports::Text("// sizes".to_string()));
        assert!(!unit.build.blend);
    }

    #[test]
    fn an_unknown_member_is_refused_by_name() {
        let step = r#"{"op":"frobnicate","n":1,"k":1,"m":0,"max_chunks":0,"planes":[]}"#;
        let team = format!(r#"{{"grid":"lanes","planes":[],"reads":[],"steps":[{step}]}}"#);
        let kernel = format!(r#"{{"$type":"TeamStage","team":{team}}}"#);
        let refused = typed(&backend(&kernel, KIND, MODE)).expect_err("refused");
        assert_eq!(refused, unknown("team step frobnicate"));
        let plain = format!(r#"{{"$type":"Kernel","sinks":[],"walk":{WALK}}}"#);
        let mode = r#"{"$ref":"wrapper:FLOAT16"}"#;
        let refused = typed(&backend(&plain, KIND, mode)).expect_err("refused");
        assert_eq!(refused, unknown("scalar mode \"FLOAT16\""));
    }
}
