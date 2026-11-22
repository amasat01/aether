//! The typed wire: a `Unit` as JSON with a closed schema. Every field is under its own name, a
//! member of a plain enum is its member name, an optional part is absent when there is none. The
//! reader is strict: a key it does not know, a required key that is missing and a member it does
//! not know are refused by name.
use super::types::*;
use super::Json;
use crate::refuse::{unknown, Refusal};

/// An object's members in order, as the writer builds them.
type Obj = Vec<(String, Json)>;

/// One kind of value on the wire: how it is written and how it is read back under a key. Each
/// record states its fields once (see `record!`) and gets both directions from the kinds of them.
trait Wire: Sized {
    /// The value as JSON.
    fn write(&self) -> Json;
    /// The value from JSON; a wrong shape is refused naming the key.
    fn read(j: &Json, key: &str) -> Result<Self, Refusal>;
    /// Adds the value to an object being written under the key.
    fn put(&self, o: &mut Obj, key: &str) {
        o.push((key.to_string(), self.write()))
    }
    /// Reads the value of a key of an object; a required key that is missing is refused.
    fn get(f: &mut Fields, key: &'static str) -> Result<Self, Refusal> {
        Self::read(f.req(key)?, key)
    }
}

/// Gives a plain enum its wire names: each member is its own name.
macro_rules! members {
    ($ty:ident, [$($member:ident),*]) => {
        impl Wire for $ty {
            fn write(&self) -> Json {
                Json::Str(match self { $($ty::$member => stringify!($member)),* }.to_string())
            }
            fn read(j: &Json, key: &str) -> Result<$ty, Refusal> {
                match j.str().ok_or_else(|| wrong(key, "not a name"))? {
                    $(stringify!($member) => Ok($ty::$member),)*
                    other => Err(unknown(format!("member {other} of {key}"))),
                }
            }
        }
    };
}

/// States a record's fields once, in wire order, under their own names; the writer and the
/// strict reader of the record both come from that list and the kind of each field.
macro_rules! record {
    ($ty:ident, $scope:literal, [$($field:ident),*]) => {
        impl $ty {
            /// Adds each field to the object, in the order stated.
            fn put_fields(&self, o: &mut Obj) {
                $(Wire::put(&self.$field, o, stringify!($field));)*
            }
            /// Reads each field from the object, in the order stated.
            fn get_fields(f: &mut Fields) -> Result<$ty, Refusal> {
                Ok($ty { $($field: Wire::get(f, stringify!($field))?,)* })
            }
        }
        impl Wire for $ty {
            fn write(&self) -> Json {
                let mut o = Obj::new();
                self.put_fields(&mut o);
                Json::Obj(o)
            }
            fn read(j: &Json, _key: &str) -> Result<$ty, Refusal> {
                let mut f = fields(j, $scope)?;
                let value = $ty::get_fields(&mut f)?;
                f.done()?;
                Ok(value)
            }
        }
    };
}

/// A number value.
fn num(n: impl ToString) -> Json {
    Json::Num(n.to_string())
}
/// A data member written as `{"Name": payload}`.
fn tag(name: &str, payload: Json) -> Json {
    Json::Obj(vec![(name.to_string(), payload)])
}

impl Wire for i64 {
    fn write(&self) -> Json {
        num(self)
    }
    fn read(j: &Json, key: &str) -> Result<i64, Refusal> {
        j.int().ok_or_else(|| wrong(key, "not an integer"))
    }
}
/// A node handle.
impl Wire for u32 {
    fn write(&self) -> Json {
        num(self)
    }
    fn read(j: &Json, key: &str) -> Result<u32, Refusal> {
        j.int()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| wrong(key, "not a node handle"))
    }
}
impl Wire for bool {
    fn write(&self) -> Json {
        Json::Bool(*self)
    }
    fn read(j: &Json, key: &str) -> Result<bool, Refusal> {
        match j {
            Json::Bool(b) => Ok(*b),
            _ => Err(wrong(key, "not a flag")),
        }
    }
}
impl Wire for String {
    fn write(&self) -> Json {
        Json::Str(self.clone())
    }
    fn read(j: &Json, key: &str) -> Result<String, Refusal> {
        j.str()
            .map(String::from)
            .ok_or_else(|| wrong(key, "not text"))
    }
}
impl<T: Wire> Wire for Vec<T> {
    fn write(&self) -> Json {
        Json::Arr(self.iter().map(T::write).collect())
    }
    fn read(j: &Json, key: &str) -> Result<Vec<T>, Refusal> {
        j.arr().iter().map(|item| T::read(item, key)).collect()
    }
}
/// A pair is a two-item list.
impl<A: Wire, B: Wire> Wire for (A, B) {
    fn write(&self) -> Json {
        Json::Arr(vec![self.0.write(), self.1.write()])
    }
    fn read(j: &Json, key: &str) -> Result<(A, B), Refusal> {
        match j.arr() {
            [a, b] => Ok((A::read(a, key)?, B::read(b, key)?)),
            _ => Err(wrong(key, "not a pair")),
        }
    }
}
/// An optional part: no key when there is none, and null counts as none.
impl<T: Wire> Wire for Option<T> {
    fn write(&self) -> Json {
        self.as_ref().map_or(Json::Null, T::write)
    }
    fn read(j: &Json, key: &str) -> Result<Option<T>, Refusal> {
        T::read(j, key).map(Some)
    }
    fn put(&self, o: &mut Obj, key: &str) {
        if let Some(value) = self {
            value.put(o, key)
        }
    }
    fn get(f: &mut Fields, key: &'static str) -> Result<Option<T>, Refusal> {
        f.opt(key).map(|j| T::read(j, key)).transpose()
    }
}

members!(Target, [Host, Device]);
members!(Scalar, [Float64, Float32, Banded]);
members!(Dtype, [F64, F32, I32, I64, Int32, Uint32, Bool]);
#[rustfmt::skip]
members!(Role, [Mutable, PlaneIn, Lookup, PlaneOut, Uniform, Terminated,
    WideIn, Nsamples, Out, Extent]);
members!(Access, [SampleLocal, CrossSampleRead, CrossSampleWrite]);
members!(MapBits, [B32, B64]);
members!(Grid, [Lanes, Keys, Chunks, Targets]);
members!(FoldForm, [Auto, Static, Rolled]);
members!(Isa, [Arm64, X86_64]);
#[rustfmt::skip]
members!(StepOp, [Lu, LuSolve, LuSolveT, LuInverse, LuDet, Chol, CholFrom, ChoSolve, TriLower,
    TriUpperT, Eigh, EighWarp, Qr, QrSolve, QrGramSolve, BlockSum, BlockPart, OrderedFold,
    ScatterZero, ScatterChunks, ScatterCells]);

// The records: each field once, in wire order, under its own name.
record!(
    Slot,
    "slot",
    [role, name, dtype, shape, index, axes, zeroed]
);
record!(Axis, "axis", [label, extent]);
record!(Walk, "walk", [access, finish, slots]);
record!(Kind, "kind", [guard, lanes, sink]);
#[rustfmt::skip]
record!(Active, "active", [map_binding, count_binding, group, map_bits, lane_set]);
record!(Step, "step", [op, n, k, m, max_chunks, planes]);
#[rustfmt::skip]
record!(TeamStage, "team stage", [grid, planes, reads, steps, active]);
record!(FusedLane, "fused lane", [name, sinks]);
#[rustfmt::skip]
record!(FormSettings, "form settings", [lazy_leaf_width, rolled_fold_min, lean_width,
    uniform_run]);
record!(LaneSettings, "lane settings", [run_max, block_run, tile]);
#[rustfmt::skip]
record!(FoldLimits, "fold limits", [k_max, regs_base, regs_column, residency_regs]);
record!(WideCard, "wide card", [row_scalars, slice_min, forced]);
#[rustfmt::skip]
record!(Build, "build", [form, fold_form, lanes, linalg_max_dim, cells_row, blend, warps,
    launch_bounds, fold_k, fold_limits, wide_card]);
record!(Caches, "caches", [l1d_bytes, l2_bytes, source]);
#[rustfmt::skip]
#[rustfmt::skip]
record!(Machine, "machine", [isa, vector_isa, exact_libm, routed_pow, caches, select_weight]);
record!(Segment, "segment", [offsets_binding, j]);
record!(ThreadBox, "thread box", [axis, t, stride]);
record!(OneStep, "one step", [sinks, walk]);
record!(TeamShape, "team", [threads, warps]);
record!(GivenBody, "given body", [text, draws_random]);
#[rustfmt::skip]
record!(Unit, "unit", [name, target, scalar, kind, form, sinks, walk, body, active, segment,
    thread_box, one_step, rows, team, build, machine, exports, literal_table]);

// The shapes below are irregular, so each has a short hand-written pair: a data member is a
// one-member object `{"Name": payload}` and a member without data is its plain name.

impl Wire for Guard {
    fn write(&self) -> Json {
        match self {
            Guard::Default => Json::Str("Default".into()),
            Guard::DataOnly => Json::Str("DataOnly".into()),
            Guard::Masks(masks) => tag("Masks", masks.write()),
        }
    }
    fn read(j: &Json, _key: &str) -> Result<Guard, Refusal> {
        match j {
            Json::Str(s) if s == "Default" => Ok(Guard::Default),
            Json::Str(s) if s == "DataOnly" => Ok(Guard::DataOnly),
            other => match tagged(other, "guard")? {
                ("Masks", masks) => Ok(Guard::Masks(Wire::read(masks, "masks")?)),
                (name, _) => Err(unknown(format!("member {name} of guard"))),
            },
        }
    }
}

impl Wire for SinkPolicy {
    fn write(&self) -> Json {
        match self {
            SinkPolicy::Plain => Json::Str("Plain".into()),
            SinkPolicy::Compensated {
                plane,
                output_slot,
                output_names,
            } => {
                let mut o = Obj::new();
                plane.put(&mut o, "plane");
                output_slot.put(&mut o, "output_slot");
                output_names.put(&mut o, "output_names");
                tag("Compensated", Json::Obj(o))
            }
        }
    }
    fn read(j: &Json, _key: &str) -> Result<SinkPolicy, Refusal> {
        match j {
            Json::Str(s) if s == "Plain" => Ok(SinkPolicy::Plain),
            other => match tagged(other, "sink")? {
                ("Compensated", body) => {
                    let mut g = fields(body, "sink")?;
                    let policy = SinkPolicy::Compensated {
                        plane: Wire::get(&mut g, "plane")?,
                        output_slot: Wire::get(&mut g, "output_slot")?,
                        output_names: Wire::get(&mut g, "output_names")?,
                    };
                    g.done()?;
                    Ok(policy)
                }
                (name, _) => Err(unknown(format!("member {name} of sink"))),
            },
        }
    }
}

impl Wire for Form {
    fn write(&self) -> Json {
        match self {
            Form::Plain => Json::Str("Plain".into()),
            Form::WarpStage => Json::Str("WarpStage".into()),
            Form::Fused(lanes) => tag("Fused", lanes.write()),
            Form::TeamStage(stage) => tag("TeamStage", stage.write()),
        }
    }
    fn read(j: &Json, _key: &str) -> Result<Form, Refusal> {
        match j {
            Json::Str(s) if s == "Plain" => Ok(Form::Plain),
            Json::Str(s) if s == "WarpStage" => Ok(Form::WarpStage),
            other => match tagged(other, "form")? {
                ("TeamStage", body) => Ok(Form::TeamStage(Wire::read(body, "form")?)),
                ("Fused", lanes) => Ok(Form::Fused(Wire::read(lanes, "form")?)),
                (name, _) => Err(unknown(format!("member {name} of form"))),
            },
        }
    }
}

impl Wire for BodySource {
    fn write(&self) -> Json {
        match self {
            BodySource::Walk => Json::Str("Walk".into()),
            BodySource::Given(given) => tag("Given", given.write()),
        }
    }
    fn read(j: &Json, _key: &str) -> Result<BodySource, Refusal> {
        match j {
            Json::Str(s) if s == "Walk" => Ok(BodySource::Walk),
            other => match tagged(other, "body")? {
                ("Given", given) => Ok(BodySource::Given(Wire::read(given, "body")?)),
                (name, _) => Err(unknown(format!("member {name} of body"))),
            },
        }
    }
}

impl Wire for Exports {
    fn write(&self) -> Json {
        match self {
            Exports::Derived => Json::Str("Derived".into()),
            Exports::Sizes(sizes) => tag("Sizes", Json::Arr(sizes.iter().map(num).collect())),
            Exports::Text(text) => tag("Text", text.write()),
        }
    }
    fn read(j: &Json, _key: &str) -> Result<Exports, Refusal> {
        match j {
            Json::Str(s) if s == "Derived" => Ok(Exports::Derived),
            other => match tagged(other, "exports")? {
                ("Text", text) => Ok(Exports::Text(Wire::read(text, "exports")?)),
                ("Sizes", sizes) => {
                    let sizes: Vec<u64> = sizes
                        .arr()
                        .iter()
                        .map(|n| {
                            n.int()
                                .and_then(|n| u64::try_from(n).ok())
                                .ok_or_else(|| wrong("exports", "not a size"))
                        })
                        .collect::<Result<_, _>>()?;
                    let sizes: [u64; 5] = sizes
                        .try_into()
                        .map_err(|_| wrong("exports", "not five sizes"))?;
                    Ok(Exports::Sizes(sizes))
                }
                (name, _) => Err(unknown(format!("member {name} of exports"))),
            },
        }
    }
}

/// The two weights are real numbers; one that is not a number reads as absent.
impl Wire for SelectWeight {
    fn write(&self) -> Json {
        let mut o = Obj::new();
        for (key, weight) in [("float64", self.float64), ("float32", self.float32)] {
            if let Some(w) = weight {
                o.push((key.to_string(), num(w)))
            }
        }
        Json::Obj(o)
    }
    fn read(j: &Json, _key: &str) -> Result<SelectWeight, Refusal> {
        let mut g = fields(j, "select weight")?;
        let mut number = |key: &'static str| match g.opt(key) {
            Some(Json::Num(n)) => n
                .parse::<f64>()
                .map(Some)
                .map_err(|_| wrong("select_weight", "not a number")),
            _ => Ok(None),
        };
        let weight = SelectWeight {
            float64: number("float64")?,
            float32: number("float32")?,
        };
        g.done()?;
        Ok(weight)
    }
}

/// The name and payload of a data member written as `{"Name": payload}`.
fn tagged<'a>(j: &'a Json, field: &str) -> Result<(&'a str, &'a Json), Refusal> {
    match j {
        Json::Obj(items) if items.len() == 1 => Ok((items[0].0.as_str(), &items[0].1)),
        _ => Err(unknown(format!("member of {field}"))),
    }
}

impl Unit {
    /// The unit as typed JSON text, led by the contract version.
    pub fn to_json(&self) -> String {
        let mut o = Obj::new();
        CONTRACT_VERSION.put(&mut o, "v");
        self.put_fields(&mut o);
        Json::Obj(o).to_text()
    }

    /// The unit a typed JSON text describes. The version is checked first; a key, a member or a
    /// version this build does not know is refused by name.
    pub fn from_json(source: &str) -> Result<Unit, Refusal> {
        Unit::from_value(&super::Contract::parse(source)?.0)
    }

    /// The unit a parsed typed JSON value describes.
    pub fn from_value(j: &Json) -> Result<Unit, Refusal> {
        let mut f = fields(j, "unit")?;
        let version = i64::get(&mut f, "v")?;
        if version != i64::from(CONTRACT_VERSION) {
            return Err(unknown(format!("contract version {version}")));
        }
        let unit = Unit::get_fields(&mut f)?;
        f.done()?;
        Ok(unit)
    }
}

impl Json {
    /// The value as JSON text.
    pub fn to_text(&self) -> String {
        match self {
            Json::Null => "null".into(),
            Json::Bool(b) => b.to_string(),
            Json::Num(n) => n.clone(),
            Json::Str(s) => quoted(s),
            Json::Arr(a) => {
                let items: Vec<String> = a.iter().map(Json::to_text).collect();
                format!("[{}]", items.join(","))
            }
            Json::Obj(o) => {
                let items: Vec<String> = o
                    .iter()
                    .map(|(k, v)| format!("{}:{}", quoted(k), v.to_text()))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

/// The text as a JSON string literal.
fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---- reading ----

/// The keys of one object, handed out by name; `done` refuses any key nobody asked for.
struct Fields<'a> {
    /// The members of the object, in the order written.
    items: &'a [(String, Json)],
    /// The keys that were asked for so far; the rest are unknown.
    asked: Vec<&'static str>,
    /// What the object stands for, for the refusal text.
    scope: &'static str,
}

/// Starts reading an object; a value that is not an object is refused by name.
fn fields<'a>(j: &'a Json, scope: &'static str) -> Result<Fields<'a>, Refusal> {
    match j {
        Json::Obj(items) => Ok(Fields {
            items,
            asked: vec![],
            scope,
        }),
        _ => Err(unknown(format!("contract key {scope}: not an object"))),
    }
}

/// The refusal for a key whose value has the wrong shape.
fn wrong(key: &str, what: &str) -> Refusal {
    unknown(format!("contract key {key}: {what}"))
}

impl<'a> Fields<'a> {
    /// The value under an optional key, none when absent (null counts as absent).
    fn opt(&mut self, key: &'static str) -> Option<&'a Json> {
        self.asked.push(key);
        let found = self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        found.filter(|v| **v != Json::Null)
    }
    /// The value under a required key.
    fn req(&mut self, key: &'static str) -> Result<&'a Json, Refusal> {
        let scope = self.scope;
        self.opt(key)
            .ok_or_else(|| unknown(format!("contract key {key} missing in {scope}")))
    }
    /// Refuses a key that was not asked for.
    fn done(self) -> Result<(), Refusal> {
        match self
            .items
            .iter()
            .find(|(k, _)| !self.asked.contains(&k.as_str()))
        {
            Some((k, _)) => Err(unknown(format!("contract key {k}"))),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small unit with the optional parts filled in or left out as given.
    fn unit(form: Form, walk: Option<Walk>) -> Unit {
        Unit {
            name: "k\"1".into(),
            target: Target::Host,
            scalar: Scalar::Float32,
            kind: Kind {
                guard: Guard::Masks(vec!["a".into()]),
                lanes: true,
                sink: SinkPolicy::Plain,
            },
            form,
            sinks: vec![1, 2],
            walk,
            body: BodySource::Given(GivenBody {
                text: "x\n".into(),
                draws_random: true,
            }),
            active: None,
            segment: Some(Segment {
                offsets_binding: "o".into(),
                j: 2,
            }),
            thread_box: Some(ThreadBox {
                axis: None,
                t: 2,
                stride: 3,
            }),
            one_step: None,
            rows: 1,
            team: Some(TeamShape {
                threads: 4,
                warps: 1,
            }),
            build: Build {
                form: FormSettings {
                    lazy_leaf_width: 8,
                    rolled_fold_min: 4,
                    lean_width: None,
                    uniform_run: Some(2),
                },
                fold_form: FoldForm::Auto,
                lanes: LaneSettings {
                    run_max: Some(8),
                    block_run: None,
                    tile: Some((4, 64)),
                },
                linalg_max_dim: None,
                cells_row: 0,
                blend: true,
                warps: 1,
                launch_bounds: None,
                fold_k: None,
                fold_limits: None,
                wide_card: None,
            },
            machine: Machine {
                isa: Some(Isa::Arm64),
                vector_isa: Some("neon".into()),
                exact_libm: true,
                routed_pow: false,
                caches: Some(Caches {
                    l1d_bytes: 65536,
                    l2_bytes: 4194304,
                    source: "probe".into(),
                }),
                select_weight: SelectWeight {
                    float64: Some(1.5),
                    float32: None,
                },
            },
            exports: Exports::Sizes([1, 2, 3, 4, 5]),
            literal_table: false,
        }
    }

    #[test]
    fn typed_contracts_round_trip() {
        let walk = Walk {
            access: Access::SampleLocal,
            finish: Some("m".into()),
            slots: vec![Slot {
                role: Role::PlaneIn,
                name: "x".into(),
                dtype: Dtype::F64,
                shape: vec![3],
                index: 0,
                axes: None,
                zeroed: false,
            }],
        };
        let stage = TeamStage {
            grid: Grid::Keys,
            planes: vec![("p".into(), 9)],
            reads: vec![3],
            steps: vec![Step {
                op: StepOp::Lu,
                n: 3,
                k: 1,
                m: 0,
                max_chunks: 0,
                planes: vec![("a".into(), "p".into())],
            }],
            active: None,
        };
        let fused = Form::Fused(vec![FusedLane {
            name: "l".into(),
            sinks: vec![1],
        }]);
        for u in [
            unit(Form::Plain, Some(walk)),
            unit(Form::TeamStage(stage), None),
            unit(fused, None),
        ] {
            assert_eq!(Unit::from_json(&u.to_json()), Ok(u));
        }
    }

    #[test]
    fn an_unknown_key_member_or_version_is_refused_by_name() {
        let good = unit(Form::Plain, None).to_json();
        let extra = good.replacen("{\"v\":1,", "{\"v\":1,\"surprise\":1,", 1);
        assert_eq!(
            Unit::from_json(&extra),
            Err(unknown("contract key surprise"))
        );
        let member = good.replacen("\"Float32\"", "\"Float16\"", 1);
        assert_eq!(
            Unit::from_json(&member),
            Err(unknown("member Float16 of scalar"))
        );
        let missing = good.replacen("\"rows\":1,", "", 1);
        assert_eq!(
            Unit::from_json(&missing),
            Err(unknown("contract key rows missing in unit"))
        );
        let version = good.replacen("{\"v\":1,", "{\"v\":2,", 1);
        assert_eq!(
            Unit::from_json(&version),
            Err(unknown("contract version 2"))
        );
    }
}
