//! A banded unit is refused by the host writer by name, as a construct not written.
use aether_writers::contract::types::{Access, Form, Grid, Target, TeamStage, Unit, Walk};
use aether_writers::unit::entry::Render;
use aether_writers::unit::host::render::render;
use aether_writers::{stage_body, RefusalKind};

/// A unit as the contract writes it, in the 32-bit float mode, with its walk left out.
const PLAIN_UNIT: &str = r#"{"v":1,"name":"k\"1","target":"Host","scalar":"Float32","kind":{"guard":{"Masks":["a"]},"lanes":true,"sink":"Plain"},"form":"Plain","sinks":[1,2],"body":{"Given":{"text":"x\n","draws_random":true}},"segment":{"offsets_binding":"o","j":2},"thread_box":{"t":2,"stride":3},"rows":1,"team":{"threads":4,"warps":1},"build":{"form":{"lazy_leaf_width":8,"rolled_fold_min":4,"uniform_run":2},"fold_form":"Auto","lanes":{"run_max":8,"tile":[4,64]},"cells_row":0,"blend":true,"warps":1},"machine":{"isa":"Arm64","vector_isa":"neon","exact_libm":true,"routed_pow":false,"caches":{"l1d_bytes":65536,"l2_bytes":4194304,"source":"probe"},"select_weight":{"float64":1.5}},"exports":{"Sizes":[1,2,3,4,5]},"literal_table":false}"#;

/// The unit with a walk of no slots, in the mode the contract names.
fn unit_in(mode: &str) -> Unit {
    let json = PLAIN_UNIT.replacen("\"Float32\"", &format!("\"{mode}\""), 1);
    let mut unit = Unit::from_json(&json).expect("a unit");
    unit.walk = Some(Walk {
        access: Access::SampleLocal,
        finish: None,
        slots: vec![],
    });
    unit
}

#[test]
fn the_host_refuses_a_banded_unit_by_name_before_it_reads_the_graph() {
    let unit = unit_in("Banded");
    let input = Render {
        unit: &unit,
        body: String::new(),
        needs_random: false,
        needs: Default::default(),
    };
    // the refusal precedes every read of the graph, so no graph is needed
    let why = render(std::ptr::null_mut(), 0, &input)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(why.kind, RefusalKind::NotWritten);
    assert!(why.reason.contains("banded mode"), "{}", why.reason);
}

/// A team stage of a unit in the mode the contract names, on the target the contract names.
fn stage_unit(mode: &str, target: Target) -> (Unit, TeamStage) {
    let mut unit = unit_in(mode);
    unit.target = target;
    let stage = TeamStage {
        grid: Grid::Keys,
        planes: vec![],
        reads: vec![],
        steps: vec![],
        active: None,
    };
    unit.form = Form::TeamStage(stage.clone());
    (unit, stage)
}

#[test]
fn the_host_refuses_a_banded_team_stage_by_name_and_the_device_writes_it() {
    let (unit, stage) = stage_unit("Banded", Target::Host);
    // the stage is written before the host entry refuses its mode, so the stage refuses on its own
    let why = stage_body(std::ptr::null_mut(), &unit, &stage)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(why.kind, RefusalKind::NotWritten);
    assert!(
        why.reason.contains("team stage in a banded scalar mode"),
        "{}",
        why.reason
    );
    let (unit, stage) = stage_unit("Banded", Target::Device);
    assert!(stage_body(std::ptr::null_mut(), &unit, &stage).is_ok());
}
