//! The sinks written beside a group are refused by name when the two sides of the group cannot be ordered: a plane one side
//! writes and the other reads at ANOTHER lane, or rewrites without accumulating. The IR does not look at the sinks beside a group
//! (a group node carries only its own body), so the host writer is where these documents are refused.
//!
//! Runs only when RAPTOR_IR_LIB names the IR core's static library (see build.rs): the body is written from a built graph.
#![cfg(raptor_ir_lib)]
use aether_writers::contract::types::{Access, Dtype, Role, Slot, Unit, Walk};
use aether_writers::door::{self, RxCtx};
use aether_writers::unit::group;
use aether_writers::RefusalKind;
use std::ffi::{c_char, CString};

extern "C" {
    fn rx_ctx_new() -> *mut RxCtx;
    fn rx_ctx_free(ctx: *mut RxCtx);
    fn rx_make(
        ctx: *mut RxCtx,
        cls: *const c_char,
        dtype: *const c_char,
        shape: *const i64,
        rank: usize,
        tag: *const c_char,
        ops: *const u32,
        nops: usize,
        fields_kt: *const c_char,
        n: usize,
        out: *mut u32,
    ) -> i32;
    fn rx_canonical(
        ctx: *mut RxCtx,
        sinks: *const u32,
        n: usize,
        declared_kt: *const c_char,
        force: *const c_char,
        exempt_kt: *const c_char,
        out: *mut u32,
    ) -> i32;
}

/// A unit as the contract writes it (the host, 64-bit floats), written from its walk.
const UNIT: &str = r#"{"v":1,"name":"k","target":"Host","scalar":"Float64","kind":{"guard":{"Masks":[]},"lanes":false,"sink":"Plain"},"form":"Plain","sinks":[],"body":"Walk","segment":null,"thread_box":null,"planed_scratch":[],"rows":1,"team":null,"build":{"form":{"lazy_leaf_width":8,"rolled_fold_min":4,"uniform_run":2},"fold_form":"Auto","lanes":{"run_max":8,"tile":[4,64]},"cells_row":0,"blend":true,"warps":1},"machine":{"isa":"Arm64","vector_isa":"neon","exact_libm":true,"routed_pow":false,"caches":{"l1d_bytes":65536,"l2_bytes":4194304,"source":"probe"},"select_weight":{"float64":1.5}},"exports":{"Sizes":[1,2,3,4,5]},"literal_table":false}"#;

/// A node of the class over the operands.
fn make(c: *mut RxCtx, class: &str, dtype: &str, ops: &[u32], fields: &str) -> u32 {
    let (class, dtype) = (CString::new(class).unwrap(), CString::new(dtype).unwrap());
    let mut out = u32::MAX;
    // SAFETY: `c` is a live context and every pointer is valid for the length given beside it.
    let status = unsafe {
        rx_make(
            c,
            class.as_ptr(),
            dtype.as_ptr(),
            [].as_ptr(),
            0,
            std::ptr::null(),
            ops.as_ptr(),
            ops.len(),
            fields.as_ptr().cast(),
            fields.len(),
            &mut out,
        )
    };
    assert_eq!(status, 0, "{}", door::last_error(c));
    out
}

/// A plane leaf the unit reads: `role` is its role in the walk, `name` its name.
fn plane(c: *mut RxCtx, role: &str, name: &str) -> u32 {
    let fields = format!(
        "(('kind', 'vocab_read'), ('role', '{role}'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False), ('axes', ()))"
    );
    make(c, "Leaf", "f64", &[], &fields)
}

/// What the sinks beside the group do.
#[derive(Clone, Copy)]
enum Beside {
    /// `z = y[2]`: reads the plane `y` the group writes at another lane.
    ReadsAtAnotherLane,
    /// `y = 1.0`: writes the plane `y` the group writes, without accumulating.
    RewritesWithoutAccumulating,
}

/// The refusal of the host writer for a count group of 8 lanes that stores `y = t[rank]` (after filling `t` from `x`) beside
/// which the sinks of `beside` stand.
fn refusal_of(beside: Beside) -> aether_writers::Refusal {
    // SAFETY: the context is made here, used only through the calls below and freed at the end.
    let c = unsafe { rx_ctx_new() };
    let x = plane(c, "plane_in", "x");
    let scratch = make(
        c,
        "Scratch",
        "f64",
        &[],
        "(('kind', 'scratch'), ('role', 'scratch'), ('name', 't'), ('quantity', None), ('wire_index', 0), ('sorted', False), ('axes', (('rank', 8),)))",
    );
    let rank = make(c, "GroupRank", "i32", &[], "()");
    let fill = make(
        c,
        "Update",
        "f64",
        &[x, rank],
        "(('kind', 'update'), ('role', 'scratch'), ('name', 't'), ('op', 'store'), ('conflict', 'unique'), ('summation', 'auto'), ('order', 'any'), ('returns_old', False), ('reduced', None), ('edge', 'unchecked'), ('axes', (('rank', 8),)))",
    );
    let read = make(
        c,
        "At",
        "f64",
        &[scratch, rank],
        "(('kind', 'at'), ('literal', None), ('edge', 'clamp'))",
    );
    let store_y = make(
        c,
        "Assign",
        "f64",
        &[read],
        "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'), ('axes', ()))",
    );
    let group = make(
        c,
        "Group",
        "bool",
        &[],
        &format!(
            "(('membership', 'count'), ('count', 8), ('grid', ()), ('axes', ()), ('max_size', None), ('scratch', (#{scratch},)), ('body_sinks', (#{fill}, #{store_y})))"
        ),
    );
    let outside = match beside {
        Beside::ReadsAtAnotherLane => {
            let y = plane(c, "plane_in", "y");
            let two = make(c, "Const", "i64", &[], "(('literal', 2),)");
            let at = make(c, "At", "f64", &[y, two], "()");
            make(
                c,
                "Assign",
                "f64",
                &[at],
                "(('kind', 'assign'), ('role', 'mutable'), ('name', 'z'), ('axes', ()))",
            )
        }
        Beside::RewritesWithoutAccumulating => {
            let one = make(c, "Const", "f64", &[], "(('literal', 1.0),)");
            make(
                c,
                "Assign",
                "f64",
                &[one],
                "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'), ('axes', ()))",
            )
        }
    };
    let sinks = [group, outside];
    let mut walk = u32::MAX;
    // SAFETY: the sink list holds live handles and `walk` receives the handle of the walk.
    let status = unsafe {
        rx_canonical(
            c,
            sinks.as_ptr(),
            sinks.len(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut walk,
        )
    };
    assert_eq!(status, 0, "{}", door::last_error(c));
    let slot = |role, name: &str, index| Slot {
        role,
        name: name.into(),
        dtype: Dtype::F64,
        shape: vec![],
        index,
        axes: None,
        zeroed: false,
    };
    let mut unit = Unit::from_json(UNIT).expect("a unit");
    unit.sinks = sinks.to_vec();
    unit.walk = Some(Walk {
        access: Access::SampleLocal,
        finish: None,
        slots: vec![
            slot(Role::PlaneOut, "y", 0),
            slot(Role::PlaneOut, "z", 1),
            slot(Role::PlaneIn, "x", 2),
        ],
    });
    let found = group::of(c, walk, &unit).expect("the unit's sinks are read");
    let node = found.expect("the unit holds a group");
    let why = group::group_body(c, walk, &unit, node)
        .map(|_| ())
        .expect_err("the document is refused");
    // SAFETY: the context is live and nothing reads it after this.
    unsafe { rx_ctx_free(c) };
    why
}

#[test]
fn a_plane_the_group_writes_and_the_sinks_beside_it_read_at_another_lane_is_refused_by_name() {
    let why = refusal_of(Beside::ReadsAtAnotherLane);
    assert_eq!(why.kind, RefusalKind::NotWritten);
    assert!(
        why.reason.contains("the plane 'y' written on one side of the group and rewritten or read at another lane by the sinks beside the group"),
        "{}",
        why.reason
    );
}

#[test]
fn a_plane_the_group_writes_and_the_sinks_beside_it_rewrite_without_accumulating_is_refused_by_name(
) {
    let why = refusal_of(Beside::RewritesWithoutAccumulating);
    assert_eq!(why.kind, RefusalKind::NotWritten);
    assert!(
        why.reason.contains("the plane 'y' written on one side of the group and rewritten or read at another lane by the sinks beside the group"),
        "{}",
        why.reason
    );
}
