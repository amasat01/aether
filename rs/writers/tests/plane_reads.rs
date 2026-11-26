//! A plane leaf that states axes and is read only through `At` gets no local of its own in the body text (its view is read by
//! index); the same plane read in the plain way, and a plane without axes, still do.
//!
//! Runs only when RAPTOR_IR_LIB names the IR core's static library (see build.rs): the body is written from a built graph.
#![cfg(raptor_ir_lib)]
use aether_writers::contract::types::{Access, Axis, Dtype, Role, Slot, Unit, Walk};
use aether_writers::door::{self, RxCtx};
use aether_writers::render_body;
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

/// How the body reads the plane `a`.
#[derive(Clone, Copy)]
enum Read {
    /// Only at the index `i`: `y = a[i]`.
    OnlyAt,
    /// At the index and also as the lane's own element: `y = a[i] + a`.
    AtAndOwn,
}

/// The body text of `y = <reads of a>` where the plane `a` states one axis of four or none.
fn body_of(with_axes: bool, read: Read) -> String {
    // SAFETY: the context is made here, used only through the calls below and freed at the end.
    let c = unsafe { rx_ctx_new() };
    let axes = if with_axes {
        ", ('axes', (('c', 4),))"
    } else {
        ""
    };
    let leaf = format!(
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'a'), ('quantity', None), ('wire_index', 0), ('sorted', False){axes})"
    );
    let a = make(c, "Leaf", "f64", &[], &leaf);
    let index = make(c, "Const", "i64", &[], "(('literal', 2),)");
    let at = make(c, "At", "f64", &[a, index], "()");
    let value = match read {
        Read::OnlyAt => at,
        Read::AtAndOwn => make(
            c,
            "Op",
            "f64",
            &[at, a],
            "(('kind', 'add'), ('literal', None))",
        ),
    };
    let y = make(c, "Assign", "f64", &[value], "(('name', 'y'),)");
    let mut walk = u32::MAX;
    // SAFETY: the sink list holds one live handle and `walk` receives the handle of the walk.
    let status = unsafe {
        rx_canonical(
            c,
            [y].as_ptr(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut walk,
        )
    };
    assert_eq!(status, 0, "{}", door::last_error(c));
    let slot = |role, name: &str, index, axes| Slot {
        role,
        name: name.into(),
        dtype: Dtype::F64,
        shape: vec![],
        index,
        axes,
        zeroed: false,
    };
    let stated = with_axes.then(|| {
        vec![Axis {
            label: "c".into(),
            extent: Some(4),
        }]
    });
    let mut unit = Unit::from_json(UNIT).expect("a unit");
    unit.sinks = vec![y];
    unit.walk = Some(Walk {
        access: Access::SampleLocal,
        finish: None,
        slots: vec![
            slot(Role::PlaneOut, "y", 0, None),
            slot(Role::PlaneIn, "a", 1, stated),
        ],
    });
    let text = render_body(c, walk, &unit, "").unwrap_or_else(|why| panic!("{}", why.reason));
    // SAFETY: the context is live and nothing reads it after this.
    unsafe { rx_ctx_free(c) };
    text
}

/// The line that reads the lane's own element of the plane `a` into a local of its own.
const OWN_LANE_READ: &str = "const auto psc_a_i = psc_a[i].eval();";

#[test]
fn a_plane_with_axes_read_only_by_index_gets_no_own_lane_read() {
    let text = body_of(true, Read::OnlyAt);
    // the view with static extents is called with the index; `psc_a[i].eval()` does not compile for it
    assert!(
        text.contains("psc_a(aether::edge::clamp(static_cast<Int>(2), psc_a.extent(0)))"),
        "{text}"
    );
    assert!(!text.contains("psc_a[i]"), "{text}");
}

#[test]
fn a_plane_read_at_the_lanes_own_element_keeps_its_own_lane_read() {
    for with_axes in [true, false] {
        let text = body_of(with_axes, Read::AtAndOwn);
        assert!(
            text.contains(OWN_LANE_READ),
            "axes stated: {with_axes}\n{text}"
        );
    }
}
