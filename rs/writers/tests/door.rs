//! Runs only when RAPTOR_IR_LIB names the IR core's static library (see build.rs): the helpers read a small graph back through the door.
#![cfg(raptor_ir_lib)]
use aether_writers::door::{self, RxCtx, Scalar};
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

fn mk(c: *mut RxCtx, cls: &str, ops: &[u32], fields: &str) -> u32 {
    let (cls, dt) = (CString::new(cls).unwrap(), CString::new("f64").unwrap());
    let mut out = u32::MAX;
    let st = unsafe {
        rx_make(
            c,
            cls.as_ptr(),
            dt.as_ptr(),
            [].as_ptr(),
            0,
            std::ptr::null(),
            ops.as_ptr(),
            ops.len(),
            fields.as_ptr() as *const c_char,
            fields.len(),
            &mut out,
        )
    };
    assert_eq!(st, 0, "{}", door::last_error(c));
    out
}

#[test]
fn helpers_read_a_graph_back() {
    let c = unsafe { rx_ctx_new() };
    let leaf = |n: &str| {
        format!("(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{n}'), ('quantity', None), ('wire_index', 0), ('sorted', False))")
    };
    let (a, b) = (
        mk(c, "Leaf", &[], &leaf("a")),
        mk(c, "Leaf", &[], &leaf("b")),
    );
    let s = mk(c, "Op", &[a, b], "(('kind', 'add'), ('literal', None))");
    let y = mk(c, "Assign", &[s], "(('name', 'y'),)");
    assert!(door::class_is(c, s, "Op").unwrap());
    assert!(door::class_is(c, y, "Assign").unwrap());
    assert_eq!(door::ttype_of(c, a).unwrap().dtype, "f64");
    assert_eq!(door::operands(c, s).unwrap(), vec![a, b]);
    assert_eq!(door::operands(c, a).unwrap(), Vec::<u32>::new());
    assert_eq!(door::text_field(c, a, "name").unwrap(), "a");
    assert_eq!(door::text_field(c, s, "kind").unwrap(), "add");
    assert!(door::text_field(c, s, "nope").is_err());
    let mut w = 0u32;
    assert_eq!(
        unsafe {
            rx_canonical(
                c,
                [y].as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                &mut w,
            )
        },
        0,
        "{}",
        door::last_error(c)
    );
    assert_eq!(door::walk_order(c, w).unwrap(), vec![a, b, s, y]);
    assert_eq!(door::digest(c, w).unwrap().len(), 64);
    let args = door::walk_args(c, w).unwrap();
    assert_eq!(
        args,
        [("mutable", "y"), ("plane_in", "a"), ("plane_in", "b")]
            .map(|(r, n)| (r.to_string(), n.to_string()))
    );
    assert!(door::walk_binds(c, w, "plane_in", "a").unwrap());
    assert!(!door::walk_binds(c, w, "mutable", "a").unwrap());
    assert!(!door::walk_binds(c, w, "terminated", "y").unwrap());
    assert!(door::walk_binds(c, 99, "plane_in", "a").is_err());
    let like = door::canonical_over(c, &[y], w).unwrap();
    assert_eq!(door::walk_args(c, like).unwrap(), args);
    assert!(door::class_is(c, s, "Op").unwrap());
    assert!(!door::class_is(c, s, "NoSuchClass").unwrap());
    assert_eq!(
        door::text_field(c, a, "role").ok(),
        door::field_text(c, a, "role").unwrap()
    );
    unsafe { rx_ctx_free(c) };
}

/// Makes a node of the given dtype, shape and tag.
fn mk_typed(
    c: *mut RxCtx,
    cls: &str,
    dtype: &str,
    shape: &[i64],
    tag: Option<&str>,
    ops: &[u32],
    fields: &str,
) -> u32 {
    let (cls, dt) = (CString::new(cls).unwrap(), CString::new(dtype).unwrap());
    let tag = tag.map(|t| CString::new(t).unwrap());
    let mut out = u32::MAX;
    let st = unsafe {
        rx_make(
            c,
            cls.as_ptr(),
            dt.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
            ops.as_ptr(),
            ops.len(),
            fields.as_ptr() as *const c_char,
            fields.len(),
            &mut out,
        )
    };
    assert_eq!(st, 0, "{}", door::last_error(c));
    out
}

#[test]
fn typed_readers_answer_for_a_small_graph() {
    let c = unsafe { rx_ctx_new() };
    let leaf = |n: &str| {
        format!(
            "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{n}'), ('quantity', None), \
             ('wire_index', 0), ('sorted', False))"
        )
    };
    let a = mk_typed(c, "Leaf", "f64", &[], None, &[], &leaf("a"));
    let t = mk_typed(c, "Leaf", "f64", &[2, 3], Some("lanes"), &[], &leaf("t"));
    let n = mk_typed(c, "Leaf", "i32", &[4], None, &[], &leaf("n"));
    let k = mk_typed(c, "Const", "f64", &[], None, &[], "(('literal', 2.5),)");
    let s = mk_typed(
        c,
        "Op",
        "f64",
        &[],
        None,
        &[a, k],
        "(('kind', 'add'), ('literal', None))",
    );
    let index = mk_typed(c, "Const", "i64", &[], None, &[], "(('literal', 2),)");
    let at = mk_typed(c, "At", "f64", &[], None, &[a, index], "()");
    let y = mk_typed(c, "Assign", "f64", &[], None, &[s], "(('name', 'y'),)");
    assert_eq!(
        door::field_text(c, a, "role").unwrap().as_deref(),
        Some("plane_in")
    );
    assert_eq!(door::field_text(c, a, "quantity").unwrap(), None);
    assert_eq!(door::field_int(c, a, "wire_index").unwrap(), Some(0));
    assert_eq!(door::field_flag(c, a, "sorted").unwrap(), Some(false));
    assert_eq!(door::field_real(c, k, "literal").unwrap(), Some(2.5));
    assert_eq!(door::field_refs(c, s, "literal").unwrap(), None);
    assert!(door::field_text(c, a, "nope").is_err());
    assert_eq!(door::field_node_list(c, a, "role").unwrap(), Some(vec![]));
    assert!(door::field_names(c, a, "feathers").unwrap().unwrap().len() < 99);
    for h in [a, t, n, k, s, at, y] {
        assert!(door::ttype_of(c, h).unwrap().shape.len() <= 2, "node {h}");
    }
    assert_eq!(door::ttype_of(c, a).unwrap().tag, None);
    assert_eq!(door::ttype_of(c, t).unwrap().tag.as_deref(), Some("lanes"));
    assert_eq!(door::ttype_of(c, t).unwrap().shape, vec![2, 3]);
    assert_eq!(door::ttype_of(c, n).unwrap().dtype, "i32");
    assert!(door::field_int(c, a, "name").is_err());
    let mut walk = 0u32;
    let st = unsafe {
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
    assert_eq!(st, 0, "{}", door::last_error(c));
    assert!(!door::in_loop_body(c, walk, s).unwrap());
    assert!(door::in_loop_body(c, walk, t).unwrap());
    assert!(door::in_loop_body(c, walk, 99).is_err());
    unsafe { rx_ctx_free(c) };
}

#[test]
fn literals_read_back_typed() {
    let c = unsafe { rx_ctx_new() };
    let leaf = "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'a'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
    let a = mk(c, "Leaf", &[], leaf);
    for (literal, want) in [
        ("None", Scalar::Absent),
        ("True", Scalar::Flag(true)),
        ("False", Scalar::Flag(false)),
        ("3", Scalar::Int(3)),
        ("-7", Scalar::Int(-7)),
        ("0.5", Scalar::Real(0.5)),
        ("1e-05", Scalar::Real(1e-5)),
        ("100.0", Scalar::Real(100.0)),
        ("1e+16", Scalar::Real(1e16)),
        ("-0.0", Scalar::Real(-0.0)),
        ("123456789.125", Scalar::Real(123456789.125)),
        ("'x'", Scalar::Text("x".into())),
        ("(2,)", Scalar::Ints(vec![2])),
        ("(1, 2)", Scalar::Ints(vec![1, 2])),
        ("()", Scalar::Ints(vec![])),
    ] {
        let op = mk(
            c,
            "Op",
            &[a],
            &format!("(('kind', 'component'), ('literal', {literal}))"),
        );
        let held = door::field_scalar(c, op, "literal").unwrap();
        assert!(held.same_value(&want), "{literal}: {held:?}");
    }
    let op = mk(c, "Op", &[a, a], "(('kind', 'add'), ('literal', None))");
    assert_eq!(door::field_node_list(c, op, "literal").unwrap(), None);
    assert_eq!(door::field_ints(c, op, "literal").unwrap(), None);
    assert_eq!(
        door::field_text(c, a, "role").unwrap().as_deref(),
        Some("plane_in")
    );
    unsafe { rx_ctx_free(c) };
}
