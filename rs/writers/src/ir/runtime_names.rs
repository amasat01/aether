//! What the host runtime and the emitted text fix, which the wing analyses must not know: the C++
//! type names of an argument slot, the spelling of derived plane names, and the most trips of a
//! loop the host compiler unrolls.

/// The most trips of a loop the host compiler fully unrolls.
pub const UNROLL_LIMIT: i64 = 16;

/// The suffix of the plane that holds the correction of a compensated accumulate.
pub const COMPANION_SUFFIX: &str = "_c";

/// The name of the plane that holds the correction of the accumulate `name`.
pub fn companion_of(name: &str) -> String {
    format!("{name}{COMPANION_SUFFIX}")
}

/// The name of the branch-mix plane of the dispatch of that name.
pub fn mix_plane(dispatch_name: &str) -> String {
    format!("{dispatch_name}_mix")
}

/// The roles whose slot is always the scalar handle, whatever its plane.
const HANDLE_ONLY: [&str; 6] = [
    "per_sample",
    "lookup",
    "terminated",
    "wide_in",
    "wide_out",
    "accum_out",
];

/// How a slot of the role crosses the launch boundary: the runtime's `GRefMirror` for a plane
/// read by reference, its `ScalarHandle`, or `value` for a by-value argument. A role that is a
/// handle only but carries a plane of a rank, or an unknown role, is refused.
pub fn mirror_of(role: &str, rank: usize) -> Result<&'static str, String> {
    if role == "mutable" {
        return Ok(if rank > 0 {
            "GRefMirror"
        } else {
            "ScalarHandle"
        });
    }
    if HANDLE_ONLY.contains(&role) && rank > 0 {
        return Err(format!(
            "role {role:?} carries a rank-{rank} plane, but the map pins it to the scalar handle"
        ));
    }
    match role {
        "out" | "vec_in" | "mat_in" => Ok("GRefMirror"),
        "uniform" | "nsamples" => Ok("value"),
        _ if HANDLE_ONLY.contains(&role) => Ok("ScalarHandle"),
        _ => Err(format!("unknown role {role:?}")),
    }
}
