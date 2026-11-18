//! What the host runtime and the emitted text fix, which the wing analyses must not know: the C++
//! type names of an argument slot, the spelling of derived plane names, and the most trips of a
//! loop the host compiler unrolls.
use crate::refuse::{bad, Refusal};

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
pub fn mirror_of(role: &str, rank: usize) -> Result<&'static str, Refusal> {
    if role == "mutable" {
        return Ok(if rank > 0 {
            "GRefMirror"
        } else {
            "ScalarHandle"
        });
    }
    if HANDLE_ONLY.contains(&role) && rank > 0 {
        return Err(bad(format!(
            "role {role:?} carries a rank-{rank} plane, but the map pins it to the scalar handle"
        )));
    }
    match role {
        "out" | "vec_in" | "mat_in" => Ok("GRefMirror"),
        "uniform" | "nsamples" => Ok("value"),
        _ if HANDLE_ONLY.contains(&role) => Ok("ScalarHandle"),
        _ => Err(bad(format!("unknown role {role:?}"))),
    }
}

#[cfg(test)]
mod tests {
    //! The launch-boundary shape of a slot by its role and rank.
    use super::mirror_of;

    /// Oracle: the table above, read by hand: a role pinned to the scalar handle refuses a plane of a rank (a plane needs the larger
    /// mirror, which that role has no packing for) and takes a rank-0 slot; a role read by reference takes any rank.
    #[test]
    fn a_ranked_plane_in_a_handle_only_role_is_refused_and_its_neighbours_are_not() {
        for role in [
            "per_sample",
            "lookup",
            "terminated",
            "wide_in",
            "wide_out",
            "accum_out",
        ] {
            let why = mirror_of(role, 1).unwrap_err();
            assert_eq!(
                why.reason,
                format!("role {role:?} carries a rank-1 plane, but the map pins it to the scalar handle")
            );
            assert_eq!(mirror_of(role, 0).unwrap(), "ScalarHandle", "{role}");
        }
        assert_eq!(mirror_of("vec_in", 1).unwrap(), "GRefMirror");
        assert_eq!(mirror_of("mutable", 2).unwrap(), "GRefMirror");
        assert_eq!(mirror_of("mutable", 0).unwrap(), "ScalarHandle");
    }
}
