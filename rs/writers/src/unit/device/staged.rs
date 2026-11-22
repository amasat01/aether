//! The frame of a device block that serves several lanes of a warp-form kernel: the vector outputs
//! of its warp stores go
//! through a shared tile, and the block's threads then store the tile into the planes, neighbouring
//! lanes together.
use crate::contract::types::Build;
use crate::ir::door::{self, RxCtx};
use crate::refuse::{bad, Refusal};
use crate::spell::binding_name;

/// The narrowest vector held sliced for scalars of `dtype` on the device of the build: the card's
/// default. A card row of the
/// dtype (chosen by the nearest device features) or a forced override would change it, and neither
/// is read here.
fn slice_min(build: &Build, dtype: &str) -> Result<i64, Refusal> {
    let card = build
        .wide_card
        .as_ref()
        .ok_or_else(|| bad("spec: no wide card"))?;
    let scalar = match dtype {
        "f32" => "float32",
        "f64" => "float64",
        other => other,
    };
    if card.row_scalars.iter().any(|row| row == scalar) || card.forced {
        return Err(crate::refuse::nib(
            "staged rows over a card row or a forced wide setting",
        ));
    }
    card.slice_min.ok_or_else(|| bad("spec: no SLICE_MIN"))
}

/// The `(opened, closed)` frame of a block that serves `rows` lanes, one warp each, around the
/// body.
pub fn frame(
    ctx: *mut RxCtx,
    sinks: &[u32],
    rows: i64,
    build: &Build,
) -> Result<(String, String), Refusal> {
    let mut stages: Vec<(String, usize)> = vec![];
    for &sink in sinks {
        if !door::is_a(ctx, sink, "warp_store")? {
            continue;
        }
        let ty = crate::body::ty_of(ctx, sink)?;
        let Some(&width) = ty.shape.first() else {
            continue;
        };
        if (width as i64) < slice_min(build, &ty.dtype)? {
            continue;
        }
        stages.push((
            binding_name(
                &door::field_text(ctx, sink, "role")?.unwrap_or_default(),
                0,
                &door::field_text(ctx, sink, "name")?.unwrap_or_default(),
            )?,
            width,
        ));
    }
    let pad = rows + 1;
    let mut opened: Vec<String> = stages
        .iter()
        .map(|(name, width)| {
            format!(
                "    __shared__ Real hawk_stg_{name}[{}];",
                *width as i64 * pad
            )
        })
        .collect();
    opened.extend([
        format!("    __shared__ int hawk_wrote[{rows}];"),
        "    const int hawk_wb = static_cast<int>(threadIdx.x >> 5);".into(),
        "    const Int hawk_wl = static_cast<Int>(threadIdx.x & 31);".into(),
        "    (void)hawk_wl;".into(),
        format!("    const long long hawk_flat = static_cast<long long>(blockIdx.x) * {rows} + hawk_wb;"),
        "    if (hawk_wl == 0) hawk_wrote[hawk_wb] = 0;".into(),
        "    if (hawk_flat < nSamples) {".into(),
        "        const long long hawk_i = base + hawk_flat;".into(),
        "        const aether::SampleIndex i =".into(),
        "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));".into(),
        "        (void)hawk_i;".into(),
        "        (void)nSamples;".into(),
    ]);
    let mut closed: Vec<String> = vec!["    }".into(), "    __syncthreads();".into()];
    for (name, width) in &stages {
        closed.extend([
            format!("    for (int hawk_c = static_cast<int>(threadIdx.x); hawk_c < {}; hawk_c += {}) {{", *width as i64 * rows, 32 * rows),
            format!("        const int hawk_e = hawk_c / {rows}, hawk_s = hawk_c % {rows};"),
            format!("        if (hawk_wrote[hawk_s]) {name}[aether::SampleIndex::make(static_cast<std::size_t>(base + static_cast<long long>(blockIdx.x) * {rows} + hawk_s))].at(hawk_e) = hawk_stg_{name}[hawk_e * {pad} + hawk_s];"),
            "    }".into(),
        ]);
    }
    Ok((opened.join("\n"), closed.join("\n")))
}
