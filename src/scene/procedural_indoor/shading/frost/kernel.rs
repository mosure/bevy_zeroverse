//! Stratified antipodal samples of the pinned Bevy transmission kernel.
//! Four quality levels use 4, 8, 16 or 32 pairs. Build the directions and
//! footprints once at shader installation, outside the per-fragment loop.
use std::fmt::Write;

// Sorted radial samples of Bevy Ultra, including both checkerboard radii.
const RADII: [f64; 64] = [
    0.05625000, 0.06250000, 0.07031250, 0.07812500, 0.08437500, 0.09375000, 0.09843750, 0.10937500,
    0.11250000, 0.12500000, 0.14062500, 0.15625000, 0.16875000, 0.16875000, 0.18750000, 0.18750000,
    0.19687500, 0.21093750, 0.21875000, 0.22500000, 0.23437500, 0.25000000, 0.25312500, 0.28125000,
    0.28125000, 0.28125000, 0.29531250, 0.31250000, 0.31250000, 0.32812500, 0.33750000, 0.33750000,
    0.35156250, 0.37500000, 0.37500000, 0.39062500, 0.39375000, 0.39375000, 0.42187500, 0.42187500,
    0.43750000, 0.43750000, 0.45000000, 0.46875000, 0.46875000, 0.49218750, 0.49218750, 0.50000000,
    0.50625000, 0.54687500, 0.54687500, 0.56250000, 0.56250000, 0.59062500, 0.59062500, 0.62500000,
    0.65625000, 0.65625000, 0.67500000, 0.68906250, 0.75000000, 0.76562500, 0.78750000, 0.87500000,
];

fn cell(pairs: usize, index: usize) -> [f64; 4] {
    let q = (((index as f64 + 0.5) / pairs as f64) * 64.0 - 0.5).clamp(0., 63.);
    let lo = q as usize;
    let r = RADII[lo] + (RADII[(lo + 1).min(63)] - RADII[lo]) * q.fract();
    // Each direction also supplies its opposite. Space the axes over half a
    // turn, with the golden ratio to avoid repeated orientations on the rings.
    let angle = index as f64 * (std::f64::consts::PI * (5f64.sqrt() - 1.) * 0.5);
    // Wider strata toward the outer, lower-density part of the scattering cone.
    let diameter = (0.31 + 1.48 * r) / ((2 * pairs) as f64).sqrt();
    [r * angle.cos(), r * angle.sin(), diameter, diameter.log2()]
}

pub(in super::super) fn radial_cdf() -> String {
    format!(
        "array<f32,64>({})",
        RADII.map(|r| format!("{r:.8}")).join(",")
    )
}

pub(in super::super) fn shader() -> String {
    let mut shader = String::from("const frost_cells=array<vec4<f32>,60>(\n");
    for pairs in [4, 8, 16, 32] {
        for i in 0..pairs {
            let [x, y, w, log_w] = cell(pairs, i);
            writeln!(shader, "vec4<f32>({x:.9},{y:.9},{w:.9},{log_w:.9}),").unwrap();
        }
    }
    shader.push_str(
        ");\nfn frost_cell(i:i32,pairs:i32)->vec4<f32>{return frost_cells[pairs-4+i];}\n",
    );
    shader
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strata_at_every_quality_span_the_same_scattering_distribution() {
        let second_moment = RADII.iter().map(|r| r * r).sum::<f64>() / 64.;
        for pairs in [4, 8, 16, 32] {
            let mut variance = 0.;
            let mut last = 0.;
            for i in 0..pairs {
                let [x, y, diameter, log_d] = cell(pairs, i);
                let radius = x.hypot(y);
                assert!((last..=1.).contains(&radius));
                assert!((log_d.exp2() - diameter).abs() < 1e-12);
                last = radius;
                variance += radius * radius;
            }
            assert!((variance / pairs as f64 - second_moment).abs() < 0.017);
        }
    }
}
