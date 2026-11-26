//! Closed fabric with shared, smooth vertices on each face. UVs follow the
//! unpleated cut length even when the heading gathers into a narrow stack.
use super::*;

pub(super) fn build(a: &mut Assembly, label: &str, f: &FittingRoomSpec, b: &CurtainBay, z: f32) {
    let span = b.right - b.left;
    let width = span - b.clear_opening;
    let left = if b.gathered_left {
        b.left
    } else {
        b.right - width
    };
    let full = span * 1.34;
    let waves = b.pleats as f32;
    let frequency = TAU * waves;
    let depth = (full * full - width * width).sqrt() / (4. * waves);
    let upper = f.rail_height - 0.046;
    let height = upper - f.hem_height;
    let g = a.part(f.curtain_stock, label);
    let base = g.positions.len() as u32;
    let steps = b.pleats * 8;
    const ROWS: usize = 3;
    let thickness = Vec3::Z * 0.003;

    let gather = if b.gathered_left { -0.075 } else { 0.075 };
    let abscissa = |u: f32| left + width * (u + gather * (PI * u).sin());
    for j in 0..=ROWS {
        let v = j as f32 / ROWS as f32;
        for i in 0..=steps {
            let u = i as f32 / steps as f32;
            let phase = frequency * u;
            let (wave, peak) = phase.sin_cos();
            let (bend, slope) = (PI * u).sin_cos();
            // Keep fold crests directly below the sewn heading. A lateral phase
            // shift between rows can turn the shading normal across a narrow fold.
            let amplitude = depth * (1. + 0.10 * (1. - v));
            let sag = 0.008 * (1. - peak);
            let bow = 0.014 * bend;
            let p = Vec3::new(
                abscissa(u),
                f.hem_height + v * (height - sag),
                z + amplitude * wave + (1. - v) * bow,
            );
            let du = Vec3::new(
                width * (1. + gather * PI * slope),
                -0.008 * v * frequency * wave,
                amplitude * frequency * peak + (1. - v) * 0.014 * PI * slope,
            );
            let dv = Vec3::new(0., height - sag, -depth * 0.10 * wave - bow);
            let normal = du.cross(dv).normalize();
            g.positions
                .extend([p.to_array(), (p - thickness).to_array()]);
            g.normals.extend([normal.to_array(), (-normal).to_array()]);
            g.uvs.extend([[u * full, v * height]; 2]);
        }
    }
    let index = |i: usize, j: usize| base + 2 * (j * (steps + 1) + i) as u32;
    for j in 0..ROWS {
        for i in 0..steps {
            let [p, q, r, s] = [
                index(i, j),
                index(i + 1, j),
                index(i + 1, j + 1),
                index(i, j + 1),
            ];
            g.indices
                .extend([p, q, r, p, r, s, q + 1, p + 1, s + 1, q + 1, s + 1, r + 1]);
            if j == 0 || j + 1 == ROWS || i == 0 || i + 1 == steps {
                let [p, q, r, s] = [p, q, r, s].map(|k| Vec3::from_array(g.positions[k as usize]));
                if j == 0 {
                    g.metric_quad([p, p - thickness, q - thickness, q], -Vec3::Y);
                }
                if j + 1 == ROWS {
                    g.metric_quad([s, r, r - thickness, s - thickness], Vec3::Y);
                }
                if i == 0 {
                    g.metric_quad([p, s, s - thickness, p - thickness], -Vec3::X);
                }
                if i + 1 == steps {
                    g.metric_quad([q, q - thickness, r - thickness, r], Vec3::X);
                }
            }
        }
    }
    // A glide joins each heading trough to the rod, on the inside of the jambs.
    let heading = upper - 0.016;
    for k in 0..b.pleats {
        let x = abscissa((k as f32 + 0.5) / waves);
        a.box_part(
            Surface::Metal,
            FIXTURE,
            Vec3::new(x, (f.rail_height + heading) * 0.5, z),
            Vec3::new(0.014, f.rail_height - heading + 0.004, 0.011),
            0.,
        );
    }
}
