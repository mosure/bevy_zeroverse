use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &LaundryCart, floor: f32) {
    if p.load == Load::Empty {
        return;
    }
    let mut r = stream(o.seed, u64::from_le_bytes(*b"LinenSet"));
    // The narrowest region of either basket bounds every fold. Disjoint lanes
    // give each pile its own contact patch, colour, length and level.
    let half = p.base_half_m * Vec2::new(0.86, 0.77);
    let columns = if p.load == Load::LooseLaundry || half.x > 0.29 {
        2
    } else {
        1
    };
    let width = 2. * half.x / columns as f32;
    let rotation = r.random_range(-0.07_f32..0.07);
    let lot = garment_stock::sample_slot(&mut r);
    for i in 0..columns {
        let x = -half.x + width * (i as f32 + 0.5);
        let depth = 2. * half.y * r.random_range(0.78..0.96);
        let z = (half.y - depth * 0.5) * r.random_range(-0.9..0.9);
        let height = (p.rim_height_m - floor) * p.fill_fraction * r.random_range(0.62..1.);
        let slot = if i == 0 || (p.load == Load::FoldedLinen && r.random_bool(0.65)) {
            lot
        } else {
            garment_stock::contrasting_slot(lot, &mut r)
        };
        let origin = Vec3::new(x, floor, z);
        if p.load == Load::FoldedLinen {
            // Folded towels have broad bearing surfaces and keep metric textile
            // UVs. Members of a stack share a cut with small offsets at each fold.
            let count = (height / 0.058).ceil().clamp(1., 7.) as usize;
            let mut towel = o.clone();
            towel.kind = ObjectKind::KitchenTowel;
            towel.seed = r.random();
            towel.size = Vec3::new(
                width * 0.89,
                (height / count as f32).min(0.074),
                depth * 0.90,
            );
            let mut folded = Assembly::default();
            crate::scene::procedural_indoor::objects::kitchen::build(&mut folded, &towel, VERSION);
            for ((surface, label), mesh) in &folded.parts {
                let g = a.part(*surface, &format!("{label}#finish{slot}"));
                for layer in 0..count {
                    let offset = Vec3::new(
                        r.random_range(-0.003..0.003),
                        layer as f32 * towel.size.y,
                        r.random_range(-0.004..0.004),
                    );
                    let yaw = rotation + r.random_range(-0.035..0.035);
                    g.append_rigid(mesh, Quat::from_rotation_y(yaw), origin + offset);
                }
            }
        } else {
            let phase = r.random_range(0.0..2. * PI);
            let g = a.part(Surface::Fabric, &format!("clothes#finish{slot}"));
            loose(g, origin, Vec3::new(width * 0.96, height, depth), phase);
        }
    }
}

// A closed, supported cloth bundle with compressed edges and broad ridges.
// Sample both neighbours of each grid cell once. UVs are in physical metres.
fn loose(g: &mut Geometry, origin: Vec3, size: Vec3, phase: f32) {
    let sample = |i: usize, j: usize| {
        let x = i as f32 / 8. * 2. - 1.;
        let z = j as f32 / 8. * 2. - 1.;
        let edge = (1. - x * x) * (1. - z * z);
        let y = 0.16
            + 0.57 * edge
            + 0.18 * (x * 8. + z * 3. + phase).sin() * (0.25 + 0.75 * edge)
            + 0.085 * (x * 3. - z * 9. + phase * 0.4).cos();
        let base = origin
            + Vec3::new(
                x * size.x * 0.5 * (1. - 0.14 * z.powi(4)),
                0.,
                z * size.z * 0.5 * (1. - 0.12 * x.powi(4)),
            );
        [base, base + Vec3::Y * size.y * y]
    };
    let grid: [[_; 9]; 9] = std::array::from_fn(|j| std::array::from_fn(|i| sample(i, j)));
    let normals: [[_; 9]; 9] = std::array::from_fn(|j: usize| {
        std::array::from_fn(|i: usize| {
            let du = grid[j][(i + 1).min(8)][1] - grid[j][i.saturating_sub(1)][1];
            let dv = grid[(j + 1).min(8)][i][1] - grid[j.saturating_sub(1)][i][1];
            dv.cross(du).normalize()
        })
    });
    for j in 1..=8 {
        let (prev, row) = (&grid[j - 1], &grid[j]);
        for i in 0..8 {
            let pts = [prev[i][1], row[i][1], row[i + 1][1], prev[i + 1][1]];
            let a = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize();
            let b = (pts[2] - pts[0]).cross(pts[3] - pts[0]).normalize();
            let crease = (a + b).normalize();
            let mut shading = [
                normals[j - 1][i],
                normals[j][i],
                normals[j][i + 1],
                normals[j - 1][i + 1],
            ];
            for n in &mut shading {
                if n.dot(a).min(n.dot(b)) <= 0. {
                    *n = crease;
                }
            }
            g.vertex_quad(pts, shading, pts.map(|v| v.xz().to_array()));
            g.metric_quad(
                [prev[i][0], prev[i + 1][0], row[i + 1][0], row[i][0]],
                Vec3::NEG_Y,
            );
            if j == 1 {
                g.metric_quad(
                    [prev[i][0], prev[i][1], prev[i + 1][1], prev[i + 1][0]],
                    Vec3::NEG_Z,
                );
            }
            if j == 8 {
                g.metric_quad(
                    [row[i][0], row[i + 1][0], row[i + 1][1], row[i][1]],
                    Vec3::Z,
                );
            }
        }
        for (i, n) in [(0, Vec3::NEG_X), (8, Vec3::X)] {
            g.metric_quad([prev[i][0], row[i][0], row[i][1], prev[i][1]], n);
        }
    }
}
