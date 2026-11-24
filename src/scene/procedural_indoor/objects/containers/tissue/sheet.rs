//! A gathered interfold with a narrow feed and an asymmetric free edge. Sample
//! each fold station once. Both optical faces share the same deformed lattice.
use super::*;
use crate::scene::procedural_indoor::materials::tissue_stock::PLY;

struct Crease {
    u: f32,
    root: f32,
    tip: f32,
    crest: f32,
    bow: f32,
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &TissueProgram, d: Dispenser) {
    let mut rng = stream(o.seed, u64::from_le_bytes(*b"PlyFold "));
    let s = o.size;
    let h = s.y * p.box_height;
    let hole = s.xz() * p.opening * 0.5;
    let reach = ((s.y - h) / (s.z * 0.55)).min(1.);
    let pull = d.mouth_offset_m.x + s.x * p.lean * reach;
    let top_half = (s.x * 0.5 - pull.abs() - 0.001) * d.flare;
    let root_half = hole.x * rng.random_range(0.40..0.74);
    let top_half = root_half + (top_half - root_half) * reach;
    let lean_z = d.curl * s.z * reach;
    // A handful of unequal folds opens into broad lobes. Interpolate the
    // crease controls before meshing so the exposed edge can curl through a
    // crest without forming a row of triangular points.
    let intervals = usize::from(d.pleats);
    let mut controls = Vec::with_capacity(intervals + 1);
    for i in 0..=intervals {
        let jitter = if i == 0 || i == intervals {
            0.
        } else {
            rng.random_range(-0.29..0.29)
        };
        let u = (i as f32 + jitter) * 2. / intervals as f32 - 1.;
        let sign = if i % 2 == 0 { 1. } else { -1. };
        controls.push((
            u,
            Vec3::new(
                sign * hole.y * rng.random_range(0.12..0.34),
                sign * s.z * p.fold * rng.random_range(0.32..1.),
                s.z * rng.random_range(-0.16..0.16),
            ),
        ));
    }
    let crown_width = rng.random_range(0.55..0.97);
    let twist = rng.random_range(-0.13..0.13) * s.z * reach;
    let profile = |u: f32, v: Vec3| {
        let crest = match d.draw {
            Draw::CornerPull => 0.22 + 0.78 * (0.5 + 0.5 * u * d.crest.signum()).powf(crown_width),
            Draw::Fan => {
                let offset = (u - d.crest * 0.72) / crown_width;
                0.30 + 0.70 / (1. + offset * offset)
            }
            _ => 0.66 + 0.34 * (p.phase + u * 2.1).cos(),
        };
        Crease {
            u,
            root: v.x,
            tip: v.y + twist * u,
            crest: crest * (1. + 0.06 * (p.phase + 3.2 * u).sin()),
            bow: v.z,
        }
    };
    let mut creases = Vec::with_capacity(4 * intervals + 1);
    for pair in controls.windows(2) {
        let [(u0, v0), (u1, v1)] = [pair[0], pair[1]];
        for step in 0..4 {
            let f = step as f32 * 0.25;
            let eased = f * f * (3. - 2. * f);
            creases.push(profile(u0 + (u1 - u0) * f, v0.lerp(v1, eased)));
        }
    }
    let (u, v) = controls[intervals];
    creases.push(profile(u, v));
    let tallest = creases.iter().map(|c| c.crest).fold(0., f32::max);
    const ROWS: usize = 7;
    let cols = creases.len();
    let mut points = Vec::with_capacity(cols * ROWS);
    for row in 0..ROWS {
        // Row zero is buried in the paper pack. Keep the feed narrow through
        // the cut before it can flare or turn towards the pulled corner.
        let v = row.saturating_sub(1) as f32 / (ROWS - 2) as f32;
        for c in &creases {
            let t = v * c.crest / tallest;
            let spread = t * t * (3. - 2. * t);
            // Each crease advances on a straight line in the projection onto
            // XY. Even at a low draw height two neighboring folds cannot cross.
            let center =
                d.mouth_offset_m + Vec2::new((pull - d.mouth_offset_m.x) * t, lean_z * t * t);
            let width = root_half + (top_half - root_half) * t;
            let y = if row == 0 {
                h * d.fill_fraction - 0.00025
            } else {
                h + (s.y - h) * t
            };
            let z =
                center.y + c.root * (1. - spread) + c.tip * spread + c.bow * reach * t * (1. - t);
            points.push(Vec3::new(center.x + c.u * width, y, z));
        }
    }
    let g = a.part(Surface::Paper, PLY);
    let base = g.positions.len() as u32;
    let gauge = rng.random_range(0.00012..0.00026);
    for y in 0..ROWS {
        for x in 0..cols {
            let across =
                points[y * cols + (x + 1).min(cols - 1)] - points[y * cols + x.saturating_sub(1)];
            let up =
                points[(y + 1).min(ROWS - 1) * cols + x] - points[y.saturating_sub(1) * cols + x];
            let normal = across.cross(up).normalize();
            for side in [1., -1.] {
                let pt = points[y * cols + x];
                g.positions
                    .push((pt + Vec3::Z * (side * gauge * 0.5)).to_array());
                g.normals.push((normal * side).to_array());
                g.uvs.push([creases[x].u * top_half, (pt.y - h)]);
            }
        }
    }
    let at = |j: usize| base + 2 * j as u32;
    for row in 0..ROWS - 1 {
        for x in 0..cols - 1 {
            let i = row * cols + x;
            let [a, b, c, d] = [i, i + 1, i + cols + 1, i + cols].map(at);
            for face in [[a, b, c], [a, c, d]] {
                let [p, q, r] = face.map(|i| Vec3::from(g.positions[i as usize]));
                let n = (q - p).cross(r - p).normalize();
                let [i, j, k] = face.map(|i| {
                    if n.dot(Vec3::from(g.normals[i as usize])) >= 0.45 {
                        return i;
                    }
                    // Split sharp creases. Interpolating across an acute fold
                    // can put its lighting normal behind the visible sheet.
                    let split = g.positions.len() as u32;
                    for (index, normal) in [(i, n), (i + 1, -n)] {
                        g.positions.push(g.positions[index as usize]);
                        g.uvs.push(g.uvs[index as usize]);
                        g.normals.push(normal.to_array());
                    }
                    split
                });
                g.indices.extend([i, j, k, i + 1, k + 1, j + 1]);
            }
        }
    }
    // Close the three exposed cut edges. The bottom edge sits inside the pack.
    let mut edge = |j, k| {
        let face =
            [at(j), at(j) + 1, at(k) + 1, at(k)].map(|i| Vec3::from(g.positions[i as usize]));
        g.metric_quad(face, (face[1] - face[0]).cross(face[2] - face[0]));
    };
    for row in 0..ROWS - 1 {
        edge((row + 1) * cols, row * cols);
        edge(row * cols + cols - 1, (row + 1) * cols + cols - 1);
    }
    for x in 0..cols - 1 {
        edge((ROWS - 1) * cols + x + 1, (ROWS - 1) * cols + x);
    }
}
