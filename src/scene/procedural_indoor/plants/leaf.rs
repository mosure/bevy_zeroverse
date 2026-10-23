use super::*;

/// Pitch the lamina independently of the branch. Available vertical clearance
/// constrains the sampled direction before tessellation, never the vertices.
pub(super) fn incline(dir: Vec3, length: f32, pitch: f32, y: f32, soil: f32, ceiling: f32) -> Vec3 {
    let dy = (length * pitch.sin()).clamp(-(y - soil).max(0.) * 0.50, (ceiling - y).max(0.) * 0.45);
    dir * length * pitch.cos() + Vec3::Y * dy
}

/// Each blade owns its UV domain: the central vein follows the midrib, even on
/// a twisted leaf. Area-weighted mesh normals follow the actual curved surface.
pub(super) fn blade(
    a: &mut Assembly,
    base: Vec3,
    delta: Vec3,
    half_width: f32,
    shape: u32,
    seed: u64,
    morphology: &Growth,
) {
    let axis = delta.normalize();
    let side = axis.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = side.cross(axis).normalize();
    let mut rng = stream(seed, 91);
    let half_width = half_width * morphology.leaf_aspect;
    let bend = rng.random_range(0.18..0.50);
    let twist = rng.random_range(-0.30..0.30);
    let asymmetry = rng.random_range(-0.13..0.13);
    let exponent = if shape == 2 {
        rng.random_range(0.25..0.6)
    } else {
        morphology.leaf_tip_exponent * rng.random_range(0.90..1.06)
    };
    let surface = if shape == 2 {
        Surface::LeafVariegated
    } else if ((seed % 1000) as f32) < 80. + (1. - morphology.maturity) * 220. {
        Surface::LeafLight
    } else {
        Surface::Leaf
    };
    let mut positions = [Vec3::ZERO; 23];
    let mut uvs = [[0.; 2]; 23];
    positions[0] = base;
    uvs[0] = [0.5, 0.0];
    for row in 1..8 {
        let t = row as f32 / 8.0;
        let envelope = (PI * t).sin().powf(exponent)
            * if shape == 4 {
                (1.30 - t).clamp(0.35, 1.0)
            } else {
                1.0
            };
        let lobes = if shape == 3 {
            0.48 + 0.52 * (t * PI * morphology.leaf_lobes).cos().abs()
        } else if shape == 5 {
            0.88 + 0.12 * (t * PI * 12.0).cos().abs()
        } else {
            1.0
        };
        let mid =
            base + delta * t + up * half_width * (bend * (PI * t).sin() - morphology.droop * t * t);
        for (column, sign) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
            let w = half_width * envelope * lobes * sign * (1.0 + sign * asymmetry);
            let i = 1 + (row - 1) * 3 + column;
            positions[i] = mid + side * w + up * (w * twist * t - w.abs() * morphology.leaf_curl);
            uvs[i] = [0.5 + sign * envelope * lobes * 0.5, t];
        }
    }
    positions[22] = base + delta - up * half_width * morphology.droop;
    uvs[22] = [0.5, 1.0];
    append(a, surface, positions, uvs);
}

const INDICES: [u32; 84] = {
    let mut indices = [0; 84];
    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 1;
    indices[3] = 0;
    indices[4] = 3;
    indices[5] = 2;
    let mut row = 0;
    while row < 6 {
        let k = 1 + row as u32 * 3;
        let quad = [
            k,
            k + 1,
            k + 3,
            k + 1,
            k + 4,
            k + 3,
            k + 1,
            k + 2,
            k + 4,
            k + 2,
            k + 5,
            k + 4,
        ];
        let mut i = 0;
        while i < 12 {
            indices[6 + row * 12 + i] = quad[i];
            i += 1;
        }
        row += 1;
    }
    indices[78] = 19;
    indices[79] = 20;
    indices[80] = 22;
    indices[81] = 20;
    indices[82] = 21;
    indices[83] = 22;
    indices
};

// A connected leaf patch has the same fixed topology for simple and palmate blades.
pub(super) fn append(
    a: &mut Assembly,
    surface: Surface,
    positions: [Vec3; 23],
    uvs: [[f32; 2]; 23],
) {
    let g = a.part(surface, "other_prop");
    let first = g.positions.len() as u32;
    let mut normals = [Vec3::ZERO; 23];
    for tri in INDICES.as_chunks::<3>().0 {
        let [i, j, k] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let n = (positions[j] - positions[i]).cross(positions[k] - positions[i]);
        for index in [i, j, k] {
            normals[index] += n;
        }
    }
    g.positions
        .extend(positions.into_iter().map(|p| p.to_array()));
    g.normals
        .extend(normals.into_iter().map(|n| n.normalize().to_array()));
    g.uvs.extend(uvs);
    g.indices.extend(INDICES.into_iter().map(|i| first + i));
}
