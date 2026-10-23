//! Palmate fronds have a joined basal web, pleated radial veins and incised tips.
//! Adjacent sectors share their basal edges; no overlapping/coplanar leaf cards.
use super::*;

pub(super) fn fan(a: &mut Assembly, base: Vec3, dir: Vec3, radius: f32, span: f32, seed: u64) {
    let mut r = stream(seed, 6454);
    let fingers = r.random_range(9..=13);
    let step = span * 2. / fingers as f32;
    let forward = (dir * 0.55 + Vec3::Y * 0.835).normalize();
    let side = dir.cross(Vec3::Y);
    let normal = side.cross(forward);
    let pleat = r.random_range(0.025..0.065);
    let droop = r.random_range(0.015..0.070);
    let web = r.random_range(0.50..0.68);
    let surface = if r.random_bool(0.12) {
        Surface::LeafLight
    } else {
        Surface::Leaf
    };
    for finger in 0..fingers {
        let mut positions = [Vec3::ZERO; 23];
        let mut uvs = [[0.; 2]; 23];
        positions[0] = base;
        uvs[0] = [0.5, 0.];
        let angle = -span + step * (finger as f32 + 0.5);
        for row in 1..8 {
            let t = row as f32 / 8.;
            let intact = ((1. - t) / (1. - web)).min(1.).powf(0.55);
            for (column, sign) in [-1_f32, 0., 1.].into_iter().enumerate() {
                let theta = angle + sign * step * intact * 0.5;
                let i = 1 + (row - 1) * 3 + column;
                positions[i] = base
                    + (forward * theta.cos() + side * theta.sin()) * radius * t
                    + normal * radius * (pleat * (1. - sign.abs()) * t - droop * t * t);
                uvs[i] = [0.5 + sign * intact * 0.5, t];
            }
        }
        positions[22] = base
            + (forward * angle.cos() + side * angle.sin()) * radius
            + normal * radius * (pleat - droop);
        uvs[22] = [0.5, 1.];
        leaf::append(a, surface, positions, uvs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_has_joined_basal_web_separate_tips_and_valid_frames() {
        for seed in 0..64 {
            let mut a = Assembly::default();
            fan(&mut a, Vec3::ZERO, Vec3::X, 0.3, 1.4, seed);
            assert_eq!(a.parts.len(), 1);
            let g = a.parts.values().next().unwrap();
            let (fingers, tail) = g.positions.as_chunks::<23>();
            assert!(tail.is_empty() && (9..=13).contains(&fingers.len()));
            for pair in fingers.windows(2) {
                // Row four (t=0.5) is inside every sampled intact web.
                assert!(Vec3::from(pair[0][12]).distance(Vec3::from(pair[1][10])) < 1e-6);
                assert!(Vec3::from(pair[0][22]).distance(Vec3::from(pair[1][22])) > 0.02);
            }
            for n in g.normals.iter().copied().map(Vec3::from) {
                assert!(n.is_finite() && (n.length() - 1.).abs() < 1e-5);
            }
        }
    }
}
