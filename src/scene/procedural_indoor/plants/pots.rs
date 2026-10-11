//! Continuously shaped, hollow planters with fitted soil and separate saucers.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PotProfile {
    pub belly: f32,
    pub neck: f32,
    pub wall_fraction: f32,
    pub flute_depth: f32,
    pub flute_count: u32,
    pub lip_height_fraction: f32,
    /// Circular, softly faceted or square/octagonal nursery and decorative pots.
    pub sides: u32,
    pub shoulder_height_fraction: f32,
    pub pedestal_fraction: f32,
}
impl PotProfile {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 6220);
        Self {
            belly: rng.random_range(0.82..1.035),
            neck: rng.random_range(0.82..1.02),
            wall_fraction: rng.random_range(0.055..0.095),
            flute_depth: rng.random_range(0.0..0.055),
            flute_count: rng.random_range(4..=8),
            lip_height_fraction: rng.random_range(0.025..0.065),
            sides: [4, 6, 8, 12, 40][rng.random_range(0..5)],
            shoulder_height_fraction: rng.random_range(0.46..0.73),
            pedestal_fraction: rng.random_range(0.025..0.12),
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    material: Surface,
    r: f32,
    h: f32,
    taper: f32,
    p: &PotProfile,
) -> (f32, f32) {
    let foot = h * p.pedestal_fraction;
    let wall = p.wall_fraction;
    let lip = p.lip_height_fraction;
    let profile = [
        (0., foot),
        (r * taper, foot),
        (r * (taper + 0.02), h * 0.15),
        (r * (taper + p.belly) * 0.5, h * 0.42),
        (r * p.belly, h * p.shoulder_height_fraction),
        (r * p.neck, h * (1. - lip)),
        (r * (p.neck + 0.025), h * (1. - lip * 0.7)),
        (r * (p.neck + 0.025), h),
        (r * (p.neck - wall), h),
        (r * (p.neck - wall), h * (1. - lip)),
        (r * (p.belly - wall), h * p.shoulder_height_fraction),
        (r * (taper - wall), h * 0.14),
        (0., h * 0.14),
    ];
    if p.sides < 16 {
        faceted_pot(a.part(material, "other_prop"), &profile, p.sides);
    } else {
        a.part(material, "other_prop").fluted_lathe(
            &profile,
            p.sides,
            p.flute_depth,
            p.flute_count,
            Transform::IDENTITY,
        );
    }
    a.part(material, "other_prop").lathe(
        &[
            (0., 0.),
            (r * 1.10, 0.),
            (r * 1.10, foot),
            (r * 1.025, foot * 1.6),
            (r * 0.96, foot),
            (0., foot),
        ],
        32,
        Transform::IDENTITY,
    );
    let soil = h * (1. - lip - 0.065);
    // Fit the inscribed disc at its actual height, including inward flutes.
    let t = ((soil / h - p.shoulder_height_fraction) / (1. - lip - p.shoulder_height_fraction))
        .clamp(0., 1.);
    let radius = r
        * ((p.belly - wall) * (1. - t) + (p.neck - wall) * t)
        * (1. - p.flute_depth)
        * (PI / p.sides as f32).cos()
        - r * 0.008;
    a.part(Surface::Soil, "other_prop").cylinder(
        radius,
        h * 0.025,
        Transform::from_xyz(0., soil - h * 0.0125, 0.),
    );
    (soil, radius)
}

/// Hard face normals for actual polygonal planter sides; radial normals would
/// falsely make a four-sided pot look like a poorly tessellated smooth cylinder.
fn faceted_pot(g: &mut super::super::geometry::Geometry, profile: &[(f32, f32)], sides: u32) {
    for pair in profile.windows(2) {
        for side in 0..sides {
            let angles = [
                side as f32 * TAU / sides as f32,
                (side + 1) as f32 * TAU / sides as f32,
            ];
            let point =
                |(r, y): (f32, f32), angle: f32| Vec3::new(r * angle.cos(), y, r * angle.sin());
            let positions = [
                point(pair[0], angles[0]),
                point(pair[1], angles[0]),
                point(pair[0], angles[1]),
                point(pair[1], angles[1]),
            ];
            let mut triangles = Vec::with_capacity(2);
            if pair[0].0 > 1e-5 {
                triangles.push([0, 1, 2]);
            }
            if pair[1].0 > 1e-5 {
                triangles.push([2, 1, 3]);
            }
            for t in triangles {
                let [a, b, c] = t.map(|i| positions[i]);
                let normal = (b - a).cross(c - a).normalize();
                let first = g.positions.len() as u32;
                for i in t {
                    let p = positions[i];
                    g.positions.push(p.to_array());
                    g.normals.push(normal.to_array());
                    g.uvs
                        .push([angles[usize::from(i >= 2)] * pair[0].0.max(pair[1].0), p.y]);
                }
                g.indices.extend([first, first + 1, first + 2]);
            }
        }
    }
}
