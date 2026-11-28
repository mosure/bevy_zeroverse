//! The coarse fraction of a large mulched bed. Its fine cover is already in
//! the material maps. Keep individual chips and stones at a physical scale.
use super::*;

#[derive(serde::Serialize)]
pub(in crate::scene::procedural_indoor::plants) struct Cover {
    seed: u64,
    pub(in crate::scene::procedural_indoor::plants) count: u8,
}

impl Cover {
    pub(in crate::scene::procedural_indoor::plants) fn sample(seed: u64, medium: Medium) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"YardChip"));
        let count = match medium {
            Medium::BarkChips | Medium::MineralGravel => r.random_range(12..=32),
            _ => 0,
        };
        Self {
            seed: r.random(),
            count,
        }
    }

    pub(in crate::scene::procedural_indoor::plants) fn mesh(
        &self,
        a: &mut Assembly,
        medium: Medium,
        radii: Vec2,
        grade: impl Fn(Vec2) -> (f32, Vec2),
    ) {
        if self.count == 0 {
            return;
        }
        let g = a.part(Surface::Soil, &medium.label(Layer::Grain));
        let mut r = stream(self.seed, u64::from_le_bytes(*b"ChipSite"));
        // One jittered azimuth per sector, with a newly rotated scattering
        // pattern on each bed. Inset the centers so the full skirt fits.
        let phase = r.random_range(0.0..TAU);
        for i in 0..self.count {
            let azimuth =
                phase + TAU * (f32::from(i) + r.random_range(0.12..0.88)) / f32::from(self.count);
            let bark = medium == Medium::BarkChips;
            let length = (if bark {
                r.random_range(0.014..0.045_f32)
            } else {
                r.random_range(0.011..0.035)
            })
            .min(radii.min_element() * 0.06);
            let width = length
                * if bark {
                    r.random_range(0.28..0.59)
                } else {
                    r.random_range(0.55..0.91)
                };
            let center = Vec2::from_angle(azimuth)
                * r.random::<f32>().sqrt()
                * (radii * 0.97 - Vec2::splat(length));
            let grain = Grain {
                center,
                half: Vec2::new(length, width),
                height: width
                    * if bark {
                        r.random_range(0.26..0.46)
                    } else {
                        r.random_range(0.40..0.82)
                    },
                yaw: r.random_range(0.0..TAU),
                shape_seed: r.random(),
                accent: false,
            };
            let first = g.positions.len();
            mesh::append(g, grain, medium, 0.);
            // A chip can straddle two triangles of the mound. Seat its entire
            // skirt in the actual surface, including at the break in slope.
            for (p, n) in g.positions[first..].iter_mut().zip(&mut g.normals[first..]) {
                let (level, slope) = grade(Vec2::new(p[0], p[2]));
                p[1] += level;
                *n = Vec3::new(n[0] - n[1] * slope.x, n[1], n[2] - n[1] * slope.y)
                    .normalize()
                    .to_array();
            }
        }
    }
}
