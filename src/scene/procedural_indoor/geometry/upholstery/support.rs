//! Height queries on the actual planar facets of the full cushion mesh.
//! All latitude rings are scaled copies of the equator. Within an azimuth
//! sector their edges are parallel, so height is linear between adjacent rings.
use super::*;

pub(crate) struct CushionSurface {
    planes: [Vec2; 8],
    radii: [f32; 7],
    heights: [f32; 7],
    slopes: [f32; 6],
}
impl CushionSurface {
    pub(crate) fn new(size: Vec3, p: CushionProfile) -> Self {
        let grid = full_grid();
        let shape = 2. / p.roundness.clamp(2., 10.);
        let crown = 2. / p.crown.clamp(2., 6.);
        let dish = p.dish.clamp(0., size.y * 0.20);
        let outline = std::array::from_fn::<_, 9, _>(|i| {
            let a = grid.angles[i];
            Vec2::new(power(a.x, shape), power(a.y, shape)) * size.xz() * 0.5
        });
        let planes = std::array::from_fn(|i| {
            let edge = outline[i + 1] - outline[i];
            let n = Vec2::new(edge.y, -edge.x);
            n / n.dot(outline[i])
        });
        let radii = std::array::from_fn(|i| grid.latitudes[i].y.powf(crown));
        let heights = std::array::from_fn(|i| {
            size.y * 0.5 * power(grid.latitudes[i].x, crown) - dish * (1. - radii[i]).powi(2)
        });
        let slopes =
            std::array::from_fn(|i| (heights[i + 1] - heights[i]) / (radii[i + 1] - radii[i]));
        Self {
            planes,
            radii,
            heights,
            slopes,
        }
    }

    pub(crate) fn height(&self, point: Vec2) -> Option<f32> {
        let q = point.abs();
        let t = self.planes.iter().fold(0_f32, |r, n| r.max(n.dot(q)));
        if t > 1. {
            return None;
        }
        let i = self.radii[1..].partition_point(|r| *r < t).min(5);
        Some(self.heights[i] + (t - self.radii[i]) * self.slopes[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cushion_support_interpolates_the_actual_rendered_triangles() {
        use crate::scene::procedural_indoor::envelope::ray_triangle;
        for (shape, crown, dish) in [(2.0, 2.0, 0.0), (4.4, 3.7, 0.016), (8.0, 5.9, 0.009)] {
            let size = Vec3::new(0.74, 0.13, 0.48);
            let profile = CushionProfile {
                roundness: shape,
                crown,
                dish,
            };
            let surface = CushionSurface::new(size, profile);
            let mut g = Geometry::default();
            g.cushion(size, profile, Transform::IDENTITY);
            for ix in -10..=10 {
                for iz in -9..=9 {
                    let pos = Vec3::new(ix as f32 * 0.0381, 0.2, iz as f32 * 0.0267);
                    let hit = g
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .filter_map(|t| {
                            ray_triangle(
                                pos,
                                Vec3::NEG_Y,
                                t.map(|i| Vec3::from_array(g.positions[i as usize])),
                            )
                        })
                        .min_by(f32::total_cmp);
                    match (surface.height(pos.xz()), hit) {
                        (Some(h), Some(d)) => {
                            assert!((h - (pos.y - d)).abs() < 2e-5, "{pos:?} {h} {d}")
                        }
                        (None, None) => {}
                        other => panic!("{pos:?} {other:?}"),
                    }
                }
            }
        }
    }
}
