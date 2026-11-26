//! A cast spider arm is wider across the radial than it is thick. Build all
//! stations in one horizontal radial frame so the die rotates with each arm.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct CastSpoke {
    pub root_width_m: f32,
    pub depth_m: f32,
    pub tip_width_fraction: f32,
    pub tip_depth_fraction: f32,
    pub crown_m: f32,
    pub chamfer_fraction: f32,
}

impl CastSpoke {
    pub(super) fn sample(r: &mut impl Rng) -> Self {
        Self {
            root_width_m: r.random_range(0.040..0.065),
            depth_m: r.random_range(0.019..0.031),
            tip_width_fraction: r.random_range(0.41..0.72),
            tip_depth_fraction: r.random_range(0.52..0.87),
            crown_m: r.random_range(0.003..0.023),
            chamfer_fraction: r.random_range(0.15..0.37),
        }
    }

    pub(in super::super) fn build(self, g: &mut Geometry, start: Vec3, end: Vec3, hub_radius: f32) {
        let radial = (end - start).with_y(0.).normalize();
        let across = Vec3::Y.cross(radial);
        let root_half = (self.root_width_m * 0.5).min(hub_radius * 0.75);
        let c = 1. - self.chamfer_fraction;
        let outline = [
            (-c, 1.),
            (c, 1.),
            (1., c),
            (1., -c),
            (c, -1.),
            (-c, -1.),
            (-1., -c),
            (-1., c),
        ];
        let ring = |t: f32| {
            let center = start.lerp(end, t) + Vec3::Y * (4. * self.crown_m * t * (1. - t));
            let width = root_half * (1. - (1. - self.tip_width_fraction) * t);
            let thick = 0.5 * self.depth_m * (1. - (1. - self.tip_depth_fraction) * t);
            (
                center,
                outline.map(|(x, y)| center + across * (x * width) + Vec3::Y * (y * thick)),
            )
        };
        let cap = |g: &mut Geometry, rim: [Vec3; 8], normal| {
            for i in [1, 3, 5] {
                g.metric_quad([rim[0], rim[i], rim[i + 1], rim[i + 2]], normal);
            }
        };
        let (mut center, mut rim) = ring(0.);
        cap(g, rim, -radial);
        for t in [0.34, 0.70, 1.] {
            let (next_center, next) = ring(t);
            for i in 0..8 {
                let j = (i + 1) % 8;
                let outside = (rim[i] + rim[j]) * 0.5 - center;
                g.stock_quad(
                    [rim[i], rim[j], next[j], next[i]],
                    outside,
                    radial,
                    Some(Vec2::ZERO),
                );
            }
            (center, rim) = (next_center, next);
        }
        cap(g, rim, radial);
    }
}
