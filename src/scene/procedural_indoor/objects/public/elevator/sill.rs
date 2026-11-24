//! Threshold stock in metres. The extruded section includes a car/landing
//! joint and a milled guide directly underneath each telescoping door plane.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Sill {
    pub stock: BrushedHardware,
    /// Cross-grain repetition of the shared brushed-metal map on long extrusions.
    pub cross_grain_scale: f32,
    pub left_m: f32,
    pub right_m: f32,
    pub back_m: f32,
    pub front_m: f32,
    pub height_m: f32,
    pub guide_centers_m: Vec<f32>,
    pub guide_width_m: f32,
    pub guide_depth_m: f32,
    pub joint_center_m: f32,
    pub joint_width_m: f32,
}

impl Sill {
    pub(super) fn sample(o: &IndoorObject, car: Vec3, p: &ElevatorProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"LiftSill"));
        let height = p.threshold * car.y;
        let guides: Vec<_> = (0..p.leaves_per_side)
            .rev()
            .map(|i| (0.46 - p.recess - i as f32 * 0.026) * car.z)
            .collect();
        let back = threshold_back(p) * car.z;
        let rear_margin = guides[0] - back;
        Self {
            stock: BrushedHardware::for_landing(o),
            cross_grain_scale: stream(
                o.finish_seed.unwrap_or(o.seed),
                u64::from_le_bytes(*b"SillGrin"),
            )
            .random_range(3.0..5.0),
            left_m: p.portal_left * car.x,
            right_m: p.portal_right * car.x,
            back_m: back,
            front_m: if is_core(o) {
                o.size.z * 0.48 + 0.006
            } else {
                car.z * 0.50
            },
            height_m: height,
            guide_width_m: r.random_range(0.008..0.015_f32).min(0.010 * car.z),
            guide_depth_m: (height * r.random_range(0.53..0.83)).min(0.0075),
            joint_center_m: back + rear_margin * 0.44,
            joint_width_m: r.random_range(0.013..0.024_f32).min(rear_margin * 0.48),
            guide_centers_m: guides,
        }
    }

    pub(super) fn build(&self, a: &mut Assembly) {
        let (left, right, top) = (self.left_m, self.right_m, self.height_m);
        // One exposed surface for each interval. The primary floor forms the
        // underside. End faces stop at the jambs without extending the sill.
        let span = |g: &mut Geometry, z0: f32, h0: f32, z1: f32, h1: f32| {
            let points = [
                Vec3::new(right, h0, z0),
                Vec3::new(left, h0, z0),
                Vec3::new(left, h1, z1),
                Vec3::new(right, h1, z1),
            ];
            g.metric_quad(points, Vec3::new(0., z1 - z0, h0 - h1));
            if z1 > z0 {
                for (x, outward) in [(left, Vec3::NEG_X), (right, Vec3::X)] {
                    g.metric_quad(
                        [
                            Vec3::new(x, 0., z0),
                            Vec3::new(x, 0., z1),
                            Vec3::new(x, h1, z1),
                            Vec3::new(x, h0, z0),
                        ],
                        outward,
                    );
                }
            }
        };
        let joint_lo = self.joint_center_m - self.joint_width_m * 0.5;
        let joint_hi = self.joint_center_m + self.joint_width_m * 0.5;
        let gasket = top * 0.32;
        let g = a.part(Surface::Metal, &self.stock.label("elevator"));
        let first_uv = g.uvs.len();
        span(g, self.back_m, top, joint_lo, top);
        span(g, joint_lo, top, joint_lo, gasket);
        span(g, joint_hi, gasket, joint_hi, top);
        let mut z = joint_hi;
        for &center in &self.guide_centers_m {
            let half = self.guide_width_m * 0.5;
            let bevel = half * 0.26;
            let bottom = top - self.guide_depth_m;
            span(g, z, top, center - half, top);
            span(g, center - half, top, center - half + bevel, bottom);
            span(
                g,
                center - half + bevel,
                bottom,
                center + half - bevel,
                bottom,
            );
            span(g, center + half - bevel, bottom, center + half, top);
            z = center + half;
        }
        span(g, z, top, self.front_m, top);
        span(g, self.front_m, top, self.front_m, 0.);
        // The brush runs along the rail. Compress its crosswise texture period
        // to roughly 1–2 mm; the shared roughness mips filter it at room scale.
        for uv in &mut g.uvs[first_uv..] {
            uv[1] *= self.cross_grain_scale;
        }
        span(
            a.part(Surface::Rubber, "elevator"),
            joint_lo,
            gasket,
            joint_hi,
            gasket,
        );
    }
}
