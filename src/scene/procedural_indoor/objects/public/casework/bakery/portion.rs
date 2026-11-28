use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::baking,
};

impl Portion {
    pub(super) fn build(self, a: &mut Assembly, recipe: Bake, pose: Transform) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"BakeDome"));
        let (rad, h) = (self.radius_m, self.height_m);
        if recipe == Bake::Biscuit {
            let mut biscuit = Assembly::default();
            crate::scene::procedural_indoor::objects::kitchen::biscuit(
                &mut biscuit,
                rad,
                0.,
                h,
                &mut r,
                VERSION,
            );
            append_at(a, biscuit, pose);
            return;
        }
        let min_aspect = if recipe == Bake::Roll { 0.69 } else { 0.86 };
        let at = pose.with_scale(Vec3::new(1., 1., r.random_range(min_aspect..1.)));
        let outline =
            RevolvedOutline::round(20, r.random_range(0.018..0.068), r.random_range(2..=4));
        let crust = baking::Finish::Crust;
        let g = a.part(crust.surface(), crust.label(VERSION));
        let first = g.uvs.len();
        match recipe {
            Bake::Bagel => {
                let hole = rad * r.random_range(0.14..0.32);
                // Closed annular section. The flat underside supports the bread while the hole stays open.
                outline.sweep(
                    g,
                    &[
                        (hole * 1.32, 0.),
                        (rad * 0.77, 0.),
                        (rad * 0.94, h * 0.16),
                        (rad, h * 0.46),
                        (rad * 0.92, h * 0.78),
                        (rad * 0.73, h * 0.97),
                        (rad * 0.55, h),
                        (hole * 1.28, h * 0.81),
                        (hole, h * 0.46),
                        (hole * 1.06, h * 0.18),
                        (hole * 1.32, 0.),
                    ],
                    ProfileJoints::Smooth,
                    at,
                );
            }
            Bake::Roll => outline.sweep(
                g,
                &[
                    (0., 0.),
                    (rad * 0.60, 0.),
                    (rad * 0.89, h * 0.15),
                    (rad, h * 0.43),
                    (rad * 0.89, h * 0.73),
                    (rad * 0.62, h * 0.92),
                    (rad * 0.29, h * 0.99),
                    (0., h),
                ],
                ProfileJoints::Smooth,
                at,
            ),
            Bake::Muffin => {
                let lip = h * r.random_range(0.54..0.67);
                let liner = rad * r.random_range(0.75..0.83);
                let pleats = r.random_range(10..=16);
                // The crown overhangs the liner. Their only contact on the exterior is the rim.
                outline.sweep(
                    g,
                    &[
                        (liner * 0.80, lip - 0.002),
                        (liner * 0.80, lip),
                        (rad * 0.91, lip),
                        (rad, lip + (h - lip) * 0.23),
                        (rad * 0.95, lip + (h - lip) * 0.53),
                        (rad * 0.68, lip + (h - lip) * 0.87),
                        (rad * 0.32, h * 0.989),
                        (0., h),
                    ],
                    ProfileJoints::Smooth,
                    at,
                );
                let liner_profile = [
                    (0., 0.),
                    (liner * 0.69, 0.),
                    (liner * 0.71, 0.002),
                    (liner, lip),
                ];
                a.part(Surface::Paper, "paper").fluted_lathe(
                    &liner_profile,
                    pleats * 3,
                    0.085,
                    pleats,
                    at,
                );
            }
            Bake::Biscuit => unreachable!(),
        }
        // Cut each piece from a different place in the same batch's baked surface.
        let offset = Vec2::new(r.random_range(0.0..0.21), r.random_range(0.0..0.21));
        for uv in &mut a.part(crust.surface(), crust.label(VERSION)).uvs[first..] {
            *uv = (Vec2::from_array(*uv) + offset).to_array();
        }
    }
}
