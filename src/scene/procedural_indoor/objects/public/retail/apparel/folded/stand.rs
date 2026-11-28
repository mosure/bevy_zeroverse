//! The frame and deck are one retail fixture in instance and semantic targets.
use super::*;

#[derive(Debug, Serialize)]
pub enum DisplayTableStand {
    Apron {
        leg_m: f32,
        end_inset_m: f32,
        side_inset_m: f32,
        rail_height_m: f32,
    },
    PanelEnds {
        end_inset_m: f32,
        panel_thickness_m: f32,
        foot_depth_m: f32,
        stretcher_center_m: f32,
    },
    Plinth {
        body_xz_m: Vec2,
        toe_height_m: f32,
        toe_recess_m: f32,
    },
}

impl DisplayTableStand {
    pub(super) fn sample(size: Vec3, frame: Surface, r: &mut ChaCha8Rng) -> Self {
        let family = r.random_range(0..10);
        if family < 5 {
            Self::Apron {
                leg_m: r.random_range(0.036..0.068),
                end_inset_m: r.random_range(0.035..0.088),
                side_inset_m: r.random_range(0.030..0.078),
                rail_height_m: r.random_range(0.064..0.128),
            }
        } else if family < 8 || frame == Surface::Metal {
            Self::PanelEnds {
                end_inset_m: size.x * r.random_range(0.09..0.20),
                panel_thickness_m: r.random_range(0.028..0.075),
                foot_depth_m: size.z * r.random_range(0.72..0.92),
                stretcher_center_m: r.random_range(0.21..0.35),
            }
        } else {
            Self::Plinth {
                body_xz_m: size.xz()
                    * Vec2::new(r.random_range(0.74..0.93), r.random_range(0.70..0.93)),
                toe_height_m: r.random_range(0.05..0.115),
                toe_recess_m: r.random_range(0.025..0.065),
            }
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, size: Vec3, table: &FoldedDisplay) {
        let d = table.deck_thickness_m;
        let h = table.deck_y_m - d;
        board(
            a,
            table.deck_surface,
            FIXTURE,
            Vec3::Y * (h + d * 0.5),
            Vec3::new(size.x, d, size.z),
        );
        let mut member = |center, extent| board(a, table.frame, FIXTURE, center, extent);
        match *self {
            Self::Apron {
                leg_m: l,
                end_inset_m,
                side_inset_m,
                rail_height_m: rail,
            } => {
                let x = (size.x - l) * 0.5 - end_inset_m;
                let z = (size.z - l) * 0.5 - side_inset_m;
                for sx in [-x, x] {
                    for sz in [-z, z] {
                        member(Vec3::new(sx, h * 0.5, sz), Vec3::new(l, h, l));
                    }
                }
                let y = h - 0.5 * rail;
                for sz in [-z, z] {
                    member(Vec3::new(0., y, sz), Vec3::new(2. * x - l, rail, l * 0.52));
                }
                for sx in [-x, x] {
                    member(Vec3::new(sx, y, 0.), Vec3::new(l * 0.52, rail, 2. * z - l));
                }
            }
            Self::PanelEnds {
                end_inset_m,
                panel_thickness_m: t,
                foot_depth_m,
                stretcher_center_m: y,
            } => {
                let x = (size.x - t) * 0.5 - end_inset_m;
                for sx in [-x, x] {
                    member(Vec3::new(sx, h * 0.5, 0.), Vec3::new(t, h, foot_depth_m));
                }
                member(Vec3::new(0., y, 0.), Vec3::new(2. * x - t, t * 2., t));
            }
            Self::Plinth {
                body_xz_m: b,
                toe_height_m: k,
                toe_recess_m: r,
            } => {
                member(
                    Vec3::Y * (k * 0.5),
                    Vec3::new(b.x - 2. * r, k, b.y - 2. * r),
                );
                member(Vec3::Y * ((h + k) * 0.5), Vec3::new(b.x, h - k, b.y));
            }
        }
    }
}
