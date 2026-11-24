//! Measured cup sections shared by the mesh, fitted accessories and dataset export.
use super::*;
use crate::scene::procedural_indoor::materials::{drinkware::Brew, variants};

mod handle;
mod lid;
mod mesh;
#[cfg(test)]
mod tests;
pub(crate) use lid::SipLid;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Saucer {
    pub center_m: Vec2,
    pub radius_m: f32,
    pub well_radius_m: f32,
    pub seat_y_m: f32,
    pub rim_y_m: f32,
    foot_fraction: f32,
    pub finish_slot: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Sleeve {
    pub low_fraction: f32,
    pub high_fraction: f32,
    pub thickness_m: f32,
    pub printed: bool,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct CupForm {
    pub center_m: Vec2,
    pub base_y_m: f32,
    pub height_m: f32,
    pub body_radius_m: f32,
    pub wall_m: f32,
    pub lip_m: f32,
    pub heel_m: f32,
    pub floor_m: f32,
    pub yaw_rad: f32,
    pub handle_gauge_m: Option<f32>,
    pub handle_reach_m: f32,
    pub sides: u32,
    pub saucer: Option<Saucer>,
    pub sleeve: Option<Sleeve>,
    pub lid: Option<SipLid>,
    pub brew: Option<Brew>,
}

impl CupForm {
    pub(crate) fn sample(o: &IndoorObject, p: &BeverageProgram) -> Self {
        let paper = o.kind == ObjectKind::CoffeeCup;
        let s = o.size;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CupBuild"));
        let yaw_rad = if paper {
            r.random_range(0.0..TAU)
        } else {
            r.random_range(-0.40..0.40) + if r.random_bool(0.5) { PI } else { 0. }
        };
        let heading = Vec2::from_angle(yaw_rad);
        let mut center_m = if paper {
            Vec2::ZERO
        } else {
            -heading * (s.x * r.random_range(0.055..0.11))
        };
        let stock = s.xz() * 0.5;
        let saucer = (!paper && p.style >= 3).then(|| {
            let radius_m = (stock - center_m.abs()).min_element() * r.random_range(0.87..0.99);
            let center = center_m;
            let offset = Vec2::from_angle(r.random_range(0.0..TAU))
                * (radius_m * r.random_range(0.0..0.085));
            center_m += offset;
            let seat_y_m = (s.y * r.random_range(0.036..0.085)).min(0.0075);
            let slot = variants::object_slot(Surface::Ceramic, o.seed, o.finish_seed);
            let finish_slot = if r.random_bool(0.32) {
                (slot + r.random_range(1..variants::COUNT)) % variants::COUNT
            } else {
                slot
            };
            Saucer {
                center_m: center,
                radius_m,
                well_radius_m: radius_m * r.random_range(0.66..0.84),
                seat_y_m,
                rim_y_m: seat_y_m + (radius_m * r.random_range(0.08..0.18)).min(s.y * 0.13),
                foot_fraction: r.random_range(0.43..0.64),
                finish_slot,
            }
        });
        let base_y_m = saucer.as_ref().map_or(0., |a| a.seat_y_m);
        let lid = (paper && p.lid).then(|| SipLid::sample(&mut r, s.y));
        let height_m = s.y - base_y_m - lid.as_ref().map_or(0., |l| l.rise_m);
        let sleeve = (paper && p.style % 3 != 2).then(|| {
            let low = r.random_range(0.17..0.37);
            Sleeve {
                low_fraction: low,
                high_fraction: low + r.random_range(0.26..0.45),
                thickness_m: r.random_range(0.0006..0.0017),
                printed: p.style % 3 == 1,
            }
        });
        let wall_m = if paper {
            r.random_range(0.00036..0.00067) * if p.style % 3 == 2 { 2.25 } else { 1. }
        } else {
            p.wall_m
        };
        let lip_m = if paper {
            r.random_range(0.0011..0.0020)
        } else {
            wall_m * 0.57
        };
        let max_scale = if paper {
            p.taper
        } else {
            p.taper.max(p.belly_fraction).max(1.)
        };
        let available = (stock - center_m.abs()).min_element();
        let margin = lip_m
            + lid.as_ref().map_or(0., |l| l.gauge_m * 2.0)
            + sleeve.as_ref().map_or(0., |sl| sl.thickness_m);
        let mut body_radius_m = (s.x.min(s.z)
            * if paper {
                r.random_range(0.32..0.42)
            } else {
                p.cup_radius_fraction
            })
        .min((available - margin) / max_scale);
        // The whole lower body clears the curved rim, including an off-centre cup.
        if let Some(a) = &saucer {
            body_radius_m = body_radius_m.min(
                (a.well_radius_m * (PI / 32.).cos() - center_m.distance(a.center_m) - 0.00025)
                    / max_scale,
            );
        }
        let heel_m = height_m * r.random_range(0.045..0.088);
        let floor_m = (heel_m * r.random_range(1.12..1.43)).min(height_m * 0.16);
        let (handle_gauge_m, handle_reach_m) = if paper {
            (None, 0.)
        } else {
            let gauge = (height_m * r.random_range(0.028..0.054)).min(body_radius_m * 0.14);
            let edge = stock - Vec2::splat(gauge + 0.00015);
            let reach_limit = ((edge - heading.signum() * center_m)
                / heading.abs().max(Vec2::splat(1e-5)))
            .min_element();
            let reach =
                (reach_limit - body_radius_m * max_scale).max(0.) * r.random_range(0.60..0.99);
            (Some(gauge.min(reach * 0.45)), reach)
        };
        Self {
            center_m,
            base_y_m,
            height_m,
            body_radius_m,
            wall_m,
            lip_m,
            heel_m,
            floor_m,
            yaw_rad,
            handle_gauge_m,
            handle_reach_m,
            sides: if paper { 32 } else { p.cup_sides },
            saucer,
            sleeve,
            lid,
            brew: (!p.empty).then(|| Brew::sample(&mut r)),
        }
    }

    fn transform(&self) -> Transform {
        Transform::from_xyz(self.center_m.x, self.base_y_m, self.center_m.y)
            .with_rotation(Quat::from_rotation_y(-self.yaw_rad))
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &BeverageProgram) {
    let form = CupForm::sample(o, p);
    let section = mesh::Section::new(&form, p);
    let label = o.kind.class_name();
    if let Some(saucer) = &form.saucer {
        saucer.draw(a, label);
    }
    section.draw(a, &form, label);
    if let Some(brew) = form.brew {
        section.drink(a, &form, p, brew, label);
    }
    if let Some(gauge) = form.handle_gauge_m {
        section.handle(a, &form, p, gauge, label);
    }
    if let Some(lid) = &form.lid {
        lid.draw(a, &form, section.mouth() + form.lip_m * 0.4, label);
    }
}
