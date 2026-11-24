//! Ceiling-mounted fans: a stationary canopy/downrod and one rigid rotor.
//! Metric envelopes include every blade angle, independently of animation speed.
use super::*;
use serde::Serialize;

mod blade;
mod legacy;
mod product;
pub(crate) use product::{canopy, installed_variant};
pub use product::{BladeFamily, BladeProfile};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FanProgram {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product: Option<BladeProfile>,
    pub style: u8,
    pub blades: u8,
    pub radius_m: f32,
    pub blade_width_m: f32,
    pub pitch_radians: f32,
    pub sweep_m: f32,
    pub taper: f32,
    pub thickness_m: f32,
    pub rotor_y: f32,
    pub rotor_top: f32,
    pub hub_radius: f32,
    pub phase: f32,
    pub blade_surface: Surface,
    pub housing_surface: Surface,
}
pub fn parameters(o: &IndoorObject) -> FanProgram {
    if installed(o) {
        product::parameters(o)
    } else {
        legacy::parameters(o)
    }
}

pub(crate) fn installed(o: &IndoorObject) -> bool {
    o.variant & product::INSTALLED != 0
}

pub fn build(a: &mut Assembly, o: &IndoorObject) {
    use super::super::motion::{construction, MotionKind};
    let p = parameters(o);
    let moving = a.motion_selected(MotionKind::Fan, o.seed);
    let mut rotor = Assembly::default();
    if let Some(section) = &p.product {
        product::build(a, &mut rotor, o, &p, section, moving);
    } else {
        legacy::build(a, &mut rotor, o, &p);
    }
    if moving {
        let [lo, hi] = a.motion_policy.as_ref().unwrap().fan_turns;
        let mut r = stream(o.seed, 0x5350494e);
        let turns = r.random_range(lo..=hi);
        let direction = if r.random_bool(0.2) { -1. } else { 1. };
        let mut joint = construction::rigid(
            MotionKind::Fan,
            o.seed,
            Vec3::Y * p.rotor_y,
            Vec3::Y,
            turns * TAU * direction,
            Vec3::ZERO,
        );
        if p.product.is_some() {
            joint.attachment = Some(product::AXLE_CONTACT.into());
        }
        a.articulated(rotor, joint);
    } else {
        a.append_assembly(rotor);
    }
}

/// Separate rotor and mount volumes allow sloped ceilings without filling the
/// entire space above the blades with a fictitious solid box.
pub(crate) fn local_reservations(o: &IndoorObject) -> [(Vec3, Vec3); 2] {
    let r = o.size.x.min(o.size.z) * 0.5;
    let (top, mount) = if installed(o) {
        let fit = product::Fit::new(o.size);
        (fit.top, fit.mount)
    } else {
        let p = legacy::parameters(o);
        (p.rotor_top, p.hub_radius.max(0.09))
    };
    [
        (Vec3::new(-r, 0., -r), Vec3::new(r, top + 0.005, r)),
        (
            Vec3::new(-mount, top, -mount),
            Vec3::new(mount, o.size.y, mount),
        ),
    ]
}

#[cfg(test)]
mod tests;
