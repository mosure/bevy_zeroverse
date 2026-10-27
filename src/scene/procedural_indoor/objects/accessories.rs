//! Small equipment with independently sampled proportions and real mechanisms.
use super::*;
#[cfg(test)]
mod audio_tests;
pub mod fan;
pub mod headphones;
pub(crate) mod lamps;
pub mod organizers;
pub mod speaker;
pub mod stationery;
pub use lamps::{lamp_parameters, TaskLampProgram};

#[derive(Debug, Clone, serde::Serialize)]
pub struct AccessoryProgram {
    pub style: u8,
    pub tilt: f32,
    pub reach: f32,
    pub sections: usize,
    pub finish: Surface,
    pub body_taper: f32,
    pub control_fraction: f32,
    pub support_fraction: f32,
    pub profile_roundness: f32,
}

pub fn parameters(o: &IndoorObject) -> AccessoryProgram {
    let mut rng = stream(o.seed, 5300);
    let mut detail = stream(o.seed, 7440);
    AccessoryProgram {
        body_taper: detail.random_range(0.65..0.97),
        control_fraction: detail.random_range(0.25..0.43),
        support_fraction: detail.random_range(0.12..0.26),
        profile_roundness: detail.random_range(2.2..8.0),
        style: rng.random_range(0..4),
        tilt: rng.random_range(-0.18..0.18),
        reach: rng.random_range(0.36..0.70),
        sections: rng.random_range(2..=4),
        finish: if rng.random_bool(0.35) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::DeskFan {
        return fan::build(a, o);
    }
    let p = parameters(o);
    match o.kind {
        ObjectKind::DeskLamp => lamps::build(a, o, &p),
        ObjectKind::Speaker => speaker::build(a, o),
        ObjectKind::Headphones => headphones::build(a, o),
        ObjectKind::Stapler => stationery::stapler(a, o),
        ObjectKind::TapeDispenser => stationery::tape(a, o),
        ObjectKind::DocumentTray | ObjectKind::DeskOrganizer => organizer(a, o, &p),
        ObjectKind::TissueBox => super::containers::tissues(a, o),
        _ => unreachable!("non-accessory object"),
    }
}

fn organizer(a: &mut Assembly, o: &IndoorObject, _: &AccessoryProgram) {
    organizers::build(a, o);
}

/// Shared direct/indirect lighting proxy for the generated task-lamp head.
pub(crate) fn lamp_photometry(o: &IndoorObject) -> (Vec3, Vec3, f32, f32, f32) {
    lamps::photometry(o)
}
