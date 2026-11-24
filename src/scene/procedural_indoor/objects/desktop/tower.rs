//! Business desktops: full towers, slim upright cases and low desktop chassis.
//! The fascia, grille, pads and port faces stay inside one measured envelope.
use super::*;
use crate::scene::procedural_indoor::materials::input_housing::Housing;

pub(crate) fn dimensions(r: &mut impl Rng, floor: bool) -> Vec3 {
    match r.random_range(if floor { 1..4 } else { 0..4 }) {
        0 => Vec3::new(
            r.random_range(0.17..0.26),
            r.random_range(0.050..0.106),
            r.random_range(0.18..0.27),
        ),
        1 => Vec3::new(
            r.random_range(0.086..0.135),
            r.random_range(0.255..0.35),
            r.random_range(0.25..0.35),
        ),
        _ => Vec3::new(
            r.random_range(0.143..0.218),
            r.random_range(0.32..0.46),
            r.random_range(0.27..0.42),
        ),
    }
}

#[derive(serde::Serialize)]
pub(crate) struct CaseProgram {
    pub feet_m: f32,
    pub fascia_depth_m: f32,
    pub intake_fraction: f32,
    pub vertical_louvers: bool,
    pub louver_pitch_m: f32,
}

pub(crate) fn parameters(o: &IndoorObject) -> CaseProgram {
    let mut r = stream(o.seed, 0x544f574552434153);
    CaseProgram {
        feet_m: o.size.y.min(0.15) * r.random_range(0.032..0.065),
        fascia_depth_m: o.size.z * r.random_range(0.027..0.067),
        intake_fraction: r.random_range(0.46..0.73),
        vertical_louvers: r.random_bool(0.44),
        louver_pitch_m: r.random_range(0.008..0.017),
    }
}

mod exhaust;
mod face;
mod fascia;
mod legacy;
mod sockets;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 231;
pub(crate) const FITTED_IO: u32 = 1 << 8;

pub(crate) fn installed(o: &IndoorObject) -> bool {
    o.variant & FITTED_IO != 0
}

pub(crate) use fascia::Fascia;
pub(super) use sockets::rear_center;

pub(crate) fn metadata(o: &IndoorObject) -> serde_json::Value {
    let p = parameters(o);
    let mut meta = serde_json::json!(p);
    if installed(o) {
        let fascia = Fascia::sample(o, &p);
        let bank = sockets::rear_sockets(o);
        meta["exhaust"] = serde_json::json!(exhaust::Exhaust::fit(o, fascia.rear_face, &bank));
        meta["fascia"] = serde_json::json!(fascia);
        meta["rear_sockets_m"] =
            serde_json::json!(bank.map(|socket| socket.aperture.center().extend(-0.5 * o.size.z)));
    }
    meta
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if version < INTRODUCED || !installed(o) {
        return legacy::build(a, o);
    }
    let p = parameters(o);
    let fascia = Fascia::sample(o, &p);
    let stock = Housing::for_object(o);
    let label = stock.case_label(o);
    let trim = stock.trim_label(o);
    let grille = crate::scene::procedural_indoor::materials::speaker::perforated_pc(o);
    face::case_shell(a, o, &p, &fascia, &stock, &label, grille);
    let front = face::Plane::Front(o.size.z * 0.5);
    fascia.build(a, front, &stock, &label, &trim, grille);
    sockets::rear(
        a,
        o,
        face::Plane::Back(-0.5 * o.size.z),
        fascia.rear_face,
        &stock,
        &label,
    );
    // Low rubber pads bear on the actual support; wide feet on small desktops.
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            let size = Vec3::new(
                (o.size.x * 0.11).clamp(0.010, 0.026),
                p.feet_m,
                (o.size.z * 0.09).clamp(0.013, 0.036),
            );
            a.box_part(
                Surface::Rubber,
                o.kind.class_name(),
                Vec3::new(x * o.size.x * 0.37, size.y * 0.5, z * o.size.z * 0.38),
                size,
                0.,
            );
        }
    }
}
