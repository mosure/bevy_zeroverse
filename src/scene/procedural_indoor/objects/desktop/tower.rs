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

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let l = o.kind.class_name();
    let stock = Housing::for_object(o);
    let case = stock.case_label(o);
    let p = parameters(o);
    let feet = p.feet_m;
    let face = p.fascia_depth_m;
    for x in [-0.36, 0.36] {
        for z in [-0.37, 0.37] {
            a.box_part(
                Surface::Rubber,
                l,
                Vec3::new(s.x * x, feet * 0.5, s.z * z),
                Vec3::new(s.x * 0.11, feet, s.z * 0.09),
                feet * 0.22,
            );
        }
    }
    let h = s.y - feet;
    a.box_part(
        stock.surface,
        &case,
        Vec3::new(0., feet + h * 0.5, -face * 0.5),
        Vec3::new(s.x, h, s.z - face),
        0.0035,
    );
    // Recessed intake backing, with raised louvers and a separate control fascia.
    let front = s.z * 0.5;
    let fraction = p.intake_fraction;
    let vent_h = h * fraction;
    a.box_part(
        Surface::Rubber,
        l,
        Vec3::new(0., feet + vent_h * 0.5, front - face * 0.57),
        Vec3::new(s.x * 0.97, vent_h, face * 0.74),
        0.002,
    );
    a.box_part(
        stock.surface,
        &case,
        Vec3::new(0., feet + vent_h + (h - vent_h) * 0.5, front - face * 0.5),
        Vec3::new(s.x * 0.99, h - vent_h, face),
        0.002,
    );
    let vertical = p.vertical_louvers;
    let run = if vertical { s.x * 0.88 } else { vent_h * 0.88 };
    let slats = (run / p.louver_pitch_m).round().clamp(3., 19.) as usize;
    for i in 0..slats {
        let f = ((i as f32 + 0.5) / slats as f32 - 0.5) * run;
        let thickness = run / slats as f32 * 0.50;
        let (x, y, w, height) = if vertical {
            (f, feet + vent_h * 0.5, thickness, vent_h * 0.91)
        } else {
            (0., feet + vent_h * 0.5 + f, s.x * 0.89, thickness)
        };
        a.box_part(
            stock.surface,
            &case,
            Vec3::new(x, y, front - face * 0.13),
            Vec3::new(w, height, face * 0.22),
            0.0009,
        );
    }
    let y = feet + vent_h + (h - vent_h) * 0.55;
    let button = s.x.min(h) * 0.052;
    a.part(Surface::Rubber, l).cylinder(
        button,
        0.002,
        Transform::from_xyz(-s.x * 0.29, y, front + 0.0006)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
    for i in [0.10, 0.28] {
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(s.x * i, y, front + 0.00045),
            Vec3::new(0.012, 0.0045, 0.001),
            0.0006,
        );
    }
    for plug in ports::TowerJack::ALL {
        let pt = ports::tower_local(o, plug);
        let (w, h) = match plug {
            ports::TowerJack::Mains => (0.020, 0.013),
            ports::TowerJack::Video(_) => (0.015, 0.0055),
            _ => (0.012, 0.0045),
        };
        a.box_part(
            Surface::Rubber,
            l,
            pt + Vec3::Z * 0.001,
            Vec3::new(w, h, 0.0014),
            0.00065,
        );
    }
}
