use super::*;
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Debug, Clone, serde::Serialize)]
pub struct DesktopItemProgram {
    pub binding: u8,
    pub leaf_fraction: f32,
    pub tip_fraction: f32,
    pub barrel_sides: u32,
    pub case_fraction: f32,
    pub bezel_fraction: f32,
    pub lens_count: usize,
    pub camera_island: bool,
    pub microphone_bend: f32,
    pub microphone_head: f32,
    pub microphone_lateral: f32,
    pub phone_roundness: f32,
    pub phone_top_border: f32,
    pub phone_bottom_border: f32,
    pub phone_controls: u32,
    pub phone_surface: Surface,
    pub phone_face_up: bool,
}
pub fn parameters(o: &IndoorObject) -> DesktopItemProgram {
    let mut r = stream(o.seed, 7450);
    let mut phone = stream(o.seed, 7451);
    DesktopItemProgram {
        binding: r.random_range(0..4),
        leaf_fraction: r.random_range(0.58..0.87),
        tip_fraction: r.random_range(0.08..0.23),
        barrel_sides: [6, 8, 20][r.random_range(0..3)],
        case_fraction: r.random_range(0.70..0.91),
        bezel_fraction: r.random_range(0.035..0.15),
        lens_count: r.random_range(1..=3),
        camera_island: r.random_bool(0.65),
        microphone_bend: r.random_range(0.18..0.57),
        microphone_head: r.random_range(0.045..0.09),
        microphone_lateral: r.random_range(-0.27..0.27),
        phone_roundness: phone.random_range(4.0..18.),
        phone_top_border: phone.random_range(0.035..0.10),
        phone_bottom_border: phone.random_range(0.035..0.16),
        phone_controls: phone.random_range(0..3),
        phone_surface: [Surface::Metal, Surface::Plastic, Surface::Chrome]
            [phone.random_range(0..3)],
        phone_face_up: phone.random_bool(0.76),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Notepad => notepad(a, o),
        ObjectKind::Pencil => pencil(a, o),
        ObjectKind::Microphone => super::microphone::build(a, o),
        ObjectKind::Phone => phone(a, o),
        _ => unreachable!(),
    }
}
fn notepad(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let mut rng = stream(o.seed, 832);
    let p = parameters(o);
    let spiral = p.binding < 2;
    let _ = rng.random_bool(0.5);
    a.box_part(
        Surface::Art,
        "paper",
        Vec3::Y * s.y * 0.06,
        Vec3::new(s.x, s.y * 0.12, s.z),
        0.0005,
    );
    a.box_part(
        Surface::Paper,
        "paper",
        Vec3::Y * s.y * (0.12 + p.leaf_fraction * 0.5),
        Vec3::new(s.x * 0.97, s.y * p.leaf_fraction, s.z * 0.975),
        0.0003,
    );
    let top = s.y * (0.12 + p.leaf_fraction);
    // Printed ruling/writing uses mipmaps, not sub-millimetre shadow-casting rods.
    a.box_part(
        Surface::PrintedPaper,
        "paper",
        Vec3::Y * (top + 0.0002),
        Vec3::new(s.x * 0.966, 0.0002, s.z * 0.97),
        0.,
    );
    if spiral {
        for i in 0..(s.z / if p.binding == 0 { 0.013 } else { 0.020 }) as usize {
            let z = -s.z * 0.45 + i as f32 * if p.binding == 0 { 0.012 } else { 0.019 };
            let wire: Vec<_> = (0..=12)
                .map(|j| {
                    let t = j as f32 * TAU / 12.;
                    Vec3::new(
                        -s.x * 0.45 + t.cos() * 0.0035,
                        s.y * 0.48 + t.sin() * s.y * 0.44,
                        z,
                    )
                })
                .collect();
            a.part(Surface::Chrome, "paper")
                .tube(&wire, (s.y * 0.035).min(0.00045), 6);
        }
    } else {
        a.box_part(
            Surface::Art,
            "paper",
            Vec3::new(0., top + 0.0005, -s.z * 0.456),
            Vec3::new(s.x * 0.97, 0.0002, s.z * 0.052),
            0.,
        );
    }
}
fn pencil(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = parameters(o);
    let radius = s.x.min(s.y) * 0.43;
    let tf =
        Transform::from_xyz(0., radius, s.z * 0.5).with_rotation(Quat::from_rotation_x(-FRAC_PI_2));
    let shaft = s.z * (1. - p.tip_fraction);
    a.part(Surface::Art, "other_prop").lathe(
        &[(0., 0.), (radius, 0.), (radius, shaft), (0., shaft)],
        p.barrel_sides,
        tf,
    );
    a.part(
        if p.barrel_sides == 20 {
            Surface::Metal
        } else {
            Surface::Wood
        },
        "other_prop",
    )
    .lathe(&[(radius, shaft), (radius * 0.13, s.z * 0.98)], 12, tf);
    a.part(Surface::Ink, "other_prop")
        .lathe(&[(radius * 0.13, s.z * 0.98), (0., s.z)], 12, tf);
    if p.binding < 2 {
        a.part(Surface::Chrome, "other_prop").lathe(
            &[(radius * 1.04, s.z * 0.035), (radius * 1.04, s.z * 0.10)],
            12,
            tf,
        );
        a.part(Surface::Rubber, "other_prop").lathe(
            &[(0., 0.), (radius, 0.), (radius, s.z * 0.035)],
            12,
            tf,
        );
    }
}

fn phone(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = parameters(o);
    let l = o.kind.class_name();
    let body = s.y * p.case_fraction;
    a.part(p.phone_surface, l).housing(
        &[
            (0., Vec2::new(s.x, s.z) * 0.475),
            (body * 0.20, Vec2::new(s.x, s.z) * 0.49),
            (body * 0.90, Vec2::new(s.x, s.z) * 0.49),
            (body, Vec2::new(s.x, s.z) * 0.475),
        ],
        p.phone_roundness,
        Transform::IDENTITY,
    );
    let up = p.phone_face_up;
    let face = body + s.y * 0.025;
    if up {
        let extent = Vec2::new(
            s.x * (1. - p.bezel_fraction).min(0.92),
            s.z * (1. - p.phone_top_border - p.phone_bottom_border),
        ) * 0.5;
        a.part(Surface::PhoneScreen, l).housing(
            &[(face - s.y * 0.0225, extent), (face + s.y * 0.0225, extent)],
            p.phone_roundness,
            Transform::from_xyz(
                0.,
                0.,
                s.z * (p.phone_top_border - p.phone_bottom_border) * 0.5,
            ),
        );
        a.part(Surface::Ink, l).cylinder(
            s.x * 0.022,
            0.00025,
            Transform::from_xyz(
                if p.camera_island { -s.x * 0.28 } else { 0. },
                face + s.y * 0.023 + 0.00015,
                -s.z * 0.40,
            ),
        );
    } else {
        a.part(Surface::Plastic, l).housing(
            &[
                (face - s.y * 0.0225, Vec2::new(s.x, s.z) * 0.465),
                (face + s.y * 0.0225, Vec2::new(s.x, s.z) * 0.465),
            ],
            p.phone_roundness,
            Transform::IDENTITY,
        );
        let cy = face + s.y * 0.053;
        if p.camera_island {
            a.part(Surface::Plastic, l).housing(
                &[
                    (cy - s.y * 0.0275, Vec2::new(s.x * 0.18, s.z * 0.115)),
                    (cy + s.y * 0.0275, Vec2::new(s.x * 0.18, s.z * 0.115)),
                ],
                4.,
                Transform::from_xyz(-s.x * 0.20, 0., -s.z * 0.30),
            );
        }
        for i in 0..p.lens_count {
            let x = -s.x * 0.30 + if i == 2 { s.x * 0.18 } else { 0. };
            let z = -s.z * 0.36 + if i == 1 { s.z * 0.10 } else { 0. };
            a.part(Surface::Ink, l).cylinder(
                s.x * 0.065,
                s.y * 0.018,
                Transform::from_xyz(
                    x,
                    if p.camera_island {
                        cy + s.y * 0.0365
                    } else {
                        face + s.y * 0.0315
                    },
                    z,
                ),
            );
        }
    }
    for side in [-1., 1.] {
        a.box_part(
            p.phone_surface,
            l,
            Vec3::new(side * s.x * 0.489, body * 0.55, -s.z * 0.10),
            Vec3::new(
                s.x * 0.014,
                body * 0.24,
                s.z * (0.08 + p.phone_controls as f32 * 0.035),
            ),
            s.y * 0.015,
        );
    }
    if up && p.phone_controls == 2 && p.phone_bottom_border > 0.09 {
        a.part(Surface::Ink, l).cylinder(
            s.x * 0.035,
            s.y * 0.008,
            Transform::from_xyz(
                0.,
                body + s.y * 0.004,
                s.z * (0.48 - p.phone_bottom_border * 0.5),
            ),
        );
    }
}
