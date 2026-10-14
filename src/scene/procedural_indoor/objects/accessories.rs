//! Small equipment with independently sampled proportions and real mechanisms.
use super::*;

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
    let p = parameters(o);
    match o.kind {
        ObjectKind::DeskLamp => lamp(a, o, &p),
        ObjectKind::Speaker => speaker(a, o, &p),
        ObjectKind::Headphones => headphones(a, o, &p),
        ObjectKind::Stapler => stapler(a, o, &p),
        ObjectKind::TapeDispenser => tape(a, o, &p),
        ObjectKind::DocumentTray | ObjectKind::DeskOrganizer => organizer(a, o, &p),
        ObjectKind::TissueBox => super::containers::tissues(a, o),
        ObjectKind::DeskFan => fan(a, o, &p),
        _ => unreachable!("non-accessory object"),
    }
}

fn lamp(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.part(p.finish, label).cylinder(
        s.x.min(s.z) * 0.43,
        s.y * 0.055,
        Transform::from_xyz(0., s.y * 0.0275, 0.),
    );
    let root = Vec3::new(0., s.y * 0.055, -s.z * 0.16);
    let elbow = Vec3::new(-s.x * p.tilt, s.y * 0.57, -s.z * 0.24);
    let head = lamp_head(o, p);
    if p.style < 2 {
        a.part(p.finish, label).rod(root, elbow, 0.007);
        a.part(p.finish, label).rod(elbow, head, 0.007);
        a.part(Surface::Metal, label)
            .ellipsoid(Vec3::splat(0.012), Transform::from_translation(elbow));
    } else {
        let points: Vec<_> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.;
                root.lerp(head, t)
                    + Vec3::new(
                        0.,
                        s.y * 0.15 * (PI * t).sin(),
                        -s.z * 0.12 * (PI * t).sin(),
                    )
            })
            .collect();
        a.part(p.finish, label).tube(&points, 0.008, 10);
    }
    // A shallow down-facing head; the inset diffuser is separated from its case.
    let width = s.x
        * if p.style.is_multiple_of(2) {
            0.78
        } else {
            0.48
        };
    let depth = s.z
        * if p.style.is_multiple_of(2) {
            0.28
        } else {
            0.40
        };
    a.box_part(
        p.finish,
        label,
        head,
        Vec3::new(width, s.y * 0.08, depth),
        0.008,
    );
    a.box_part(
        Surface::Light,
        label,
        head - Vec3::Y * (s.y * 0.041),
        Vec3::new(width * 0.88, 0.001, depth * 0.83),
        0.001,
    );
}

fn lamp_head(o: &IndoorObject, p: &AccessoryProgram) -> Vec3 {
    Vec3::new(
        o.size.x * p.tilt * 0.45,
        o.size.y * 0.91,
        o.size.z * (p.reach - 0.35).clamp(-0.15, 0.27),
    )
}

/// The direct renderer and diffuse transport use the same metric emitter. Its
/// position sits just below the inset diffuser, outside the opaque housing.
pub(crate) fn lamp_photometry(o: &IndoorObject) -> (Vec3, Vec3, f32, f32, f32) {
    let p = parameters(o);
    let mut rng = stream(o.seed, 5320);
    let position = o
        .transform()
        .transform_point(lamp_head(o, &p) - Vec3::Y * o.size.y * 0.044);
    let color = super::super::materials::kelvin_rgb(rng.random_range(2700.0..5800.0));
    (
        position,
        color,
        rng.random_range(150.0..550.0),
        rng.random_range(0.32..0.56),
        rng.random_range(0.82..1.10),
    )
}

fn speaker(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.part(
        if p.style == 0 {
            Surface::WoodEdge
        } else {
            p.finish
        },
        label,
    )
    .housing(
        &[
            (0., Vec2::new(s.x * 0.47 * p.body_taper, s.z * 0.46)),
            (s.y * 0.35, Vec2::new(s.x * 0.49, s.z * 0.46)),
            (s.y, Vec2::new(s.x * 0.49 * p.body_taper, s.z * 0.46)),
        ],
        p.profile_roundness,
        Transform::IDENTITY,
    );
    if p.style < 2 {
        for (y, r) in [
            (0.34, p.control_fraction.min(0.36)),
            (0.78, p.support_fraction.min(0.18)),
        ] {
            a.part(Surface::Rubber, label).cylinder(
                s.x.min(s.y) * r,
                s.z * 0.055,
                Transform::from_xyz(0., s.y * y, s.z * 0.45)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
            a.part(Surface::Fabric, label).ellipsoid(
                Vec3::new(s.x * r * 0.82, s.x * r * 0.82, s.z * 0.022),
                Transform::from_xyz(0., s.y * y, s.z * 0.475),
            );
        }
    } else {
        a.box_part(
            Surface::FabricAlt,
            label,
            Vec3::new(0., s.y * 0.5, s.z * 0.47),
            Vec3::new(s.x * p.body_taper * 0.93, s.y * 0.94, s.z * 0.04),
            s.x * 0.07,
        );
    }
}

fn headphones(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let half = s.x * (0.32 + p.support_fraction * 0.20);
    let cup_h = s.y * p.control_fraction;
    let cup_y = cup_h;
    let band_top = s.y * 0.95;
    let points: Vec<_> = (0..=20)
        .map(|i| {
            let t = i as f32 * PI / 20.;
            Vec3::new(t.cos() * half, cup_y + (band_top - cup_y) * t.sin(), 0.)
        })
        .collect();
    a.part(p.finish, l).tube(&points, s.x.min(s.z) * 0.038, 8);
    for side in [-1., 1.] {
        let c = Vec3::new(side * half, cup_y, 0.);
        a.part(p.finish, l).cushion(
            Vec3::new(s.x * 0.20, cup_h * 2., s.z * (0.66 + p.support_fraction)),
            super::super::geometry::CushionProfile {
                roundness: p.profile_roundness,
                crown: 2.4,
                dish: 0.,
            },
            Transform::from_translation(c),
        );
        a.part(Surface::Leather, l).ellipsoid(
            Vec3::new(s.x * 0.055, cup_h * 0.83, s.z * 0.33),
            Transform::from_translation(c - Vec3::X * side * s.x * 0.08),
        );
    }
}

fn stapler(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let base = s.y * p.support_fraction;
    a.box_part(
        Surface::Rubber,
        l,
        Vec3::Y * base * 0.5,
        Vec3::new(s.x, base, s.z),
        s.x * 0.07,
    );
    let jaw = s.y * p.control_fraction;
    a.box_part(
        Surface::Metal,
        l,
        Vec3::new(0., base + jaw * 0.48, -s.z * 0.02),
        Vec3::new(s.x * 0.70, jaw, s.z * 0.86),
        s.x * 0.025,
    );
    let upper = s.y - base - jaw;
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., base + jaw + upper * 0.5, -s.z * 0.035),
        Vec3::new(
            s.x * (0.73 + p.support_fraction),
            upper,
            s.z * (0.70 + p.support_fraction),
        ),
        s.x * (0.08 + 0.10 * p.body_taper),
    );
    a.part(Surface::Metal, l).cylinder(
        jaw * 0.38,
        s.x * 0.96,
        Transform::from_xyz(0., base + jaw * 0.68, -s.z * 0.30)
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    );
}

fn tape(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        p.finish,
        label,
        Vec3::Y * s.y * p.support_fraction * 0.5,
        Vec3::new(s.x, s.y * p.support_fraction, s.z),
        0.007,
    );
    let roll = Transform::from_xyz(
        0.,
        s.y * (0.48 + p.support_fraction * 0.4),
        -s.z * (0.06 + p.support_fraction * 0.5),
    )
    .with_rotation(Quat::from_rotation_z(FRAC_PI_2));
    let r = (s.y * p.control_fraction).min(s.z * 0.28);
    for side in [-1., 1.] {
        a.part(p.finish, label).rod(
            Vec3::new(side * s.x * 0.38, s.y * p.support_fraction, -s.z * 0.2),
            Vec3::new(side * s.x * 0.38, roll.translation.y, roll.translation.z),
            s.x * 0.065,
        );
    }
    a.part(Surface::Plastic, label)
        .cylinder(r * 0.49, s.x * 0.86, roll);
    a.part(Surface::Paper, label).lathe(
        &[
            (r * 0.52, -s.x * 0.30),
            (r, -s.x * 0.30),
            (r, s.x * 0.30),
            (r * 0.52, s.x * 0.30),
            (r * 0.52, -s.x * 0.30),
        ],
        24,
        roll,
    );
    a.box_part(
        Surface::Metal,
        label,
        Vec3::new(0., s.y * 0.39, s.z * 0.39),
        Vec3::new(s.x * 0.93, s.y * 0.12, s.z * 0.08),
        0.001,
    );
}

fn organizer(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let levels = if o.kind == ObjectKind::DocumentTray {
        p.sections.min(3)
    } else {
        1
    };
    let t = s.y.min(s.z) * (0.025 + p.support_fraction * 0.12);
    if levels > 1 {
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(
                        side * (s.x * 0.5 - t * 1.75),
                        (s.y + t) * 0.5,
                        end * (s.z * 0.5 - t * 1.75),
                    ),
                    Vec3::new(t, s.y - t, t),
                    t * 0.2,
                );
            }
        }
    }
    for level in 0..levels {
        let base = level as f32 * s.y / levels as f32;
        let height = s.y / levels as f32 * p.body_taper;
        a.box_part(
            p.finish,
            label,
            Vec3::new(0., base + t * 0.5, 0.),
            Vec3::new(s.x, t, s.z),
            0.001,
        );
        for side in [-1., 1.] {
            a.box_part(
                p.finish,
                label,
                Vec3::new(side * (s.x - t) * 0.5, base + (height + t) * 0.5, 0.),
                Vec3::new(t, height - t, s.z),
                0.001,
            );
        }
        a.box_part(
            p.finish,
            label,
            Vec3::new(0., base + (height + t) * 0.5, -(s.z - t) * 0.5),
            Vec3::new(s.x - 2. * t, height - t, t),
            0.001,
        );
        if o.kind == ObjectKind::DocumentTray {
            a.box_part(
                Surface::PrintedPaper,
                "paper",
                Vec3::new(0., base + t + height * 0.09, s.z * 0.02),
                Vec3::new(s.x * 0.83, height * 0.18, s.z * 0.87),
                0.001,
            );
        } else {
            for section in 1..p.sections {
                let x = s.x * (section as f32 / p.sections as f32 - 0.5);
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(x, (s.y * p.body_taper + t) * 0.5, 0.),
                    Vec3::new(t, s.y * p.body_taper - t, s.z * 0.92),
                    0.001,
                );
            }
            for i in 0..p.sections {
                let x = s.x * ((i as f32 + 0.5) / p.sections as f32 - 0.5);
                a.part(Surface::Wood, "other_prop").rod(
                    Vec3::new(x, t, 0.),
                    Vec3::new(x, s.y * 0.94, -s.z * 0.10),
                    0.003,
                );
            }
        }
    }
}

fn fan(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        p.finish,
        label,
        Vec3::Y * s.y * 0.04,
        Vec3::new(s.x * 0.66, s.y * 0.08, s.z),
        0.008,
    );
    a.part(p.finish, label)
        .rod(Vec3::Y * s.y * 0.08, Vec3::Y * s.y * 0.69, 0.012);
    let center = Vec3::Y * s.y * (0.66 + p.support_fraction * 0.20);
    let r = (s.x * (0.30 + p.support_fraction * 0.6)).min(s.y * 0.26);
    for z in [-0.20, 0.20] {
        let ring: Vec<_> = (0..=32)
            .map(|i| {
                let t = i as f32 * TAU / 32.;
                center + Vec3::new(t.cos() * r, t.sin() * r, s.z * z)
            })
            .collect();
        a.part(Surface::Metal, label).tube(&ring, 0.0025, 6);
        for i in 0..12 {
            let t = i as f32 * TAU / 12.;
            a.part(Surface::Metal, label).rod(
                center + Vec3::Z * s.z * z,
                center + Vec3::new(t.cos() * r, t.sin() * r, s.z * z),
                0.0012,
            );
        }
    }
    a.part(p.finish, label).cylinder(
        r * 0.23,
        s.z * 0.46,
        Transform::from_translation(center).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
    for i in 0..(3 + p.style as usize % 3) {
        let n = 3 + p.style as usize % 3;
        let t = i as f32 * TAU / n as f32;
        a.part(Surface::Plastic, label).ellipsoid(
            Vec3::new(r * 0.42, r * 0.17, s.z * 0.035),
            Transform::from_translation(
                center + Vec3::new(t.cos() * r * 0.48, t.sin() * r * 0.48, 0.),
            )
            .with_rotation(Quat::from_rotation_z(t + 0.28)),
        );
    }
}
