//! Appliance cavities, doors, controls and supported interior contents.
use super::super::super::motion::{construction as motion, MotionKind};
use super::*;
// The cooktop sheet is 0.07 * 0.24 of total height; its work plane is at 0.94.
const RANGE_BODY_TOP: f32 = 0.94 - 0.07 * 0.24;
fn knobs(a: &mut Assembly, base: Vec3, step: Vec3, count: u32, radius: f32, label: &str) {
    for i in 0..count {
        let p = base + step * i as f32;
        a.part(Surface::Metal, label).cylinder(
            radius,
            0.016,
            Transform::from_translation(p).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
        a.box_part(
            Surface::Ink,
            label,
            p + Vec3::new(0., radius * 0.42, 0.009),
            Vec3::new(0.002, radius * 0.65, 0.001),
            0.0001,
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn oven_leaf(
    a: &mut Assembly,
    width: f32,
    height: f32,
    bottom: Vec2,
    front: f32,
    p: &KitchenProgram,
    label: &str,
    seed: u64,
) {
    let rotation = Quat::from_rotation_x(p.opening_radians);
    let tf = Transform::from_translation(
        Vec3::new(bottom.x, bottom.y, front + 0.014) + rotation * Vec3::Y * height * 0.5,
    )
    .with_rotation(rotation);
    let joint = || {
        let mut joint = motion::rigid(
            MotionKind::Appliance,
            seed,
            Vec3::new(bottom.x, bottom.y, front + 0.014),
            Vec3::X,
            motion::endpoint(seed, p.opening_radians, 1.35) - p.opening_radians,
            Vec3::ZERO,
        );
        joint.frame = tf;
        joint
    };
    motion::hinged(a, MotionKind::Appliance, seed, joint, |a| {
        cabinet::panel(
            a,
            Vec2::new(width, height),
            tf,
            Surface::Metal,
            label,
            p.viewport,
        );
        cabinet::handle(a, tf, 0., height * 0.36, p, true, label);
    });
}
fn side_leaf(
    a: &mut Assembly,
    center: Vec3,
    width: f32,
    height: f32,
    side: f32,
    p: &KitchenProgram,
    o: &IndoorObject,
) {
    let label = o.kind.class_name();
    let seed = o.seed ^ ((center.x.to_bits() as u64) << 32) ^ center.y.to_bits() as u64;
    let rotation = Quat::from_rotation_y(side * p.opening_radians);
    let tf = Transform::from_translation(
        center + Vec3::X * side * width * 0.5 + rotation * Vec3::X * (-side * width * 0.5),
    )
    .with_rotation(rotation);
    let joint = || {
        let mut joint = motion::rigid(
            MotionKind::Appliance,
            seed,
            center + Vec3::X * side * width * 0.5,
            Vec3::Y,
            side * (motion::endpoint(seed, p.opening_radians, 1.35) - p.opening_radians),
            Vec3::ZERO,
        );
        joint.frame = tf;
        joint
    };
    motion::hinged(a, MotionKind::Appliance, seed, joint, |a| {
        cabinet::panel(
            a,
            Vec2::new(width - 0.004, height),
            tf,
            Surface::Metal,
            label,
            p.viewport,
        );
        cabinet::handle(a, tf, -side * (width * 0.5 - 0.045), 0., p, false, label);

        if o.kind == ObjectKind::Refrigerator {
            for n in 0..p.door_notes {
                let x = (if p.viewport {
                    side * 0.32
                } else {
                    -0.15 + n as f32 * 0.10
                }) * width;
                let y = o.size.y * (0.28 - n as f32 * 0.11);
                box_at(
                    a,
                    if n.is_multiple_of(2) {
                        Surface::Art
                    } else {
                        Surface::Paper
                    },
                    "paper",
                    Vec3::new(x, y, 0.030),
                    Vec3::new(width * 0.13, o.size.y * 0.065, 0.002),
                    tf,
                );
                box_at(
                    a,
                    Surface::Plastic,
                    "paper",
                    Vec3::new(x, y + o.size.y * 0.026, 0.033),
                    Vec3::new(0.015, 0.012, 0.003),
                    tf,
                );
                if !n.is_multiple_of(2) {
                    for line in 0..3 {
                        box_at(
                            a,
                            Surface::Ink,
                            "paper",
                            Vec3::new(x, y - line as f32 * o.size.y * 0.012, 0.0315),
                            Vec3::new(width * 0.09, 0.0015, 0.0005),
                            tf,
                        );
                    }
                }
            }
        }
    });
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram) {
    if o.kind == ObjectKind::Cooktop {
        cooktop(a, o.size, Vec3::ZERO, p);
        return;
    }
    let s = o.size;
    let (d, z) = body(o, p);
    let w = s.x - 0.018;
    let front = z + d * 0.5;
    let label = o.kind.class_name();
    let toe = if o.kind == ObjectKind::Microwave {
        0.012
    } else {
        0.075
    };
    a.box_part(
        Surface::Rubber,
        label,
        Vec3::new(0., toe * 0.5, z),
        Vec3::new(w * 0.89, toe, d * 0.89),
        0.004,
    );
    cabinet::shell(
        a,
        Vec3::new(
            w,
            if o.kind == ObjectKind::Range {
                s.y * RANGE_BODY_TOP - toe
            } else {
                s.y - toe - 0.006
            },
            d,
        ),
        toe,
        z,
        Surface::Metal,
        label,
    );
    // A distinct inner lining makes open doors reveal a cavity rather than a dark decal.
    a.box_part(
        Surface::Plastic,
        label,
        Vec3::new(0., s.y * 0.51, z - d * 0.5 + 0.025),
        Vec3::new(w * 0.91, s.y * 0.88, 0.008),
        0.002,
    );
    match o.kind {
        ObjectKind::Refrigerator => {
            for i in 0..p.shelves {
                let y = toe + 0.04 + i as f32 * (s.y - toe - 0.15) / p.shelves as f32;
                a.box_part(
                    Surface::ContainerGlass,
                    label,
                    Vec3::new(0., y, z),
                    Vec3::new(w * 0.92, 0.012, d * 0.88),
                    0.001,
                );
                if p.contents {
                    let items = contents::shelf_contents(
                        o,
                        i,
                        (
                            Vec2::new(-w * 0.45, z - d * 0.42),
                            Vec2::new(w * 0.45, z + d * 0.42),
                        ),
                        y + 0.007,
                        (s.y - toe - 0.15) / p.shelves as f32 - 0.015,
                    );
                    contents::build(a, &items);
                }
            }
            let count = if p.paired { 2 } else { 1 };
            let leaf = (w - 0.018) / count as f32;
            for i in 0..count {
                let side = if i == 0 { -1. } else { 1. };
                let cx = if count == 2 { side * leaf * 0.5 } else { 0. };
                side_leaf(
                    a,
                    Vec3::new(cx, (s.y + toe) * 0.5, front + 0.014),
                    leaf,
                    s.y - toe - 0.02,
                    side,
                    p,
                    o,
                );
            }
        }
        ObjectKind::Microwave => {
            let lower = p.controls == 2;
            let sign = if p.controls == 1 { -1. } else { 1. };
            let door_w = if lower { w * 0.94 } else { w * 0.77 };
            let cx = if lower { 0. } else { -sign * w * 0.10 };
            side_leaf(
                a,
                Vec3::new(cx, s.y * if lower { 0.64 } else { 0.52 }, front + 0.014),
                door_w,
                s.y * if lower { 0.60 } else { 0.82 },
                -sign,
                p,
                o,
            );
            let center = if lower {
                Vec3::new(0., s.y * 0.17, front + 0.016)
            } else {
                Vec3::new(sign * w * 0.40, s.y * 0.51, front + 0.016)
            };
            let panel_size = if lower {
                Vec3::new(w * 0.96, s.y * 0.21, 0.025)
            } else {
                Vec3::new(w * 0.16, s.y * 0.86, 0.025)
            };
            a.box_part(Surface::Plastic, label, center, panel_size, 0.003);
            if lower {
                knobs(
                    a,
                    center + Vec3::new(-w * 0.29, 0., 0.021),
                    Vec3::X * w * 0.18,
                    3,
                    0.012,
                    label,
                );
            } else {
                knobs(
                    a,
                    Vec3::new(sign * w * 0.40, s.y * 0.24, front + 0.037),
                    Vec3::Y * s.y * 0.19,
                    2,
                    0.016,
                    label,
                );
            }
            a.box_part(
                Surface::Screen,
                label,
                if lower {
                    center + Vec3::new(w * 0.30, 0., 0.014)
                } else {
                    Vec3::new(sign * w * 0.40, s.y * 0.77, front + 0.030)
                },
                Vec3::new(w * 0.12, s.y * 0.11, 0.002),
                0.001,
            );
            let tray_y = if lower {
                s.y * 0.34 + 0.01
            } else {
                toe + 0.025
            };
            if lower {
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(0., tray_y - 0.013, z),
                    Vec3::new(w * 0.94, 0.018, d * 0.92),
                    0.001,
                );
            }
            a.part(Surface::ContainerGlass, "dishware").cylinder(
                (w * 0.30).min(d * 0.38),
                0.007,
                Transform::from_xyz(cx, tray_y, z),
            );
            if p.contents {
                let mut bowl = dishware::parameters(
                    Vec3::new(w * 0.30, s.y * 0.48, d * 0.38),
                    1,
                    1,
                    o.seed ^ 0x4d4943524f444953,
                );
                // Microwave contents use a microwave-safe substrate.
                bowl.surface = Surface::Ceramic;
                dishware::build(
                    a,
                    Transform::from_xyz(cx, tray_y + 0.0035, z),
                    &bowl,
                    o.seed,
                );
            }
        }
        ObjectKind::Dishwasher => {
            let bottom = toe + 0.024;
            // Fit below the control fascia even on short compact appliances.
            // A fixed fraction plus the toe height overlapped the fascia.
            let height = (s.y * 0.865 - bottom - 0.012).max(0.10);
            oven_leaf(
                a,
                w * 0.96,
                height,
                Vec2::new(0., bottom),
                front,
                p,
                label,
                o.seed,
            );
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0., s.y * 0.925, front + 0.014),
                Vec3::new(w * 0.97, s.y * 0.12, 0.022),
                0.003,
            );
            knobs(
                a,
                Vec3::new(-w * 0.30, s.y * 0.925, front + 0.03),
                Vec3::X * w * 0.18,
                4,
                0.012,
                label,
            );
            for (index, level) in [0.22, 0.55].into_iter().enumerate() {
                props::rack(
                    a,
                    Vec3::new(0., s.y * level, z),
                    Vec3::new(w * 0.86, s.y * 0.23, d * 0.79),
                    if p.contents { 4 } else { 0 },
                    o.seed ^ (0x444953485241434b + index as u64),
                );
            }
        }
        ObjectKind::Oven | ObjectKind::Range => {
            let tall = o.kind == ObjectKind::Oven && s.y > 1.3;
            let layers = if tall && p.stacked_ovens { 2 } else { 1 };
            let dh = if tall { s.y * 0.28 } else { s.y * 0.73 };
            let mut apertures = Vec::new();
            for level in 0..layers {
                let bottom = if tall {
                    s.y * (if layers == 2 {
                        0.26 + level as f32 * 0.32
                    } else {
                        0.47
                    })
                } else {
                    toe + 0.012
                };
                let mut q = p.clone();
                if level > 0 {
                    q.opening_radians = 0.;
                }
                let offset = match p.controls {
                    1 => w * 0.08,
                    2 => -w * 0.08,
                    _ => 0.,
                };
                let width = if p.controls == 0 { w * 0.92 } else { w * 0.74 };
                apertures.push([
                    Vec2::new(offset - width * 0.5 - 0.003, bottom - 0.014),
                    Vec2::new(offset + width * 0.5 + 0.003, bottom + dh + 0.003),
                ]);
                oven_leaf(
                    a,
                    width,
                    dh,
                    Vec2::new(offset, bottom),
                    front,
                    &q,
                    label,
                    o.seed ^ level as u64,
                );
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(0., bottom + 0.008, z),
                    Vec3::new(w * 0.86, 0.016, d * 0.83),
                    0.002,
                );
                let y = bottom + dh * 0.38;
                for n in 0..9 {
                    let x = (n as f32 / 8. - 0.5) * w * 0.80;
                    pipe(
                        a,
                        Surface::Metal,
                        label,
                        &[Vec3::new(x, y, z - d * 0.36), Vec3::new(x, y, z + d * 0.36)],
                        0.003,
                    );
                }
                if p.contents {
                    a.box_part(
                        Surface::Metal,
                        "dishware",
                        Vec3::new(0., y + 0.008, z),
                        Vec3::new(w * 0.57, 0.011, d * 0.55),
                        0.004,
                    );
                }
            }
            cabinet::aperture_front(
                a,
                if tall { p.finish } else { Surface::Metal },
                label,
                [
                    Vec2::new(-w * 0.485, toe),
                    Vec2::new(
                        w * 0.485,
                        s.y * if o.kind == ObjectKind::Range {
                            RANGE_BODY_TOP
                        } else {
                            0.99
                        },
                    ),
                ],
                &apertures,
                front + 0.014,
            );
            let y = if tall { s.y * 0.90 } else { s.y * 0.86 };
            if p.controls != 0 {
                let x = if p.controls == 1 { -w * 0.42 } else { w * 0.42 };
                let cy = if tall { s.y * 0.64 } else { s.y * 0.48 };
                knobs(
                    a,
                    Vec3::new(x, cy - s.y * 0.15, front + 0.031),
                    Vec3::Y * s.y * 0.14,
                    3,
                    0.015,
                    label,
                );
            } else {
                knobs(
                    a,
                    Vec3::new(-w * 0.30, y, front + 0.032),
                    Vec3::X * w * 0.20,
                    4,
                    0.016,
                    label,
                );
            }
            if o.kind == ObjectKind::Range {
                cooktop(
                    a,
                    Vec3::new(w, s.y * 0.07, d),
                    Vec3::new(0., s.y * RANGE_BODY_TOP, z),
                    p,
                );
            }
        }
        _ => unreachable!(),
    }
}
fn cooktop(a: &mut Assembly, s: Vec3, base: Vec3, p: &KitchenProgram) {
    a.box_part(
        if p.gas { Surface::Metal } else { Surface::Ink },
        "stove",
        base + Vec3::Y * s.y * 0.12,
        Vec3::new(s.x, s.y * 0.24, s.z),
        0.005,
    );
    for i in 0..4 {
        let x = if i % 2 == 0 { -0.22 } else { 0.22 };
        let z = if i < 2 { -0.22 } else { 0.20 };
        let c = base + Vec3::new(s.x * x, s.y * 0.30, s.z * z);
        let radius = s.x.min(s.z) * if i == 0 { 0.145 } else { 0.115 };
        a.part(Surface::Ink, "stove")
            .cylinder(radius, s.y * 0.18, Transform::from_translation(c));
        if p.gas {
            a.part(Surface::Metal, "stove").cylinder(
                radius * 0.72,
                s.y * 0.12,
                Transform::from_translation(c + Vec3::Y * s.y * 0.15),
            );
            for axis in [Vec3::X, Vec3::Z] {
                pipe(
                    a,
                    Surface::Ink,
                    "stove",
                    &[
                        c - axis * radius * 1.30 + Vec3::Y * s.y * 0.20,
                        c + axis * radius * 1.30 + Vec3::Y * s.y * 0.20,
                    ],
                    0.006,
                );
            }
        } else {
            for ring in [0.72, 0.94] {
                let points: Vec<_> = (0..=32)
                    .map(|k| {
                        let t = TAU * k as f32 / 32.;
                        c + Vec3::new(t.cos() * radius * ring, s.y * 0.12, t.sin() * radius * ring)
                    })
                    .collect();
                pipe(a, Surface::Metal, "stove", &points, 0.0012);
            }
        }
    }
    // Controls share the chosen front/back/side placement with the physical knobs.
    for i in 0..4 {
        let t = (i as f32 - 1.5) * 0.16;
        let (x, z) = match p.controls {
            0 => (t, 0.43),
            1 => (t, -0.43),
            _ => (0.43, t),
        };
        a.part(Surface::Plastic, "stove").cylinder(
            s.x.min(s.z) * 0.025,
            s.y * 0.32,
            Transform::from_translation(base + Vec3::new(x * s.x, s.y * 0.42, z * s.z)),
        );
    }
}
