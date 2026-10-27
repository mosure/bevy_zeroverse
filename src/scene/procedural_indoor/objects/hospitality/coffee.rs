//! Countertop brewers with actual group heads, cup wells, filter baskets and
//! removable carafes. All parts stay inside the admitted metric envelope.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Brewer {
    Espresso,
    Capsule,
    Automatic,
    Filter,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CoffeeProgram {
    pub brewer: Brewer,
    pub body_width: f32,
    pub body_depth: f32,
    pub shoulder_height: f32,
    pub corner_exponent: f32,
    pub head_height: f32,
    pub controls: u32,
    pub tray_fraction: f32,
    pub thermal_carafe: bool,
    pub finish: Surface,
    pub trim: Surface,
}

pub fn parameters(o: &IndoorObject) -> CoffeeProgram {
    let mut r = stream(o.seed, 0x434f_4646_4545_4745);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x434f_4646_4545_4649);
    CoffeeProgram {
        brewer: [
            Brewer::Espresso,
            Brewer::Capsule,
            Brewer::Automatic,
            Brewer::Filter,
        ][r.random_range(0..4)],
        body_width: r.random_range(0.78..0.96),
        body_depth: r.random_range(0.40..0.53),
        shoulder_height: r.random_range(0.78..0.90),
        corner_exponent: r.random_range(2.8..7.0),
        head_height: r.random_range(0.49..0.60),
        controls: r.random_range(1..=4),
        tray_fraction: r.random_range(0.68..0.90),
        thermal_carafe: r.random_bool(0.42),
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][f.random_range(0..3)],
        trim: [Surface::Chrome, Surface::Metal, Surface::Ink][f.random_range(0..3)],
    }
}

pub fn dimensions(seed: u64) -> Vec3 {
    let mut r = stream(seed, 0x434f_4646_4545_5349);
    // Width, capacity and depth vary independently before placement; compact
    // capsule machines and larger espresso units are not one scaled chassis.
    let mut shape = stream(seed, 0x434f_4646_4545_4745);
    let (width, height, depth) = match shape.random_range(0..4usize) {
        0 => ((0.28, 0.46), (0.32, 0.48), (0.31, 0.45)),
        1 => ((0.17, 0.26), (0.25, 0.36), (0.29, 0.39)),
        2 => ((0.25, 0.37), (0.34, 0.47), (0.34, 0.46)),
        _ => ((0.22, 0.33), (0.31, 0.45), (0.26, 0.38)),
    };
    Vec3::new(
        r.random_range(width.0..width.1),
        r.random_range(height.0..height.1),
        r.random_range(depth.0..depth.1),
    )
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let base = s.y * 0.07;
    a.box_part(
        Surface::Rubber,
        l,
        Vec3::Y * base * 0.5,
        Vec3::new(s.x * 0.98, base, s.z * 0.98),
        0.004,
    );
    // Rear service connector remains at the published equipment-port location.
    a.box_part(
        Surface::Ink,
        l,
        Vec3::new(0., s.y * 0.10, -s.z * 0.49),
        Vec3::new(s.x * 0.11, s.y * 0.06, s.z * 0.02),
        0.001,
    );
    match p.brewer {
        Brewer::Filter => filter(a, o, &p, base),
        Brewer::Capsule => capsule(a, o, &p, base),
        _ => pressure_brewer(a, o, &p, base),
    }
}

fn capsule(a: &mut Assembly, o: &IndoorObject, p: &CoffeeProgram, base: f32) {
    let s = o.size;
    let l = o.kind.class_name();
    // A compact curved pressure shell and separate rear reservoir, not an
    // espresso chassis with its portafilter omitted.
    a.part(p.finish, l).housing(
        &[
            (base, Vec2::new(s.x * 0.35, s.z * 0.20)),
            (s.y * 0.32, Vec2::new(s.x * p.body_width * 0.48, s.z * 0.23)),
            (
                s.y * p.shoulder_height,
                Vec2::new(s.x * p.body_width * 0.47, s.z * 0.22),
            ),
            (s.y * 0.94, Vec2::new(s.x * 0.28, s.z * 0.17)),
        ],
        2.0 + p.corner_exponent * 0.18,
        Transform::from_xyz(0., 0., -s.z * 0.08),
    );
    a.part(Surface::Plastic, l).housing(
        &[
            (base, Vec2::new(s.x * 0.29, s.z * 0.075)),
            (s.y * 0.74, Vec2::new(s.x * 0.26, s.z * 0.071)),
        ],
        3.4,
        Transform::from_xyz(0., 0., -s.z * 0.414),
    );
    a.box_part(
        p.trim,
        l,
        Vec3::new(0., s.y * p.head_height, s.z * 0.22),
        Vec3::new(s.x * 0.24, s.y * 0.11, s.z * 0.24),
        0.005,
    );
    a.part(Surface::Ink, l).cylinder(
        s.x * 0.025,
        s.y * 0.035,
        Transform::from_xyz(0., s.y * (p.head_height - 0.07), s.z * 0.29),
    );
    a.box_part(
        p.trim,
        l,
        Vec3::new(0., base + s.y * 0.025, s.z * 0.29),
        Vec3::new(s.x * p.tray_fraction, s.y * 0.05, s.z * 0.36),
        0.004,
    );
    for i in 0..5 {
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(
                (i as f32 - 2.) * s.x * p.tray_fraction * 0.16,
                base + s.y * 0.05 + 0.0006,
                s.z * 0.29,
            ),
            Vec3::new(s.x * 0.018, 0.001, s.z * 0.27),
            0.,
        );
    }
    let lever = [
        Vec3::new(0., s.y * 0.92, -s.z * 0.20),
        Vec3::new(0., s.y * 0.97, -s.z * 0.16),
        Vec3::new(0., s.y * 0.97, s.z * 0.045),
        Vec3::new(0., s.y * 0.92, s.z * 0.06),
    ];
    for edge in lever.windows(2) {
        a.part(p.trim, l).rod(edge[0], edge[1], s.y * 0.01);
    }
}

fn pressure_brewer(a: &mut Assembly, o: &IndoorObject, p: &CoffeeProgram, base: f32) {
    let s = o.size;
    let l = o.kind.class_name();
    let w = s.x * p.body_width;
    let d = s.z * p.body_depth;
    let rear = -s.z * 0.49 + d * 0.5;
    let top = s.y
        * if p.brewer == Brewer::Automatic {
            0.86
        } else {
            0.96
        };
    a.part(p.finish, l).housing(
        &[
            (base, Vec2::new(w * 0.48, d * 0.5)),
            (top * p.shoulder_height, Vec2::new(w * 0.5, d * 0.5)),
            (top, Vec2::new(w * 0.46, d * 0.46)),
        ],
        p.corner_exponent,
        Transform::from_xyz(0., 0., rear),
    );
    // The forward head joins the rear body but leaves a real open cup bay.
    let bottom = s.y * p.head_height;
    let front = s.z * 0.29;
    let back = rear;
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., (top + bottom) * 0.5, (front + back) * 0.5),
        Vec3::new(w * 0.93, top - bottom, front - back),
        s.x * 0.035,
    );
    let panel = Vec3::new(0., bottom + (top - bottom) * 0.56, front + 0.001);
    a.box_part(
        p.trim,
        l,
        panel,
        Vec3::new(w * 0.78, (top - bottom) * 0.57, 0.002),
        0.001,
    );
    for i in 0..p.controls {
        let x = w * 0.59 * ((i as f32 + 0.5) / p.controls as f32 - 0.5);
        a.part(Surface::Ink, l).cylinder(
            s.x * 0.028,
            0.004,
            Transform::from_xyz(x, panel.y, front + 0.004)
                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
    let tray_top = base + s.y * 0.055;
    a.box_part(
        p.trim,
        l,
        Vec3::new(0., (base + tray_top) * 0.5, s.z * 0.27),
        Vec3::new(s.x * p.tray_fraction, tray_top - base, s.z * 0.40),
        0.003,
    );
    for i in 0..5 {
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(
                (i as f32 - 2.) * s.x * p.tray_fraction * 0.16,
                tray_top + 0.0006,
                s.z * 0.27,
            ),
            Vec3::new(s.x * 0.018, 0.001, s.z * 0.29),
            0.,
        );
    }
    if p.brewer == Brewer::Espresso {
        let head = Vec3::new(0., bottom - s.y * 0.035, s.z * 0.17);
        a.part(p.trim, l)
            .cylinder(s.x * 0.11, s.y * 0.07, Transform::from_translation(head));
        a.part(Surface::Rubber, l).rod(
            head,
            head + Vec3::new(s.x * 0.31, 0., s.z * 0.21),
            s.x * 0.026,
        );
        let wand = [
            Vec3::new(-w * 0.35, bottom + s.y * 0.05, front * 0.65),
            Vec3::new(-w * 0.39, bottom - s.y * 0.13, front * 0.85),
            Vec3::new(-w * 0.30, bottom - s.y * 0.20, s.z * 0.33),
        ];
        for pair in wand.windows(2) {
            a.part(Surface::Chrome, l).rod(pair[0], pair[1], 0.003);
        }
    } else {
        a.box_part(
            p.trim,
            l,
            Vec3::new(0., bottom - s.y * 0.035, front * 0.80),
            Vec3::new(w * 0.25, s.y * 0.07, s.z * 0.12),
            0.003,
        );
        if p.brewer == Brewer::Automatic {
            a.part(Surface::Plastic, l).housing(
                &[
                    (top, Vec2::new(w * 0.29, d * 0.30)),
                    (s.y * 0.98, Vec2::new(w * 0.34, d * 0.36)),
                ],
                4.,
                Transform::from_xyz(0., 0., rear),
            );
        }
    }
}

fn filter(a: &mut Assembly, o: &IndoorObject, p: &CoffeeProgram, base: f32) {
    let s = o.size;
    let l = o.kind.class_name();
    let rear = -s.z * 0.30;
    a.part(p.finish, l).housing(
        &[
            (base, Vec2::new(s.x * 0.37, s.z * 0.19)),
            (s.y * 0.95, Vec2::new(s.x * 0.34, s.z * 0.18)),
        ],
        p.corner_exponent,
        Transform::from_xyz(0., 0., rear),
    );
    let r = (s.x * 0.30).min(s.z * 0.30);
    let z = s.z * 0.16;
    // A bridge from the tank actually supports the tapered filter basket.
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., s.y * 0.89, -s.z * 0.055),
        Vec3::new(s.x * 0.68, s.y * 0.12, s.z * 0.52),
        0.006,
    );
    a.part(p.finish, l).lathe(
        &[
            (0., 0.),
            (r * 0.72, 0.),
            (r * 1.08, s.y * 0.25),
            (0., s.y * 0.25),
        ],
        24,
        Transform::from_xyz(0., s.y * 0.60, z),
    );
    let bottom = base + 0.004;
    let h = s.y * 0.43;
    a.part(
        if p.thermal_carafe {
            p.trim
        } else {
            Surface::ContainerGlass
        },
        l,
    )
    .lathe(
        &[
            (0., 0.),
            (r * 0.88, 0.),
            (r, h * 0.25),
            (r * 0.89, h * 0.76),
            (r * 0.72, h),
            (r * 0.72 - 0.002, h),
            (r * 0.89 - 0.002, h * 0.76),
            (r - 0.002, h * 0.25),
            (0., 0.003),
        ],
        24,
        Transform::from_xyz(0., bottom, z),
    );
    a.part(Surface::Plastic, l).cylinder(
        r * 0.77,
        s.y * 0.028,
        Transform::from_xyz(0., bottom + h + s.y * 0.014, z),
    );
    let handle = [
        Vec3::new(r * 0.85, bottom + h * 0.79, z),
        Vec3::new(s.x * 0.45, bottom + h * 0.72, z),
        Vec3::new(s.x * 0.45, bottom + h * 0.22, z),
        Vec3::new(r * 0.94, bottom + h * 0.18, z),
    ];
    for edge in handle.windows(2) {
        a.part(Surface::Plastic, l)
            .rod(edge[0], edge[1], s.x * 0.025);
    }
}

#[cfg(test)]
mod tests;
