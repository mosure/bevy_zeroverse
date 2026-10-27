//! Thermal urns, gravity beverage tanks and enclosed fountain dispensers.
use super::*;
#[derive(Clone, Debug, serde::Serialize)]
pub struct DispenserProgram {
    pub construction: u8,
    pub tanks: usize,
    pub body_bottom: f32,
    pub shoulder: f32,
    pub tank_radius: f32,
    pub neck: f32,
    pub fill: f32,
    pub tap_height: f32,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> DispenserProgram {
    let mut r = stream(o.seed, 0x44495350454e5345);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4449535046494e);
    let construction = r.random_range(0..4);
    DispenserProgram {
        construction,
        tanks: if construction == 1 {
            r.random_range(2..=3)
        } else {
            1
        },
        body_bottom: r.random_range(0.22..0.35),
        shoulder: r.random_range(0.74..0.87),
        tank_radius: r.random_range(0.31..0.43),
        neck: r.random_range(0.58..0.96),
        fill: r.random_range(0.24..0.84),
        tap_height: r.random_range(0.20..0.30),
        finish: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Plastic, // More common coated/polymer housings; no wall treatment.
        ][f.random_range(0..4)],
    }
}
pub(crate) fn power_port(o: &IndoorObject) -> Option<Vec3> {
    (parameters(o).construction >= 2).then_some(Vec3::new(0., o.size.y * 0.035, -o.size.z * 0.485))
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let base = s.y * 0.065;
    a.box_part(
        p.finish,
        l,
        Vec3::Y * base * 0.5,
        Vec3::new(s.x * 0.96, base, s.z * 0.96),
        0.005,
    );
    if p.construction >= 2 {
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(0., s.y * 0.035, -s.z * 0.482),
            Vec3::new(s.x * 0.10, s.y * 0.025, s.z * 0.004),
            0.001,
        );
    }
    let count = p.tanks.min((s.x / 0.14).floor().max(1.) as usize);
    if p.construction == 3 {
        // Enclosed fountain with a recessed cup bay and a real projecting head.
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., s.y * 0.49, -s.z * 0.23),
            Vec3::new(s.x * 0.91, s.y * 0.85, s.z * 0.46),
            0.009,
        );
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., s.y * 0.75, s.z * 0.08),
            Vec3::new(s.x * 0.94, s.y * 0.42, s.z * 0.68),
            0.010,
        );
        for i in 0..3 {
            let x = (i as f32 - 1.) * s.x * 0.25;
            a.box_part(
                Surface::Art,
                l,
                Vec3::new(x, s.y * 0.76, s.z * 0.424),
                Vec3::new(s.x * 0.19, s.y * 0.19, 0.002),
                0.001,
            );
            a.part(Surface::Rubber, "faucet").rod(
                Vec3::new(x, s.y * 0.55, s.z * 0.27),
                Vec3::new(x, s.y * 0.47, s.z * 0.29),
                s.x.min(s.z) * 0.027,
            );
        }
    } else {
        for i in 0..count {
            let x = s.x * ((i as f32 + 0.5) / count as f32 - 0.5);
            let radius = (s.x / count as f32).min(s.z) * p.tank_radius;
            let z = -s.z * 0.075;
            let bottom = s.y * p.body_bottom;
            let top = s.y * 0.91;
            let metal = p.construction == 2;
            a.part(p.finish, l).cylinder(
                radius * 0.72,
                bottom - base,
                Transform::from_xyz(x, (bottom + base) * 0.5, z),
            );
            let tf = Transform::from_xyz(x, bottom, z);
            let h = top - bottom;
            let profile = [
                (0., 0.),
                (radius * 0.88, 0.),
                (radius, h * 0.08),
                (radius, h * p.shoulder),
                (radius * p.neck, h),
                (radius * p.neck - 0.0015, h),
                (radius - 0.0015, h * p.shoulder),
                (radius - 0.0015, h * 0.08),
                (0., 0.004),
            ];
            a.part(
                if metal {
                    p.finish
                } else {
                    Surface::ContainerGlass
                },
                l,
            )
            .lathe(&profile, 24, tf);
            if !metal {
                a.part(Surface::Liquid, l).cylinder(
                    radius - 0.003,
                    h * p.fill.min(p.shoulder - 0.03) - 0.006,
                    Transform::from_xyz(
                        x,
                        bottom + 0.004 + (h * p.fill.min(p.shoulder - 0.03) - 0.006) * 0.5,
                        z,
                    ),
                );
            }
            a.part(p.finish, l).cylinder(
                radius * p.neck * 1.04,
                s.y * 0.025,
                Transform::from_xyz(x, top + s.y * 0.0125, z),
            );
            a.part(Surface::Rubber, l).ellipsoid(
                Vec3::new(radius * 0.18, s.y * 0.018, radius * 0.18),
                Transform::from_xyz(x, top + s.y * 0.043, z),
            );
            let tap_y = bottom + h * 0.06;
            let tip = Vec3::new(x, tap_y, s.z * 0.43);
            a.part(p.finish, "faucet").rod(
                Vec3::new(x, tap_y, z + radius * 0.90),
                tip,
                radius * 0.10,
            );
            a.part(p.finish, "faucet")
                .rod(tip, tip - Vec3::Y * s.y * 0.06, radius * 0.11);
            a.part(Surface::Rubber, "faucet").rod(
                tip - Vec3::Z * s.z * 0.05,
                tip - Vec3::Z * s.z * 0.05 + Vec3::Y * s.y * p.tap_height * 0.25,
                radius * 0.06,
            );
        }
    }
    a.box_part(
        Surface::Rubber,
        l,
        Vec3::new(0., base + 0.003, s.z * 0.32),
        Vec3::new(s.x * 0.83, 0.006, s.z * 0.26),
        0.003,
    );
    for i in 0..7 {
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new((i as f32 - 3.) * s.x * 0.10, base + 0.007, s.z * 0.32),
            Vec3::new(s.x * 0.035, 0.002, s.z * 0.23),
            0.0005,
        );
    }
}
