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
    pub tap_count: usize,
    pub base_fraction: f32,
    pub top_fraction: f32,
    pub body_width: f32,
    pub body_depth: f32,
    pub front: f32,
    pub bay_top: f32,
    pub tap_span: f32,
    pub label_height: f32,
    pub label_center: f32,
    pub tray_width: f32,
    pub tray_depth: f32,
    pub grille_count: usize,
    pub grille_crosswise: bool,
    pub trim: Surface,
}
pub fn parameters(o: &IndoorObject) -> DispenserProgram {
    let mut r = stream(o.seed, 0x44495350454e5345);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4449535046494e);
    let construction = r.random_range(0..4);
    DispenserProgram {
        construction,
        tanks: if construction == 1 {
            r.random_range(2..=3)
                .min((o.size.x / 0.14).floor().max(1.) as usize)
        } else {
            1
        },
        tap_count: r
            .random_range(1..=4)
            .min((o.size.x / 0.09).floor().max(1.) as usize),
        base_fraction: r.random_range(0.055..0.095),
        top_fraction: r.random_range(0.80..0.92),
        body_width: r.random_range(0.72..0.96),
        body_depth: r.random_range(0.36..0.54),
        front: r.random_range(0.24..0.42),
        bay_top: r.random_range(0.48..0.64),
        tap_span: r.random_range(0.63..0.85),
        label_height: r.random_range(0.30..0.68),
        label_center: r.random_range(0.43..0.61),
        tray_width: r.random_range(0.70..0.94),
        tray_depth: r.random_range(0.20..0.30),
        grille_count: r.random_range(4..=11),
        grille_crosswise: r.random_bool(0.5),
        trim: [Surface::Metal, Surface::Chrome, Surface::Rubber][f.random_range(0..3)],
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
    let base = s.y * p.base_fraction;
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
    let count = p.tanks;
    if p.construction == 3 {
        // A connected rear chassis and head leave an actual open cup bay.
        let width = s.x * p.body_width;
        let depth = s.z * p.body_depth;
        let rear = -s.z * 0.47 + depth * 0.5;
        let top = s.y * (p.top_fraction + 0.04);
        let bottom = s.y * p.bay_top;
        let front = s.z * p.front;
        a.part(p.finish, l).housing(
            &[
                (base, Vec2::new(width * 0.48, depth * 0.5)),
                (top * 0.92, Vec2::new(width * 0.5, depth * 0.5)),
                (top, Vec2::new(width * 0.47, depth * 0.46)),
            ],
            3.0 + p.neck * 6.,
            Transform::from_xyz(0., 0., rear),
        );
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., (top + bottom) * 0.5, (front + rear) * 0.5),
            Vec3::new(width, top - bottom, front - rear),
            0.006,
        );
        for i in 0..p.tap_count {
            let x = width * p.tap_span * ((i as f32 + 0.5) / p.tap_count as f32 - 0.5);
            let label_y = bottom + (top - bottom) * p.label_center;
            a.box_part(
                Surface::Art,
                l,
                Vec3::new(x, label_y, front + 0.0015),
                Vec3::new(
                    width * p.tap_span / p.tap_count as f32 * 0.72,
                    (top - bottom) * p.label_height,
                    0.002,
                ),
                0.0005,
            );
            let radius = (width / p.tap_count as f32 * 0.07).min(s.z * 0.03);
            let tap = Vec3::new(x, bottom + radius * 0.5, front * 0.78);
            a.part(p.trim, "faucet")
                .rod(tap, tap - Vec3::Y * s.y * p.tap_height * 0.3, radius);
            a.box_part(
                Surface::Rubber,
                "faucet",
                // Embed the actuator into the nozzle; tangent rounded faces
                // would leave only a point contact and can appear detached.
                tap + Vec3::new(0., -s.y * p.tap_height * 0.3, radius * 0.8),
                Vec3::new(radius * 2.4, s.y * 0.045, radius),
                radius * 0.2,
            );
        }
    } else {
        for i in 0..count {
            let x = s.x * ((i as f32 + 0.5) / count as f32 - 0.5);
            let radius = (s.x / count as f32).min(s.z) * p.tank_radius;
            let z = -s.z * 0.075;
            let bottom = s.y * p.body_bottom;
            let top = s.y * p.top_fraction;
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
            let tap_y = bottom + h * (p.tap_height * 0.25);
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
    let tray = Vec2::new(s.x * p.tray_width, s.z * p.tray_depth);
    a.box_part(
        Surface::Rubber,
        l,
        Vec3::new(0., base + 0.003, s.z * 0.32),
        Vec3::new(tray.x, 0.006, tray.y),
        0.002,
    );
    for i in 0..p.grille_count {
        let t = ((i as f32 + 0.5) / p.grille_count as f32 - 0.5) * 0.88;
        let (offset, size) = if p.grille_crosswise {
            (
                Vec3::Z * tray.y * t,
                Vec3::new(tray.x * 0.90, 0.002, tray.y / p.grille_count as f32 * 0.34),
            )
        } else {
            (
                Vec3::X * tray.x * t,
                Vec3::new(tray.x / p.grille_count as f32 * 0.34, 0.002, tray.y * 0.90),
            )
        };
        a.box_part(
            p.trim,
            l,
            Vec3::new(0., base + 0.007, s.z * 0.32) + offset,
            size,
            0.0003,
        );
    }
}

#[cfg(test)]
mod tests;
