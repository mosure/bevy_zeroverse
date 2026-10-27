//! Flat-based refillable dispensers and squeeze bottles; pumps meet the neck.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DispenserProgram {
    pub body_height: f32,
    pub shoulder: f32,
    pub waist: f32,
    pub roundness: f32,
    pub pump_yaw: f32,
    pub nozzle_reach: f32,
    pub plunger_rise: f32,
    pub finish: Surface,
    pub pump_finish: Surface,
}
pub fn parameters(seed: u64, pump: bool) -> DispenserProgram {
    let mut r = stream(seed, 0x534f4150424f4459);
    DispenserProgram {
        body_height: r.random_range(0.65..0.84),
        shoulder: r.random_range(0.66..0.90),
        waist: r.random_range(0.73..0.99),
        roundness: r.random_range(2.0..7.5),
        pump_yaw: r.random_range(-PI..PI),
        nozzle_reach: r.random_range(0.28..0.43),
        plunger_rise: r.random_range(0.025..0.095),
        finish: if pump {
            [Surface::Plastic, Surface::Ceramic, Surface::Metal][r.random_range(0..3)]
        } else {
            Surface::Plastic
        },
        pump_finish: if r.random_bool(0.52) {
            Surface::Chrome
        } else {
            Surface::Plastic
        },
    }
}
pub(in super::super) fn build(a: &mut Assembly, base: Vec3, s: Vec3, pump: bool, seed: u64) {
    let p = parameters(seed, pump);
    let h = s.y * p.body_height;
    let half = s.xz() * 0.46;
    let label = "soap_dispenser";
    a.part(p.finish, label).housing(
        &[
            (0., half * 0.82),
            (h * 0.10, half),
            (h * p.shoulder, half * p.waist),
            (h, half * 0.38),
        ],
        p.roundness,
        Transform::from_translation(base),
    );
    let neck = s.x.min(s.z) * 0.12;
    let collar_h = (s.y * 0.07).min(0.012);
    a.part(p.pump_finish, label).cylinder(
        neck,
        collar_h,
        Transform::from_translation(base + Vec3::Y * (h + collar_h * 0.5)),
    );
    if pump {
        let radius = (neck * 0.30).min(s.y * 0.03);
        let y = (h + collar_h + s.y * p.plunger_rise).min(s.y - radius * 1.4);
        let reach = s.x.min(s.z) * p.nozzle_reach;
        let direction = Quat::from_rotation_y(p.pump_yaw) * Vec3::Z;
        // A pump is a plunger, pressed head and outlet, not a tight swept elbow.
        // Separate joined parts avoid an inverted tube at a short 90-degree bend.
        a.part(p.pump_finish, label).rod(
            base + Vec3::Y * (h + collar_h * 0.7),
            base + Vec3::Y * y,
            radius * 0.7,
        );
        a.part(p.pump_finish, label).cuboid(
            Vec3::new(radius * 3., radius * 2.1, reach + radius * 2.),
            radius * 0.35,
            Transform::from_translation(base + Vec3::Y * y + direction * reach * 0.45)
                .with_rotation(Quat::from_rotation_y(p.pump_yaw)),
        );
        a.part(p.pump_finish, label).rod(
            base + Vec3::Y * y + direction * reach * 0.85,
            base + Vec3::Y * (y - radius * 2.5) + direction * reach * 0.85,
            radius * 0.65,
        );
    } else {
        let height = s.y * 0.075;
        a.part(Surface::Plastic, label).housing(
            &[
                (h + collar_h, Vec2::splat(neck)),
                (h + collar_h + height, Vec2::splat(neck * 0.7)),
            ],
            p.roundness,
            Transform::from_translation(base),
        );
    }
}
