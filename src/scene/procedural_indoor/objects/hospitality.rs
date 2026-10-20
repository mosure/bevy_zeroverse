//! Refreshment equipment and hollow vessels; no opaque caps across vessel mouths.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct HospitalityProgram {
    pub style: u8,
    pub belly: f32,
    pub neck: f32,
    pub taper: f32,
    pub flutes: u32,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> HospitalityProgram {
    let mut rng = stream(o.seed, 5400);
    HospitalityProgram {
        style: rng.random_range(0..4),
        belly: rng.random_range(0.68..0.96),
        neck: rng.random_range(0.28..0.67),
        taper: rng.random_range(0.50..0.90),
        flutes: rng.random_range(12..=32),
        finish: if rng.random_bool(0.45) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::CoffeeMaker => coffee(a, o, &p),
        ObjectKind::Kettle => kettle(a, o, &p),
        ObjectKind::Vase | ObjectKind::Bowl => super::vessels::build(a, o),
        _ => unreachable!("non-hospitality object"),
    }
}

fn coffee(a: &mut Assembly, o: &IndoorObject, p: &HospitalityProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        Surface::Rubber,
        label,
        Vec3::Y * s.y * 0.04,
        Vec3::new(s.x, s.y * 0.08, s.z),
        0.006,
    );
    a.box_part(
        p.finish,
        label,
        Vec3::new(0., s.y * 0.51, -s.z * 0.19),
        Vec3::new(
            s.x * p.belly,
            s.y * (0.78 + p.neck * 0.24),
            s.z * (0.44 + p.taper * 0.16),
        ),
        0.013,
    );
    a.box_part(
        Surface::Metal,
        label,
        Vec3::new(0., s.y * (0.63 + p.neck * 0.14), s.z * 0.12),
        Vec3::new(
            s.x * p.belly * 0.91,
            s.y * (0.18 + p.neck * 0.28),
            s.z * 0.14,
        ),
        0.008,
    );
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(-s.x * 0.16, s.y * 0.78, s.z * 0.195),
        Vec3::new(s.x * 0.28, s.y * 0.08, 0.001),
        0.001,
    );
    for x in [0.05, 0.28] {
        a.part(Surface::Chrome, label).cylinder(
            s.x * 0.035,
            0.004,
            Transform::from_xyz(s.x * x, s.y * 0.77, s.z * 0.195)
                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
    a.box_part(
        Surface::Metal,
        label,
        Vec3::new(0., s.y * 0.12, s.z * 0.24),
        Vec3::new(s.x * 0.84, s.y * 0.055, s.z * 0.38),
        0.005,
    );
    for i in 0..7 {
        a.box_part(
            Surface::Ink,
            label,
            Vec3::new((i as f32 - 3.) * s.x * 0.1, s.y * 0.149, s.z * 0.24),
            Vec3::new(s.x * 0.025, 0.001, s.z * 0.30),
            0.0,
        );
    }
    if p.style < 2 {
        a.part(Surface::Chrome, label).cylinder(
            s.x * 0.10,
            s.y * 0.08,
            Transform::from_xyz(0., s.y * 0.48, s.z * 0.18),
        );
        a.part(Surface::Rubber, label).rod(
            Vec3::new(0., s.y * 0.51, s.z * 0.20),
            Vec3::new(s.x * 0.33, s.y * 0.51, s.z * 0.32),
            0.009,
        );
        a.part(Surface::Chrome, label).rod(
            Vec3::new(-s.x * 0.34, s.y * 0.52, s.z * 0.12),
            Vec3::new(-s.x * 0.30, s.y * 0.30, s.z * 0.23),
            0.004,
        );
    } else {
        let r = (s.x * 0.27).min(s.z * 0.20);
        a.part(Surface::ContainerGlass, label).lathe(
            &[
                (0., 0.),
                (r * 0.76, 0.),
                (r, s.y * 0.13),
                (r * 0.94, s.y * 0.28),
                (r * 0.70, s.y * 0.29),
                (r * 0.65, s.y * 0.28),
                (r * 0.89, s.y * 0.13),
                (r * 0.70, 0.012),
                (0., 0.012),
            ],
            24,
            Transform::from_xyz(0., s.y * 0.15, s.z * 0.24),
        );
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(s.x * 0.28, s.y * 0.30, s.z * 0.24),
            Vec3::new(s.x * 0.10, s.y * 0.22, s.z * 0.16),
            0.008,
        );
    }
}

fn kettle(a: &mut Assembly, o: &IndoorObject, p: &HospitalityProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let r = (s.x * 0.46).min(s.z * 0.31);
    let body = if p.style < 2 {
        Surface::Metal
    } else {
        Surface::Ceramic
    };
    let tf = Transform::from_xyz(0., 0., -s.z * 0.045);
    a.part(body, label).lathe(
        &[
            (0., 0.),
            (r * p.taper, 0.),
            (r, s.y * 0.33),
            (r * p.belly, s.y * 0.76),
            (r * 0.55, s.y * 0.83),
            (0., s.y * 0.83),
        ],
        32,
        tf,
    );
    a.part(Surface::Plastic, label).cylinder(
        r * 0.53,
        s.y * 0.035,
        Transform::from_xyz(0., s.y * 0.845, -s.z * 0.045),
    );
    a.part(Surface::Rubber, label).ellipsoid(
        Vec3::new(r * 0.16, s.y * 0.04, r * 0.16),
        Transform::from_xyz(0., s.y * 0.90, -s.z * 0.045),
    );
    let handle: Vec<_> = (0..=16)
        .map(|i| {
            let t = i as f32 * PI / 16.;
            Vec3::new(
                0.,
                s.y * (0.48 - t.cos() * 0.29),
                -s.z * (0.23 + t.sin() * 0.23),
            )
        })
        .collect();
    a.part(Surface::Rubber, label).tube(&handle, s.z * 0.03, 10);
    let points = [
        Vec3::new(0., s.y * 0.50, s.z * 0.15),
        Vec3::new(0., s.y * 0.64, s.z * 0.34),
        Vec3::new(0., s.y * 0.78, s.z * 0.43),
    ];
    // The tube's dark end is visibly recessed into a small flared nozzle.
    a.part(body, label).tube(&points, s.x.min(s.z) * 0.055, 12);
    a.part(Surface::Ink, label).ellipsoid(
        Vec3::new(s.x.min(s.z) * 0.035, s.y * 0.013, s.z * 0.01),
        Transform::from_xyz(0., s.y * 0.795, s.z * 0.431),
    );
}
