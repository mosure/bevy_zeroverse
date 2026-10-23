//! Refreshment equipment and hollow vessels; no opaque caps across vessel mouths.
use super::*;
pub mod coffee;

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
    match o.kind {
        ObjectKind::CoffeeMaker => coffee::build(a, o),
        ObjectKind::Kettle => kettle(a, o, &parameters(o)),
        ObjectKind::Vase | ObjectKind::Bowl => super::vessels::build(a, o),
        _ => unreachable!("non-hospitality object"),
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
