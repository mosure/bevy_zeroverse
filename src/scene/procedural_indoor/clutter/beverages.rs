//! Hollow vessels with separate rims, closures, liquids and printed sleeves.
use super::*;
use std::f32::consts::{PI, TAU};

#[derive(Debug, serde::Serialize)]
pub struct BeverageProgram {
    pub style: u32,
    pub taper: f32,
    pub fill_fraction: f32,
    pub wall_m: f32,
    pub lid: bool,
}
pub fn parameters(o: &IndoorObject) -> BeverageProgram {
    let mut rng = stream(o.seed, 831);
    BeverageProgram {
        style: rng.random_range(0..6),
        taper: if o.kind == ObjectKind::CoffeeCup {
            rng.random_range(1.10..1.28)
        } else {
            rng.random_range(0.72..1.08)
        },
        fill_fraction: rng.random_range(0.32..0.85),
        wall_m: rng.random_range(0.0014..0.0034),
        lid: rng.random_bool(0.72),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::Mug | ObjectKind::CoffeeCup => cup(a, o, &p),
        ObjectKind::WaterBottle => bottle(a, o, &p),
        ObjectKind::SodaCan => can(a, o, &p),
        _ => unreachable!(),
    }
}
fn cup(a: &mut Assembly, o: &IndoorObject, p: &BeverageProgram) {
    let takeaway = o.kind == ObjectKind::CoffeeCup;
    let h = o.size.y * if takeaway { 0.9 } else { 1. };
    let mut r = o.size.z.min(o.size.x) * if takeaway { 0.365 } else { 0.37 };
    let offset = if takeaway { 0. } else { -o.size.x * 0.1 };
    if !takeaway && p.style >= 3 {
        // The saucer shares the cup's off-center axis. Reserve its real rim,
        // rather than letting it overhang a narrow mug footprint by several mm.
        r = r.min((o.size.x * 0.5 + offset).min(o.size.z * 0.5) / 1.23);
    }
    let top = r * p.taper;
    let inner = top - p.wall_m;
    let belly = r * if !takeaway && p.style.is_multiple_of(3) {
        1.08
    } else {
        (1. + p.taper) * 0.5
    };
    let finish = if takeaway {
        Surface::Paper
    } else {
        Surface::Ceramic
    };
    a.part(finish, "other_prop").lathe(
        &[
            (0., 0.),
            (r * 0.8, 0.),
            (r, h * 0.06),
            (belly, h * 0.52),
            (top, h * 0.97),
            (top * 0.997, h),
            (inner, h),
            (belly - p.wall_m, h * 0.52),
            (r - p.wall_m, h * 0.09),
            (0., h * 0.09),
        ],
        32,
        Transform::from_xyz(offset, 0., 0.),
    );
    let fill_radius = if p.fill_fraction < 0.52 {
        r + (belly - r) * ((p.fill_fraction - 0.09) / 0.43)
    } else {
        belly + (top - belly) * ((p.fill_fraction - 0.52) / 0.45)
    };
    a.part(Surface::Drink, "other_prop").cylinder(
        fill_radius - p.wall_m * 1.1,
        0.001,
        Transform::from_xyz(offset, h * p.fill_fraction, 0.),
    );
    if takeaway {
        // Kraft or printed sleeves fit the taper; double-wall cups omit one.
        if p.style % 3 != 2 {
            let sleeve = if p.style.is_multiple_of(3) {
                Surface::Terracotta
            } else {
                Surface::Art
            };
            a.part(sleeve, "other_prop").lathe(
                &[
                    (r * (1. + (p.taper - 1.) * 0.24) + 0.001, h * 0.24),
                    (r * (1. + (p.taper - 1.) * 0.65) + 0.001, h * 0.65),
                ],
                32,
                Transform::IDENTITY,
            );
        }
        if p.lid {
            a.part(Surface::Plastic, "other_prop").lathe(
                &[
                    (0., h),
                    (top * 1.04, h),
                    (top * 1.045, h * 1.027),
                    (top * 0.90, h * 1.04),
                    (top * 0.85, h * 1.08),
                    (0., h * 1.08),
                ],
                32,
                Transform::IDENTITY,
            );
            a.box_part(
                Surface::Ink,
                "other_prop",
                Vec3::new(0., h * 1.08 + 0.0004, top * 0.55),
                Vec3::new(top * 0.45, 0.0006, top * 0.13),
                0.0002,
            );
        }
    } else {
        let start = r * p.taper * 0.87 + offset;
        let reach = (o.size.x * 0.5 - start - 0.005).max(0.008);
        let points: Vec<_> = (0..=24)
            .map(|i| {
                let t = i as f32 / 24. * PI;
                Vec3::new(start + reach * t.sin(), h * (0.50 + 0.32 * t.cos()), 0.)
            })
            .collect();
        a.part(finish, "other_prop")
            .tube(&points, (h * 0.046).min(0.005), 10);
        if p.style >= 3 {
            a.part(Surface::Ceramic, "other_prop").lathe(
                &[
                    (0., 0.001),
                    (r * 1.19, 0.001),
                    (r * 1.23, 0.006),
                    (r * 1.16, 0.008),
                    (0., 0.005),
                ],
                32,
                Transform::from_xyz(offset, 0., 0.),
            );
        }
    }
}
fn bottle(a: &mut Assembly, o: &IndoorObject, p: &BeverageProgram) {
    let h = o.size.y;
    let r = o.size.x.min(o.size.z) * 0.45;
    let clear = p.style <= 2;
    let surface = if clear {
        Surface::ContainerGlass
    } else if p.style == 3 {
        Surface::Chrome
    } else {
        Surface::Metal
    };
    let shoulder = h * if p.style == 1 { 0.56 } else { 0.76 };
    let neck = r * if p.style == 2 { 0.67 } else { 0.43 };
    let mut profile = vec![(0., 0.), (r * 0.88, 0.), (r, h * 0.04)];
    for i in 1..=16 {
        let y = h * 0.04 + (shoulder - h * 0.04) * i as f32 / 16.;
        let rib = if p.style == 0 && i % 2 == 0 { 0.96 } else { 1. };
        profile.push((r * rib, y));
    }
    profile.extend([
        (r * 0.87, shoulder + h * 0.05),
        (neck, h * 0.88),
        (neck, h * 0.94),
    ]);
    if clear {
        profile.extend([
            (neck - 0.001, h * 0.94),
            (neck - 0.001, h * 0.88),
            (r * 0.94, shoulder),
            (r * 0.94, h * 0.025),
            (0., h * 0.025),
        ]);
        a.part(Surface::Liquid, "other_prop").cylinder(
            r * 0.925,
            h * p.fill_fraction.min(0.53),
            Transform::from_xyz(0., h * (0.025 + p.fill_fraction.min(0.53) * 0.5), 0.),
        );
    } else {
        profile.push((0., h * 0.94));
    }
    a.part(surface, "other_prop")
        .lathe(&profile, 32, Transform::IDENTITY);
    a.part(Surface::Plastic, "other_prop").cylinder(
        neck * 1.12,
        h * 0.055,
        Transform::from_xyz(0., h * 0.9525, 0.),
    );
    for i in 0..16 {
        let angle = i as f32 * TAU / 16.;
        a.part(Surface::Plastic, "other_prop").rod(
            Vec3::new(
                angle.cos() * neck * 1.12,
                h * 0.936,
                angle.sin() * neck * 1.12,
            ),
            Vec3::new(
                angle.cos() * neck * 1.12,
                h * 0.969,
                angle.sin() * neck * 1.12,
            ),
            0.0007,
        );
    }
    if p.style < 2 {
        a.part(Surface::Art, "other_prop").lathe(
            &[(r + 0.0007, h * 0.28), (r + 0.0007, h * 0.46)],
            32,
            Transform::IDENTITY,
        );
    } else if p.style == 5 {
        let points: Vec<_> = (0..=24)
            .map(|i| {
                let t = i as f32 * PI / 24.;
                Vec3::new(t.cos() * neck * 0.65, h * 0.97 + t.sin() * h * 0.025, 0.)
            })
            .collect();
        a.part(Surface::Rubber, "other_prop")
            .tube(&points, h * 0.003, 8);
    }
}
fn can(a: &mut Assembly, o: &IndoorObject, p: &BeverageProgram) {
    let h = o.size.y;
    let r = o.size.x.min(o.size.z) * 0.47;
    a.part(Surface::Art, "other_prop").lathe(
        &[
            (0., h * 0.02),
            (r * 0.86, 0.),
            (r, h * 0.055),
            (r, h * 0.88),
            (r * 0.84, h * 0.96),
            (0., h * 0.96),
        ],
        32,
        Transform::IDENTITY,
    );
    for (y, radius) in [(h * 0.022, r * 0.86), (h * 0.976, r * 0.9)] {
        a.part(Surface::Chrome, "other_prop").lathe(
            &[
                (radius * 0.94, y - 0.001),
                (radius, y),
                (radius, y + 0.0012),
                (radius * 0.94, y + 0.0014),
            ],
            32,
            Transform::IDENTITY,
        );
    }
    a.part(Surface::Chrome, "other_prop").cylinder(
        r * 0.87,
        0.0015,
        Transform::from_xyz(0., h * 0.974, 0.),
    );
    let tab: Vec<_> = (0..=24)
        .map(|i| {
            let t = TAU * i as f32 / 24.;
            Vec3::new(t.cos() * r * 0.23, h * 0.987, r * 0.14 + t.sin() * r * 0.34)
        })
        .collect();
    a.part(Surface::Chrome, "other_prop")
        .tube(&tab, r * 0.035, 8);
    a.part(Surface::Chrome, "other_prop").cylinder(
        r * 0.085,
        0.001,
        Transform::from_xyz(0., h * 0.987, -r * 0.20),
    );
    if p.lid {
        a.part(Surface::Ink, "other_prop").ellipsoid(
            Vec3::new(r * 0.29, 0.0003, r * 0.33),
            Transform::from_xyz(0., h * 0.984, -r * 0.37),
        );
    }
    // Abstract printed label strokes wrap around the can, without a logo asset.
    let mut rng = stream(o.seed, 835);
    let count = 1 + p.style;
    let start = rng.random_range(0.2..0.38);
    let step = 0.42 / count as f32;
    for i in 0..count {
        let y = start + i as f32 * step;
        let width = rng.random_range(0.012..step * 0.7);
        a.part(Surface::Paper, "other_prop").lathe(
            &[(r + 0.0004, h * y), (r + 0.0004, h * (y + width))],
            32,
            Transform::IDENTITY,
        );
    }
}
