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
    pub shoulder_fraction: f32,
    pub neck_fraction: f32,
    pub heel_fraction: f32,
    pub belly_height: f32,
    pub belly_fraction: f32,
    pub handle_bottom: f32,
    pub handle_top: f32,
    pub handle_shape: f32,
    pub cup_radius_fraction: f32,
    pub cup_sides: u32,
    pub empty: bool,
    pub waist: f32,
    pub waist_height: f32,
    pub ribs: usize,
    pub rib_depth: f32,
}
pub fn parameters(o: &IndoorObject) -> BeverageProgram {
    let mut rng = stream(o.seed, 831);
    let mut profile = stream(o.seed, 7455);
    BeverageProgram {
        belly_height: profile.random_range(0.30..0.68),
        belly_fraction: profile.random_range(0.76..1.20),
        handle_bottom: profile.random_range(0.12..0.30),
        handle_top: profile.random_range(0.72..0.90),
        handle_shape: profile.random_range(0.45..1.80),
        cup_radius_fraction: profile.random_range(0.29..0.39),
        cup_sides: if profile.random_bool(0.20) {
            profile.random_range(8..=12)
        } else {
            32
        },
        empty: profile.random_bool(0.32),
        waist: profile.random_range(0.76..1.0),
        waist_height: profile.random_range(0.32..0.70),
        ribs: profile.random_range(3..=10),
        rib_depth: profile.random_range(0.008..0.055),
        shoulder_fraction: profile.random_range(0.54..0.80),
        neck_fraction: profile.random_range(0.24..0.80),
        heel_fraction: profile.random_range(0.035..0.10),
        style: rng.random_range(0..6),
        taper: if o.kind == ObjectKind::CoffeeCup {
            rng.random_range(1.10..1.28)
        } else {
            rng.random_range(0.70..1.22)
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
    let mut r = o.size.z.min(o.size.x)
        * if takeaway {
            0.365
        } else {
            p.cup_radius_fraction
        };
    let offset = if takeaway { 0. } else { -o.size.x * 0.1 };
    // Profile shape and handle clearances are independent, but the complete
    // vessel still fits the same envelope used by surface placement and labels.
    r = r
        .min((o.size.x * 0.5 + offset).min(o.size.z * 0.5) / p.taper.max(p.belly_fraction).max(1.));
    let sides = if takeaway { 32 } else { p.cup_sides };
    if !takeaway && p.style >= 3 {
        // The saucer shares the cup's off-center axis. Reserve its real rim,
        // rather than letting it overhang a narrow mug footprint by several mm.
        r = r.min((o.size.x * 0.5 + offset).min(o.size.z * 0.5) / 1.23);
    }
    let top = r * p.taper;
    let belly = r * p.belly_fraction;
    let finish = if takeaway {
        Surface::Paper
    } else {
        Surface::Ceramic
    };
    let radius_at = |t: f32| {
        let (lo, hi, t) = if t < p.belly_height {
            (r, belly, t / p.belly_height)
        } else {
            (belly, top, (t - p.belly_height) / (1. - p.belly_height))
        };
        let t = t * t * (3. - 2. * t);
        lo + (hi - lo) * t
    };
    let mut profile = vec![(0., 0.), (r * 0.8, 0.)];
    for i in 0..=16 {
        let t = i as f32 / 16.;
        profile.push((radius_at(t), h * (0.05 + t * 0.95)));
    }
    for i in (1..=16).rev() {
        let t = i as f32 / 16.;
        profile.push((radius_at(t) - p.wall_m, h * (0.05 + t * 0.95)));
    }
    profile.push((0., h * 0.10));
    a.part(finish, "other_prop")
        .lathe(&profile, sides, Transform::from_xyz(offset, 0., 0.));
    if !p.empty {
        let fill_radius = (radius_at((p.fill_fraction - 0.05) / 0.95) - p.wall_m * 1.1)
            * (PI / sides as f32).cos();
        a.part(Surface::Drink, "other_prop").cylinder(
            fill_radius,
            0.001,
            Transform::from_xyz(offset, h * p.fill_fraction, 0.),
        );
    }
    if takeaway {
        // Kraft or printed sleeves fit the taper; double-wall cups omit one.
        if p.style % 3 != 2 {
            let sleeve = if p.style.is_multiple_of(3) {
                Surface::Terracotta
            } else {
                Surface::Art
            };
            a.part(sleeve, "other_prop").lathe(
                &(0..=10)
                    .map(|i| {
                        let t = 0.24 + i as f32 * 0.041;
                        (radius_at((t - 0.05) / 0.95) + 0.001, h * t)
                    })
                    .collect::<Vec<_>>(),
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
        let tube_r = (h * 0.04).min(0.0045).min(r * 0.10);
        let lower_x = radius_at((p.handle_bottom - 0.05) / 0.95) + offset - tube_r * 0.5;
        let upper_x = radius_at((p.handle_top - 0.05) / 0.95) + offset - tube_r * 0.5;
        let reach = (o.size.x * 0.5 - lower_x.max(upper_x) - tube_r - 0.001).max(0.003);
        let points: Vec<_> = (0..=24)
            .map(|i| {
                let t = i as f32 / 24. * PI;
                let u = (1. - t.cos()) * 0.5;
                Vec3::new(
                    upper_x
                        + (lower_x - upper_x) * u
                        + reach * t.sin().max(0.).powf(p.handle_shape),
                    h * (p.handle_top + (p.handle_bottom - p.handle_top) * u),
                    0.,
                )
            })
            .collect();
        a.part(finish, "other_prop").tube(&points, tube_r, 10);
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
    let shoulder = h * p.shoulder_fraction;
    let neck = r * p.neck_fraction;
    let body_radius = |t: f32| {
        let waist = 1. - (1. - p.waist) * (-(t - p.waist_height).powi(2) / 0.035).exp();
        let rib = if p.style == 0 {
            1. - p.rib_depth * (t * p.ribs as f32 * TAU).sin().powi(2)
        } else {
            1.
        };
        r * waist * rib
    };
    let mut profile = vec![(0., 0.), (r * 0.83, 0.)];
    let mut wall = Vec::with_capacity(35);
    for i in 0..=32 {
        let t = i as f32 / 32.;
        let y = h * 0.04 + (shoulder - h * 0.04) * t;
        let radius = body_radius(t);
        profile.push((radius, y));
        wall.push((radius - 0.0015, y));
    }
    profile.extend([
        (r * 0.87, shoulder + h * 0.025),
        (neck, h * 0.88),
        (neck, h * 0.94),
    ]);
    if clear {
        profile.extend([
            (neck - 0.0015, h * 0.94),
            (neck - 0.0015, h * 0.88),
            (r * 0.87 - 0.0015, shoulder + h * 0.025),
        ]);
        profile.extend(wall.into_iter().rev());
        profile.push((0., h * 0.025));
        if !p.empty {
            let fill = h * p.fill_fraction.min(p.shoulder_fraction - 0.03);
            let mut liquid = vec![(0., h * 0.026)];
            for i in 0..=16 {
                let y = h * 0.04 + (fill - h * 0.04) * i as f32 / 16.;
                liquid.push((
                    body_radius((y - h * 0.04) / (shoulder - h * 0.04)) - 0.003,
                    y,
                ));
            }
            liquid.push((0., fill));
            a.part(Surface::Liquid, "other_prop")
                .lathe(&liquid, 32, Transform::IDENTITY);
        }
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
            &(0..=12)
                .map(|i| {
                    let t = 0.28 + i as f32 * 0.015;
                    (
                        body_radius((h * t - h * 0.04) / (shoulder - h * 0.04)) + 0.0007,
                        h * t,
                    )
                })
                .collect::<Vec<_>>(),
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
            (r, h * p.heel_fraction),
            (r, h * (0.81 + p.heel_fraction)),
            (r * (0.74 + p.neck_fraction * 0.24), h * 0.96),
            (0., h * 0.96),
        ],
        32,
        Transform::IDENTITY,
    );
    for (y, radius) in [
        (h * 0.022, r * 0.86),
        (h * 0.976, r * (0.77 + p.neck_fraction * 0.20)),
    ] {
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
        r * (0.745 + p.neck_fraction * 0.20),
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
