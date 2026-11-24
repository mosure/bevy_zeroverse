//! Blow-moulded or reusable bottles. Cap knurling is one continuous shell.
use super::*;
use crate::scene::procedural_indoor::materials::package_print;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Closure {
    pub flutes: u32,
    pub grip_depth: f32,
    pub cap_height_fraction: f32,
    pub tamper_band: bool,
    pub label_band: Option<Vec2>,
}
impl Closure {
    pub fn sample(o: &IndoorObject, p: &BeverageProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CapStock"));
        let commercial = p.style < 3;
        let flutes = r.random_range(if commercial { 12..=32 } else { 6..=18 });
        Self {
            flutes,
            grip_depth: r.random_range(if commercial {
                0.020..0.052
            } else {
                0.028..0.085
            }),
            cap_height_fraction: r.random_range(if commercial {
                0.038..0.067
            } else {
                0.055..0.092
            }),
            tamper_band: commercial,
            label_band: (p.style < 2).then(|| {
                let lo = r.random_range(0.16..0.31);
                Vec2::new(
                    lo,
                    r.random_range((lo + 0.13)..(p.shoulder_fraction - 0.035)),
                )
            }),
        }
    }
}

fn radius_at(profile: &[(f32, f32)], y: f32) -> f32 {
    let i = profile
        .partition_point(|&(_, h)| h < y)
        .clamp(1, profile.len() - 1);
    let (a, h) = profile[i - 1];
    let (b, k) = profile[i];
    a + (b - a) * ((y - h) / (k - h)).clamp(0., 1.)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &BeverageProgram) {
    let c = Closure::sample(o, p);
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
    let label = o.kind.class_name();
    let shoulder = h * p.shoulder_fraction;
    let neck = r * p.neck_fraction;
    let base = h * p.heel_fraction;
    let thickness = (r * 0.035).min(0.0014);
    let throat = 0.872 * h;
    let cap_top = if p.style == 5 { 0.938 * h } else { 0.993 * h };
    let cap_height = h * c.cap_height_fraction;
    let cap_low = cap_top - cap_height;
    let band = if c.tamper_band {
        (cap_height * 0.18).min(0.0035)
    } else {
        0.
    };
    // The moulded neck always enters the closure, including tall caps on flasks.
    let lip = cap_low + cap_height * 0.36;
    let mut outer = vec![(r * 0.825, 0.)];
    let rows = if p.style == 0 { 4 * p.ribs } else { 10 };
    for i in 0..=rows {
        let u = i as f32 / rows as f32;
        let waist = 1. - (1. - p.waist) * (-(u - p.waist_height).powi(2) / 0.035).exp();
        let rib = if p.style == 0 {
            1. - p.rib_depth * (0.5 - 0.5 * (TAU * p.ribs as f32 * u).cos())
        } else {
            1.
        };
        outer.push((r * waist * rib, base + (shoulder - base) * u));
    }
    outer.extend([
        (r * 0.90, shoulder + h * 0.023),
        (neck * 1.05, throat.min(lip - h * 0.024)),
        (neck, lip - h * 0.014),
        (neck, lip),
    ]);
    let round = outline(32);
    let g = a.part(surface, label);
    round.sweep(
        g,
        &[(0., h * 0.010), (r * 0.64, h * 0.006), (r * 0.825, 0.)],
        ProfileJoints::Smooth,
        Transform::IDENTITY,
    );
    if clear {
        let floor = h * 0.019;
        let mut inner = vec![(0., floor), (outer[0].0 - thickness, floor)];
        inner.extend(outer[1..].iter().map(|&(rad, y)| (rad - thickness, y)));
        if !p.empty {
            let level = h * p.fill_fraction;
            let mut liquid = Vec::with_capacity(inner.len() + 2);
            // Sample the same mould stations up to the actual fill line, even when
            // the shoulder is submerged. A submillimetre separation avoids coincident interfaces.
            let gap = (thickness * 0.20).min(0.0003);
            let start = floor + gap;
            liquid.extend([(0., start), (radius_at(&inner[1..], start) - gap, start)]);
            liquid.extend(
                inner[2..]
                    .iter()
                    .copied()
                    .take_while(|&(_, y)| y < level - 1e-5)
                    .map(|(rad, y)| (rad - gap, y)),
            );
            let end = radius_at(&inner[1..], level) - gap;
            liquid.extend([(end, level), (0., level)]);
            round.sweep(
                a.part(Surface::Liquid, label),
                &liquid,
                ProfileJoints::Smooth,
                Transform::IDENTITY,
            );
        }
        inner.reverse();
        inner.insert(0, (neck, lip));
        round.sweep(
            a.part(surface, label),
            &inner,
            ProfileJoints::Smooth,
            Transform::IDENTITY,
        );
    }
    wrap::side(
        a,
        (surface, label),
        outer,
        c.label_band.map(|b| wrap::Wrap::new(o, b * h)),
        Transform::IDENTITY,
    );
    let cap_label = package_print::cap(o);
    let g = a.part(Surface::Plastic, cap_label);
    // A smooth rolled seal and a separate tamper-evident band leave a visible neck reveal.
    let cap_r = neck * 1.13;
    if c.tamper_band {
        let sep = band * 0.25;
        round.sweep(
            g,
            &[
                (neck, cap_low - band - sep),
                (cap_r * 0.94, cap_low - band - sep),
                (cap_r * 0.98, cap_low - band * 0.72 - sep),
                (cap_r * 0.98, cap_low - sep),
                (neck, cap_low - sep),
            ],
            ProfileJoints::Creased,
            Transform::IDENTITY,
        );
    }
    let closure = RevolvedOutline::round(c.flutes * 4, c.grip_depth, c.flutes);
    closure.sweep(
        g,
        &[
            (neck * 0.90, cap_low),
            (cap_r * 0.97, cap_low),
            (cap_r, cap_low + cap_height * 0.09),
            (cap_r, cap_top - cap_height * 0.15),
            (cap_r * 0.96, cap_top - cap_height * 0.035),
            (cap_r * 0.87, cap_top),
            (0., cap_top),
        ],
        ProfileJoints::Smooth,
        Transform::IDENTITY,
    );
    if p.style == 5 {
        let gauge = h * 0.0045;
        let span = neck * 0.72;
        let rise = (h - cap_top - gauge) * 0.92;
        let foot = cap_top - gauge;
        let points: Vec<_> = (0..=16)
            .map(|i| {
                let (sin, cos) = (PI * i as f32 / 16.).sin_cos();
                Vec3::new(cos * span, foot + sin * rise, 0.)
            })
            .collect();
        a.part(Surface::Rubber, label).tube(&points, gauge, 8);
    }
}
