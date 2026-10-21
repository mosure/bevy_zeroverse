//! Bounded, reproducible cable/bulb program shared by construction, reservations,
//! emission and receipts. Variant parity selects wall or ceiling suspension.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StringProgram {
    pub suspended: bool,
    pub bulbs: usize,
    pub bays: usize,
    pub sag: f32,
    pub skew: f32,
    pub bow: f32,
    pub radius_m: f32,
    pub drop_m: f32,
    pub shape: u32,
    pub colors_srgb: Vec<[f32; 3]>,
    pub luminance: f32,
    pub spacing_jitter: f32,
}

pub fn program(o: &IndoorObject) -> StringProgram {
    let mut r = stream(o.seed, 0x535452494e4753);
    let suspended = o.variant % 2 == 1;
    let radius = r
        .random_range(0.009_f32..0.054)
        .min(o.size.z * 0.16)
        .min(o.size.y * 0.085);
    let mut p = StringProgram {
        suspended,
        bulbs: (o.size.x / r.random_range(0.16..0.46))
            .round()
            .clamp(4., 36.) as usize,
        bays: if o.size.x > 3. {
            r.random_range(1..=3)
        } else {
            1
        },
        sag: r.random_range(0.21..0.57),
        skew: r.random_range(-0.55..0.55),
        bow: r.random_range(-0.29..0.29),
        radius_m: radius,
        drop_m: o.size.y * r.random_range(0.025..0.14),
        shape: r.random_range(0..4),
        colors_srgb: Vec::new(),
        luminance: r.random_range(300.0..1100.),
        spacing_jitter: r.random_range(0.06..0.32),
    };
    // A larger globe needs more pitch. Bound the worst pair of independent
    // spacing/size jitters, so decorative variation cannot intersect neighbors.
    let capacity = (o.size.x * 0.96 * (1. - 2. * p.spacing_jitter) / (2.24 * p.radius_m + 0.006))
        .floor()
        .max(1.) as usize;
    p.bulbs = p.bulbs.min(capacity);
    p.sag = p
        .sag
        .min(0.71 - p.drop_m / o.size.y - 3.7 * radius / o.size.y);
    // Coherent white installations, alternating tones, festive and pastel runs.
    // Palette size is bounded: never a material or shadow light per bulb.
    let style = r.random_range(0..5);
    if style < 2 {
        p.colors_srgb.push(
            crate::scene::procedural_indoor::photometry::tinted_white(
                r.random_range(1900.0..6200.),
                r.random_range(-0.025..0.025),
            )
            .to_array(),
        );
    } else {
        let count = if style == 2 { 2 } else { r.random_range(3..=6) };
        let hue: f32 = r.random_range(0.0..360.);
        for i in 0..count {
            let c = Color::hsl(
                (hue + 360. * i as f32 / count as f32 + r.random_range(-12.0..12.))
                    .rem_euclid(360.),
                if style == 4 {
                    r.random_range(0.40..0.66)
                } else {
                    r.random_range(0.72..0.96)
                },
                if style == 4 {
                    r.random_range(0.70..0.83)
                } else {
                    r.random_range(0.50..0.64)
                },
            )
            .to_srgba();
            p.colors_srgb.push([c.red, c.green, c.blue]);
        }
    }
    p.colors_srgb.truncate(p.bulbs);
    p
}

impl StringProgram {
    pub fn point(&self, size: Vec3, t: f32) -> Vec3 {
        let bay = (t * self.bays as f32).floor();
        let u = (t * self.bays as f32).fract();
        // Gravity-loaded cable lies in a vertical plane between supports.
        // Change direction at a supported bay, never bend sideways in mid-air.
        let left = (PI * bay + self.skew).cos();
        let right = (PI * (bay + 1.) + self.skew).cos();
        let transverse = left + (right - left) * u;
        Vec3::new(
            (t - 0.5) * size.x * 0.96,
            size.y * (0.82 + self.skew * 0.08 * (t - 0.5) - self.sag * 4. * u * (1. - u)),
            size.z * self.bow * transverse,
        )
    }

    pub fn anchors(&self, size: Vec3) -> impl Iterator<Item = Vec3> + '_ {
        (0..=self.bays).map(move |i| self.point(size, i as f32 / self.bays as f32))
    }

    fn bulbs(&self, o: &IndoorObject) -> impl Iterator<Item = (Vec3, f32, f32, usize)> + '_ {
        let mut r = stream(o.seed, 0x42554c4253);
        let size = o.size;
        (0..self.bulbs).map(move |i| {
            let t = (i as f32 + 0.5 + r.random_range(-self.spacing_jitter..self.spacing_jitter))
                / self.bulbs as f32;
            (
                self.point(size, t),
                self.radius_m * r.random_range(0.78..1.12),
                self.drop_m * r.random_range(0.65..1.15),
                i % self.colors_srgb.len(),
            )
        })
    }

    fn line(&self, size: Vec3) -> Vec<Vec3> {
        // Each bay has an exact support vertex, including three-bay strands.
        let n = self.bays * 24;
        (0..=n)
            .map(|i| self.point(size, i as f32 / n as f32))
            .collect()
    }

    pub fn local_reservations(&self, o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
        let mut boxes = Vec::with_capacity(self.bays * 25 + self.bulbs + 1);
        let segment =
            |a: Vec3, b: Vec3, r: f32| (a.min(b) - Vec3::splat(r), a.max(b) + Vec3::splat(r));
        for pair in self.line(o.size).windows(2) {
            boxes.push(segment(pair[0], pair[1], 0.003));
        }
        for q in self.anchors(o.size) {
            let end = if self.suspended {
                q.with_y(o.size.y - 0.008)
            } else {
                q.with_z(-o.size.z * 0.5 + 0.008)
            };
            boxes.push(segment(q, end, 0.003));
            boxes.push((end - Vec3::splat(0.008), end + Vec3::splat(0.008)));
        }
        for (q, r, drop, _) in self.bulbs(o) {
            boxes.push((
                q - Vec3::new(r, drop + r * 3.0, r),
                q + Vec3::new(r, r * 0.11, r),
            ));
        }
        boxes
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = program(o);
    let l = o.kind.class_name();
    a.part(Surface::Rubber, l).tube(&p.line(o.size), 0.002, 6);
    for q in p.anchors(o.size) {
        let end = if p.suspended {
            q.with_y(o.size.y - 0.008)
        } else {
            q.with_z(-o.size.z * 0.5 + 0.008)
        };
        a.part(Surface::Metal, l).rod(q, end, 0.0025);
        a.box_part(Surface::Metal, l, end, Vec3::splat(0.016), 0.002);
    }
    for (q, r, drop, color) in p.bulbs(o) {
        let socket = q - Vec3::Y * drop;
        a.part(Surface::Rubber, l).rod(q, socket, 0.002);
        a.part(Surface::Plastic, l).cylinder(
            r * 0.44,
            r * 0.65,
            Transform::from_translation(socket - Vec3::Y * r * 0.22),
        );
        let label = format!("{l}#bulb{color}");
        if p.shape == 1 {
            let profile = [
                (0., -2.5),
                (0.55, -2.35),
                (0.90, -1.9),
                (1., -1.3),
                (0.75, -0.8),
                (0.38, -0.5),
                (0., -0.5),
            ]
            .map(|(x, y)| (x * r, y * r));
            a.part(Surface::Light, &label)
                .lathe(&profile, 12, Transform::from_translation(socket));
        } else {
            let radii = match p.shape {
                2 => Vec3::new(r * 0.48, r * 1.22, r * 0.48),
                3 => Vec3::new(r * 0.65, r * 0.82, r * 0.65),
                _ => Vec3::splat(r),
            };
            a.part(Surface::Light, &label).ellipsoid(
                radii,
                Transform::from_translation(socket - Vec3::Y * (r * 0.48 + radii.y)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_bulbs_and_jitter_preserve_neighbor_clearance() {
        for seed in 0..512 {
            let mut o = crate::scene::procedural_indoor::objects::audit::fixture(
                ObjectKind::StringLights,
                seed,
            );
            for size in [
                Vec3::new(1.4, 0.60, 0.16),
                Vec3::new(3., 0.9, 1.2),
                Vec3::new(7.5, 0.9, 1.2),
            ] {
                o.size = size;
                let p = program(&o);
                let bulbs: Vec<_> = p.bulbs(&o).collect();
                for pair in bulbs.windows(2) {
                    assert!(
                        pair[1].0.x - pair[0].0.x > pair[0].1 + pair[1].1 + 0.005,
                        "{seed}: {pair:?}"
                    );
                }
                assert!(p.bulbs <= 36 && p.colors_srgb.len() <= p.bulbs);
            }
        }
    }
}
