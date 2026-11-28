use super::*;
use crate::scene::procedural_indoor::materials::glass::carboy;
const SIDES: u32 = 32;
const WALL_M: f32 = 0.0015;

#[derive(Debug, Clone, Serialize)]
pub struct Bottle {
    pub capacity_l: f32,
    pub fill_fraction: f32,
    pub radius_m: f32,
    pub height_m: f32,
    pub insertion_m: f32,
    pub ribs: usize,
    pub shoulder: f32,
}

// Inner profile of a blow-moulded inverted carboy in radius/height fractions.
impl Bottle {
    pub(super) fn sample(seed: u64, width: f32) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"Carboy3D"));
        let large = r.random_bool(0.64);
        let mut b = Self {
            capacity_l: if large { 18.927 } else { 11.356 },
            fill_fraction: if r.random_bool(0.064) {
                0.
            } else {
                r.random_range(0.045..0.964)
            },
            radius_m: if large {
                r.random_range(0.130..0.145)
            } else {
                r.random_range(0.112..0.127)
            },
            height_m: 1.,
            insertion_m: r.random_range(0.022..0.031),
            ribs: r.random_range(2..=5),
            shoulder: r.random_range(0.245..0.315),
        };
        b.radius_m = b.radius_m.min(width * 0.473 - WALL_M);
        b.height_m = b.capacity_l / (b.profile_volume() * 1000.);
        b
    }
    pub(super) fn fit(mut self, sy: f32, footprint: f32) -> Self {
        let radius = self.radius_m.min(footprint * 0.474 - WALL_M);
        self.capacity_l *= sy * (radius / self.radius_m).powi(2);
        self.radius_m = radius;
        self.height_m *= sy;
        self.insertion_m *= sy;
        self
    }
    fn profile(&self) -> Vec<(f32, f32)> {
        let mut rings = Vec::with_capacity(self.ribs * 4 + 12);
        rings.extend([
            (0.176, 0.),
            (0.176, 0.078),
            (0.207, 0.081),
            (0.207, 0.099),
            (0.179, 0.107),
            (0.187, 0.142),
            (0.61, self.shoulder - 0.055),
            (0.943, self.shoulder),
            (0.972, 0.35),
        ]);
        for i in 0..self.ribs {
            let y = 0.395 + 0.48 * i as f32 / (self.ribs - 1) as f32;
            rings.extend([
                (0.972, y - 0.018),
                (1., y - 0.007),
                (1., y + 0.006),
                (0.972, y + 0.019),
            ]);
        }
        rings.extend([(0.97, 0.945), (0.953, 0.973), (0.868, 1.)]);
        for (r, y) in &mut rings {
            *r *= self.radius_m;
            *y *= self.height_m;
        }
        rings
    }
    fn profile_volume(&self) -> f32 {
        self.profile()
            .windows(2)
            .map(|p| segment_volume(p[0], p[1]))
            .sum()
    }
    fn liquid_joint(&self, rings: &[(f32, f32)]) -> Option<(usize, (f32, f32))> {
        if self.fill_fraction <= 0. {
            return None;
        }
        let mut remaining = self.fill_fraction * self.capacity_l * 0.001;
        for (i, p) in rings.windows(2).enumerate() {
            let v = segment_volume(p[0], p[1]);
            if remaining <= v {
                let (mut lo, mut hi) = (0., 1.);
                let at = |t: f32| {
                    (
                        p[0].0 + (p[1].0 - p[0].0) * t,
                        p[0].1 + (p[1].1 - p[0].1) * t,
                    )
                };
                for _ in 0..18 {
                    let t = (lo + hi) * 0.5;
                    if segment_volume(p[0], at(t)) < remaining {
                        lo = t;
                    } else {
                        hi = t;
                    }
                }
                return Some((i, at((lo + hi) * 0.5)));
            }
            remaining -= v;
        }
        None
    }
    pub(super) fn build(&self, a: &mut Assembly, deck: f32) {
        let mut rings = self.profile();
        let wet = self.liquid_joint(&rings);
        let tf = Transform::from_xyz(0., deck - self.insertion_m, 0.);
        let joint = wet.map(|(i, point)| {
            a.part(Surface::Liquid, "drink_container").lathe(
                &[
                    (point.0, point.1),
                    (point.0 * 0.96, point.1 - 0.0007),
                    (0., point.1 - 0.0007),
                ],
                SIDES,
                tf,
            );
            // Snap only sub-millimetre coincidences to the nearest profile vertex.
            if (point.1 - rings[i].1).abs() < 0.00008 {
                i
            } else if (point.1 - rings[i + 1].1).abs() < 0.00008 {
                i + 1
            } else {
                rings.insert(i + 1, point);
                i + 1
            }
        });
        for (r, _) in &mut rings {
            *r += WALL_M;
        }
        rings.push((0., self.height_m));
        if let Some(j) = joint {
            let mut dry = Geometry::default();
            a.part(Surface::ContainerGlass, carboy::WET)
                .split_lathe(&mut dry, &rings, j, SIDES, tf);
            a.part(Surface::ContainerGlass, carboy::DRY).append(dry);
        } else {
            a.part(Surface::ContainerGlass, carboy::DRY)
                .lathe(&rings, SIDES, tf);
        }
    }
}
fn segment_volume((r0, y0): (f32, f32), (r1, y1): (f32, f32)) -> f32 {
    let area = 0.5 * SIDES as f32 * (std::f32::consts::TAU / SIDES as f32).sin();
    area * (y1 - y0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3.
}
