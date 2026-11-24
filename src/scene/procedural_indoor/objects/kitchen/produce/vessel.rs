use super::*;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Carrier {
    Glazed,
    TurnedWood,
    Wire,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Vessel {
    pub kind: Carrier,
    pub rim_axes_m: Vec2,
    pub floor_m: f32,
    pub rim_y_m: f32,
    pub base_fraction: f32,
    pub gauge_m: f32,
    pub foot_fraction: f32,
    pub ribs: usize,
    pub hoops: usize,
    pub spoke_sweep: f32,
    pub finish: Surface,
    pub wall_curve: f32,
}

impl Vessel {
    pub(crate) fn is_basket(&self) -> bool {
        self.kind == Carrier::Wire
    }
    pub(super) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"FruitBin"));
        let mut f = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"BowlCoat"),
        );
        let kind = match r.random_range(0..100) {
            0..=45 => Carrier::Glazed,
            46..=66 => Carrier::TurnedWood,
            _ => Carrier::Wire,
        };
        let axes =
            o.size.xz() * Vec2::new(r.random_range(0.41..0.482), r.random_range(0.42..0.482));
        let gauge = match kind {
            Carrier::Wire => r.random_range(0.0014..0.0025),
            Carrier::Glazed => r.random_range(0.0028..0.0055),
            Carrier::TurnedWood => r.random_range(0.0048..0.009),
        };
        Self {
            kind,
            rim_axes_m: axes,
            floor_m: if kind == Carrier::Wire {
                r.random_range(0.005..0.010)
            } else {
                gauge * r.random_range(1.35..2.3)
            },
            rim_y_m: (o.size.y * r.random_range(0.29..0.51)).min(axes.min_element() * 0.66),
            base_fraction: r.random_range(0.59..0.80),
            gauge_m: gauge,
            foot_fraction: r.random_range(0.35..0.51),
            ribs: r.random_range(10..=17),
            hoops: r.random_range(2..=4),
            spoke_sweep: r.random_range(-0.080..0.080),
            finish: match kind {
                Carrier::Glazed => Surface::Ceramic,
                Carrier::TurnedWood => Surface::Wood,
                _ => {
                    if f.random_bool(0.57) {
                        Surface::Chrome
                    } else {
                        Surface::Metal
                    }
                }
            },
            wall_curve: r.random_range(-0.38..0.96),
        }
    }
    fn radius_fraction(&self, t: f32) -> f32 {
        self.base_fraction + (1. - self.base_fraction) * (t + self.wall_curve * t * (1. - t))
    }
    fn wall_fraction(&self) -> f32 {
        self.gauge_m / self.rim_axes_m.min_element()
    }
    /// Inscribed free radius in ellipse coordinates. The polygonal ring,
    /// welded wire, manufacturing thickness and a small assembly gap are included.
    pub(super) fn clear_fraction(&self, above_floor: f32) -> f32 {
        let t = (above_floor / (self.rim_y_m - self.floor_m)).clamp(0., 1.);
        // A linear lower bound covers the whole fruit edge between stations,
        // including where it crosses one of the faceted vessel rings.
        let radius =
            self.base_fraction + (1. - self.base_fraction) * (t + self.wall_curve.min(0.) * 0.25);
        // A swept rib chords its path between hoops. Bound its inward bend
        // and the finite thickness of stock including the sloping wire axis.
        let sides = if self.is_basket() {
            (self.ribs * 3).min(32)
        } else {
            32
        };
        radius * (PI / sides as f32).cos() - self.wall_fraction() * 1.75 - 0.005
    }
    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject) {
        let outer = self.rim_axes_m;
        let aspect = Vec2::new(1., outer.y / outer.x);
        let section = RevolvedOutline::superellipse(32, 2., aspect);
        let base = self.base_fraction * outer.x;
        let top = self.rim_y_m;
        let g = self.gauge_m;
        let floor = self.floor_m;
        if self.kind != Carrier::Wire {
            let foot = self.foot_fraction * outer.x;
            let wall = self.wall_fraction() * outer.x;
            let side = |t, offset| {
                (
                    self.radius_fraction(t) * outer.x + wall * offset,
                    floor + (top - floor) * t,
                )
            };
            let profile = [
                (0., g * 0.35),
                (foot * 0.89, g * 0.35),
                (foot * 0.90, 0.),
                (foot, 0.),
                (foot + g, g * 0.55),
                (base + wall * 0.26, floor - g * 0.70),
                side(1. / 3., 0.15),
                side(2. / 3., 0.05),
                (outer.x, top - g * 0.50),
                (outer.x - wall * 0.28, top),
                (outer.x - wall * 0.74, top),
                (outer.x - wall, top - g * 0.40),
                side(2. / 3., -1.),
                side(1. / 3., -1.),
                (base - wall, floor),
                (0., floor),
            ];
            section.sweep(
                a.part(self.finish, "dishware"),
                &profile,
                ProfileJoints::Smooth,
                Transform::IDENTITY,
            );
            return;
        }
        // A closed plywood or metal base bears the load. Every rib starts
        // inside its edge and meets each hoop at a shared tessellation vertex.
        section.sweep(
            a.part(
                if o.seed & 1 == 0 {
                    Surface::Wood
                } else {
                    self.finish
                },
                "dishware",
            ),
            &[
                (0., 0.),
                (base, 0.),
                (base + g * 0.70, g * 0.60),
                (base + g * 0.70, floor - g * 0.40),
                (base, floor),
                (0., floor),
            ],
            ProfileJoints::Creased,
            Transform::IDENTITY,
        );
        let position = |j: usize, t: f32, n: usize| {
            let angle = TAU * j as f32 / n as f32 + self.spoke_sweep * t;
            let (s, c) = angle.sin_cos();
            let xz = outer * Vec2::new(c, s) * self.radius_fraction(t);
            Vec3::new(xz.x, floor - g * 0.6 + (top - floor) * t, xz.y)
        };
        let metal = a.part(self.finish, "dishware");
        let n = self.ribs * 3;
        for k in 1..=self.hoops {
            let t = k as f32 / self.hoops as f32;
            let mut ring: Vec<_> = (0..n).map(|i| position(i, t, n)).collect();
            ring.push(ring[0]);
            metal.tube(&ring, if k == self.hoops { g * 1.08 } else { g * 0.80 }, 8);
        }
        for i in 0..self.ribs {
            let mut points = [Vec3::ZERO; 5];
            for (k, p) in points[..=self.hoops].iter_mut().enumerate() {
                *p = position(i * 3, k as f32 / self.hoops as f32, n);
            }
            metal.tube(&points[..=self.hoops], g * 0.78, 7);
        }
    }
}
