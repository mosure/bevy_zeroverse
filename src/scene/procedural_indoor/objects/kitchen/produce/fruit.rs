use super::*;
use crate::scene::procedural_indoor::materials::produce_skin;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Variety {
    Apple,
    Pear,
    Mandarin,
    Lemon,
    Plum,
}

impl Variety {
    pub(super) fn draw(r: &mut impl Rng) -> Self {
        match r.random_range(0..100) {
            0..=29 => Self::Apple,
            30..=49 => Self::Pear,
            50..=72 => Self::Mandarin,
            73..=87 => Self::Lemon,
            _ => Self::Plum,
        }
    }
    // Meridian stations (fraction of equatorial radius, fraction of body
    // height). Apples and pears return into the stem cavity after the shoulder.
    fn profile(self) -> [(f32, f32); 9] {
        match self {
            Self::Apple => [
                (0., 0.),
                (0.33, 0.),
                (0.65, 0.11),
                (0.89, 0.33),
                (1., 0.61),
                (0.90, 0.84),
                (0.61, 0.99),
                (0.28, 0.97),
                (0., 0.85),
            ],
            Self::Pear => [
                (0., 0.),
                (0.27, 0.),
                (0.76, 0.10),
                (1., 0.30),
                (0.91, 0.50),
                (0.57, 0.70),
                (0.32, 0.91),
                (0.19, 1.),
                (0., 0.97),
            ],
            Self::Mandarin => [
                (0., 0.),
                (0.20, 0.),
                (0.67, 0.10),
                (0.94, 0.31),
                (1., 0.52),
                (0.92, 0.75),
                (0.64, 0.93),
                (0.29, 1.),
                (0., 0.97),
            ],
            Self::Lemon => [
                (0., 0.),
                (0.20, 0.),
                (0.62, 0.11),
                (0.91, 0.30),
                (1., 0.50),
                (0.92, 0.69),
                (0.67, 0.87),
                (0.30, 0.98),
                (0., 1.),
            ],
            Self::Plum => [
                (0., 0.),
                (0.17, 0.),
                (0.57, 0.12),
                (0.91, 0.37),
                (1., 0.58),
                (0.88, 0.79),
                (0.60, 0.95),
                (0.28, 1.),
                (0., 0.97),
            ],
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Fruit {
    pub variety: Variety,
    pub skin: Skin,
    pub skin_meridian: u8,
    pub center_m: Vec2,
    pub axes_m: Vec2,
    pub height_m: f32,
    pub yaw: f32,
    pub lean_m: f32,
    pub lobing: f32,
    pub phase: f32,
    pub stem_m: f32,
}

impl Fruit {
    pub(super) fn sample(variety: Variety, r: &mut impl Rng, finish: &mut impl Rng) -> Self {
        use Variety::*;
        let (width, height, aspect, lean, lobing, stem) = match variety {
            Apple => (
                r.random_range(0.030..0.043),
                r.random_range(1.68..2.04),
                r.random_range(0.89..1.0),
                0.060,
                r.random_range(0.030..0.060),
                r.random_bool(0.80),
            ),
            Pear => (
                r.random_range(0.029..0.041),
                r.random_range(2.15..2.76),
                r.random_range(0.90..1.0),
                0.18,
                r.random_range(0.010..0.034),
                r.random_bool(0.91),
            ),
            Mandarin => (
                r.random_range(0.026..0.041),
                r.random_range(1.42..1.87),
                r.random_range(0.93..1.0),
                0.035,
                r.random_range(0.011..0.030),
                false,
            ),
            Lemon => (
                r.random_range(0.032..0.052),
                r.random_range(1.13..1.42),
                r.random_range(0.55..0.74),
                0.04,
                r.random_range(0.0..0.018),
                false,
            ),
            Plum => (
                r.random_range(0.023..0.035),
                r.random_range(1.75..2.24),
                r.random_range(0.86..0.99),
                0.06,
                r.random_range(0.032..0.068),
                r.random_bool(0.13),
            ),
        };
        let skin = match variety {
            Apple => [
                Skin::RedApple,
                Skin::BlushApple,
                Skin::GreenApple,
                Skin::GoldenApple,
            ][finish.random_range(0..4)],
            Pear => [
                Skin::GreenPear,
                Skin::RussetPear,
                Skin::RedPear,
                Skin::YellowPear,
            ][finish.random_range(0..4)],
            Mandarin => {
                if finish.random_bool(0.55) {
                    Skin::Tangerine
                } else {
                    Skin::Orange
                }
            }
            Lemon => {
                if finish.random_bool(0.79) {
                    Skin::Lemon
                } else {
                    Skin::Lime
                }
            }
            Plum => [
                Skin::VioletPlum,
                Skin::RedPlum,
                Skin::BluePlum,
                Skin::Greengage,
            ][finish.random_range(0..4)],
        };
        Self {
            variety,
            skin,
            skin_meridian: finish.random_range(0..24),
            center_m: Vec2::ZERO,
            axes_m: Vec2::new(width, width * aspect),
            height_m: width * height,
            yaw: r.random_range(-PI..PI),
            lean_m: width * r.random_range(-lean..lean),
            lobing,
            phase: r.random_range(0.0..TAU),
            stem_m: if stem {
                r.random_range(0.005..0.018)
            } else {
                0.
            },
        }
    }
    pub(super) fn total_height(&self) -> f32 {
        self.height_m
            .max(self.variety.profile()[8].1 * self.height_m + self.stem_m)
    }
    pub(super) fn clearance_radius(&self) -> f32 {
        self.axes_m.max_element() + self.lean_m.abs()
    }
    pub(super) fn center_limit(&self, v: &Vessel) -> f32 {
        let min_axis = v.rim_axes_m.min_element();
        self.variety
            .profile()
            .into_iter()
            .map(|(radius, y)| {
                v.clear_fraction(y * self.height_m)
                    - (radius * self.axes_m.max_element() + self.lean_m.abs() * y + 0.00065)
                        / min_axis
            })
            .fold(f32::INFINITY, f32::min)
    }

    pub(super) fn build(&self, a: &mut Assembly, floor: f32) {
        const SIDES: usize = 24;
        static CIRCLE: OnceLock<[(f32, f32); SIDES + 1]> = OnceLock::new();
        let angles = CIRCLE.get_or_init(|| {
            std::array::from_fn(|j| (TAU * (j % SIDES) as f32 / SIDES as f32).sin_cos())
        });
        let pose = Transform::from_xyz(self.center_m.x, floor, self.center_m.y)
            .with_rotation(Quat::from_rotation_y(self.yaw));
        let sections = self.variety.profile();
        let (uv0, span) = self.skin.chart();
        {
            let g = a.part(Surface::Ink, produce_skin::LABEL);
            let base = g.positions.len() as u32;
            // Seven shared rings, one seam per ring. The smooth normal uses the
            // actual elliptical, lobed outline and its drifting centre.
            let lobes = match self.variety {
                Variety::Apple => 5.,
                Variety::Mandarin => 8.,
                Variety::Pear => 3.,
                Variety::Lemon => 2.,
                Variety::Plum => 1.,
            };
            let meridians = std::array::from_fn::<_, { SIDES + 1 }, _>(|j| {
                let longitude = (j + usize::from(self.skin_meridian)) % SIDES;
                let (sin, cos) = angles[longitude];
                let (s, c) = (TAU * longitude as f32 / SIDES as f32 * lobes + self.phase).sin_cos();
                let (factor, slope) = if self.variety == Variety::Plum {
                    // The stone leaves a shallow suture along one meridian.
                    let seam = (1. + c) * 0.5;
                    (
                        1. - self.lobing * seam.powi(12),
                        6. * self.lobing * s * seam.powi(11),
                    )
                } else {
                    (
                        1. - self.lobing * (1. - c) * 0.5,
                        -self.lobing * lobes * s * 0.5,
                    )
                };
                let axis = Vec2::new(self.axes_m.x * cos, self.axes_m.y * sin);
                let tangent =
                    Vec2::new(-self.axes_m.x * sin, self.axes_m.y * cos) * factor + axis * slope;
                (axis * factor, tangent)
            });
            for k in 1..8 {
                let (radius, h) = sections[k];
                let (dr, dh) = (
                    sections[k + 1].0 - sections[k - 1].0,
                    sections[k + 1].1 - sections[k - 1].1,
                );
                for (j, &(axis, tangent)) in meridians.iter().enumerate() {
                    let mut xz = axis * radius;
                    let mut delta = axis * dr;
                    let mut bearing = tangent;
                    if self.variety == Variety::Lemon {
                        // A sideways citrus narrows into unequal terminal tips. Apply
                        // the same derivative to the tangents as to the silhouette.
                        let x = xz.x / self.axes_m.x;
                        let tip = x.powi(14);
                        let pinch = 0.135 + 0.045 * x;
                        let scale = 1. + pinch * (tip - 1.);
                        let gradient = scale + 14. * pinch * tip + 0.045 * x * (tip - 1.);
                        xz.x *= scale;
                        delta.x *= gradient;
                        bearing.x *= gradient;
                    }
                    xz.x += self.lean_m * h;
                    delta.x += self.lean_m * dh;
                    let n = Vec3::new(delta.x, self.height_m * dh, delta.y)
                        .cross(Vec3::new(bearing.x, 0., bearing.y))
                        .normalize();
                    g.positions.push(
                        pose.transform_point(Vec3::new(xz.x, h * self.height_m, xz.y))
                            .to_array(),
                    );
                    g.normals.push((pose.rotation * n).to_array());
                    g.uvs.push(
                        (uv0 + span * Vec2::new(j as f32 / SIDES as f32, 1. - k as f32 / 8.))
                            .to_array(),
                    );
                }
            }
            for ring in 0..6 {
                for j in 0..SIDES {
                    let i = base + (ring * (SIDES + 1) + j) as u32;
                    let next = i + (SIDES + 1) as u32;
                    g.indices.extend([i, next, i + 1, i + 1, next, next + 1]);
                }
            }
            for (top, k, n) in [(false, 0, Vec3::NEG_Y), (true, 8, Vec3::Y)] {
                let h = sections[k].1;
                let p = pose.transform_point(Vec3::new(self.lean_m * h, self.height_m * h, 0.));
                let first = base + if top { (6 * (SIDES + 1)) as u32 } else { 0 };
                for j in 0..SIDES as u32 {
                    let i = g.positions.len() as u32;
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs.push(
                        (uv0 + span
                            * Vec2::new(
                                (j as f32 + 0.5) / SIDES as f32,
                                if top { 0. } else { 1. },
                            ))
                        .to_array(),
                    );
                    let edge = first + j;
                    g.indices.extend(if top {
                        [i, edge + 1, edge]
                    } else {
                        [i, edge, edge + 1]
                    });
                }
            }
        }
        if self.stem_m > 0. {
            let h = sections[8].1;
            let root = Vec3::new(self.lean_m * h, self.height_m * h - 0.0006, 0.);
            let s = self.stem_m;
            let path = [
                root,
                root + Vec3::new(s * 0.12, s * 0.48, s * 0.07),
                root + Vec3::new(s * 0.32, s, s * 0.16),
            ]
            .map(|p| pose.transform_point(p));
            a.part(Surface::WoodEdge, "food")
                .tube(&path, (self.axes_m.x * 0.051).min(0.0020), 6);
        }
    }
}
