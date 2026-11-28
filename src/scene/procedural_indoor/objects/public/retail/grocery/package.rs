use super::*;

mod paper_bag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pack {
    Gable,
    Tub,
    Carton,
    PaperBag,
    Can,
    Bottle,
}

impl Pack {
    pub(super) fn face(self) -> (Surface, &'static str) {
        match self {
            Self::Gable => (Surface::Paper, "drink_container"),
            Self::Tub => (Surface::Plastic, "food"),
            Self::Carton | Self::PaperBag => (Surface::Paper, "food"),
            Self::Can => (Surface::Metal, "drink_container"),
            Self::Bottle => (Surface::Plastic, "drink_container"),
        }
    }
    pub(super) fn size(self, pitch: f32, clear: f32, r: &mut ChaCha8Rng) -> Vec3 {
        let (width, h, depth) = match self {
            Self::Gable => {
                let d = r.random_range(0.061_f32..0.097).min(pitch * 0.78);
                (d, r.random_range(0.15..0.249), d)
            }
            Self::Tub => {
                let d = r.random_range(0.085_f32..0.147).min(pitch * 0.78);
                (d, r.random_range(0.055..0.108), d)
            }
            Self::Carton => (
                r.random_range(0.105_f32..0.208),
                r.random_range(0.15_f32..0.294),
                r.random_range(0.052..0.117),
            ),
            Self::PaperBag => (
                r.random_range(0.093..0.166),
                r.random_range(0.143..0.25),
                r.random_range(0.052..0.103),
            ),
            Self::Can => {
                let d = r.random_range(0.061..0.082);
                (d, r.random_range(0.103..0.165), d)
            }
            Self::Bottle => {
                let d = r.random_range(0.065..0.094);
                (d, r.random_range(0.185..0.286), d)
            }
        };
        Vec3::new(width.min(pitch * 0.88), h.min(clear * 0.92), depth)
    }
    pub(super) fn build(self, a: &mut Assembly, base: Vec3, size: Vec3, yaw: f32, print: &str) {
        let (surface, class) = self.face();
        let rot = Quat::from_rotation_y(yaw);
        let tf = Transform::from_translation(base).with_rotation(rot);
        if matches!(self, Self::Gable) {
            return gable(a, size, tf, print);
        }
        if matches!(self, Self::Tub) {
            return tub(a, size, tf, print);
        }
        if matches!(self, Self::PaperBag) {
            return paper_bag::build(a, size, tf, print);
        }
        if matches!(self, Self::Carton) {
            a.part(surface, print).mapped_cuboid(
                size,
                0.,
                Transform::from_translation(base + Vec3::Y * (size.y * 0.5)).with_rotation(rot),
                |normal, _, _, du, dv, _| {
                    move |x, y| {
                        // A single wrap with a wordmark, side seam and nutrition panel.
                        let interval = if normal.z > 0.5 {
                            (0.03, 0.62)
                        } else if normal.z < -0.5 {
                            (0.65, 0.99)
                        } else {
                            (0.57, 0.74)
                        };
                        Vec2::new(
                            interval.0 + (x + du) / (2. * du) * (interval.1 - interval.0),
                            0.5 - y / (2. * dv),
                        )
                    }
                },
            );
            return;
        }
        let (r, h) = (size.x * 0.5, size.y);
        let bottle = matches!(self, Self::Bottle);
        let (low, high) = if bottle {
            (h * 0.15, h * 0.72)
        } else {
            (h * 0.055, h * 0.943)
        };
        let skin = if bottle {
            Surface::Plastic
        } else {
            Surface::Metal
        };
        let body = a.part(skin, class);
        let segments = 12;
        body.lathe(&[(0., 0.), (r * 0.88, 0.), (r, low)], segments, tf);
        if bottle {
            body.lathe(
                &[
                    (r, high),
                    (r * 0.92, h * 0.79),
                    (r * 0.42, h * 0.88),
                    (r * 0.42, h * 0.92),
                ],
                segments,
                tf,
            );
            a.part(Surface::Accent, class).lathe(
                &[
                    (r * 0.42, h * 0.92),
                    (r * 0.47, h * 0.925),
                    (r * 0.47, h),
                    (0., h),
                ],
                segments,
                tf,
            );
        } else {
            body.lathe(
                &[(r, high), (r * 0.89, h * 0.972), (r * 0.92, h), (0., h)],
                segments,
                tf,
            );
        }
        let g = a.part(surface, print);
        let start = g.uvs.len();
        g.lathe(&[(r, low), (r, high)], segments, tf);
        for (i, uv) in g.uvs[start..].iter_mut().enumerate() {
            *uv = [1. - (i % 13) as f32 / 12., if i < 13 { 1. } else { 0. }];
        }
    }
}

/// A single folded shell: a square printed body, pitched shoulders and the
/// sealed fin. Adjacent sections meet at their folds without hidden box faces.
fn gable(a: &mut Assembly, s: Vec3, tf: Transform, print: &str) {
    let ring = |y: f32, z: f32| {
        let x = s.x * 0.5;
        [
            Vec3::new(-x, y, -z),
            Vec3::new(x, y, -z),
            Vec3::new(x, y, z),
            Vec3::new(-x, y, z),
        ]
    };
    let rings = [
        ring(0., s.z * 0.5),
        ring(s.y * 0.73, s.z * 0.5),
        ring(s.y * 0.945, 0.0017),
        ring(s.y, 0.0017),
    ];
    let paper = "drink_container";
    for (i, section) in rings.windows(2).enumerate() {
        let g = a.part(Surface::Paper, if i == 0 { print } else { paper });
        for j in 0..4 {
            let k = (j + 1) % 4;
            let p = [section[0][j], section[1][j], section[1][k], section[0][k]];
            let n = tf.rotation * (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            let (left, right) = if j == 2 {
                (0.03, 0.62)
            } else if j == 0 {
                (0.65, 0.99)
            } else {
                (0.57, 0.74)
            };
            let uv = if i == 0 {
                [[right, 1.], [right, 0.], [left, 0.], [left, 1.]]
            } else {
                [
                    [0., 0.],
                    [0., p[1].distance(p[0])],
                    [p[2].distance(p[1]), p[1].distance(p[0])],
                    [p[3].distance(p[0]), 0.],
                ]
            };
            g.vertex_quad(p.map(|v| tf.transform_point(v)), [n; 4], uv);
        }
    }
    let g = a.part(Surface::Paper, paper);
    for (i, n) in [(0, Vec3::NEG_Y), (3, Vec3::Y)] {
        let mut p = rings[i];
        if i == 3 {
            p.reverse();
        }
        let uv = p.map(|v| [v.x, v.z]);
        g.vertex_quad(p.map(|v| tf.transform_point(v)), [tf.rotation * n; 4], uv);
    }
}

fn tub(a: &mut Assembly, size: Vec3, tf: Transform, print: &str) {
    let (r, h) = (size.x * 0.5, size.y);
    let side = a.part(Surface::Plastic, print);
    let start = side.uvs.len();
    side.lathe(&[(0.77 * r, h * 0.045), (0.945 * r, h * 0.88)], 12, tf);
    for (i, uv) in side.uvs[start..].iter_mut().enumerate() {
        *uv = [1. - (i % 13) as f32 / 12., if i < 13 { 1. } else { 0. }];
    }
    let body = a.part(Surface::Plastic, "food");
    body.lathe(&[(0., 0.), (0.73 * r, 0.), (0.77 * r, h * 0.045)], 12, tf);
    let lid = a.part(Surface::Paper, "food");
    lid.lathe(
        &[
            (0.945 * r, h * 0.88),
            (r, h * 0.91),
            (r, h * 0.97),
            (0.95 * r, h),
            (0., h),
        ],
        12,
        tf,
    );
}
