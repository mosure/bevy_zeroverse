//! Seeded food packages shared by pantry and refrigerator shelf admission/build.
use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PackageProgram {
    /// Carton, jar, can, bottle, pouch, lidded tub.
    pub style: u8,
    pub body: Vec3,
    pub shoulder: f32,
    pub neck: f32,
    pub foot: f32,
    pub lid: f32,
    pub roundness: f32,
    pub fill: f32,
    pub label_band: Option<Vec2>,
    pub finish: Surface,
    pub cap: Surface,
}
pub fn parameters(size: Vec3, style: u8, seed: u64) -> PackageProgram {
    let mut r = stream(seed, 0x5041434b41474550);
    let style = style % 6;
    let height = if style == 5 {
        r.random_range(0.42..0.72)
    } else {
        r.random_range(0.76..0.98)
    };
    PackageProgram {
        style,
        body: size
            * Vec3::new(
                r.random_range(0.54..0.81),
                height,
                r.random_range(0.55..0.82),
            ),
        shoulder: r.random_range(0.58..0.81),
        neck: if style == 3 {
            r.random_range(0.24..0.46)
        } else {
            r.random_range(0.68..0.92)
        },
        foot: r.random_range(0.74..0.96),
        lid: size.y * r.random_range(0.025..0.055),
        roundness: r.random_range(2.4..8.0),
        fill: r.random_range(0.20..0.70),
        label_band: if r.random_bool(0.62) {
            Some(Vec2::new(
                r.random_range(0.12..0.25),
                r.random_range(0.47..0.59),
            ))
        } else {
            None
        },
        finish: if style == 2 {
            Surface::Metal
        } else if matches!(style, 1 | 3) {
            Surface::ContainerGlass
        } else if style == 0 || r.random_bool(0.4) {
            Surface::Paper
        } else {
            Surface::Plastic
        },
        cap: if r.random_bool(0.5) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}
impl PackageProgram {
    pub(crate) fn half_extents(&self, yaw: f32) -> Vec2 {
        if matches!(self.style, 1..=3) {
            Vec2::splat(self.body.x.min(self.body.z) * 0.5 + 0.00025)
        } else {
            crate::scene::procedural_indoor::placement::rotated_half(self.body.xz(), yaw)
        }
    }
}
pub(crate) fn build(a: &mut Assembly, tf: Transform, p: &PackageProgram, seed: u64) {
    let s = p.body;
    let l = if p.style == 3 {
        "drink_container"
    } else {
        "food"
    };
    let key = |surface| {
        let slot = super::super::super::materials::variants::object_slot(surface, seed, None);
        format!("{l}#finish{slot}")
    };
    let radius = s.x.min(s.z) * 0.5;
    match p.style {
        0 => {
            // Gable carton: the narrow top ridge retains finite thickness.
            a.part(p.finish, &key(p.finish)).housing(
                &[
                    (0., s.xz() * 0.5),
                    (s.y * 0.77, s.xz() * 0.5),
                    (s.y, s.xz() * Vec2::new(0.49, 0.035)),
                ],
                9.,
                tf,
            );
        }
        1 | 3 => {
            let h = s.y - p.lid;
            let w = (radius * 0.055).min(0.003);
            a.part(p.finish, &key(p.finish)).lathe(
                &[
                    (0., 0.),
                    (radius * p.foot, 0.),
                    (radius, h * 0.15),
                    (radius, h * p.shoulder),
                    (radius * p.neck, h * 0.93),
                    (radius * p.neck, h),
                    (radius * p.neck - w, h),
                    (radius - w, h * p.shoulder),
                    (radius - w, h * 0.15),
                    (radius * p.foot - w, w),
                    (0., w),
                ],
                32,
                tf,
            );
            a.part(p.cap, &key(p.cap)).cylinder(
                radius * p.neck,
                p.lid,
                tf.mul_transform(Transform::from_xyz(0., h + p.lid * 0.5, 0.)),
            );
            let fill_top = h * p.fill.min(p.shoulder - 0.025);
            // Contents follow the narrowing foot, avoiding liquid outside glass.
            let fill_surface = if p.style == 3 {
                Surface::Drink
            } else {
                Surface::Terracotta
            };
            let label = if p.style == 3 {
                l.to_owned()
            } else {
                "food#finish103".into()
            };
            a.part(fill_surface, &label).lathe(
                &[
                    (0., w * 1.2),
                    (radius * p.foot - w * 1.4, w * 1.2),
                    (radius - w * 1.4, (h * 0.15).min(fill_top * 0.8)),
                    (radius - w * 1.4, fill_top),
                    (0., fill_top),
                ],
                24,
                tf,
            );
        }
        2 => {
            let edge = (radius * 0.035).min(0.002);
            a.part(p.finish, &key(p.finish)).lathe(
                &[
                    (0., 0.),
                    (radius - edge, 0.),
                    (radius, edge),
                    (radius, s.y - edge),
                    (radius - edge, s.y),
                    (0., s.y),
                ],
                32,
                tf,
            );
        }
        4 => {
            a.part(p.finish, &key(p.finish)).housing(
                &[
                    (0., s.xz() * Vec2::new(0.45, 0.32)),
                    (s.y * 0.2, s.xz() * 0.5),
                    (s.y * 0.80, s.xz() * Vec2::new(0.47, 0.42)),
                    (s.y, s.xz() * Vec2::new(0.48, 0.055)),
                ],
                p.roundness,
                tf,
            );
        }
        _ => {
            a.part(p.finish, &key(p.finish)).housing(
                &[(0., s.xz() * 0.5 * p.foot), (s.y - p.lid, s.xz() * 0.48)],
                p.roundness,
                tf,
            );
            a.part(p.cap, &key(p.cap)).housing(
                &[(s.y - p.lid + 0.0004, s.xz() * 0.5), (s.y, s.xz() * 0.495)],
                p.roundness,
                tf,
            );
        }
    }
    if let Some(band) = p.label_band.filter(|_| matches!(p.style, 1..=3)) {
        let slot = super::super::super::materials::variants::object_slot(
            Surface::PrintedPaper,
            seed,
            None,
        );
        let label = format!("{l}#finish{slot}#uv_local");
        let h = if p.style == 2 { s.y } else { s.y - p.lid };
        let hi = (band.y * h).min(h * p.shoulder - 0.002);
        let lo = (band.x * h).max(h * 0.15 + 0.0005);
        if hi > lo + 0.003 {
            let mut wrap = Geometry::default();
            wrap.lathe(&[(radius + 0.00025, lo), (radius + 0.00025, hi)], 32, tf);
            // Each physical label has its own unwrapped image. Parent X/Z
            // projection would erase the label's vertical dimension, and a
            // shared finish may be used by several differently sized packages.
            for uv in &mut wrap.uvs {
                uv[0] /= TAU * (radius + 0.00025);
                uv[1] = (uv[1] - (radius + 0.00025) - lo) / (hi - lo);
            }
            a.part(Surface::PrintedPaper, &label).append(wrap);
        }
    }
}
