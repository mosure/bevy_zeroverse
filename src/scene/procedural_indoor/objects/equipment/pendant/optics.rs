//! The material seam is also the mechanical seat. Only the outward lens or
//! exposed bulb envelope emits; mesh and photometry integrate the same meridian.
use super::*;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};
use std::sync::LazyLock;

const SIDES: u32 = 32;
static OUTLINE: LazyLock<RevolvedOutline> = LazyLock::new(|| RevolvedOutline::round(SIDES, 0., 0));

pub(super) fn rim_scale(angle: f32) -> f32 {
    let half = PI / SIDES as f32;
    half.cos() / ((angle.rem_euclid(2. * half)) - half).cos()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", content = "bulb")]
pub enum Optic {
    OpenBulb(BulbEnvelope),
    Recessed,
    OpalBowl,
}

impl Optic {
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenBulb(_) => "OpenBulb",
            Self::Recessed => "Recessed",
            Self::OpalBowl => "OpalBowl",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BulbEnvelope {
    Globe,
    Pear,
    Candle,
    Tubular,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PendantLens {
    pub optic: Optic,
    pub radius_m: f32,
    pub bottom_y_m: f32,
    pub rise_m: f32,
}

impl PendantLens {
    pub(super) fn sample(
        o: &IndoorObject,
        shade: &ShadeProgram,
        radius: f32,
        height: f32,
        taper: f32,
    ) -> (f32, Self) {
        let opening = luminaires::shade_opening(shade, radius, radius);
        let wall = luminaires::wall_thickness(radius);
        // Keep the driver and its retainer inside even the sharply turned
        // lower shoulder of a bell. The bezel follows a straight radial section.
        let slope = if shade.shape == 2 {
            radius * 0.20 / 0.18
        } else {
            radius - luminaires::upper_radius(shade, radius, taper)
        } / (height - 2. * wall);
        let stock = [
            BulbEnvelope::Globe,
            BulbEnvelope::Pear,
            BulbEnvelope::Candle,
            BulbEnvelope::Tubular,
        ][shade.bulb as usize];

        let mut r = stream(o.seed, u64::from_le_bytes(*b"PendLens"));
        let optic = match (shade.shape, r.random_range(0..10)) {
            (4, _) | (_, 0..=3) => Optic::OpenBulb(stock),
            (_, 4..=7) => Optic::Recessed,
            _ => Optic::OpalBowl,
        };
        let bottom = o.size.y * r.random_range(0.006..0.032);
        let (lip, radius_m, bottom_y_m, rise_m) = match optic {
            Optic::OpenBulb(bulb) => {
                let aspect = r.random_range(match bulb {
                    BulbEnvelope::Globe => 1.62_f32..2.05,
                    BulbEnvelope::Pear => 2.05..2.65,
                    BulbEnvelope::Candle => 2.65..3.80,
                    BulbEnvelope::Tubular => 2.40..3.50,
                });
                let radius = (opening * r.random_range(0.19..0.47)).min(height * 0.55 / aspect);
                let lip = bottom + height * r.random_range(0.06..0.31);
                (
                    lip,
                    radius,
                    (lip + height * r.random_range(-0.22..0.19)).max(bottom),
                    radius * aspect,
                )
            }
            Optic::Recessed => {
                let recess = height
                    * if shade.shape == 2 {
                        r.random_range(0.03..0.12)
                    } else {
                        r.random_range(0.03..0.26)
                    };
                let available = opening - slope * (recess + height * 0.045);
                (
                    bottom,
                    available * r.random_range(0.70..0.96),
                    bottom + recess,
                    0.,
                )
            }
            Optic::OpalBowl => {
                let radius = (opening - slope * height * 0.045) * r.random_range(0.74..0.96);
                let rise = (radius * r.random_range(0.23..0.66)).min(o.size.y * 0.17);
                (bottom + rise, radius, bottom, rise)
            }
        };
        (
            lip,
            Self {
                optic,
                radius_m,
                bottom_y_m,
                rise_m,
            },
        )
    }

    fn with_profile<T>(&self, use_profile: impl FnOnce(&[(f32, f32)]) -> T) -> T {
        let stock: &[(f32, f32)] = match self.optic {
            Optic::Recessed => &[(0., 0.), (1., 0.)],
            Optic::OpalBowl => &[
                (0., 0.),
                (0.22, 0.022),
                (0.52, 0.132),
                (0.77, 0.35),
                (0.94, 0.69),
                (1., 1.),
            ],
            Optic::OpenBulb(bulb) => match bulb {
                BulbEnvelope::Globe => &[
                    (0., 0.),
                    (0.38, 0.045),
                    (0.74, 0.17),
                    (0.94, 0.33),
                    (1., 0.5),
                    (0.94, 0.67),
                    (0.72, 0.84),
                    (0.40, 0.94),
                    (0.32, 1.),
                ],
                BulbEnvelope::Pear => &[
                    (0., 0.),
                    (0.34, 0.035),
                    (0.75, 0.15),
                    (0.96, 0.32),
                    (1., 0.48),
                    (0.87, 0.66),
                    (0.58, 0.80),
                    (0.34, 0.93),
                    (0.32, 1.),
                ],
                BulbEnvelope::Candle => &[
                    (0., 0.),
                    (0.17, 0.11),
                    (0.52, 0.27),
                    (0.81, 0.44),
                    (1., 0.63),
                    (0.96, 0.77),
                    (0.73, 0.87),
                    (0.36, 0.97),
                    (0.32, 1.),
                ],
                BulbEnvelope::Tubular => &[
                    (0., 0.),
                    (0.6, 0.022),
                    (0.92, 0.07),
                    (1., 0.14),
                    (1., 0.72),
                    (0.94, 0.82),
                    (0.70, 0.91),
                    (0.36, 0.97),
                    (0.32, 1.),
                ],
            },
        };
        let mut section = [(0., 0.); 9];
        for (out, &(r, h)) in section.iter_mut().zip(stock) {
            *out = (r * self.radius_m, self.bottom_y_m + h * self.rise_m);
        }
        use_profile(&section[..stock.len()])
    }

    /// Lateral area of the tessellated frusta, including a disk or a pointed end.
    pub fn emitting_area_m2(&self) -> f32 {
        let (s, c) = (PI / SIDES as f32).sin_cos();
        self.with_profile(|section| {
            section
                .windows(2)
                .map(|ab| {
                    let [(a, ay), (b, by)] = [ab[0], ab[1]];
                    SIDES as f32 * s * (a + b) * (by - ay).hypot(c * (b - a))
                })
                .sum()
        })
    }

    pub(crate) fn write_aperture(
        &self,
        geometry: &mut crate::scene::procedural_indoor::geometry::Geometry,
    ) {
        self.with_profile(|section| {
            OUTLINE.sweep(
                geometry,
                section,
                ProfileJoints::Smooth,
                Transform::IDENTITY,
            );
        });
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        label: &str,
        trim: Surface,
        opening: f32,
        lip: f32,
        top: f32,
    ) -> f32 {
        self.write_aperture(a.part(Surface::Light, label));
        let r = self.radius_m;
        let y = self.bottom_y_m + self.rise_m;
        if matches!(self.optic, Optic::OpenBulb(_)) {
            let neck = r * 0.32;
            let back = y + ((top - y) * 0.43).min(r * 0.79);
            a.part(Surface::Ceramic, label).lathe(
                &[
                    (neck, y),
                    (neck * 1.32, y + (back - y) * 0.13),
                    (neck * 1.32, back - (back - y) * 0.12),
                    (neck, back),
                    (0., back),
                ],
                SIDES,
                Transform::IDENTITY,
            );
            back
        } else {
            let back = y + ((top - lip) * 0.045).min(r * 0.12);
            // The annular bezel meets the 32-sided lens and shade opening
            // edge-for-edge. A closed, non-emitting back carries the driver.
            a.part(trim, label).lathe(
                &[(r, y), (opening, lip), (r, back), (0., back)],
                SIDES,
                Transform::IDENTITY,
            );
            back
        }
    }
}
