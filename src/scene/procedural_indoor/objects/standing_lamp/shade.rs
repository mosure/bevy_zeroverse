//! Floor shades are cut around one perimeter. The lining, rolled hems and
//! attachment wires all meet the same polygon, including on oblique faces.
use super::*;
use crate::scene::procedural_indoor::geometry::RevolvedOutline;
use std::sync::LazyLock;
#[cfg(test)]
mod tests;

static CIRCLE: LazyLock<RevolvedOutline> = LazyLock::new(|| RevolvedOutline::round(32, 0., 0));

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ShadeSection {
    Round,
    Oval { minor_ratio: f32 },
    SoftRectangle { minor_ratio: f32, exponent: f32 },
    Panelled { minor_ratio: f32, panels: u32 },
}

impl ShadeSection {
    fn outline(self) -> Option<RevolvedOutline> {
        let (sides, exponent, minor) = match self {
            Self::Round => return None,
            Self::Oval { minor_ratio } => (32, 2., minor_ratio),
            Self::SoftRectangle {
                minor_ratio,
                exponent,
            } => (32, exponent, minor_ratio),
            Self::Panelled {
                panels,
                minor_ratio,
            } => (panels, 2., minor_ratio),
        };
        let section = RevolvedOutline::superellipse(sides, exponent, Vec2::new(1., minor));
        Some(if matches!(self, Self::Panelled { .. }) {
            section.faceted()
        } else {
            section
        })
    }

    /// Lower bound on distance to any face of the unit cutting pattern.
    fn inscribed(self) -> f32 {
        match self {
            Self::Round => (PI / 32.).cos(),
            Self::Oval { minor_ratio } | Self::SoftRectangle { minor_ratio, .. } => {
                minor_ratio * (PI / 32.).cos()
            }
            Self::Panelled {
                minor_ratio,
                panels,
            } => minor_ratio * (PI / panels as f32).cos(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fitter {
    SocketRing,
    Harp,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ShadeCut {
    pub section: ShadeSection,
    pub rotation_rad: f32,
    pub lining: Option<Surface>,
    pub fitter: Fitter,
    pub spokes: u32,
    pub spoke_phase_turns: f32,
}

impl ShadeCut {
    pub fn sample(o: &IndoorObject, p: &LampProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ShadeCut"));
        let section = if p.shade.shape >= 3 {
            ShadeSection::Round // Pleats and wire cages have dedicated radial construction.
        } else if p.shade_sides < 32 {
            ShadeSection::Panelled {
                minor_ratio: r.random_range(0.74..=1.),
                panels: p.shade_sides as u32,
            }
        } else {
            match r.random_range(0..10) {
                0..=2 => ShadeSection::Round,
                3..=6 => ShadeSection::Oval {
                    minor_ratio: r.random_range(0.65..0.94),
                },
                _ => ShadeSection::SoftRectangle {
                    minor_ratio: r.random_range(0.67..0.98),
                    exponent: r.random_range(3.2..6.),
                },
            }
        };
        let lining = match (p.shade.shape, p.shade.finish) {
            (0..=2, Surface::FabricAlt | Surface::WoodEdge) if r.random_bool(0.72) => {
                Some(Surface::Paper)
            }
            (0..=2, surface) => Some(surface),
            (3, _) => Some(Surface::FabricAlt),
            _ => None,
        };
        let vertical_clearance =
            p.shade_height * 0.62 - luminaires::bulb_radii(&p.shade, p.bulb_radius).y;
        Self {
            section,
            lining,
            rotation_rad: if matches!(section, ShadeSection::Round) {
                0.
            } else {
                r.random_range(-PI..PI)
            },
            fitter: if vertical_clearance > 0.026 && r.random_bool(0.54) {
                Fitter::Harp
            } else {
                Fitter::SocketRing
            },
            spokes: r.random_range(3..=4),
            spoke_phase_turns: r.random_range(0.0..1.),
        }
    }

    pub(super) fn build(
        self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &LampProgram,
        nominal: f32,
        bulb: Vec3,
        root: Vec3,
    ) {
        let shaped = self.section.outline();
        let outline = shaped.as_ref().unwrap_or(&CIRCLE);
        let rotation = Quat::from_rotation_y(self.rotation_rad);
        let bottom = bulb.with_y(o.size.y - p.shade_height);
        // A rotated rounded rectangle can extend beyond its axial radii.
        let fit = |axis: Vec3, half: f32| {
            let (lo, hi) = outline.projected_range((rotation.inverse() * axis).xz());
            (half - bottom.dot(axis).abs()) / hi.max(-lo)
        };
        let widest = fit(Vec3::X, o.size.x * 0.5).min(fit(Vec3::Z, o.size.z * 0.5));
        let radius = nominal.min(widest / p.shade_taper.max(1.));
        let wall = luminaires::wall_thickness(radius);
        let bulb_width = luminaires::bulb_radii(&p.shade, p.bulb_radius).x;
        let inscribed = self.section.inscribed() * if p.shade.shape == 3 { 0.945 } else { 1. };
        let harp_width = bulb_width + 0.005;
        let taper = p
            .shade_taper
            .max(((harp_width + 0.002) / inscribed + wall) / radius);
        let stock = if p.shade.shape < 3 {
            luminaires::ShadeStock::Cut {
                outline,
                rotation,
                lining: self.lining.unwrap_or(p.shade.finish),
            }
        } else {
            luminaires::ShadeStock::Round
        };
        stock.build(a, "lamp", &p.shade, bottom, radius, p.shade_height, taper);

        let top_ring = self.fitter == Fitter::Harp;
        let (rim, y) = if top_ring {
            (
                luminaires::upper_radius(&p.shade, radius, taper),
                o.size.y - wall * 0.5,
            )
        } else {
            (radius, bottom.y + wall * 0.5)
        };
        let wire = (wall * inscribed * 0.38).min(0.0015);
        let seat = (rim + luminaires::shade_opening(&p.shade, radius, rim)) * 0.5;
        let metal = a.part(Surface::Metal, "lamp");
        let socket = root + Vec3::Y * (0.13 * p.bulb_radius);
        let hub = if top_ring {
            let hub = bulb.with_y(y);
            let spread = rotation * Vec3::X;
            for sign in [-1., 1.] {
                let side = spread * sign;
                let bend = (bulb.y + luminaires::bulb_radii(&p.shade, p.bulb_radius).y + 0.005)
                    .min(y - 0.009);
                metal.tube(
                    &[
                        socket + side * p.bulb_radius * 0.35,
                        socket + side * harp_width,
                        bulb.with_y(bend) + side * harp_width,
                        hub + side * harp_width * 0.48,
                        hub,
                    ],
                    wire,
                    6,
                );
            }
            // An enclosed washer/finial joins the harp and the upper spider.
            let hub_radius = (p.bulb_radius * 0.16).min(0.005);
            metal.lathe(
                &[
                    (0., -hub_radius),
                    (hub_radius, -hub_radius * 0.65),
                    (hub_radius, 0.),
                    (0., 0.),
                ],
                12,
                Transform::from_translation(hub.with_y(o.size.y)),
            );
            hub
        } else {
            socket
        };
        for i in 0..self.spokes {
            let point =
                outline.perimeter_point(self.spoke_phase_turns + i as f32 / self.spokes as f32);
            let radial = rotation * Vec3::new(point.x, 0., point.y);
            let end = bottom.with_y(y) + radial * seat;
            let start = if top_ring {
                hub
            } else {
                hub + radial.normalize() * (p.bulb_radius * 0.35)
            };
            let delta = end - start;
            metal.lathe(
                &[(wire, 0.), (wire, delta.length())],
                6,
                Transform::from_translation(start)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
            );
        }
    }
}
