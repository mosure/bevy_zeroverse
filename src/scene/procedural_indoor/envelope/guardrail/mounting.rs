//! Glazed and opaque guard infills are gripped on both faces. Each installation
//! uses one hardware stock, cut to its actual bays and stair rake.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
mod section;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Retainer {
    EdgeClips {
        height_m: f32,
        lower_fraction: f32,
        upper_fraction: f32,
    },
    PostChannel {
        end_setback_m: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PanelMount {
    pub retainer: Retainer,
    /// Distance from post face to the cut edge of the panel.
    pub edge_clearance_m: f32,
    /// Nominal depth of metal overlapping each face of the panel.
    pub bite_m: f32,
    pub jaw_wall_m: f32,
    pub gasket_m: f32,
    pub finish: Surface,
}

impl PanelMount {
    pub(super) fn sample(section: &GuardSection) -> Option<Self> {
        let clip_chance = match section.infill {
            Infill::Glass => 0.68,
            Infill::Solid => 0.32,
            _ => return None,
        };
        let mut r = stream(section.seed, u64::from_le_bytes(*b"RailGrip"));
        let (retainer, clearance, bite) = if r.random_bool(clip_chance) {
            (
                Retainer::EdgeClips {
                    height_m: r.random_range(0.038..0.075),
                    lower_fraction: r.random_range(0.11..0.27),
                    upper_fraction: r.random_range(0.73..0.89),
                },
                0.009..0.020,
                0.018..0.045,
            )
        } else {
            (
                Retainer::PostChannel {
                    end_setback_m: r.random_range(0.007..0.035),
                },
                0.006..0.015,
                0.012..0.028,
            )
        };
        Some(Self {
            retainer,
            edge_clearance_m: r.random_range(clearance),
            bite_m: r.random_range(bite),
            jaw_wall_m: r.random_range(0.0028..0.0058),
            gasket_m: r.random_range(0.0014..0.0036),
            finish: if r.random_bool(0.62) {
                Surface::Metal
            } else {
                Surface::Chrome
            },
        })
    }

    pub(super) fn validate(&self, s: &GuardSection) -> Result<(), String> {
        let usable = match self.retainer {
            Retainer::EdgeClips {
                height_m,
                lower_fraction,
                upper_fraction,
            } => {
                (0.035..=0.080).contains(&height_m)
                    && (0.10..=0.30).contains(&lower_fraction)
                    && (0.70..=0.90).contains(&upper_fraction)
            }
            Retainer::PostChannel { end_setback_m } => (0.006..=0.04).contains(&end_setback_m),
        };
        if usable
            && matches!(s.infill, Infill::Glass | Infill::Solid)
            && matches!(self.finish, Surface::Metal | Surface::Chrome)
            && (0.005..=0.021).contains(&self.edge_clearance_m)
            && (0.01..=0.05).contains(&self.bite_m)
            && (0.0025..=0.006).contains(&self.jaw_wall_m)
            && (0.001..=0.004).contains(&self.gasket_m)
        {
            Ok(())
        } else {
            Err("invalid guard panel mount".into())
        }
    }

    pub(super) fn bay(
        &self,
        a: &mut Assembly,
        s: &GuardSection,
        start: Vec3,
        end: Vec3,
        rise: f32,
        height: f32,
    ) {
        let span = end - start;
        let length = span.xz().length();
        let inset = s.post_width * 0.5 + self.edge_clearance_m;
        if inset >= 0.45 * length {
            return;
        }
        let margin = span * (inset / length);
        let (p, q) = (start + margin, end - margin);
        let along = span.with_y(0.) / length;
        let normal = along.cross(Vec3::Y);
        let bottom = rise + s.bottom_gap;
        let rail_depth = if s.round_handrail {
            s.handrail_width
        } else {
            s.handrail_depth
        };
        let slope = span.y / length;
        // Vertical distance to the underside of a sloping rail at fixed XZ.
        let top = height - 0.5 * rail_depth * 1_f32.hypot(slope) - 0.015;
        let surface = if s.infill == Infill::Glass {
            Surface::GlassInterior
        } else {
            s.panel
        };
        let panel = a.part(surface, "other_structure");
        let face = [
            p + Vec3::Y * bottom,
            q + Vec3::Y * bottom,
            q + Vec3::Y * top,
            p + Vec3::Y * top,
        ];
        let half = normal * (s.panel_thickness * 0.5);
        let front = face.map(|v| v + half);
        let back = face.map(|v| v - half);
        let cut = surface.is_cut_timber().then(|| {
            let mut r = stream(
                s.seed ^ u64::from(start.x.to_bits()),
                u64::from(start.z.to_bits()) ^ 0x474c415a435554,
            );
            Vec2::new(r.random_range(0.0..4.0), r.random_range(0.0..4.0))
        });
        // Face veneers run vertically with a different stock cut in each bay.
        panel.stock_quad(front, normal, Vec3::Y, cut);
        panel.stock_quad(back, -normal, Vec3::Y, cut);
        for i in 0..4 {
            let j = (i + 1) % 4;
            panel.metric_quad(
                [front[i], front[j], back[j], back[i]],
                (face[j] - face[i]).cross(normal),
            );
        }
        let bite = self.bite_m.min((length - 2. * inset) * 0.24);
        let stock = section::Section::fit(self, s, bite);
        for (lining, surface) in [(false, self.finish), (true, Surface::Rubber)] {
            let mesh = a.part(surface, "other_structure");
            for (edge, direction, rake) in [(p, along, slope), (q, -along, -slope)] {
                let placement = section::Edge {
                    edge,
                    along: direction,
                    normal: direction.cross(Vec3::Y),
                    rake: if matches!(self.retainer, Retainer::PostChannel { .. }) {
                        rake
                    } else {
                        0.
                    },
                };
                match self.retainer {
                    Retainer::EdgeClips {
                        height_m,
                        lower_fraction,
                        upper_fraction,
                    } => {
                        for fraction in [lower_fraction, upper_fraction] {
                            let center = bottom + (top - bottom) * fraction;
                            placement.extrude(
                                mesh,
                                &stock,
                                (center - height_m * 0.5, center + height_m * 0.5),
                                lining,
                            );
                        }
                    }
                    Retainer::PostChannel { end_setback_m } => {
                        placement.extrude(
                            mesh,
                            &stock,
                            (bottom + end_setback_m, top - end_setback_m),
                            lining,
                        );
                    }
                }
            }
        }
    }
}
