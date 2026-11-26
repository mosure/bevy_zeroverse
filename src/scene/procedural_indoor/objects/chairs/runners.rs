//! Bent side frames for visitor and dining chairs. Each side has two glides
//! seated on its straight floor run and shares its opposite side's tooling.
use super::*;

use super::laminate;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 224;
const ARC_STEPS: usize = 5;
const PATH_NODES: usize = 2 + 2 * (ARC_STEPS + 1);

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerSection {
    RoundTube,
    BentPlywood,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize)]
pub struct RunnerStock {
    pub section: RunnerSection,
    pub table_order: bool,
    pub width_m: f32,
    pub thickness_m: f32,
    pub bend_radius_m: f32,
    pub rail_surface: Surface,
    pub glide_surface: Surface,
    pub glide_length_m: f32,
    pub glide_height_m: f32,
    pub glide_inset_fraction: f32,
    pub side_span_fraction: f32,
    pub floor_front_fraction: f32,
    pub floor_rear_fraction: f32,
    pub mount_front_fraction: f32,
    pub mount_rear_fraction: f32,
}

impl ChairProgram {
    pub fn runner_stock(&self, o: &IndoorObject, version: u32) -> Option<RunnerStock> {
        if version < INTRODUCED
            || o.kind != ObjectKind::Chair
            || !matches!(o.variant, 2..=4)
            || !matches!(self.base_style, 5 | 6)
        {
            return None;
        }
        let order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| stream(o.seed, u64::from_le_bytes(*b"RailOrdr")).random_bool(0.79));
        let lot = order.map_or(o.seed, |(finish, table)| {
            finish ^ (table as u64).rotate_left(19) ^ u64::from(o.variant).rotate_left(45)
        });
        let mut r = stream(
            lot,
            u64::from_le_bytes(*b"SeatRail") ^ u64::from(self.base_style),
        );
        let timber = if o.variant == 3 {
            r.random_bool(0.81)
        } else {
            r.random_bool(0.14)
        };
        let (section, width_m, thickness_m, bend_radius_m, rail_surface) = if timber {
            let thickness = if self.base_style == 6 {
                r.random_range(0.024..0.036)
            } else {
                r.random_range(0.016..0.027)
            };
            (
                RunnerSection::BentPlywood,
                r.random_range(0.033..0.055),
                thickness,
                r.random_range(0.073..0.113),
                Surface::Wood,
            )
        } else {
            let diameter = if self.base_style == 6 {
                r.random_range(0.021..0.031)
            } else {
                r.random_range(0.016..0.027)
            };
            let finish = if self.frame_surface == Surface::WoodEdge {
                Surface::Metal
            } else {
                self.frame_surface
            };
            (
                RunnerSection::RoundTube,
                diameter,
                diameter,
                r.random_range(0.034..0.084),
                finish,
            )
        };
        Some(RunnerStock {
            section,
            table_order: order.is_some(),
            width_m,
            thickness_m,
            bend_radius_m,
            rail_surface,
            glide_surface: if r.random_bool(0.30) {
                Surface::Plastic
            } else {
                Surface::Rubber
            },
            glide_length_m: r.random_range(0.032..0.070),
            glide_height_m: r.random_range(0.003..0.009),
            glide_inset_fraction: r.random_range(0.025..0.14),
            side_span_fraction: if self.formed_seat.is_some() {
                0.43
            } else {
                r.random_range(0.34..0.43)
            },
            floor_front_fraction: r.random_range(0.36..0.48),
            floor_rear_fraction: r.random_range(0.37..0.49),
            mount_front_fraction: if self.base_style == 6 {
                r.random_range(0.39..0.46)
            } else {
                r.random_range(0.30..0.445)
            },
            mount_rear_fraction: r.random_range(0.356..0.44),
        })
    }
}

struct SideFrame {
    path: [Vec3; PATH_NODES],
    glides_z: [f32; 2],
    glide_length: f32,
}

impl RunnerStock {
    fn fit(&self, style: u8, depth: f32, mounting: f32) -> SideFrame {
        let floor = self.glide_height_m + self.thickness_m * 0.5;
        // Intersect the underside of the pan/carrier. The sled terminates
        // against it. The cantilever bears on it along its upper surface.
        let top = if style == 5 {
            mounting + 0.003
        } else {
            mounting + 0.003 - self.thickness_m * 0.5
        };
        let joint = |y, fraction| Vec3::new(0., y, depth * fraction);
        let front = joint(floor, -self.floor_front_fraction);
        let rear = joint(floor, self.floor_rear_fraction);
        let upper_front = joint(top, -self.mount_front_fraction);
        let upper_rear = joint(top, self.mount_rear_fraction);
        let corners = if style == 5 {
            [upper_front, front, rear, upper_rear]
        } else {
            [rear, front, upper_front, upper_rear]
        };
        // The forward crossbar on a formed shell bears at -0.25 depth.
        // Preserve straight stock beneath it even in a shallow cantilever.
        let upper_setback = if style == 6 {
            (-depth * 0.25 - upper_front.z - 0.002).max(0.)
        } else {
            f32::INFINITY
        };
        let path = rounded_side(corners, self.bend_radius_m, upper_setback);
        let (z0, z1) = if style == 5 {
            (path[ARC_STEPS + 1].z, path[ARC_STEPS + 2].z)
        } else {
            (path[1].z, path[0].z)
        };
        let straight = z1 - z0;
        let length = self.glide_length_m.min(straight * 0.30);
        let margin = length * 0.5 + self.glide_inset_fraction * straight;
        SideFrame {
            path,
            glides_z: [z0 + margin, z1 - margin],
            glide_length: length,
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        p: &ChairProgram,
        width: f32,
        depth: f32,
        mounting: f32,
        seed: u64,
    ) {
        let fitted = self.fit(p.base_style, depth, mounting);
        // A cut carries its own metric grain phase around both bends.
        let mut cuts = (self.section == RunnerSection::BentPlywood)
            .then(|| TimberCuts::new(seed, u64::from_le_bytes(*b"RailCuts")));
        for side in [-1., 1.] {
            let x = width * self.side_span_fraction * side;
            let path = fitted.path.map(|p| p.with_x(x));
            if let Some(cuts) = &mut cuts {
                laminate::build(a, &path, self.width_m, self.thickness_m, cuts.origin());
            } else {
                a.part(self.rail_surface, "chair")
                    .tube(&path, self.thickness_m * 0.5, 10);
            }
            for z in fitted.glides_z {
                let g = a.part(self.glide_surface, "chair");
                if self.section == RunnerSection::RoundTube {
                    tube_glide(
                        g,
                        x,
                        z,
                        fitted.glide_length,
                        self.thickness_m * 0.5,
                        self.glide_height_m,
                    );
                } else {
                    let height = self.glide_height_m + 0.0008;
                    g.cuboid(
                        Vec3::new(self.width_m * 0.91, height, fitted.glide_length),
                        0.0013,
                        Transform::from_xyz(x, height * 0.5, z),
                    );
                }
            }
        }
    }
}

/// Two circular bends joined by straight stock. Limit tangent setbacks on short
/// seat depths so adjacent bends leave a load-bearing length between them.
fn rounded_side(c: [Vec3; 4], radius: f32, upper_setback: f32) -> [Vec3; PATH_NODES] {
    let mut path = [Vec3::ZERO; PATH_NODES];
    path[0] = c[0];
    path[PATH_NODES - 1] = c[3];
    for i in 1..=2 {
        let a = c[i] - c[i - 1];
        let b = c[i + 1] - c[i];
        let (incoming, outgoing) = (a.normalize(), b.normalize());
        let cos = incoming.dot(outgoing);
        let half_tan = ((1. - cos) / (1. + cos)).sqrt();
        let trim = (radius * half_tan).min(0.42 * a.length().min(b.length()));
        let trim = if i == 2 {
            trim.min(upper_setback)
        } else {
            trim
        };
        let from = c[i] - incoming * trim;
        let to = c[i] + outgoing * trim;
        let normal = outgoing.reject_from_normalized(incoming).normalize();
        let center = from + normal * (trim / half_tan);
        let j = 1 + (i - 1) * (ARC_STEPS + 1);
        path[j] = from;
        path[j + ARC_STEPS] = to;
        for k in 1..ARC_STEPS {
            let offset = from.lerp(to, k as f32 / ARC_STEPS as f32) - center;
            path[j + k] = center + offset.normalize() * (trim / half_tan);
        }
    }
    path
}

/// A longitudinal clip with a concave saddle. Its flat sole meets the floor,
/// and its raised wings engage the metal tube on either side of bottom centre.
fn tube_glide(g: &mut Geometry, x: f32, z: f32, length: f32, radius: f32, sole: f32) {
    let strips = [-1.07_f32, -0.86, -0.50, 0., 0.50, 0.86, 1.07];
    let section = strips.map(|u| {
        let h = if u.abs() > 1. {
            0.69
        } else {
            1. - (1. - u * u).sqrt()
        };
        let edge = x + u * radius;
        let roof = sole + radius * h + 0.0009;
        let (front, rear) = (z - length * 0.5, z + length * 0.5);
        [
            Vec3::new(edge, 0., front),
            Vec3::new(edge, roof, front),
            Vec3::new(edge, roof, rear),
            Vec3::new(edge, 0., rear),
        ]
    });
    for pair in section.windows(2) {
        let [a, b, c, d] = pair[0];
        let [e, f, h, i] = pair[1];
        g.metric_quad([a, e, f, b], -Vec3::Z);
        g.metric_quad([b, f, h, c], Vec3::Y);
        g.metric_quad([c, h, i, d], Vec3::Z);
        g.metric_quad([d, i, e, a], -Vec3::Y);
    }
    let right = section[section.len() - 1];
    g.metric_quad(section[0], -Vec3::X);
    g.metric_quad(right, Vec3::X);
}
