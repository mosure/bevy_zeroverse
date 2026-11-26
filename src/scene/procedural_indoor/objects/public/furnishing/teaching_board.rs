//! Writing leaves, hinges and pen ledges fitted inside the admitted stand.
use super::*;
use crate::scene::procedural_indoor::{materials::variants::CONTENT_COUNT, objects::presentation};
mod mesh;
#[cfg(test)]
mod tests;

pub const INTRODUCED: u32 = 251;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arrangement {
    Single,
    FixedBank,
    Folding,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Leaf {
    /// Location of the center in the fully open, coplanar board.
    pub flat_x_m: f32,
    pub width_m: f32,
    pub front_slot: usize,
    pub front_u: [f32; 2],
    pub reverse_slot: Option<usize>,
    pub hinge: Option<Hinge>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Hinge {
    pub axis_x_m: f32,
    /// Signed yaw about the real pin. Outer edges swing toward the audience.
    pub angle_rad: f32,
    /// Signed forward travel limit for this fitted leaf, measured from flat.
    pub stop_rad: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Ledge {
    /// Usable tray in the level stand frame, with X across and Z toward the class.
    pub lo_m: Vec2,
    pub hi_m: Vec2,
    pub top_m: f32,
    pub lip_m: f32,
    pub contents: Vec<presentation::TrayItem>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FootPads {
    pub radius_m: f32,
    pub height_m: f32,
    /// Distance from the stand center to either pad along a runner.
    pub center_z_m: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Installation {
    pub arrangement: Arrangement,
    pub continuous_front: bool,
    pub border_m: f32,
    pub recess_m: f32,
    pub bevel_m: f32,
    pub rear_stock: Surface,
    pub carrier_x_m: f32,
    pub carrier_width_m: f32,
    pub post_center_x_m: f32,
    pub pin_radius_m: f32,
    pub hinge_height_m: f32,
    pub hinge_inset_fraction: f32,
    pub leaves: Vec<Leaf>,
    pub ledge: Option<Ledge>,
    pub foot_pads: Option<FootPads>,
}

impl Installation {
    pub(super) fn fit(o: &IndoorObject, p: &TeachingProgram, size: Vec3, pose: Transform) -> Self {
        let mut product = stream(o.seed, u64::from_le_bytes(*b"BoardSet"));
        let mut use_state = stream(o.seed, u64::from_le_bytes(*b"BoardUse"));
        let mut ink = stream(o.seed, u64::from_le_bytes(*b"LeafInks"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"BoardBak"),
        );
        let arrangement = match product.random_range(0..100) {
            _ if size.x < 0.82 => Arrangement::Single,
            0..45 => Arrangement::Single,
            45..65 => Arrangement::FixedBank,
            _ => Arrangement::Folding,
        };
        let count = match arrangement {
            Arrangement::Single => 1,
            _ if size.x > 1.25 && product.random_bool(0.50) => 3,
            _ => 2,
        };
        let pin = product.random_range(0.004..0.008);
        let gap = if arrangement == Arrangement::Folding {
            2. * pin + product.random_range(0.0015..0.004)
        } else {
            product.random_range(0.006..0.024)
        };
        let stock_width = size.x - (count - 1) as f32 * gap;
        let fixed = if count == 2 {
            usize::from(product.random_bool(0.5))
        } else {
            count / 2
        };
        let mut widths = vec![1.; count];
        if arrangement == Arrangement::Folding {
            widths[fixed] = product.random_range(1.32..2.05);
        }
        for w in &mut widths {
            *w *= product.random_range(0.90..1.10);
        }
        let total: f32 = widths.iter().sum();
        for w in &mut widths {
            *w *= stock_width / total;
        }
        let narrowest = widths.iter().copied().fold(size.y, f32::min);
        let border_m = product.random_range(0.011_f32..0.036).min(narrowest * 0.08);
        let recess_m = product.random_range(0.0015_f32..0.0045).min(size.z * 0.20);
        let bevel_m = product
            .random_range(0.0006..0.0021_f32)
            .min(recess_m * 0.75);
        let rear_stock =
            [Surface::Metal, Surface::Wood, Surface::Plastic][finish.random_range(0..3)];
        let continuous_front = count > 1 && ink.random_bool(0.44);
        let slots = rand::seq::index::sample(&mut ink, CONTENT_COUNT, 2 * count).into_vec();
        let writable_rear = product.random_bool(0.53);
        // Budget for the tilted top, hinge projection and the whole rotating leaf.
        // Every angle between flat and the sampled stop stays inside this depth.
        let tilt_z = (pose.rotation * Vec3::Y).z.abs() * size.y * 0.5;
        let pivot_z = size.z * 0.5 + pin * 0.70;
        let available_z = (o.size.z * 0.5 - pose.translation.z - tilt_z - pivot_z - 0.006).max(0.);
        let chart_width = stock_width - 2. * border_m * count as f32;
        let mut x = -size.x * 0.5;
        let mut chart_x = 0.;
        let mut leaves = Vec::with_capacity(count);
        for (i, &width) in widths.iter().enumerate() {
            let flat_x_m = x + width * 0.5;
            let hinge = (arrangement == Arrangement::Folding && i != fixed).then(|| {
                let side = if i < fixed { -1. } else { 1. };
                let limit = (available_z / (width + gap * 0.5))
                    .clamp(0., 1.)
                    .asin()
                    .min(1.18);
                let opened = if use_state.random_bool(0.17) {
                    0.
                } else {
                    limit * use_state.random_range(0.12..1.)
                };
                Hinge {
                    axis_x_m: flat_x_m - side * (width + gap) * 0.5,
                    angle_rad: -side * opened,
                    stop_rad: -side * limit,
                }
            });
            let start = chart_x;
            chart_x += width - 2. * border_m;
            let front_u = if continuous_front {
                [
                    start / chart_width,
                    if i + 1 == count {
                        1.
                    } else {
                        chart_x / chart_width
                    },
                ]
            } else {
                [0., 1.]
            };
            leaves.push(Leaf {
                flat_x_m,
                width_m: width,
                front_slot: if continuous_front {
                    slots[0]
                } else {
                    slots[2 * i]
                },
                front_u,
                reverse_slot: writable_rear.then_some(slots[2 * i + 1]),
                hinge,
            });
            x += width + gap;
        }
        let (carrier_x_m, carrier_width_m) = if arrangement == Arrangement::Folding {
            (leaves[fixed].flat_x_m, widths[fixed])
        } else {
            (0., size.x)
        };
        // The fixed leaf carries the feet. Move their center toward the
        // combined leaf mass, keeping both post heads on the carrier rail.
        let post = o.size.x.min(o.size.z) * 0.022;
        let post_travel = (carrier_width_m * (1. - p.post_span) * 0.5 - post).max(0.);
        let post_center_x_m = if arrangement == Arrangement::Folding {
            let loaded_x = leaves
                .iter()
                .map(|leaf| {
                    let mut center = leaf.flat_x_m;
                    if let Some(h) = leaf.hinge {
                        let (sin, cos) = h.angle_rad.sin_cos();
                        center = h.axis_x_m + cos * (center - h.axis_x_m) - sin * pivot_z;
                    }
                    center * leaf.width_m
                })
                .sum::<f32>()
                / stock_width;
            loaded_x.clamp(carrier_x_m - post_travel, carrier_x_m + post_travel)
        } else {
            0.
        };
        let ledge = p.tray.then(|| {
            let center = pose.transform_point(Vec3::new(
                carrier_x_m,
                -size.y * 0.5 - post * 0.45,
                size.z.max(post) * 0.5,
            ));
            let width = carrier_width_m * product.random_range(0.44..0.94);
            let x = center.x + (carrier_width_m - width) * product.random_range(-0.43..0.43);
            let z = center.z - 0.001;
            let lo_m = Vec2::new(x - width * 0.5, z);
            let hi_m = Vec2::new(x + width * 0.5, z + product.random_range(0.036..0.079));
            Ledge {
                lo_m,
                hi_m,
                top_m: center.y,
                lip_m: product.random_range(0.006..0.015),
                contents: presentation::contents::tray_items(
                    o,
                    (lo_m + Vec2::new(0.002, 0.005), hi_m - Vec2::splat(0.003)),
                ),
            }
        });
        Self {
            arrangement,
            continuous_front,
            border_m,
            recess_m,
            bevel_m,
            rear_stock,
            carrier_x_m,
            carrier_width_m,
            post_center_x_m,
            pin_radius_m: pin,
            hinge_height_m: product.random_range(0.037..0.069),
            hinge_inset_fraction: product.random_range(0.26..0.41),
            leaves,
            ledge,
            foot_pads: (p.structure != 3).then(|| {
                // Seat both ends of each runner on its flat underside, inside
                // the bevel. The rubber skirt remains visible from floor level.
                let radius_m = (post * 1.4 - 0.004).max(0.002) * product.random_range(0.70..0.98);
                FootPads {
                    radius_m,
                    height_m: product.random_range(0.0045..0.012),
                    center_z_m: (o.size.z * p.foot_depth * 0.5
                        - radius_m
                        - product.random_range(0.008..0.024))
                    .max(0.),
                }
            }),
        }
    }

    fn pivot(&self, h: Hinge, depth: f32) -> Vec3 {
        Vec3::new(h.axis_x_m, 0., depth * 0.5 + self.pin_radius_m * 0.70)
    }

    /// Hinge rotation applied in the unfolded board frame, before stand tilt.
    fn swing(&self, hinge: Option<Hinge>, depth: f32) -> Transform {
        let Some(h) = hinge else {
            return Transform::IDENTITY;
        };
        let rotation = Quat::from_rotation_y(h.angle_rad);
        let pivot = self.pivot(h, depth);
        Transform::from_translation(pivot - rotation * pivot).with_rotation(rotation)
    }
}
