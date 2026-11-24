//! Fitted optical hardware in handset-local X/Z coordinates (metres).
use super::*;
use crate::scene::procedural_indoor::materials::handset;
use rand::seq::SliceRandom;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontMount {
    Bezel,
    Punch,
    SensorPill,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct FrontCamera {
    pub mount: FrontMount,
    pub center: Vec2,
    pub half_span: Vec2,
}

impl FrontCamera {
    pub(super) fn sample(seed: u64, p: &PhoneProgram) -> Self {
        use crate::scene::procedural_indoor::geometry::housing_front_z;
        let span = p.span;
        let screen = p.screen_span;
        let screen_z = p.screen_z;
        let rim = p.rim_width;
        let mut r = stream(seed, u64::from_le_bytes(*b"FrontCam"));
        let radius = span.x * r.random_range(0.014..0.026);
        let top = screen_z - screen.y * 0.5;
        // Keep the complete aperture between the display and the inside of the bumper.
        let bezel_edge = -span.y * 0.49 + rim + 0.0004;
        let available = top - bezel_edge;
        if available > radius * 2.9 && r.random_bool(0.34) {
            let x = screen.x * r.random_range(-0.14..0.14);
            let bezel_edge = housing_front_z(
                span * 0.49 - Vec2::splat(rim),
                p.roundness,
                x.abs() + radius,
            ) + 0.0004;
            if top - bezel_edge > radius * 2.3 {
                return Self {
                    mount: FrontMount::Bezel,
                    center: Vec2::new(x, (bezel_edge + top) * 0.5),
                    half_span: Vec2::splat(radius),
                };
            }
        }
        let pill = r.random_bool(0.32);
        let x = if !pill && r.random_bool(0.44) {
            (if r.random_bool(0.5) { -1. } else { 1. }) * r.random_range(0.25..0.34)
        } else {
            r.random_range(-0.018..0.018)
        };
        let half_span = Vec2::new(
            if pill {
                radius * r.random_range(2.5..4.6)
            } else {
                radius
            },
            radius,
        );
        let x = screen.x * x;
        let edge = screen_z + housing_front_z(screen * 0.5, p.roundness, x.abs() + half_span.x);
        Self {
            mount: if pill {
                FrontMount::SensorPill
            } else {
                FrontMount::Punch
            },
            half_span,
            center: Vec2::new(
                x,
                (top + radius * r.random_range(2.6..4.4)).max(edge + radius * 1.2),
            ),
        }
    }

    /// Replace a housing's standalone top fan with two exactly adjoining materials.
    /// Both convex outlines start at +X. Merging their edge directions walks
    /// corresponding support points: at an inner edge the outer vertex is beyond
    /// its supporting line, and at an outer edge the inner vertex is inside it.
    /// This works even for small off-centre apertures near a rounded corner.
    pub(super) fn inlay(self, cap: &mut Geometry, lens: &mut Geometry) {
        const SIDES: usize = 16;
        static SHAPES: OnceLock<[[Vec2; SIDES]; 2]> = OnceLock::new();
        let shapes = SHAPES.get_or_init(|| {
            [1., 0.58].map(|power| {
                std::array::from_fn(|i| {
                    let (z, x) = (TAU * i as f32 / SIDES as f32).sin_cos();
                    let coord = |v: f32| {
                        if v.abs() < 1e-6 {
                            0.
                        } else {
                            v.signum() * v.abs().powf(power)
                        }
                    };
                    Vec2::new(coord(x), coord(z))
                })
            })
        });
        let inner = shapes[usize::from(self.mount == FrontMount::SensorPill)]
            .map(|unit| self.center + unit * self.half_span);
        let y = cap.positions[0][1];
        let outer_len = cap.positions.len() - 1;
        // The old centre becomes the first vertex of the aperture.
        let point = |p: Vec2| [p.x, y, p.y];
        cap.positions[0] = point(inner[0]);
        cap.uvs[0] = inner[0].to_array();
        let base = lens.positions.len() as u32;
        for (i, p) in inner.into_iter().enumerate() {
            if i != 0 {
                cap.positions.push(point(p));
                cap.normals.push(Vec3::Y.to_array());
                cap.uvs.push(p.to_array());
            }
            lens.positions.push(point(p));
            lens.normals.push(Vec3::Y.to_array());
            lens.uvs.push(p.to_array());
            if i >= 2 {
                lens.indices
                    .extend([base, base + i as u32, base + i as u32 - 1]);
            }
        }
        let outer_index = |i: usize| (1 + i % outer_len) as u32;
        let inner_index = |i: usize| match i % SIDES {
            0 => 0,
            j => (outer_len + j) as u32,
        };
        cap.indices.clear();
        let (mut i, mut j) = (0, 0);
        while i < outer_len || j < SIDES {
            let index = outer_index(i);
            let next = outer_index(i + 1);
            let edge = (Vec3::from_array(cap.positions[next as usize])
                - Vec3::from_array(cap.positions[index as usize]))
            .xz();
            let cut_edge = inner[(j + 1) % SIDES] - inner[j % SIDES];
            let advance_outer = j == SIDES || (i < outer_len && edge.perp_dot(cut_edge) >= 0.);
            let to = if advance_outer {
                next
            } else {
                inner_index(j + 1)
            };
            cap.indices.extend([index, inner_index(j), to]);
            if advance_outer {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
}

pub(super) fn rear_radii(seed: u64, count: usize, available: f32) -> (f32, [f32; 4]) {
    let mut r = stream(seed, u64::from_le_bytes(*b"LensPack"));
    let cap = if r.random_bool(0.09) {
        0.009..0.0125
    } else {
        0.0048..0.009
    };
    let main = available.min(r.random_range(cap));
    let primary = r.random_range(0..count);
    let matched = r.random_bool(0.30);
    let radii = std::array::from_fn(|i| {
        if i >= count {
            0.
        } else if i == primary || matched {
            main
        } else {
            main * r.random_range(0.57..0.96)
        }
    });
    (main, radii)
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RearFlash {
    pub center: Vec2,
    pub radius: f32,
    pub on_island: bool,
}

pub(super) fn rear_flash(seed: u64, p: &PhoneProgram) -> Option<RearFlash> {
    let mut r = stream(seed, u64::from_le_bytes(*b"Rear LED"));
    if r.random_bool(0.06) {
        return None;
    }
    let radius = (p.span.x * r.random_range(0.020..0.039)).min(p.lens_radius * 0.85);
    let covered = |offset: Vec2, extent: Vec2, exponent: f32| {
        let q = (offset.abs() + Vec2::splat(radius * 1.18)) / extent;
        q.x.powf(exponent) + q.y.powf(exponent) < 0.94
    };
    let center = p.island_center;
    let mut corners = [
        Vec2::ONE,
        Vec2::new(-1., 1.),
        -Vec2::ONE,
        Vec2::new(1., -1.),
    ];
    corners.shuffle(&mut r);
    let inset = p.island_span * r.random_range(0.23..0.33);
    let clearance = p.lens_radius + radius * 1.45;
    for dir in corners {
        let at = center + dir * inset;
        if covered(at - center, p.island_span * 0.48, p.island_roundness)
            && p.lens_centers[..p.lens_count]
                .iter()
                .all(|c| c.distance_squared(at) > clearance * clearance)
        {
            return Some(RearFlash {
                center: at,
                radius,
                on_island: true,
            });
        }
    }
    let mut sides = [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y];
    sides.shuffle(&mut r);
    let offset = p.island_span * 0.5 + Vec2::splat(radius * r.random_range(1.55..2.35));
    for direction in sides {
        let at = center + direction * offset;
        if covered(at, p.span * 0.47, p.roundness) {
            return Some(RearFlash {
                center: at,
                radius,
                on_island: false,
            });
        }
    }
    None
}

pub(super) fn rear(a: &mut Assembly, p: &PhoneProgram, label: &str) {
    let face_top = p.body_height + p.face_height;
    let top = face_top + p.island_height;
    let rings = [
        (face_top - p.face_height * 0.2, p.island_span * 0.5),
        (top, p.island_span * 0.48),
    ];
    let tf = Transform::from_xyz(p.island_center.x, 0., p.island_center.y);
    if p.glazed_island {
        let mut cover = Geometry::default();
        a.part(p.shell, label)
            .housing_with_top(&mut cover, &rings, p.island_roundness, tf);
        a.part(Surface::Ink, handset::COVER).append(cover);
    } else {
        a.part(p.shell, label)
            .housing(&rings, p.island_roundness, tf);
    }
    let h = p.face_height;
    for (center, radius) in p.lens_centers[..p.lens_count].iter().zip(p.lens_radii) {
        let throat = radius * p.optical_fraction;
        let at = Transform::from_xyz(center.x, top, center.y);
        // The turned rim and shallow crowned glass meet exactly at the throat.
        // Only exposed sections are meshed, including the recess around the optic.
        a.part(Surface::Metal, label).lathe(
            &[
                (radius, -h * 0.1),
                (radius * 0.94, h),
                (throat, h),
                (throat, h * 0.34),
            ],
            16,
            at,
        );
        a.part(Surface::Ink, handset::LENS).lathe(
            &[
                (throat, h * 0.34),
                (throat * 0.78, h * 0.55),
                (0., h * 0.63),
            ],
            16,
            at,
        );
    }
    if let Some(flash) = p.rear_flash {
        let y = if flash.on_island { top } else { face_top };
        a.part(Surface::Ink, handset::DIFFUSER).lathe(
            &[
                (flash.radius, -h * 0.10),
                (flash.radius * 0.93, h * 0.43),
                (flash.radius * 0.74, h * 0.60),
                (0., h * 0.66),
            ],
            12,
            Transform::from_xyz(flash.center.x, y, flash.center.y),
        );
    }
}
