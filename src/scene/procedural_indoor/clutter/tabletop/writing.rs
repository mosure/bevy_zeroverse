//! Sharpened, mechanical and capped writing tools with fitted tips and clips.
use super::*;
use crate::scene::procedural_indoor::geometry::{Geometry, ProfileJoints, RevolvedOutline};
use std::sync::LazyLock;

mod clip;
mod rest;

const MARKER_BARREL_RADIUS: f32 = 0.84;
const MARKER_SHOULDER_HEIGHT: f32 = 0.94;

#[derive(Clone, Debug, serde::Serialize)]
pub struct WritingProgram {
    pub mechanism: u8,
    pub length: f32,
    pub radius: f32,
    pub sides: u32,
    pub tip_fraction: f32,
    pub grip_fraction: f32,
    pub grip_radius: f32,
    pub tail_fraction: f32,
    pub eraser: bool,
    pub clip: bool,
    pub finish: Surface,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub enum WritingEnd {
    PencilPoint,
    Ballpoint,
    MarkerCap,
    PushButton,
    Eraser,
    BarrelEnd,
}
pub fn parameters(o: &IndoorObject) -> WritingProgram {
    stock(o.seed, o.finish_seed.unwrap_or(o.seed), o.size)
}

pub(in crate::scene::procedural_indoor::clutter) fn stock(
    seed: u64,
    finish: u64,
    size: Vec3,
) -> WritingProgram {
    let mut r = stream(seed, 0x57524954494e4750);
    let mut f = stream(finish, 0x57524954494e4746);
    let mechanism = r.random_range(0..3);
    let clip = mechanism != 0 && r.random_bool(0.8);
    let radius = size.x.min(size.y) * r.random_range(0.30..if clip { 0.37 } else { 0.45 });
    WritingProgram {
        mechanism,
        length: size.z * r.random_range(0.76..0.99),
        radius,
        sides: if mechanism == 0 {
            [6, 8, 16][r.random_range(0..3)]
        } else {
            16
        },
        tip_fraction: r.random_range(0.06..0.19),
        grip_fraction: r.random_range(0.14..0.28),
        grip_radius: r.random_range(0.78..1.0),
        tail_fraction: r.random_range(0.04..0.12),
        eraser: mechanism == 0 && r.random_bool(0.55),
        clip,
        finish: if mechanism == 0 && f.random_bool(0.25) {
            Surface::Wood
        } else if mechanism != 0 && f.random_bool(0.40) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}
// The same cross-section is used on both sides of every material joint.
static HEX: LazyLock<RevolvedOutline> = LazyLock::new(|| RevolvedOutline::polygon(6));
static OCT: LazyLock<RevolvedOutline> = LazyLock::new(|| RevolvedOutline::polygon(8));
static ROUND: LazyLock<RevolvedOutline> = LazyLock::new(|| RevolvedOutline::round(16, 0., 0));

impl WritingProgram {
    pub(crate) fn upper_end(&self, axis_up: bool) -> WritingEnd {
        match (self.mechanism, axis_up) {
            (0, true) => WritingEnd::PencilPoint,
            (0, false) if self.eraser => WritingEnd::Eraser,
            (1, true) => WritingEnd::Ballpoint,
            (1, false) => WritingEnd::PushButton,
            (2, false) => WritingEnd::MarkerCap,
            _ => WritingEnd::BarrelEnd,
        }
    }

    fn outline(&self) -> &'static RevolvedOutline {
        match self.sides {
            6 => &HEX,
            8 => &OCT,
            _ => &ROUND,
        }
    }

    /// Closed external meridian. The cap starts at local zero on a marker.
    /// Contiguous stocks meet on a ring and exposed steps have annular end faces.
    pub(super) fn profiles(&self, mut emit: impl FnMut(Surface, &[(f32, f32)])) {
        let (h, r) = (self.length, self.radius);
        if self.mechanism == 2 {
            let cap = self.cap_length();
            let barrel = r * MARKER_BARREL_RADIUS;
            emit(
                self.finish,
                &[(0., 0.), (r * 0.75, 0.), (r, cap * 0.07), (r, cap * 0.96)],
            );
            emit(Surface::Metal, &[(r, cap * 0.96), (r, cap), (barrel, cap)]);
            emit(
                self.finish,
                &[
                    (barrel, cap),
                    (barrel, h * MARKER_SHOULDER_HEIGHT),
                    (r * 0.50, h),
                    (0., h),
                ],
            );
            return;
        }

        let tail = if self.eraser || self.mechanism == 1 {
            h * self.tail_fraction
        } else {
            0.
        };
        if self.eraser {
            emit(
                Surface::Rubber,
                &[(0., 0.), (r * 0.88, 0.), (r * 0.88, tail * 0.65)],
            );
            emit(
                Surface::Metal,
                &[(r * 0.88, tail * 0.65), (r, tail * 0.65), (r, tail)],
            );
        } else if self.mechanism == 1 {
            emit(
                Surface::Metal,
                &[(0., 0.), (r * 0.68, 0.), (r * 0.68, tail), (r, tail)],
            );
        } else {
            emit(self.finish, &[(0., 0.), (r, 0.)]);
        }
        let end = h * (1. - self.tip_fraction);
        let grip = if self.mechanism == 1 {
            end - h * self.grip_fraction
        } else {
            end
        };
        emit(self.finish, &[(r, tail), (r, grip)]);
        let taper = if self.mechanism == 1 {
            emit(Surface::Rubber, &[(r, grip), (r * self.grip_radius, end)]);
            (Surface::Metal, r * self.grip_radius)
        } else {
            (Surface::Wood, r)
        };
        emit(taper.0, &[(taper.1, end), (r * 0.16, h * 0.98)]);
        emit(Surface::Ink, &[(r * 0.16, h * 0.98), (0., h)]);
    }

    fn cap_length(&self) -> f32 {
        self.length * (0.20 + self.grip_fraction * 0.40)
    }

    /// Exact low and high vertices under a rigid rotation. The clip stock is
    /// included, so a tool can rest on either end or lie flat on a worktop.
    pub(crate) fn vertical_range(&self, rot: Quat) -> (f32, f32) {
        self.projected_extent(rot.inverse() * Vec3::Y)
    }

    pub(crate) fn projected_extent(&self, y: Vec3) -> (f32, f32) {
        let (low, high) = self.outline().projected_range(y.xz());
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        self.profiles(|_, path| {
            for &(r, t) in path {
                lo = lo.min(y.y * t + r * low);
                hi = hi.max(y.y * t + r * high);
            }
        });
        if self.clip {
            for v in self.clip_rings().into_iter().flatten() {
                let h = y.dot(v);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        (lo, hi)
    }

    pub(crate) fn shaft_at(&self, height: f32, base_y: f32, vertical: f32) -> Option<(f32, f32)> {
        // The whole horizontal slice must cut a cylindrical length of stock.
        // The lower transition is used to clear any wider sections below it.
        let lift = self.radius * (1. - vertical * vertical).max(0.).sqrt();
        let mut bearing = None;
        self.profiles(|_, path| {
            for pair in path.windows(2) {
                let [(r, a), (other, b)] = [pair[0], pair[1]];
                if r != other || r <= 0. || a >= b {
                    continue;
                }
                let a = base_y + vertical * a;
                let b = base_y + vertical * b;
                let lower = a.min(b) + lift;
                if lower < height && height < a.max(b) - lift {
                    bearing = Some((r, lower));
                }
            }
        });
        bearing
    }

    pub(crate) fn draw(&self, a: &mut Assembly, tf: Transform, label: &str, barrel_label: &str) {
        self.profiles(|surface, path| {
            let label = if surface == self.finish {
                barrel_label
            } else {
                label
            };
            self.outline()
                .sweep(a.part(surface, label), path, ProfileJoints::Creased, tf);
        });
        if self.clip {
            self.draw_clip(a.part(Surface::Metal, label), tf);
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let (tf, _) = p.table_rest(o.seed);
    let label = o.kind.class_name();
    p.draw(a, tf, label, label);
}
