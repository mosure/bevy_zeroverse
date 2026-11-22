//! A single continuous room-side surface. The ceramic grout, panel backing
//! and bevels share their edges with the stock and are cut at the actual jambs.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::solid_rectangles,
    envelope::lining::{LiningStock, WallCourses},
};

#[derive(Clone, Copy)]
struct Joints {
    origin: f32,
    pitch: f32,
    half: f32,
}

impl Joints {
    fn inset(self, x: f32) -> f32 {
        let u = (x - self.origin) / self.pitch;
        let distance = (u - u.round()).abs() * self.pitch;
        ((self.half - distance) / (self.half * 0.56)).clamp(0., 1.)
    }

    fn cuts(self, lo: f32, hi: f32, ends: &[f32]) -> Vec<f32> {
        let mut cuts = vec![lo, hi];
        cuts.extend(ends.iter().copied().filter(|&v| v > lo && v < hi));
        let first = ((lo - self.origin - self.half) / self.pitch).ceil() as i32;
        let last = ((hi - self.origin + self.half) / self.pitch).floor() as i32;
        for i in first..=last {
            let joint = self.origin + i as f32 * self.pitch;
            for k in [-1., -0.44, 0.44, 1.] {
                let v = joint + k * self.half;
                if v > lo && v < hi {
                    cuts.push(v);
                }
            }
        }
        cuts.sort_unstable_by(f32::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        cuts
    }
}

pub(super) fn build(
    a: &mut Assembly,
    tf: Transform,
    lining: &WallLining,
    courses: WallCourses,
    span: f32,
    height: f32,
    holes: &[WindowOpening],
) {
    let lo = Vec2::new(-span * 0.5, 0.);
    let hi = Vec2::new(span * 0.5, height);
    let fade = lining.joint_depth_m * 1.5;
    let mut edge_cuts = [
        vec![lo.x + fade, hi.x - fade],
        vec![lo.y + fade, hi.y - fade],
    ];
    for h in holes {
        for (i, axis) in edge_cuts.iter_mut().enumerate() {
            axis.extend([h.min[i] - fade, h.min[i], h.max[i], h.max[i] + fade]);
        }
    }
    let edge_fade = |p: Vec2| {
        let outside = (p - lo).min(hi - p).min_element();
        let distance = holes.iter().fold(outside, |d, h| {
            d.min((h.min - p).max(p - h.max).max_element())
        });
        (distance / fade).clamp(0., 1.)
    };
    let y_joint = Joints {
        origin: courses.phase * courses.height_m,
        pitch: courses.height_m,
        half: lining.joint_width_m * 0.5,
    };
    let mut skin = Skin {
        board: Geometry::default(),
        joint: Geometry::default(),
        tf,
    };
    for (min, max) in solid_rectangles(lo, hi, holes) {
        // Alternate-row x phases are sampled once per wall. Both meet in the
        // full-depth horizontal joint, including where a window cuts a course.
        let xs = [0., courses.bond_offset].map(|offset| {
            let joints = Joints {
                origin: (lining.phase + offset) * lining.module_width_m,
                pitch: lining.module_width_m,
                half: y_joint.half,
            };
            (joints, joints.cuts(min.x, max.x, &edge_cuts[0]))
        });
        let ys = y_joint.cuts(min.y, max.y, &edge_cuts[1]);
        for y in ys.windows(2) {
            let mid_y = (y[0] + y[1]) * 0.5;
            let row = ((mid_y - y_joint.origin) / y_joint.pitch).floor() as i32;
            let (x_joint, cuts) = &xs[row.rem_euclid(2) as usize];
            let dy = [y_joint.inset(y[0]), y_joint.inset(y[1])];
            let horizontal_grout = y_joint.inset(mid_y) >= 0.999;
            for x in cuts.windows(2) {
                let mid_x = (x[0] + x[1]) * 0.5;
                let col = ((mid_x - x_joint.origin) / x_joint.pitch).floor();
                let dx = [x_joint.inset(x[0]), x_joint.inset(x[1])];
                let at = |i: usize, j: usize| {
                    let p = Vec2::new(x[i], y[j]);
                    let inset = dx[i] + (1. - dx[i]) * dy[j];
                    let depth = if inset > 0. {
                        -lining.joint_depth_m * inset * edge_fade(p)
                    } else {
                        0.
                    };
                    p.extend(depth)
                };
                let grout = horizontal_grout || x_joint.inset(mid_x) >= 0.999;
                let uv = if grout {
                    Vec2::ZERO
                } else {
                    Vec2::new(
                        col * 0.7548777 + row as f32 * 0.47,
                        row as f32 * 0.683 + col * 0.5698403,
                    )
                };
                skin.patch([at(0, 0), at(1, 0), at(1, 1), at(0, 1)], uv, grout);
            }
        }
    }
    let backing = match lining.stock {
        LiningStock::CeramicSlab => Surface::Concrete,
        LiningStock::AcousticFabric => Surface::Rubber,
        LiningStock::Veneer => Surface::WoodEdge,
    };
    a.part(
        lining.stock.surface(),
        &format!("wall#finish{}", lining.finish_slot),
    )
    .append(skin.board);
    a.part(backing, "wall").append(skin.joint);
}
