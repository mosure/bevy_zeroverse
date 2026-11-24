//! A single continuous room-side surface. The ceramic grout, panel backing
//! and bevels share their edges with the stock and are cut at the actual jambs.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::{solid_cells, solid_rectangles},
    envelope::lining::{LiningStock, WallCourses},
    materials::masonry as finish,
};

#[derive(Clone, Copy)]
struct Joints {
    origin: f32,
    pitch: f32,
    half: f32,
    trough: f32,
}

impl Joints {
    fn inset(self, x: f32) -> f32 {
        let u = (x - self.origin) / self.pitch;
        let distance = (u - u.round()).abs() * self.pitch;
        ((self.half - distance) / (self.half * (1. - self.trough))).clamp(0., 1.)
    }

    fn cuts(self, lo: f32, hi: f32, ends: &[f32]) -> Vec<f32> {
        let mut cuts = vec![lo, hi];
        cuts.extend(ends.iter().copied().filter(|&v| v > lo && v < hi));
        let first = ((lo - self.origin - self.half) / self.pitch).ceil() as i32;
        let last = ((hi - self.origin + self.half) / self.pitch).floor() as i32;
        for i in first..=last {
            let joint = self.origin + i as f32 * self.pitch;
            for k in [-1., -self.trough, self.trough, 1.] {
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
        trough: lining.joint_bottom_fraction(),
    };
    let mut skin = Skin {
        board: Geometry::default(),
        joint: Geometry::default(),
        tf,
    };
    if lining.stock.is_masonry() {
        // One continuous bed is visible through every head and bed joint. Even
        // subpixel raster seams between the beveled pieces hit real mortar.
        for (min, max) in solid_cells(lo, hi, holes) {
            masonry::mortar(
                &tf,
                &mut skin.joint,
                Rect { min, max },
                lining.joint_depth_m,
            );
        }
    }
    let mut tones: [Geometry; 3] = Default::default();
    for (min, max) in solid_rectangles(lo, hi, holes) {
        let field = Rect {
            min: min + fade,
            max: max - fade,
        };
        // All phases meet in the full-depth bed joint. Reuse the grids across
        // rows; apertures only clip a unit, they never restart the bond.
        let xs: Vec<_> = (0..courses.phase_count(lining.stock))
            .map(|i| {
                let offset = i as f32 * courses.bond_offset;
                let joints = Joints {
                    origin: (lining.phase + offset) * lining.module_width_m,
                    pitch: lining.module_width_m,
                    ..y_joint
                };
                (joints, joints.cuts(min.x, max.x, &edge_cuts[0]))
            })
            .collect();
        let mut bed = Vec::new();
        if lining.stock.is_masonry() {
            bed.extend([min.x, max.x]);
            bed.extend(
                edge_cuts[0]
                    .iter()
                    .copied()
                    .filter(|&x| min.x < x && x < max.x),
            );
            bed.sort_unstable_by(f32::total_cmp);
            bed.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        }
        let ys = y_joint.cuts(min.y, max.y, &edge_cuts[1]);
        for y in ys.windows(2) {
            let mid_y = (y[0] + y[1]) * 0.5;
            let row = ((mid_y - y_joint.origin) / y_joint.pitch).floor() as i32;
            let (x_joint, cuts) = &xs[row.rem_euclid(xs.len() as i32) as usize];
            let dy = [y_joint.inset(y[0]), y_joint.inset(y[1])];
            let horizontal_grout = y_joint.inset(mid_y) >= 0.999;
            let y_origin = y_joint.origin + row as f32 * y_joint.pitch;
            let cuts = if lining.stock.is_masonry() && horizontal_grout {
                &bed
            } else {
                cuts
            };
            for x in cuts.windows(2) {
                let mid_x = (x[0] + x[1]) * 0.5;
                let col = ((mid_x - x_joint.origin) / x_joint.pitch).floor();
                let grout = horizontal_grout || x_joint.inset(mid_x) >= 0.999;
                let uv = if grout {
                    Vec2::ZERO
                } else {
                    Vec2::new(
                        col * 0.7548777 + row as f32 * 0.47,
                        row as f32 * 0.683 + col * 0.5698403,
                    )
                };
                if lining.stock.is_masonry() {
                    if grout {
                        if Vec2::new(x[0], y[0]).cmpge(field.min).all()
                            && Vec2::new(x[1], y[1]).cmple(field.max).all()
                        {
                            continue;
                        }
                        // Only model the mortar rising to meet a jamb or wall end.
                        // The field-depth triangles already belong to the continuous bed.
                        let p = [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(i, j)| {
                            let v = Vec2::new(x[i], y[j]);
                            v.extend(-lining.joint_depth_m * edge_fade(v))
                        });
                        for ids in [[0, 1, 2], [0, 2, 3]] {
                            let tri = ids.map(|i| p[i]);
                            if tri.iter().any(|v| v.z > -lining.joint_depth_m + 1e-7) {
                                Skin::triangle(&tf, &mut skin.joint, tri, Vec2::ZERO);
                            }
                        }
                        continue;
                    }
                    let pitch = Vec2::new(x_joint.pitch, y_joint.pitch);
                    let min = Vec2::new(x_joint.origin + col * pitch.x, y_origin);
                    let bounds = Rect {
                        min,
                        max: min + pitch,
                    };
                    if bounds.min.cmpge(field.min).all() && bounds.max.cmple(field.max).all() {
                        let center = bounds.center();
                        if x[0] <= center.x
                            && center.x < x[1]
                            && y[0] <= center.y
                            && center.y < y[1]
                        {
                            let g = &mut tones[finish::tone_at(col as i32, row, lining.phase)];
                            masonry::unit(
                                &tf,
                                g,
                                bounds,
                                y_joint.half,
                                y_joint.half * y_joint.trough,
                                lining.joint_depth_m,
                                uv,
                            );
                        }
                        continue;
                    }
                }
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
                let p = [at(0, 0), at(1, 0), at(1, 1), at(0, 1)];
                if lining.stock.is_masonry() {
                    let g = &mut tones[finish::tone_at(col as i32, row, lining.phase)];
                    patch(&tf, g, p, uv);
                } else {
                    skin.patch(p, uv, grout);
                }
            }
        }
    }
    if lining.stock.is_masonry() {
        for (tone, g) in tones.into_iter().enumerate() {
            if !g.indices.is_empty() {
                a.part(
                    lining.stock.surface(),
                    &finish::label(lining.finish_slot, tone),
                )
                .append(g);
            }
        }
        a.part(
            Surface::Concrete,
            &finish::label(lining.finish_slot, finish::MORTAR),
        )
        .append(skin.joint);
        return;
    }
    let backing = match lining.stock {
        LiningStock::CeramicSlab | LiningStock::FiredBrick | LiningStock::ConcreteBlock => {
            Surface::Concrete
        }
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
