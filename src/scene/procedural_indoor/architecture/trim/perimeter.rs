//! Cut primary-room skirting and cornice from their actual stock profiles.
//! Adjacent lengths share a bisector, with a millimetre-scale assembly joint.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::EnvelopeProgram,
    geometry::{stock::TimberCuts, Geometry},
};

const BACK: f32 = -0.002;
const CUT_TOLERANCE: f32 = 0.00005;

mod facing;
#[cfg(test)]
mod tests;

struct Run {
    center: Vec2,
    along: Vec2,
    inward: Vec2,
    span: Vec2,
    ends: [Vec2; 2],
    datum: f32,
    gradient: Vec2,
    timber_origin: Option<Vec2>,
}

impl Run {
    fn new(e: &EnvelopeProgram, size: Vec3, edge: usize, datum: f32, crown: bool) -> Self {
        let end = (edge + 1) % e.footprint.len();
        let a = e.footprint[edge];
        let b = e.footprint[end];
        let along = (b - a).normalize();
        let half = a.distance(b) * 0.5;
        Self {
            center: (a + b) * 0.5,
            along,
            inward: along.perp(),
            span: Vec2::new(-half, half),
            ends: [
                polygon::inset_direction(&e.footprint, edge),
                polygon::inset_direction(&e.footprint, end),
            ],
            timber_origin: None,
            datum: if crown {
                e.roof_height(size, Vec2::ZERO) - datum
            } else {
                datum
            },
            gradient: if crown {
                -e.ceiling_drop / size.xz()
            } else {
                Vec2::ZERO
            },
        }
    }

    fn level(&self, x: f32, z: f32) -> f32 {
        self.datum
            + self
                .gradient
                .dot(self.center + self.along * x + self.inward * z)
    }

    // The carpenter scribes one straight relief across each fixture or casing.
    // On a pitched ceiling, keep only stock that clears the head across its
    // entire width and depth. Ordinary skirting cuts follow the sill exactly.
    fn cut_rect(&self, min: Vec2, max: Vec2, section: &TrimSection) -> Option<Rect> {
        let lo = min.x.max(self.span.x);
        let hi = max.x.min(self.span.y);
        if hi <= lo {
            return None;
        }
        let levels = [
            self.level(lo, BACK),
            self.level(lo, section.depth_m),
            self.level(hi, BACK),
            self.level(hi, section.depth_m),
        ];
        let bottom = (min.y - levels.into_iter().fold(f32::NEG_INFINITY, f32::max)).max(0.);
        let top = (max.y - levels.into_iter().fold(f32::INFINITY, f32::min)).min(section.height_m);
        (top > bottom).then_some(Rect {
            min: Vec2::new(lo, bottom),
            max: Vec2::new(hi, top),
        })
    }

    fn fit_ends(&mut self, depth: f32, joint: f32, cuts: &[Rect]) {
        let length = self.span.y - self.span.x;
        for i in 0..2 {
            let sign = if i == 0 { 1. } else { -1. };
            let offset = self.ends[i];
            let tangent = sign * self.along.dot(offset);
            // Advance along the board so the perpendicular distance to the
            // common cut plane is half the recorded joint, at any wall angle.
            let seam = 0.5 * joint * offset.length();
            let outer = (tangent * depth).max(tangent * BACK);
            let reach = seam + outer;
            let end = self.span[i];
            let beyond_cut = cuts.iter().flat_map(|cut| [cut.min.x, cut.max.x]).any(|x| {
                let d = (x - end) * sign;
                d > 0. && d <= reach + CUT_TOLERANCE
            });
            // A short return or a casing inside the saw cut cannot carry the
            // full mitre. Square it on this side of the bisector so it can
            // terminate cleanly without folding a face or entering a jamb.
            // At a narrow re-entrant notch the face projects out past the
            // wall end. Limit that cut as well as inward cuts at sharp tips.
            let inset = if beyond_cut || reach.max(tangent.abs() * depth) >= length * 0.45 {
                self.ends[i] = self.inward;
                reach
            } else {
                seam
            };
            self.span[i] += sign * inset;
        }
    }

    fn point(&self, x: f32, height: f32, depth: f32) -> Vec3 {
        let offset = if x == self.span.x {
            self.ends[0]
        } else if x == self.span.y {
            self.ends[1]
        } else {
            self.inward
        };
        let p = self.center + self.along * x + offset * depth;
        Vec3::new(p.x, self.datum + self.gradient.dot(p) + height, p.y)
    }
}

pub(crate) fn build(
    a: &mut Assembly,
    scene: &IndoorManifest,
    edge: usize,
    trim: &TrimProgram,
    section: &TrimSection,
    holes: &[WindowOpening],
    crown: bool,
) {
    let datum = if crown {
        trim.joint_m + section.height_m
    } else {
        trim.joint_m
    };
    let envelope = scene.envelope.as_ref().unwrap();
    let mut run = Run::new(envelope, scene.room_size, edge, datum, crown);
    if section.surface.is_cut_timber() {
        run.timber_origin =
            Some(TimberCuts::new(scene.seed, edge as u64 * 2 + u64::from(crown)).origin());
    }
    let joint = trim.miter_joint_m.expect("mitred installation");
    // Keep a fitting clearance at jambs and sills. Otherwise their end faces
    // can occupy exactly the same plane as the edge of the window surround.
    let clearance = Vec2::splat(joint * 0.5);
    let mut cuts: Vec<_> = holes
        .iter()
        .filter_map(|o| run.cut_rect(o.min - clearance, o.max + clearance, section))
        .collect();
    if envelope.shared_edge(edge, scene.room_size) {
        // The structural hole ends on the jamb centre lines. Skirting and
        // cornice must instead terminate at the outside of the installed stock.
        let half_stock = if scene.entry_door.is_some() {
            crate::scene::procedural_indoor::architecture::doors::FRAME_WIDTH_M * 0.5
        } else {
            0.032
        };
        let half = scene.entry_width() * 0.5 + half_stock + joint;
        if let Some(casing) = run.cut_rect(
            Vec2::new(-scene.door_x - half, 0.),
            Vec2::new(
                -scene.door_x + half,
                scene.entry_height() + half_stock + joint,
            ),
            section,
        ) {
            cuts.push(casing);
        }
    }
    run.fit_ends(section.depth_m, joint, &cuts);
    if run.span.y - run.span.x <= CUT_TOLERANCE {
        return;
    }
    let mut profile = section.knots();
    if crown {
        profile.reverse();
        for p in &mut profile {
            p.x = 1. - p.x;
        }
    }
    for p in &mut profile {
        *p *= Vec2::new(section.height_m, section.depth_m);
    }
    let g = a.part(
        section.surface,
        &format!(
            "{}#finish{}",
            if crown { "ceiling" } else { "wall" },
            section.finish_slot
        ),
    );
    sweep(g, &run, &profile, &cuts, Some((section, crown)));
}

/// A cut grid in running metres and stock height. Emit only boundary faces
/// of the occupied cells, including the underside of a notch above a doorway
/// and the top of retained skirting below a low sill. No internal knot caps.
fn sweep(
    g: &mut Geometry,
    run: &Run,
    profile: &[Vec2],
    cuts: &[Rect],
    section: Option<(&TrimSection, bool)>,
) {
    let mut xs = vec![run.span.x, run.span.y];
    let mut levels: Vec<_> = profile.iter().map(|p| p.x).collect();
    for cut in cuts {
        xs.extend([
            cut.min.x.clamp(run.span.x, run.span.y),
            cut.max.x.clamp(run.span.x, run.span.y),
        ]);
        levels.extend([cut.min.y, cut.max.y]);
    }
    for axis in [&mut xs, &mut levels] {
        axis.sort_unstable_by(f32::total_cmp);
        // Preserve the upper physical endpoint when a nearby cut rounds to it.
        let end = *axis.last().unwrap();
        axis.dedup_by(|a, b| (*a - *b).abs() < CUT_TOLERANCE);
        *axis.last_mut().unwrap() = end;
    }
    let depth: Vec<_> = levels
        .iter()
        .map(|&h| {
            let i = profile
                .partition_point(|p| p.x < h)
                .clamp(1, profile.len() - 1);
            let (a, b) = (profile[i - 1], profile[i]);
            a.y.lerp(b.y, ((h - a.x) / (b.x - a.x)).clamp(0., 1.))
        })
        .collect();
    let nx = xs.len() - 1;
    let ny = levels.len() - 1;
    let solid: Vec<_> = levels
        .windows(2)
        .flat_map(|y| {
            xs.windows(2).map(move |x| {
                let p = Vec2::new((x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5);
                !cuts.iter().any(|r| r.contains(p))
            })
        })
        .collect();
    let along = Vec3::new(run.along.x, 0., run.along.y);
    let normal = Vec3::new(run.inward.x, 0., run.inward.y);
    let mut facing = facing::Facing::new(run, section);
    for (y, heights) in levels.windows(2).enumerate() {
        let band = facing.band(heights[0], heights[1], depth[y + 1] - depth[y]);
        for (x, span) in xs.windows(2).enumerate() {
            let i = y * nx + x;
            if !solid[i] {
                continue;
            }
            let corners = [(0, 0), (1, 0), (1, 1), (0, 1)];
            let front = corners.map(|(u, v)| run.point(span[u], heights[v], depth[y + v]));
            let back = corners.map(|(u, v)| run.point(span[u], heights[v], BACK));
            band.front(g, front);
            g.stock_quad(back, -normal, facing.along, run.timber_origin);
            let boundaries = [
                (y == 0 || !solid[i - nx], Vec3::NEG_Y),
                (x + 1 == nx || !solid[i + 1], along),
                (y + 1 == ny || !solid[i + nx], Vec3::Y),
                (x == 0 || !solid[i - 1], -along),
            ];
            for (j, (exposed, outward)) in boundaries.into_iter().enumerate() {
                if exposed {
                    let k = (j + 1) % 4;
                    let points = [back[j], back[k], front[k], front[j]];
                    if j % 2 == 0 {
                        g.stock_quad(points, outward, facing.along, run.timber_origin);
                    } else {
                        g.metric_quad(points, outward);
                    }
                }
            }
        }
    }
}
