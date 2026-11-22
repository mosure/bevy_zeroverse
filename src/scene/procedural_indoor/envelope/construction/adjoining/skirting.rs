use super::*;
use crate::scene::procedural_indoor::layout::AdjoiningWindows;
use architecture::trim::TrimSection;

pub(super) fn build(
    a: &mut Assembly,
    outline: &[Vec2],
    joint: f32,
    section: &TrimSection,
    windows: Option<&AdjoiningWindows>,
) {
    let g = a.part(
        section.surface,
        &format!("wall#finish{}#exterior", section.finish_slot),
    );
    let n = outline.len();
    let inward = |edge: usize| (outline[(edge + 1) % n] - outline[edge]).normalize().perp();
    // A unit offset along this bisector lies one metre from both wall planes,
    // including at a re-entrant corner. Adjacent pieces share this cross section.
    let miter = |i| {
        let (left, right) = (inward((i + n - 1) % n), inward(i));
        (left + right) / (1. + left.dot(right))
    };
    let stock: Vec<_> = section
        .knots()
        .into_iter()
        .map(|p| Vec2::new(joint + p.x * section.height_m, p.y * section.depth_m))
        .collect();
    let back = -0.0015;
    for (edge, (a, b)) in polygon::edges(outline).enumerate().skip(1) {
        let end = (edge + 1) % n;
        let length = a.distance(b);
        let along = (b - a) / length;
        let normal = inward(edge);
        let first = edge == 1;
        let last = end == 0;
        // Stop against the actual back face of the shared structural wall.
        let start = if first {
            SHARED_WALL_THICKNESS_M + 0.002
        } else {
            0.
        };
        let stop = if last {
            length - SHARED_WALL_THICKNESS_M - 0.002
        } else {
            length
        };
        if start >= stop {
            continue;
        }
        let mut cuts: Vec<_> = windows
            .filter(|w| w.edge == edge)
            .into_iter()
            .flat_map(|w| w.facade.trim_openings())
            .filter(|o| o.min.y < joint + section.height_m && o.max.y > joint)
            .map(|o| {
                (
                    (o.min.x + length * 0.5 - 0.002).clamp(start, stop),
                    (o.max.x + length * 0.5 + 0.002).clamp(start, stop),
                )
            })
            .collect();
        cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
        cuts.push((stop, stop));
        let mut cursor = start;
        for (hi, next) in cuts {
            if hi > cursor {
                let left = if cursor == 0. { miter(edge) } else { normal };
                let right = if hi == length { miter(end) } else { normal };
                let at = |x: f32, offset: Vec2, p: Vec2| {
                    let base = if x == length { b } else { a + along * x };
                    let pos = base + offset * p.y;
                    Vec3::new(pos.x, p.x, pos.y)
                };
                let low = |p| at(cursor, left, p);
                let high = |p| at(hi, right, p);
                let bottom = Vec2::new(joint, back);
                let top = Vec2::new(joint + section.height_m, back);
                let face = |g: &mut Geometry, p: Vec2, q: Vec2| {
                    let side = normal * (q.x - p.x);
                    g.metric_quad(
                        [low(p), high(p), high(q), low(q)],
                        Vec3::new(side.x, p.y - q.y, side.y),
                    );
                };
                face(g, bottom, stock[0]);
                for s in stock.windows(2) {
                    face(g, s[0], s[1]);
                    face(g, Vec2::new(s[1].x, back), Vec2::new(s[0].x, back));
                    for (cap, x, offset, direction) in [
                        (cursor > 0., cursor, left, -along),
                        (hi < length, hi, right, along),
                    ] {
                        if cap {
                            g.metric_quad(
                                [
                                    at(x, offset, s[0]),
                                    at(x, offset, s[1]),
                                    at(x, offset, Vec2::new(s[1].x, back)),
                                    at(x, offset, Vec2::new(s[0].x, back)),
                                ],
                                Vec3::new(direction.x, 0., direction.y),
                            );
                        }
                    }
                }
                face(g, *stock.last().unwrap(), top);
            }
            cursor = cursor.max(next);
        }
    }
}
