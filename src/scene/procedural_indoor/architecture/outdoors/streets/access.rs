//! Paths connect built thresholds to clear gaps in the perimeter sidewalk.
//! Their end edges follow the two structures even when the approach is skewed.
use super::*;
use crate::scene::procedural_indoor::footprint::Footprint;

pub(super) struct Walk {
    pub side: usize,
    pub opening: [f32; 2],
    pub start: [Vec2; 2],
    pub end: [Vec2; 2],
    pub span: Footprint,
    pub seed: u64,
}
impl Walk {
    pub fn length(&self) -> f32 {
        (self.start[0] + self.start[1] - self.end[0] - self.end[1]).length() * 0.5
    }
}

pub(super) fn plan(
    sides: &[Strip; 4],
    sites: &[Parcel],
    terraces: &[midground::Terrace],
    entrances: &[(usize, Entrance)],
) -> Vec<Walk> {
    let occupied: Vec<_> = sites
        .iter()
        .map(|s| Footprint::oriented(s.center, s.size, s.yaw))
        .collect();
    let mut walks: Vec<Walk> = Vec::new();
    for &(index, entry) in entrances {
        if walks.len() == 5 {
            break;
        }
        let mut r = stream(sites[index].seed, 0x444f_4f52_5041_5448);
        let width = (entry.width + r.random_range(0.08..0.33)).clamp(0.86, 1.62);
        let start_width = width.min(entry.width + 0.10);
        let apron_axis = Vec2::new(entry.outward.y, -entry.outward.x);
        let source = entry.threshold + entry.outward * 0.003;
        let mut best: Option<(f32, Walk)> = None;
        for (side, strip) in sides.iter().enumerate() {
            let axis = strip.axis();
            let lip = strip.lip();
            let half = strip.length() * 0.5 - width * 0.5 - 0.20;
            let closest = (source - lip).dot(axis);
            let distance = (source - lip).dot(strip.outward);
            if half < 0. || !(0.45..10.5).contains(&distance) {
                continue;
            }
            for shift in [0., -1.35, 1.35, -2.75, 2.75] {
                let u = (closest + shift).clamp(-half, half);
                let landing = lip + axis * u;
                let delta = landing - source;
                let dir = delta.normalize();
                if dir.dot(entry.outward) < 0.65 || dir.dot(-strip.outward) < 0.70 {
                    continue;
                }
                // Pair the two cross-sections in the same order. Their four
                // corners fit in this oriented collision envelope.
                let across = Vec2::new(dir.y, -dir.x);
                let end_axis = axis * axis.dot(across).signum() * width * 0.5;
                let start_axis = apron_axis * apron_axis.dot(across).signum() * start_width * 0.5;
                let start = [source - start_axis, source + start_axis];
                let end = [landing - end_axis, landing + end_axis];
                let corners = [start[0], start[1], end[1], end[0]];
                if (0..4).any(|i| {
                    (corners[(i + 1) % 4] - corners[i])
                        .perp_dot(corners[(i + 2) % 4] - corners[(i + 1) % 4])
                        <= 0.015
                }) {
                    continue;
                }
                let c = (source + landing) * 0.5;
                let extent = start.into_iter().chain(end).fold(Vec2::ZERO, |size, v| {
                    size.max(Vec2::new((v - c).dot(across).abs(), (v - c).dot(dir).abs()))
                });
                let span = Footprint::oriented(c, extent * 2., dir.x.atan2(dir.y));
                if occupied
                    .iter()
                    .enumerate()
                    .any(|(i, p)| i != index && span.overlaps(*p, 0.25))
                    || terraces.iter().any(|p| span.overlaps(p.footprint(), 0.12))
                    || walks.iter().any(|p| span.overlaps(p.span, 0.10))
                {
                    continue;
                }
                let cost = delta.length() + 1.2 * (1. - dir.dot(entry.outward));
                if best.as_ref().is_none_or(|(score, _)| cost < *score) {
                    best = Some((
                        cost,
                        Walk {
                            side,
                            opening: [u - width * 0.5 - 0.05, u + width * 0.5 + 0.05],
                            start,
                            end,
                            span,
                            seed: sites[index].seed,
                        },
                    ));
                }
            }
        }
        if let Some((_, walk)) = best {
            walks.push(walk);
        }
    }
    walks
}

pub(super) fn pave(a: &mut Assembly, paths: &[Walk], finish: &str) {
    let g = a.part(Surface::Concrete, finish);
    for path in paths {
        let mut r = stream(path.seed, 0x5041_5645_4a4f_494e);
        let length = path.length();
        let count = (length / r.random_range(0.70..1.70)).round().max(1.) as usize;
        let joint = r.random_range(0.005..0.011) / length;
        let mut begin = 0.;
        for i in 0..count {
            let last = i + 1 == count;
            let limit = if last {
                1.
            } else {
                (i as f32 + 1. + r.random_range(-0.11..0.11)) / count as f32
            };
            let end = if last { limit } else { limit - joint };
            let corner = |k: usize, t| path.start[k].lerp(path.end[k], t).extend(-0.005).xzy();
            let top = [
                corner(0, begin),
                corner(1, begin),
                corner(1, end),
                corner(0, end),
            ];
            g.metric_quad(top, Vec3::Y);
            for k in 0..4 {
                let (p, q) = (top[k], top[(k + 1) % 4]);
                let normal = Vec3::Y.cross(q - p);
                g.metric_quad([p, q, q - Vec3::Y * 0.068, p - Vec3::Y * 0.068], normal);
            }
            begin = limit;
        }
    }
}

/// Gaps are sorted and may overlap at a corner. Return just the usable spans.
pub(super) fn intervals(
    lo: f32,
    hi: f32,
    gaps: &[[f32; 2]],
) -> impl Iterator<Item = (f32, f32)> + '_ {
    let mut from = lo;
    gaps.iter()
        .copied()
        .chain(std::iter::once([hi, hi]))
        .filter_map(move |[left, right]| {
            let end = left.min(hi);
            let start = from;
            from = from.max(right);
            (end - start > 0.022).then_some((start, end))
        })
}
