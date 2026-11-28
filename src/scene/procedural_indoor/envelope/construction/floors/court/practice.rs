//! Replace timber only inside a fitted stencil. Primary volleyball paint keeps
//! right of way at crossings. Both pigments and the wood share the walk plane.
use super::*;

struct Strip {
    corners: [Vec2; 4],
    bounds: Rect,
    paint: bool,
}

pub(super) struct Stencil {
    bounds: Rect,
    strips: Vec<Strip>,
}

impl Stencil {
    pub(super) fn new(mark: FreeThrowMark) -> Self {
        let r = mark.line_length_m * 0.5;
        let mut stencil = Self {
            bounds: mark.bounds(),
            strips: Vec::new(),
        };
        let mut strip = |z0, z1, left: [f32; 2], right: [f32; 2], paint| {
            if left == right {
                return;
            }
            let corners = [
                mark.world(left[0], z0),
                mark.world(right[0], z0),
                mark.world(right[1], z1),
                mark.world(left[1], z1),
            ];
            let bounds = Rect::from_corners(
                mark.world(left[0].min(left[1]), z0),
                mark.world(right[0].max(right[1]), z1),
            );
            stencil.strips.push(Strip {
                corners,
                bounds,
                paint,
            });
        };
        strip(-LINE * 0.5, LINE * 0.5, [-r; 2], [r; 2], true);
        if mark.semicircle {
            // Include the crown of both edges exactly. Angular sampling limits
            // the sagitta to 4 mm at the 1.8 m outer radius.
            let inner = r - LINE;
            let mut levels = vec![LINE * 0.5];
            for radius in [inner, r] {
                for i in 1..=12 {
                    levels.push(radius * (std::f32::consts::FRAC_PI_2 * (i as f32 / 12.)).sin());
                }
            }
            levels.sort_unstable_by(f32::total_cmp);
            let section = |z: f32| {
                let chord = |radius: f32| ((radius - z) * (radius + z)).max(0.).sqrt();
                let (a, b) = (chord(r), chord(inner));
                [-r, -a, -b, b, a, r]
            };
            let mut lower = section(levels[0]);
            for pair in levels.windows(2) {
                let upper = section(pair[1]);
                for k in 0..5 {
                    strip(
                        pair[0],
                        pair[1],
                        [lower[k], upper[k]],
                        [lower[k + 1], upper[k + 1]],
                        k == 1 || k == 3,
                    );
                }
                lower = upper;
            }
        }
        stencil
    }

    fn draw(&self, a: &mut Assembly, rect: Rect) {
        if self.strips.len() == 1 {
            horizontal_quad(a.part(Surface::Paint, sport::PRACTICE), rect, 0.);
            return;
        }
        for (paint, surface, label) in [
            (false, Surface::Floor, "floor"),
            (true, Surface::Paint, sport::PRACTICE),
        ] {
            let mut possible = self
                .strips
                .iter()
                .filter(|strip| strip.paint == paint && !rect.intersect(strip.bounds).is_empty())
                .peekable();
            if possible.peek().is_none() {
                continue;
            }
            let g = a.part(surface, label);
            for strip in possible {
                let clipped;
                let points = if rect.contains(strip.bounds.min) && rect.contains(strip.bounds.max) {
                    &strip.corners[..]
                } else {
                    clipped = polygon::clip_box(&strip.corners, rect.min, rect.max);
                    &clipped
                };
                if points.len() < 3 {
                    continue;
                }
                for i in 1..points.len() - 1 {
                    triangle(
                        g,
                        [points[0], points[i], points[i + 1]].map(|p| Vec3::new(p.x, 0., p.y)),
                        Vec3::Y,
                    );
                }
            }
        }
    }
}

pub(super) fn timber(a: &mut Assembly, rect: Rect, stencils: &[Stencil]) {
    let Some((stencil, rest)) = stencils.split_first() else {
        horizontal_quad(a.part(Surface::Floor, "floor"), rect, 0.);
        return;
    };
    let hit = rect.intersect(stencil.bounds);
    if hit.is_empty() {
        timber(a, rect, rest);
        return;
    }
    stencil.draw(a, hit);
    // At most two non-overlapping stations. Leave the large unmarked floor
    // patches as rectangles, independent of the arc tessellation.
    for (min, max) in [
        (rect.min, Vec2::new(hit.min.x, rect.max.y)),
        (Vec2::new(hit.max.x, rect.min.y), rect.max),
        (
            Vec2::new(hit.min.x, rect.min.y),
            Vec2::new(hit.max.x, hit.min.y),
        ),
        (
            Vec2::new(hit.min.x, hit.max.y),
            Vec2::new(hit.max.x, rect.max.y),
        ),
    ] {
        if min.cmplt(max).all() {
            timber(a, Rect { min, max }, rest);
        }
    }
}

#[cfg(test)]
mod tests;
