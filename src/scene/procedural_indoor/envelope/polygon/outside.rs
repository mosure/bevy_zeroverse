use super::{clip, edges};
use bevy::prelude::*;

/// Convex pieces of the rectangle outside a simple footprint, in either winding.
/// Between consecutive vertex X coordinates the boundary edges cannot change
/// order. Fill the exterior intervals directly from those edges. Subtracting
/// individual room triangles can leave thin polygons along their shared diagonals.
pub fn outside(poly: &[Vec2], lo: Vec2, hi: Vec2) -> Vec<Vec<Vec2>> {
    if lo.x >= hi.x || lo.y >= hi.y {
        return Vec::new();
    }
    let mut cuts = vec![lo.x, hi.x];
    let mut bottom = lo.y;
    let mut top = hi.y;
    for p in poly {
        if lo.x < p.x && p.x < hi.x {
            cuts.push(p.x);
        }
        bottom = bottom.min(p.y);
        top = top.max(p.y);
    }
    cuts.sort_unstable_by(f32::total_cmp);
    cuts.dedup();

    let mut pieces = Vec::new();
    let mut crossings = Vec::with_capacity(poly.len());
    for xs in cuts.windows(2) {
        let (left, right) = (xs[0], xs[1]);
        crossings.clear();
        for (a, b) in edges(poly) {
            let (a, b) = if a.x < b.x { (a, b) } else { (b, a) };
            if a.x <= left && right <= b.x {
                let height = |x: f32| {
                    if x == a.x {
                        a.y
                    } else if x == b.x {
                        b.y
                    } else {
                        let (a, b) = (a.as_dvec2(), b.as_dvec2());
                        (a.y + (f64::from(x) - a.x) / (b.x - a.x) * (b.y - a.y)) as f32
                    }
                };
                crossings.push(Vec2::new(height(left), height(right)));
            }
        }
        // The average endpoint height orders edges at the middle of the strip.
        crossings.sort_unstable_by(|a, b| {
            a.as_dvec2()
                .element_sum()
                .total_cmp(&b.as_dvec2().element_sum())
        });
        let mut gap = |lower: Vec2, upper: Vec2| {
            if lower == upper || lower.min_element() >= hi.y || upper.max_element() <= lo.y {
                return;
            }
            let mut p = vec![
                Vec2::new(left, lower.x),
                Vec2::new(right, lower.y),
                Vec2::new(right, upper.y),
                Vec2::new(left, upper.x),
            ];
            p.dedup();
            if p.first() == p.last() {
                p.pop();
            }
            // A caller can request a rectangle smaller than the footprint.
            if lower.min_element() < lo.y {
                p = clip(&p, Vec2::Y, lo.y);
            }
            if upper.max_element() > hi.y {
                p = clip(&p, Vec2::NEG_Y, -hi.y);
            }
            if p.len() >= 3 {
                pieces.push(p);
            }
        };
        let mut lower = Vec2::splat(bottom);
        for &[entry, exit] in crossings.as_chunks::<2>().0 {
            gap(lower, entry);
            lower = exit;
        }
        gap(lower, Vec2::splat(top));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::envelope::polygon::{area, contains, triangles};

    #[test]
    fn diagonal_clips_and_a_single_float_wide_exterior_notch_keep_their_area() {
        let triangle = [Vec2::new(0., -2.), Vec2::new(4., 2.), Vec2::new(0., 2.)];
        let pieces = outside(&triangle, Vec2::new(0., -1.), Vec2::new(4., 1.));
        assert_eq!(pieces.iter().map(|p| area(p)).sum::<f32>(), 4.);

        // A physical notch can be thinner than a clipping tolerance. Its
        // jambs lie on real boundary edges, even when its X limits are adjacent floats.
        let left = 0.5_f32;
        let right = left.next_up();
        let outline = [
            [0., 0.],
            [2., 0.],
            [2., 2.],
            [right, 2.],
            [right, 1.],
            [left, 1.],
            [left, 2.],
            [0., 2.],
        ]
        .map(Vec2::from_array);
        let pieces = outside(&outline, Vec2::ZERO, Vec2::splat(2.));
        assert_eq!(pieces.len(), 1);
        assert_eq!(area(&pieces[0]), right - left);
    }

    #[test]
    fn exterior_spans_cover_notches_and_clip_to_the_requested_rectangle() {
        // A vertical slice passes through two occupied intervals on the right.
        let mut room = [
            [-3., -3.],
            [3., -3.],
            [3., -1.],
            [-1., -1.],
            [-1., 1.],
            [3., 1.],
            [3., 3.],
            [-3., 3.],
        ]
        .map(Vec2::from_array);
        for _ in 0..2 {
            for (lo, hi, expected) in [
                (Vec2::splat(-4.), Vec2::splat(4.), 36.),
                (Vec2::splat(-3.), Vec2::splat(3.), 8.),
                (Vec2::splat(-2.), Vec2::splat(2.), 6.),
                (Vec2::new(-2.5, -2.5), Vec2::new(-1.5, 2.5), 0.),
                (Vec2::splat(4.), Vec2::splat(6.), 4.),
            ] {
                let pieces = outside(&room, lo, hi);
                assert_eq!(pieces.iter().map(|p| area(p)).sum::<f32>(), expected);
                for t in pieces.iter().flat_map(|p| triangles(p)) {
                    assert!(t.iter().all(|p| p.cmpge(lo).all() && p.cmple(hi).all()));
                    assert!(!contains(&room, (t[0] + t[1] + t[2]) / 3., 0.));
                }
            }
            room.reverse();
        }
    }
}
