//! Simple polygon operations in metric X/Z coordinates. Concave outlines are
//! triangulated before clipping, so a courtyard notch never acquires a floor.
use bevy::prelude::*;

pub fn area(p: &[Vec2]) -> f32 {
    signed_area(p) as f32
}

// Widen before subtracting: clipped f32 vertices can delimit a valid sliver
// much smaller than a fixed metric epsilon. Use the same determinant/sign for
// degeneracy, convexity and triangle containment, without inflating its edges.
fn orient(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    (b.as_dvec2() - a.as_dvec2()).perp_dot(c.as_dvec2() - a.as_dvec2())
}

fn signed_area(p: &[Vec2]) -> f64 {
    p.get(1..).map_or(0., |tail| {
        tail.windows(2)
            .map(|edge| orient(p[0], edge[0], edge[1]))
            .sum::<f64>()
            * 0.5
    })
}

pub fn edges(p: &[Vec2]) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    p.iter()
        .copied()
        .zip(p.iter().copied().cycle().skip(1))
        .take(p.len())
}

fn point_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    if a.distance_squared(b) < 1e-12 {
        return p.distance(a);
    }
    p.distance(a + (b - a) * ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0., 1.))
}

pub fn contains(poly: &[Vec2], p: Vec2, margin: f32) -> bool {
    let mut inside = false;
    for (a, b) in edges(poly) {
        let distance = point_segment(p, a, b);
        if distance < margin - 1e-6 {
            return false;
        }
        if distance < 1e-6 && margin <= 0. {
            return true;
        }
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) {
            inside = !inside;
        }
    }
    inside
}

fn intersects(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let ab = b - a;
    let cd = d - c;
    let denominator = ab.perp_dot(cd);
    if denominator.abs() < 1e-8 {
        if ab.length_squared() < 1e-12 {
            return point_segment(a, c, d) < 1e-6;
        }
        if (c - a).perp_dot(ab).abs() > 1e-6 {
            return false;
        }
        let lo = (c - a).dot(ab) / ab.length_squared();
        let hi = (d - a).dot(ab) / ab.length_squared();
        return lo.min(hi) <= 1. && lo.max(hi) >= 0.;
    }
    let t = (c - a).perp_dot(cd) / denominator;
    let u = (c - a).perp_dot(ab) / denominator;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

/// Continuous capsule containment, including concave boundaries and tangencies.
pub fn segment_inside(poly: &[Vec2], a: Vec2, b: Vec2, margin: f32) -> bool {
    contains(poly, a, margin)
        && contains(poly, b, margin)
        && edges(poly).all(|(c, d)| {
            !intersects(a, b, c, d)
                && [
                    point_segment(a, c, d),
                    point_segment(b, c, d),
                    point_segment(c, a, b),
                    point_segment(d, a, b),
                ]
                .into_iter()
                .all(|x| x + 1e-6 >= margin)
        })
}

pub fn box_inside(poly: &[Vec2], lo: Vec2, hi: Vec2, margin: f32) -> bool {
    let corners = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
    let inside = edges(&corners).all(|(a, b)| segment_inside(poly, a, b, margin));
    inside
}

/// Triangulate a simple polygon into CCW triangles, preserving thin pieces.
/// Degenerate/non-finite inputs produce no triangles. Untriangulable input is
/// reported and discarded as a whole, never emitted as a partial surface.
pub fn triangles(poly: &[Vec2]) -> Vec<[Vec2; 3]> {
    if poly.len() < 3 || !poly.iter().all(|p| p.is_finite()) {
        return Vec::new();
    }
    let winding = signed_area(poly);
    if winding == 0. {
        return Vec::new();
    }
    let mut indices: Vec<_> = (0..poly.len()).collect();
    if winding < 0. {
        indices.reverse();
    }
    let mut result = Vec::with_capacity(poly.len() - 2);
    // Half-plane clipping can repeat a vertex on a clip plane or retain a
    // collinear run. Ear removal can expose another such run, so normalize on
    // every iteration. Only exact zero-area vertices are redundant.
    loop {
        if indices.len() < 3 {
            return result;
        }
        let redundant = (0..indices.len()).find(|&i| {
            let a = poly[indices[(i + indices.len() - 1) % indices.len()]];
            let b = poly[indices[i]];
            let c = poly[indices[(i + 1) % indices.len()]];
            orient(a, b, c) == 0.
        });
        if let Some(i) = redundant {
            indices.remove(i);
            continue;
        }
        let ear = (0..indices.len()).find(|&i| {
            let ids = [
                indices[(i + indices.len() - 1) % indices.len()],
                indices[i],
                indices[(i + 1) % indices.len()],
            ];
            let [a, b, c] = ids.map(|j| poly[j]);
            orient(a, b, c) > 0.
                && indices.iter().all(|j| {
                    ids.contains(j) || {
                        let p = poly[*j];
                        orient(a, b, p) < 0. || orient(b, c, p) < 0. || orient(c, a, p) < 0.
                    }
                })
        });
        let Some(ear) = ear else {
            warn!("cannot triangulate envelope polygon: {poly:?}");
            return Vec::new();
        };
        result.push([
            poly[indices[(ear + indices.len() - 1) % indices.len()]],
            poly[indices[ear]],
            poly[indices[(ear + 1) % indices.len()]],
        ]);
        if indices.len() == 3 {
            return result;
        }
        indices.remove(ear);
    }
}

// A clipped convex polygon is still convex. Rounding intersections back to
// f32 can introduce a one-ULP backtracking edge (seed 106), so restore that
// invariant with sign predicates, not a distance/area epsilon. This helper is
// deliberately confined to convex clipping, never to a concave footprint.
fn convex_hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    fn append(hull: &mut Vec<Vec2>, p: Vec2) {
        while hull.len() >= 2 && orient(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0. {
            hull.pop();
        }
        hull.push(p);
    }
    let mut lower = Vec::with_capacity(points.len());
    let mut upper = Vec::with_capacity(points.len());
    for (&a, &b) in points.iter().zip(points.iter().rev()) {
        append(&mut lower, a);
        append(&mut upper, b);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Clip a convex polygon by one half plane, n.dot(p) >= offset.
/// The result is a convex CCW ring, or fewer than three vertices if collapsed.
pub fn clip(poly: &[Vec2], n: Vec2, offset: f32) -> Vec<Vec2> {
    let mut result = Vec::new();
    let n = n.as_dvec2();
    let offset = f64::from(offset);
    for (a, b) in edges(poly) {
        let da = n.dot(a.as_dvec2()) - offset;
        let db = n.dot(b.as_dvec2()) - offset;
        if da >= 0. {
            result.push(a);
        }
        if (da >= 0.) != (db >= 0.) {
            result
                .push((a.as_dvec2() + (b.as_dvec2() - a.as_dvec2()) * (da / (da - db))).as_vec2());
        }
    }
    convex_hull(result)
}

pub fn clip_box(poly: &[Vec2], lo: Vec2, hi: Vec2) -> Vec<Vec2> {
    let mut p = poly.to_vec();
    for (n, d) in [
        (Vec2::X, lo.x),
        (-Vec2::X, -hi.x),
        (Vec2::Y, lo.y),
        (-Vec2::Y, -hi.y),
    ] {
        p = clip(&p, n, d);
    }
    p
}

pub fn line_intervals(poly: &[Vec2], axis: usize, coordinate: f32) -> Vec<(f32, f32)> {
    let other = 1 - axis;
    let mut hits = Vec::new();
    for (a, b) in edges(poly) {
        if (a[axis] <= coordinate && b[axis] > coordinate)
            || (b[axis] <= coordinate && a[axis] > coordinate)
        {
            hits.push(
                a[other] + (coordinate - a[axis]) / (b[axis] - a[axis]) * (b[other] - a[other]),
            );
        }
    }
    hits.sort_by(f32::total_cmp);
    hits.as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0], p[1]))
        .collect()
}

/// Convex pieces of a bounding rectangle outside the concave envelope.
pub fn outside(poly: &[Vec2], lo: Vec2, hi: Vec2) -> Vec<Vec<Vec2>> {
    let mut pieces = vec![vec![lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]];
    for tri in triangles(poly) {
        pieces = pieces
            .into_iter()
            .flat_map(|piece| {
                let mut inside = piece;
                let mut outside = Vec::new();
                for (a, b) in edges(&tri) {
                    let n = Vec2::new(a.y - b.y, b.x - a.x);
                    let d = n.dot(a);
                    let p = clip(&inside, -n, -d);
                    if p.len() >= 3 && area(&p) > 1e-7 {
                        outside.push(p);
                    }
                    inside = clip(&inside, n, d);
                }
                outside
            })
            .collect();
    }
    pieces
}

pub fn validate(poly: &[Vec2]) -> Result<(), String> {
    if poly.len() < 3 || poly.len() > 32 || !poly.iter().all(|p| p.is_finite()) || area(poly) < 4. {
        return Err("invalid envelope footprint".into());
    }
    let e: Vec<_> = edges(poly).collect();
    for (i, &(a, b)) in e.iter().enumerate() {
        if a.distance(b) < 0.08 {
            return Err("collapsed footprint edge".into());
        }
        for (j, &(c, d)) in e.iter().enumerate().skip(i + 1) {
            if j == i + 1 || (i == 0 && j + 1 == e.len()) {
                continue;
            }
            if intersects(a, b, c, d) {
                return Err("self-intersecting footprint".into());
            }
        }
    }
    Ok(())
}
