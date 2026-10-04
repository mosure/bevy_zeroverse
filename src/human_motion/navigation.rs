//! Continuous travel programs and checked corner fillets on the placed scene.
//! Every returned segment is tested; a rejected fillet retains its original corner.
use super::planning::{route, segment_clear};
use crate::scene::procedural_indoor::layout::IndoorManifest;
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationRecipe {
    pub kind: String,
    pub lateral_fraction: f32,
    pub return_fraction: f32,
    pub corner_radius_m: f32,
    pub path_length_m: f32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample(
    scene: &IndoorManifest,
    a: Vec3,
    b: Vec3,
    height: f32,
    radius: f32,
    boxes: &[(Vec3, Vec3)],
    rng: &mut impl Rng,
) -> Option<(Vec<Vec3>, NavigationRecipe)> {
    let delta = b - a;
    let lateral = Vec3::new(-delta.z, 0.0, delta.x);
    let bend = rng.random_range(-0.38..0.38);
    let corner_radius_m = rng.random_range(0.18..0.55);
    let return_fraction = if rng.random_bool(0.22) {
        rng.random_range(0.0..0.35)
    } else {
        1.0
    };
    let mut anchors = vec![a];
    let kind = if return_fraction < 1.0 {
        anchors = turnaround(a, b, return_fraction, bend, corner_radius_m);
        "return"
    } else if bend.abs() > 0.08 {
        anchors.extend([a.lerp(b, rng.random_range(0.35..0.65)) + lateral * bend, b]);
        "bend"
    } else {
        anchors.push(b);
        "direct"
    };
    let mut path = vec![a];
    for pair in anchors.windows(2) {
        let segment = route(scene, pair[0], pair[1], height, radius, boxes)?;
        path.extend(segment.into_iter().skip(1));
    }
    let path = fillet(&path, corner_radius_m, height, radius, boxes);
    let length = path.windows(2).map(|p| p[0].distance(p[1])).sum();
    Some((
        path,
        NavigationRecipe {
            kind: kind.into(),
            lateral_fraction: bend,
            return_fraction,
            corner_radius_m,
            path_length_m: length,
        },
    ))
}

/// A reversal needs physical turning space. A quadratic fillet of an out-and-
/// back line has a zero tangent at its cusp and creates an abrupt 180° heading.
/// An explicit semicircle keeps travel direction finite; routing checks every
/// chord afterward and rejects turns that cannot fit the furnished room.
fn turnaround(a: Vec3, b: Vec3, fraction: f32, bend: f32, radius: f32) -> Vec<Vec3> {
    let forward = (b - a).normalize_or(Vec3::Z);
    let side = Vec3::new(-forward.z, 0., forward.x) * if bend < 0. { -1. } else { 1. };
    let radius = radius.min(a.distance(b) * 0.22);
    let centre = b - forward * radius;
    let mut path = vec![a, a.lerp(b, 0.32) - side * radius];
    for i in 0..=12 {
        let angle = i as f32 * std::f32::consts::PI / 12.;
        path.push(centre - side * (radius * angle.cos()) + forward * (radius * angle.sin()));
    }
    path.push(a.lerp(b, fraction) + side * radius);
    path
}

fn fillet(
    path: &[Vec3],
    corner: f32,
    height: f32,
    radius: f32,
    boxes: &[(Vec3, Vec3)],
) -> Vec<Vec3> {
    let mut rounded = vec![path[0]];
    for points in path.windows(3) {
        let [a, b, c] = [points[0], points[1], points[2]];
        let cut = corner.min(a.distance(b) * 0.35).min(c.distance(b) * 0.35);
        let enter = b + (a - b).normalize_or_zero() * cut;
        let leave = b + (c - b).normalize_or_zero() * cut;
        let curve: Vec<_> = (0..=4)
            .map(|i| {
                let t = i as f32 * 0.25;
                enter.lerp(b, t).lerp(b.lerp(leave, t), t)
            })
            .collect();
        let clear = std::iter::once(*rounded.last().unwrap())
            .chain(curve.iter().copied())
            .collect::<Vec<_>>()
            .windows(2)
            .all(|p| segment_clear(p[0], p[1], height, radius, boxes));
        if clear {
            rounded.extend(curve);
        } else {
            rounded.push(b);
        }
    }
    rounded.push(*path.last().unwrap());
    rounded.dedup_by(|a, b| a.distance_squared(*b) < 1e-8);
    rounded
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn return_walks_have_no_cusp_or_instantaneous_heading_reversal() {
        for direction in [-1., 1.] {
            for length in [1.1, 2.0, 4.5] {
                let p = turnaround(Vec3::ZERO, Vec3::Z * length, 0.2, direction, 0.45);
                let p = fillet(&p, 0.3, 1.8, 0.1, &[]);
                let headings: Vec<_> = p.windows(2).map(|p| (p[1] - p[0]).normalize()).collect();
                assert!(headings.iter().all(|h| h.is_finite()));
                assert!(
                    headings.windows(2).all(|h| h[0].dot(h[1]) > 0.8),
                    "abrupt reversal"
                );
                assert!(headings.first().unwrap().z > 0.8 && headings.last().unwrap().z < -0.8);
            }
        }
    }
    #[test]
    fn corners_are_smoothed_only_when_the_swept_curve_is_clear() {
        let path = [Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO, Vec3::Z];
        let clear = fillet(&path, 0.4, 1.8, 0.1, &[]);
        assert!(clear.len() > 3);
        assert!(clear.iter().all(|p| p.is_finite()));
        let block = [(Vec3::new(-0.5, 0.0, 0.2), Vec3::new(-0.2, 1.0, 0.5))];
        let checked = fillet(&path, 0.4, 1.8, 0.1, &block);
        assert!(checked
            .windows(2)
            .all(|p| segment_clear(p[0], p[1], 1.8, 0.1, &block)));
        assert_eq!(checked.first(), path.first());
        assert_eq!(checked.last(), path.last());
    }
}
