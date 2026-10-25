//! Renderer-independent measurements of synchronized camera poses.
//! Thresholds describe pose similarity, not image duplication or training utility.
use serde::{Deserialize, Serialize};

pub const TIMES: usize = 33;
pub type Point = [f64; 3];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Path {
    pub positions: Vec<Point>,
    pub forwards: Vec<Point>,
    pub target: Point,
    pub fov_degrees: f64,
    pub length_m: f64,
    /// Preserve calibrated lens/mount fields from the live generator wire type.
    #[serde(default, flatten)]
    pub metadata: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Group {
    pub seed: u64,
    pub baseline: f64,
    pub camera_count: usize,
    pub error: Option<String>,
    pub cameras: Vec<Path>,
    pub policy: serde_json::Value,
    pub geometry: serde_json::Value,
    /// Production placement proxy, including both directions and all five times.
    pub reference_overlap: serde_json::Value,
    /// Keep peer-overlap, mixed-rig and coherence diagnostics in published data.
    #[serde(default, flatten)]
    pub metadata: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub start_nearest_neighbor_m: Vec<f64>,
    pub pair_baseline_m: Vec<f64>,
    pub path_lengths_m: Vec<f64>,
    pub pair_optical_angle_degrees: Vec<f64>,
    pub max_start_optical_angle_degrees: Option<f64>,
    pub max_optical_angle_degrees: Option<f64>,
    pub start_azimuth_span_degrees: Option<f64>,
    pub target_diameter_m: Option<f64>,
    pub common_axis_residual_m: Option<f64>,
    pub exact_pose_duplicate_pairs: usize,
    pub similar_pose_pairs: usize,
    pub pair_count: usize,
    pub similar_moving_shape_pairs: usize,
    pub moving_pair_count: usize,
    pub pair_relative_motion: Vec<f64>,
}

pub fn sub(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}
pub fn dot(a: Point, b: Point) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
pub fn length(a: Point) -> f64 {
    dot(a, a).sqrt()
}
fn angle(a: Point, b: Point) -> f64 {
    (dot(a, b) / (length(a) * length(b)))
        .clamp(-1., 1.)
        .acos()
        .to_degrees()
}

/// Least-squares distance to infinite optical-axis lines, at the initial pose.
/// Parallel / ill-conditioned lines have no well-defined common focus.
fn common_axis_residual(paths: &[Path]) -> Option<f64> {
    if paths.len() < 2 {
        return None;
    }
    let mut a = [[0.; 3]; 3];
    let mut b = [0.; 3];
    for path in paths {
        let p = path.positions[0];
        let f = path.forwards[0];
        for i in 0..3 {
            for j in 0..3 {
                let v = f64::from(i == j) - f[i] * f[j];
                a[i][j] += v;
                b[i] += v * p[j];
            }
        }
    }
    let mut m: [[f64; 4]; 3] = std::array::from_fn(|i| [a[i][0], a[i][1], a[i][2], b[i]]);
    for i in 0..3 {
        let pivot = (i..3).max_by(|&x, &y| m[x][i].abs().total_cmp(&m[y][i].abs()))?;
        if m[pivot][i].abs() < 1e-9 * paths.len() as f64 {
            return None;
        }
        m.swap(i, pivot);
        let divisor = m[i][i];
        for value in &mut m[i][i..] {
            *value /= divisor;
        }
        for k in 0..3 {
            if k != i {
                let factor = m[k][i];
                for j in i..4 {
                    m[k][j] -= factor * m[i][j];
                }
            }
        }
    }
    let center = [m[0][3], m[1][3], m[2][3]];
    Some(
        (paths
            .iter()
            .map(|path| {
                let d = sub(center, path.positions[0]);
                (dot(d, d) - dot(d, path.forwards[0]).powi(2)).max(0.)
            })
            .sum::<f64>()
            / paths.len() as f64)
            .sqrt(),
    )
}

pub fn measure(paths: &[Path]) -> Metrics {
    let mut m = Metrics {
        start_nearest_neighbor_m: Vec::new(),
        pair_baseline_m: Vec::new(),
        path_lengths_m: paths.iter().map(|p| p.length_m).collect(),
        pair_optical_angle_degrees: Vec::new(),
        max_start_optical_angle_degrees: None,
        max_optical_angle_degrees: None,
        start_azimuth_span_degrees: None,
        target_diameter_m: None,
        common_axis_residual_m: common_axis_residual(paths),
        exact_pose_duplicate_pairs: 0,
        similar_pose_pairs: 0,
        pair_count: paths.len() * paths.len().saturating_sub(1) / 2,
        similar_moving_shape_pairs: 0,
        moving_pair_count: 0,
        pair_relative_motion: Vec::new(),
    };
    if paths.len() < 2 {
        return m;
    }
    let mut target_diameter: f64 = 0.;
    let mut max_start_angle: f64 = 0.;
    let mut max_angle: f64 = 0.;
    for (i, a) in paths.iter().enumerate() {
        m.start_nearest_neighbor_m.push(
            paths
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, b)| length(sub(a.positions[0], b.positions[0])))
                .fold(f64::INFINITY, f64::min),
        );
        for b in &paths[i + 1..] {
            target_diameter = target_diameter.max(length(sub(a.target, b.target)));
            let (mut separation, mut heading, mut residual, mut travel_a, mut travel_b) =
                (0., 0., 0., 0., 0.);
            let mut exact = true;
            for t in 0..TIMES {
                let d = sub(a.positions[t], b.positions[t]);
                let degrees = angle(a.forwards[t], b.forwards[t]);
                m.pair_baseline_m.push(length(d));
                m.pair_optical_angle_degrees.push(degrees);
                max_angle = max_angle.max(degrees);
                if t == 0 {
                    max_start_angle = max_start_angle.max(degrees);
                }
                separation += dot(d, d);
                heading += degrees * degrees;
                exact &= a.positions[t] == b.positions[t] && a.forwards[t] == b.forwards[t];
                let da = sub(a.positions[t], a.positions[0]);
                let db = sub(b.positions[t], b.positions[0]);
                residual += dot(sub(da, db), sub(da, db));
                travel_a += dot(da, da);
                travel_b += dot(db, db);
            }
            m.exact_pose_duplicate_pairs += usize::from(exact);
            // Intentionally close stereo rigs can satisfy this diagnostic.
            m.similar_pose_pairs += usize::from(
                (separation / TIMES as f64).sqrt() < 0.15 && (heading / TIMES as f64).sqrt() < 5.,
            );
            let travel = travel_a.max(travel_b);
            if travel > 1e-8 {
                let relative = (residual / travel).sqrt();
                m.pair_relative_motion.push(relative);
                m.moving_pair_count += 1;
                m.similar_moving_shape_pairs += usize::from(relative < 0.25);
            }
        }
    }
    let center: Point = std::array::from_fn(|axis| {
        paths.iter().map(|p| p.target[axis]).sum::<f64>() / paths.len() as f64
    });
    let mut angles: Vec<_> = paths
        .iter()
        .map(|p| {
            let d = sub(p.positions[0], center);
            d[2].atan2(d[0]).rem_euclid(std::f64::consts::TAU)
        })
        .collect();
    angles.sort_by(f64::total_cmp);
    let largest_gap = angles
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .chain(std::iter::once(
            angles[0] + std::f64::consts::TAU - angles[angles.len() - 1],
        ))
        .fold(0., f64::max);
    m.start_azimuth_span_degrees = Some((std::f64::consts::TAU - largest_gap).to_degrees());
    m.target_diameter_m = Some(target_diameter);
    m.max_start_optical_angle_degrees = Some(max_start_angle);
    m.max_optical_angle_degrees = Some(max_angle);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(start: Point, end: Point, forward: Point) -> Path {
        Path {
            positions: (0..TIMES)
                .map(|i| std::array::from_fn(|a| start[a] + (end[a] - start[a]) * i as f64 / 32.))
                .collect(),
            forwards: vec![forward; TIMES],
            target: [0.; 3],
            length_m: length(sub(end, start)),
            fov_degrees: 60.,
            metadata: Default::default(),
        }
    }

    #[test]
    fn singleton_and_static_groups_do_not_invent_pair_or_motion_observations() {
        let a = path([0.; 3], [0.; 3], [0., 0., -1.]);
        let single = measure(std::slice::from_ref(&a));
        assert_eq!(single.pair_count, 0);
        assert_eq!(single.max_optical_angle_degrees, None);
        assert!(single.start_nearest_neighbor_m.is_empty());
        let duplicate = measure(&[a.clone(), a]);
        assert_eq!(duplicate.exact_pose_duplicate_pairs, 1);
        assert_eq!(duplicate.similar_pose_pairs, 1);
        assert_eq!(duplicate.moving_pair_count, 0);
        assert_eq!(duplicate.common_axis_residual_m, None);
    }

    #[test]
    fn translated_rig_shapes_are_repeated_even_when_absolute_paths_are_distinct() {
        let m = measure(&[
            path([0.; 3], [0., 0., 1.], [0., 0., -1.]),
            path([2., 0., 0.], [2., 0., 1.], [0., 0., -1.]),
        ]);
        assert_eq!(m.similar_pose_pairs, 0);
        assert_eq!(m.exact_pose_duplicate_pairs, 0);
        assert_eq!(m.similar_moving_shape_pairs, 1);
        assert_eq!(m.pair_relative_motion, [0.]);
        assert_eq!(m.start_nearest_neighbor_m, [2., 2.]);
    }

    #[test]
    fn common_focus_is_distinct_from_narrow_viewing_angle() {
        let m = measure(&[
            path([-1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]),
            path([1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]),
        ]);
        assert_eq!(m.max_optical_angle_degrees, Some(180.));
        assert_eq!(m.start_azimuth_span_degrees, Some(180.));
        // Opposing collinear rays still do not identify a unique focus.
        assert_eq!(m.common_axis_residual_m, None);
        let crossing = measure(&[
            path([-1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]),
            path([0., 0., -1.], [0., 0., -1.], [0., 0., 1.]),
        ]);
        assert!(crossing.common_axis_residual_m.unwrap() < 1e-8);
    }
}
