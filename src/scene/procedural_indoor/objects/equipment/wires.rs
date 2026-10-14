use super::*;
mod catenary;
pub(crate) use catenary::hanging_curve;

pub(crate) fn radius(o: &IndoorObject) -> f32 {
    stream(o.seed, 5750).random_range(0.0018..0.0033)
}

/// Connector strain relief is wider than the flexible jacket. The exact same
/// radius pads the archived object bounds and is used by segment clearance.
pub(crate) fn bound_radius(o: &IndoorObject) -> f32 {
    radius(o) * 1.8
}

/// Realized, sampled routes are bounded even when they contain curved bends.
pub(crate) const MAX_PATH_POINTS: usize = 128;

/// C1 circular fillets with a realistic five-jacket-radius bend. The realized
/// samples are archived before bounds and collision tests; no hidden renderer
/// deformation can move the wire outside its qualified route.
pub(crate) fn fillet_path(points: &[Vec3], jacket: f32) -> Option<Vec<Vec3>> {
    if !(2..=64).contains(&points.len()) || points.iter().any(|p| !p.is_finite()) {
        return None;
    }
    let bend_radius = jacket * 5.;
    let mut output = Vec::with_capacity(points.len() * 6);
    output.push(points[0]);
    for i in 1..points.len() - 1 {
        let before = points[i] - points[i - 1];
        let after = points[i + 1] - points[i];
        let before_length = before.length();
        let after_length = after.length();
        if before_length < 0.0001 || after_length < 0.0001 {
            return None;
        }
        let u = before / before_length;
        let v = after / after_length;
        let cosine = u.dot(v).clamp(-1., 1.);
        let angle = cosine.acos();
        if angle < 0.015 {
            output.push(points[i]);
            continue;
        }
        let cos_half = ((1. + cosine) * 0.5).sqrt();
        if cos_half < 0.01 {
            return None;
        }
        let cut = bend_radius * ((1. - cosine) * 0.5).sqrt() / cos_half;
        if cut > before_length.min(after_length) * 0.45
            || i == 1 && before_length - cut < 0.012
            || i == points.len() - 2 && after_length - cut < 0.012
        {
            return None;
        }
        let axis = u.cross(v).try_normalize()?;
        let entry = points[i] - u * cut;
        let center = entry + axis.cross(u) * bend_radius;
        let radial = entry - center;
        let steps = (angle / 15_f32.to_radians()).ceil().max(1.) as usize;
        for step in 0..=steps {
            output.push(
                center + Quat::from_axis_angle(axis, angle * step as f32 / steps as f32) * radial,
            );
        }
        if output.len() >= MAX_PATH_POINTS {
            return None;
        }
    }
    output.push(*points.last()?);
    if !path_points_qualified(&output, jacket) {
        return None;
    }
    Some(output)
}

/// The exact rendered route keeps a gentle frame turn and finite positive
/// curvature. Radius tolerance covers sampled f32 arc coordinates only; kinked
/// or reversed polylines are rejected before constructing their sweep mesh.
pub(crate) fn path_qualified(o: &IndoorObject) -> bool {
    o.cable_path
        .as_ref()
        .is_some_and(|p| path_points_qualified(p, radius(o)))
}

fn path_points_qualified(points: &[Vec3], jacket: f32) -> bool {
    if !(2..=MAX_PATH_POINTS).contains(&points.len())
        || points.iter().any(|p| !p.is_finite())
        || points[0].distance(points[1]) < 0.012
        || points[points.len() - 1].distance(points[points.len() - 2]) < 0.012
        || points
            .windows(2)
            .any(|s| s[0].distance_squared(s[1]) < 1e-8)
    {
        return false;
    }
    let turn_cosine = 15.2_f64.to_radians().cos();
    let minimum_radius = jacket as f64 * 4.8;
    for triple in points.windows(3) {
        let a = (triple[1] - triple[0]).as_dvec3();
        let b = (triple[2] - triple[1]).as_dvec3();
        let length_a = a.length();
        let length_b = b.length();
        if a.dot(b) / (length_a * length_b) < turn_cosine {
            return false;
        }
        let cross = a.cross(b).length();
        if cross > length_a * length_b * 1e-6 {
            let curvature_radius = length_a * length_b * (a + b).length() / (2. * cross);
            if curvature_radius < minimum_radius {
                return false;
            }
        }
    }
    true
}

/// Stable metric distance between two closed segments, evaluated in f64 to
/// preserve near-parallel and short millimetre connector cases symmetrically.
pub(crate) fn segment_distance_squared(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let u = (b - a).as_dvec3();
    let v = (d - c).as_dvec3();
    let w = (a - c).as_dvec3();
    let aa = u.length_squared();
    let bb = u.dot(v);
    let cc = v.length_squared();
    let dd = u.dot(w);
    let ee = v.dot(w);
    let mut s;
    let t;
    if aa < 1e-20 && cc < 1e-20 {
        return w.length_squared();
    } else if aa < 1e-20 {
        s = 0.;
        t = (ee / cc).clamp(0., 1.);
    } else if cc < 1e-20 {
        t = 0.;
        s = (-dd / aa).clamp(0., 1.);
    } else {
        let denominator = aa * cc - bb * bb;
        s = if denominator > 32. * f64::EPSILON * aa * cc {
            ((bb * ee - cc * dd) / denominator).clamp(0., 1.)
        } else {
            0.
        };
        let proposed = (bb * s + ee) / cc;
        if proposed < 0. {
            t = 0.;
            s = (-dd / aa).clamp(0., 1.);
        } else if proposed > 1. {
            t = 1.;
            s = ((bb - dd) / aa).clamp(0., 1.);
        } else {
            t = proposed;
        }
    }
    (w + u * s - v * t).length_squared()
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let Some(points) = &o.cable_path else {
        return;
    };
    if !path_qualified(o) {
        return;
    }
    let r = radius(o);
    let label = o.kind.class_name();
    a.part(Surface::Rubber, label).tube(points, r, 8);
    for (end, next) in [
        (points[0], points[1]),
        (*points.last().unwrap(), points[points.len() - 2]),
    ] {
        let direction = (next - end).normalize_or_zero();
        a.part(Surface::Plastic, label).rod(
            end,
            end + direction * end.distance(next).min(0.012),
            bound_radius(o),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(points: Vec<Vec3>, seed: u64) -> IndoorObject {
        IndoorObject {
            finish_seed: None,
            id: 0,
            kind: ObjectKind::Cable,
            position: Vec3::ZERO,
            size: Vec3::ONE,
            yaw: 0.,
            variant: 0,
            seed,
            solid: false,
            support: None,
            neighbor: false,
            interaction_target: None,
            cable_path: Some(points),
            cable_target: None,
        }
    }

    #[test]
    fn realized_cable_bends_keep_realistic_radius_positive_winding_and_exact_bounds() {
        for seed in 0..96 {
            for degrees in [5_f32, 30., 60., 90., 105., 120., 125.] {
                let angle = degrees.to_radians();
                let mut o = wire(
                    vec![
                        Vec3::ZERO,
                        Vec3::X * 0.08,
                        Vec3::X * 0.08 + Vec3::new(angle.cos(), angle.sin(), 0.) * 0.30,
                    ],
                    seed,
                );
                let original = o.cable_path.clone().unwrap();
                let realized = fillet_path(&original, radius(&o)).expect("physical bend");
                assert_eq!(realized[0], original[0]);
                assert_eq!(realized.last(), original.last());
                assert!(realized.len() <= MAX_PATH_POINTS);
                o.cable_path = Some(realized);
                assert!(path_qualified(&o));
                let points = o.cable_path.as_ref().unwrap();
                let mut g = Geometry::default();
                g.tube(points, radius(&o), 8);
                let lo = points
                    .iter()
                    .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
                    - Vec3::splat(bound_radius(&o));
                let hi = points
                    .iter()
                    .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
                    + Vec3::splat(bound_radius(&o));
                for p in &g.positions {
                    assert!(Vec3::from_array(*p).cmpge(lo - Vec3::splat(1e-6)).all());
                    assert!(Vec3::from_array(*p).cmple(hi + Vec3::splat(1e-6)).all());
                }
                for triangle in g.indices.as_chunks::<3>().0 {
                    let p = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let face = (p[1] - p[0]).cross(p[2] - p[0]);
                    let normal = triangle
                        .iter()
                        .map(|&i| Vec3::from_array(g.normals[i as usize]))
                        .sum::<Vec3>();
                    assert!(face.length_squared() > 1e-20);
                    assert!(face.dot(normal) > 0., "seed {seed} bend {degrees}");
                }
            }
        }
        let sharp = wire(
            vec![Vec3::ZERO, Vec3::X * 0.025, Vec3::X * 0.025 + Vec3::Y * 0.1],
            0,
        );
        assert!(!path_qualified(&sharp));
        let short = [Vec3::ZERO, Vec3::X * 0.003, Vec3::X * 0.003 + Vec3::Y * 0.1];
        assert!(fillet_path(&short, 0.003).is_none());
        let reversal = [Vec3::ZERO, Vec3::X * 0.025, Vec3::X * 0.025 - Vec3::X * 0.3];
        assert!(fillet_path(&reversal, 0.003).is_none());
    }

    #[test]
    fn capsule_clearance_is_symmetric_for_short_parallel_skew_and_degenerate_segments() {
        for (a, b, c, d, expected) in [
            (
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y * 0.01,
                Vec3::X + Vec3::Y * 0.01,
                0.0001,
            ),
            (
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(0.5, -1., 0.),
                Vec3::new(0.5, 1., 0.),
                0.,
            ),
            (
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(0.5, 0., 0.01),
                Vec3::new(0.5, 1., 0.01),
                0.0001,
            ),
            (Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y * 2., 1.),
            (
                Vec3::ZERO,
                Vec3::X * 0.001,
                Vec3::X * 0.002,
                Vec3::X * 0.003,
                0.000001,
            ),
        ] {
            let distance = segment_distance_squared(a, b, c, d);
            assert!((distance - expected).abs() < 1e-8);
            assert!((distance - segment_distance_squared(c, d, a, b)).abs() < 1e-12);
            assert!((distance - segment_distance_squared(b, a, d, c)).abs() < 1e-12);
        }
    }
}
