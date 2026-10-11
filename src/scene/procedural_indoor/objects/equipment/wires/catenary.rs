//! Gravity-shaped cable centerlines, including a supported floor section.
//! The analytic catenary is sampled before collision checks and serialization.
use bevy::prelude::*;

pub(crate) fn hanging_curve(start: Vec3, end: Vec3, floor: f32, tension: f32) -> Option<Vec<Vec3>> {
    if !start.is_finite()
        || !end.is_finite()
        || !floor.is_finite()
        || !tension.is_finite()
        || !(0.10..=0.60).contains(&tension)
        || start.y < floor
        || end.y < floor
    {
        return None;
    }
    let span = start.xz().distance(end.xz()) as f64;
    if !(0.025..=4.).contains(&span) {
        return None;
    }
    let a = (span * tension as f64).max(0.08);
    let dy = (end.y - start.y) as f64;
    let center = span * 0.5 - a * (dy / (2. * a * (span / (2. * a)).sinh())).asinh();
    let offset = start.y as f64 - a * (center / a).cosh();
    let free_height = |x: f64| a * ((x - center) / a).cosh() + offset;
    let supported = free_height(center.clamp(0., span)) < floor as f64;
    let left = a * (1. + (start.y - floor) as f64 / a).acosh();
    let right = span - a * (1. + (end.y - floor) as f64 / a).acosh();
    if supported && left > right + 1e-6 {
        return None;
    }
    let profile = |x: f64| -> (f64, f64) {
        if !supported {
            let u = (x - center) / a;
            (free_height(x), u.sinh())
        } else if x < left {
            let u = (x - left) / a;
            (floor as f64 + a * (u.cosh() - 1.), u.sinh())
        } else if x > right {
            let u = (x - right) / a;
            (floor as f64 + a * (u.cosh() - 1.), u.sinh())
        } else {
            (floor as f64, 0.)
        }
    };
    let mut samples = vec![0.];
    let mut intervals = vec![(0., span, 0u8)];
    while let Some((lo, hi, depth)) = intervals.pop() {
        let mid = (lo + hi) * 0.5;
        let (yl, dl) = profile(lo);
        let (yh, dh) = profile(hi);
        let ym = profile(mid).0;
        if (hi - lo).hypot(yh - yl) > 0.28
            || (dh.atan() - dl.atan()).abs() > 0.16
            || ((yl + yh) * 0.5 - ym).abs() > 0.0015
        {
            if depth >= 14 || samples.len() + intervals.len() >= 58 {
                return None;
            }
            intervals.push((mid, hi, depth + 1));
            intervals.push((lo, mid, depth + 1));
        } else {
            samples.push(hi);
        }
    }
    let mut points: Vec<_> = samples
        .into_iter()
        .map(|x| {
            let mut p = start.lerp(end, (x / span) as f32);
            p.y = profile(x).0 as f32;
            p
        })
        .collect();
    // Avoid accumulated trig/interpolation error at the actual ports/edge.
    points[0] = start;
    *points.last_mut()? = end;
    Some(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_curves_sag_and_rest_on_the_floor_without_penetrating_it() {
        for span in [0.25, 0.7, 1.5, 2.6] {
            for end_y in [0.26, 0.65, 1.1] {
                for tension in [0.10, 0.25, 0.60] {
                    let start = Vec3::new(0., 0.78, 0.);
                    let end = Vec3::new(span, end_y, 0.);
                    let p = hanging_curve(start, end, 0.015, tension).unwrap();
                    assert_eq!(p[0], start);
                    assert_eq!(*p.last().unwrap(), end);
                    assert!(p.len() <= 60);
                    assert!(p.iter().all(|p| p.is_finite() && p.y >= 0.015 - 1e-6));
                    assert!(p
                        .windows(2)
                        .all(|s| s[0].x < s[1].x && s[0].distance(s[1]) <= 0.281));
                    let length: f32 = p.windows(2).map(|s| s[0].distance(s[1])).sum();
                    assert!(length > start.distance(end));
                    if span > 2.5 && tension <= 0.25 {
                        assert!(p.iter().any(|p| p.y < 0.08));
                    }
                }
            }
        }
    }

    #[test]
    fn gravity_route_is_reversible_and_rejects_invalid_endpoints() {
        let a = Vec3::new(1., 0.78, -2.);
        let b = Vec3::new(-1., 0.26, -1.5);
        let forward = hanging_curve(a, b, 0.015, 0.2).unwrap();
        let mut reverse = hanging_curve(b, a, 0.015, 0.2).unwrap();
        reverse.reverse();
        assert_eq!(forward.len(), reverse.len());
        assert!(forward
            .iter()
            .zip(&reverse)
            .all(|(a, b)| a.distance(*b) < 1e-5));
        assert!(hanging_curve(a, b, 0.5, 0.2).is_none());
        assert!(hanging_curve(a, b, 0., f32::NAN).is_none());
        assert!(hanging_curve(a, a, 0., 0.2).is_none());
    }
}
