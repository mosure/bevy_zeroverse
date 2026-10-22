//! Conservative angular-to-perspective Jacobian bound on the portion of each
//! source view that can reach the sensor. Interval cells exclude only rays
//! certified outside the image/domain; no pixel or scene-dependent heuristic.
use crate::calibration::{CameraCalibration, Fisheye};

#[derive(Clone, Copy)]
pub(super) struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    pub(super) fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }
    pub(super) fn point(x: f64) -> Self {
        Self::new(x, x)
    }
    pub(super) fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
    fn abs_min(self) -> f64 {
        if self.lo <= 0. && self.hi >= 0. {
            0.
        } else {
            self.lo.abs().min(self.hi.abs())
        }
    }
    fn abs_max(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    fn mul(self, b: Self) -> Self {
        let x = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self::new(
            x.into_iter().fold(f64::INFINITY, f64::min),
            x.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn add(self, b: Self) -> Self {
        Self::new(self.lo + b.lo, self.hi + b.hi)
    }
}

pub(super) fn angular_bounds(d: [Interval; 3]) -> (Interval, Interval) {
    let r = Interval::new(
        d[0].abs_min().hypot(d[1].abs_min()),
        d[0].abs_max().hypot(d[1].abs_max()),
    );
    let z = d[2].neg();
    let a = [
        r.lo.atan2(z.lo),
        r.lo.atan2(z.hi),
        r.hi.atan2(z.lo),
        r.hi.atan2(z.hi),
    ];
    (
        r,
        Interval::new(
            a.into_iter().fold(f64::INFINITY, f64::min),
            a.into_iter().fold(0., f64::max),
        ),
    )
}

pub(super) fn may_reach_sensor(
    c: &CameraCalibration,
    lens: Fisheye,
    d: [Interval; 3],
    r: Interval,
    theta: Interval,
) -> bool {
    if theta.lo > lens.max_angle as f64 {
        return false;
    }
    let radial = Interval::new(
        lens.radius(theta.lo),
        lens.radius(theta.hi.min(lens.max_angle as f64)),
    );
    let direction = |a: Interval| {
        if r.lo <= 1e-12 {
            Interval::new(-1., 1.)
        } else {
            let v = a.mul(Interval::new(1. / r.hi, 1. / r.lo));
            Interval::new(v.lo.max(-1.), v.hi.min(1.))
        }
    };
    let x = direction(d[0]).mul(radial);
    let y = direction(d[1].neg()).mul(radial);
    let px = x
        .mul(Interval::point(c.k[0][0] as f64))
        .add(y.mul(Interval::point(c.k[0][1] as f64)))
        .add(Interval::point(c.k[0][2] as f64));
    let py = y
        .mul(Interval::point(c.k[1][1] as f64))
        .add(Interval::point(c.k[1][2] as f64));
    // Include the bilinear reconstruction footprint at the sensor boundary.
    px.hi >= -2.
        && px.lo <= c.image_size[0] as f64 + 2.
        && py.hi >= -2.
        && py.lo <= c.image_size[1] as f64 + 2.
}

fn polynomial_bound(lens: Fisheye, theta: Interval, derivative: bool) -> f64 {
    1. + lens
        .coefficients
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            let power = 2 * (i as i32 + 1);
            let a = if k < 0. { theta.lo } else { theta.hi };
            k as f64 * a.powi(power) * if derivative { (power + 1) as f64 } else { 1. }
        })
        .sum::<f64>()
}

pub(in crate::camera::lens) fn density(c: &CameraCalibration, lens: Fisheye) -> f32 {
    // At tiny sizes a guard can be large relative to the core; use the full
    // angular bound instead of assuming the usual <=25% perspective overlap.
    let tiny = c.image_size.into_iter().min().unwrap() < 64;
    let domain = Interval::new(0., lens.max_angle as f64);
    let angular = |theta: Interval| {
        let tangential = polynomial_bound(lens, theta, false)
            * if theta.hi < 1e-9 {
                1.
            } else {
                theta.hi / theta.hi.sin()
            };
        polynomial_bound(lens, theta, true).max(tangential)
    };
    if tiny {
        return angular(domain).max(1.) as f32;
    }
    let mut maximum = 1_f64;
    const CELLS: u32 = 24;
    for face in 0..5 {
        for y in 0..CELLS {
            for x in 0..CELLS {
                let interval = |i: u32| {
                    Interval::new(
                        -1.25 + 2.5 * i as f64 / CELLS as f64,
                        -1.25 + 2.5 * (i + 1) as f64 / CELLS as f64,
                    )
                };
                let (u, v) = (interval(x), interval(y));
                let one = Interval::point(1.);
                let d = match face {
                    0 => [u, v, one.neg()],
                    1 => [one, v, u],
                    2 => [one.neg(), v, u.neg()],
                    3 => [u, one, v],
                    _ => [u, one.neg(), v.neg()],
                };
                let (r, mut theta) = angular_bounds(d);
                if !may_reach_sensor(c, lens, d, r, theta) {
                    continue;
                }
                theta.hi = theta.hi.min(domain.hi);
                // D(normalize([u,v,1])) has operator norm 1/|[u,v,1]|.
                // The angular projection has radial/tangential singular values
                // g'(theta), g(theta)/sin(theta). Bound their product per cell.
                let length = (1. + u.abs_min().powi(2) + v.abs_min().powi(2)).sqrt();
                maximum = maximum.max(angular(theta) / length);
            }
        }
    }
    // A heavily shifted circular image may be tiny inside a large sensor.
    // Its minimum guard would extend past the certified source-coordinate box.
    if 2. * c.k[0][0].max(c.k[1][1]) as f64 * maximum < 32. {
        maximum = angular(domain).max(1.);
    }
    // Outward inflation covers f32 calibration and interval arithmetic roundoff.
    (maximum * 1.00001) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::{default, Vec3};
    #[test]
    fn certified_density_covers_the_projection_jacobian_for_varied_sensors() {
        use crate::camera::lens::LensSampling;
        for seed in 0..96 {
            let prior = LensSampling {
                probability: 1.,
                ..default()
            };
            let sensor = prior.sample(seed, 0).unwrap();
            let size = if seed % 2 == 0 {
                [512, 512]
            } else {
                [960, 540]
            };
            let c = sensor.calibration(size, [0.12, -0.08]).unwrap();
            let limit = density(&c, sensor.lens) as f64;
            for face in 0..5 {
                let orientation = super::super::super::runtime::orientations()[face];
                for y in 0..=48 {
                    for x in 0..=48 {
                        let u = -1.2 + 2.4 * x as f32 / 48.;
                        let v = -1.2 + 2.4 * y as f32 / 48.;
                        let d = orientation * Vec3::new(u, v, -1.);
                        if !c.project(d.to_array()).is_some_and(|p| c.in_image(p)) {
                            continue;
                        }
                        let theta = (-d.z / d.length()).clamp(-1., 1.).acos() as f64;
                        let t = Interval::point(theta);
                        let radial = polynomial_bound(sensor.lens, t, true);
                        let tangential = if theta < 1e-6 {
                            1.
                        } else {
                            sensor.lens.radius(theta) / theta.sin()
                        };
                        let actual_bound = radial.max(tangential) / d.length() as f64;
                        assert!(
                            actual_bound <= limit + 0.00002,
                            "{seed} {face} {u} {v}: {actual_bound} > {limit}"
                        );
                    }
                }
            }
        }
    }
}
