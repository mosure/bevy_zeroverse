//! Central angular cameras. No renderer, allocation or random-state dependency.
use super::{CameraCalibration, LensModel, PixelConvention};
use serde::{Deserialize, Serialize};

/// Sensor coverage, independent of the angular projection law.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FisheyeFit {
    /// Full-frame / diagonal fisheye: the calibrated domain covers every
    /// sensor corner, including optical-center shifts and unequal focal axes.
    #[default]
    #[serde(alias = "full_frame")]
    Diagonal,
    /// A complete circular image inside the sensor. Requires isotropic pixels.
    Circular,
    /// A complete elliptical image inside the sensor; unequal focal axes are
    /// explicit rather than silently stretching/cropping a circular image.
    Elliptical,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fisheye {
    /// r(theta) = theta * (1 + k1 theta² + ... + k4 theta⁸), theta in radians.
    pub coefficients: [f32; 4],
    /// Maximum polar angle from -Z. Values beyond pi/2 see behind the camera.
    pub max_angle: f32,
}

/// Resolution-independent realized lens. Store this in scene manifests; export
/// its full pixel calibration for every captured view. FOV describes the fitted
/// diameter, not a pinhole vertical FOV.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FisheyeSensor {
    pub lens: Fisheye,
    pub fit: FisheyeFit,
    pub anisotropy: f32,
}
impl FisheyeSensor {
    pub fn calibration(
        &self,
        size: [u32; 2],
        offset: [f32; 2],
    ) -> Result<CameraCalibration, String> {
        self.lens
            .calibration(size, self.fit, offset, self.anisotropy)
    }
}

impl Fisheye {
    pub fn radius(&self, theta: f64) -> f64 {
        let x = theta * theta;
        theta
            * (1.
                + x * self
                    .coefficients
                    .iter()
                    .rev()
                    .fold(0., |a, &k| a * x + k as f64))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.max_angle.is_finite()
            || !(0.15..=2.0).contains(&self.max_angle)
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(
                "fisheye requires finite coefficients and polar domain in [0.15,2] radians".into(),
            );
        }
        // The public contract is executed in f32 WGSL as well as f64 CPU math.
        // Bound the absolute Horner terms to reject overflowing/cancelling
        // custom coefficients even when their f64 derivative is monotonic.
        let x = f64::from(self.max_angle).powi(2);
        let magnitude = 1.
            + self
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, k)| f64::from(k.abs()) * x.powi(i as i32 + 1))
                .sum::<f64>();
        if magnitude * f64::from(self.max_angle).max(9.) > f64::from(f32::MAX) * 0.25 {
            return Err("fisheye coefficients exceed portable f32 arithmetic".into());
        }
        // Interval lower bounds certify monotonicity throughout the whole
        // domain. This is stronger than evaluating a finite set of derivatives.
        let hi = f64::from(self.max_angle).powi(2);
        let derivative: [f64; 4] =
            std::array::from_fn(|i| self.coefficients[i] as f64 * (2 * i + 3) as f64);
        fn certify(k: &[f64; 4], lo: f64, hi: f64, remaining: u32) -> bool {
            let lower = 1.
                + k.iter()
                    .enumerate()
                    .map(|(i, &v)| v * if v >= 0. { lo } else { hi }.powi(i as i32 + 1))
                    .sum::<f64>();
            if lower > 0.02 {
                return true;
            }
            if remaining == 0 {
                return false;
            }
            let mid = (lo + hi) * 0.5;
            certify(k, lo, mid, remaining - 1) && certify(k, mid, hi, remaining - 1)
        }
        if !certify(&derivative, 0., hi, 12) {
            return Err(
                "fisheye polynomial is folded or cannot certify a derivative above 0.02".into(),
            );
        }
        Ok(())
    }

    pub(super) fn project(&self, c: &CameraCalibration, point: [f32; 3]) -> Option<[f32; 2]> {
        if point.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let [x, y, z] = point.map(f64::from);
        let transverse = x.hypot(y);
        if transverse == 0. && z >= 0. {
            return None;
        }
        let theta = transverse.atan2(-z);
        if theta > self.max_angle as f64 + 1e-7 {
            return None;
        }
        let scale = if transverse > 1e-12 {
            self.radius(theta) / transverse
        } else {
            0.
        };
        let a = x * scale;
        let b = -y * scale;
        Some([
            (c.k[0][0] as f64 * a + c.k[0][1] as f64 * b + c.k[0][2] as f64) as f32,
            (c.k[1][1] as f64 * b + c.k[1][2] as f64) as f32,
        ])
    }

    pub(super) fn ray(&self, c: &CameraCalibration, pixel: [f32; 2]) -> Option<[f32; 3]> {
        if pixel.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let y = (pixel[1] as f64 - c.k[1][2] as f64) / c.k[1][1] as f64;
        let x = (pixel[0] as f64 - c.k[0][2] as f64 - c.k[0][1] as f64 * y) / c.k[0][0] as f64;
        let r = x.hypot(y);
        let max = self.radius(self.max_angle as f64);
        if r > max + 1e-7 {
            return None;
        }
        if r < 1e-12 {
            return Some([0., 0., -1.]);
        }
        // Bounded monotonic inverse: no convergence-dependent capture latency.
        let (mut lo, mut hi) = (0., self.max_angle as f64);
        for _ in 0..32 {
            let mid = (lo + hi) * 0.5;
            if self.radius(mid) < r {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let theta = (lo + hi) * 0.5;
        let s = theta.sin() / r;
        Some([(x * s) as f32, (-y * s) as f32, -theta.cos() as f32])
    }

    pub fn calibration(
        self,
        size: [u32; 2],
        fit: FisheyeFit,
        offset: [f32; 2],
        anisotropy: f32,
    ) -> Result<CameraCalibration, String> {
        self.validate()?;
        if size.contains(&0)
            || offset.iter().any(|x| !x.is_finite() || x.abs() > 0.2)
            || !(0.8..=1.25).contains(&anisotropy)
        {
            return Err("invalid fisheye sensor size, optical center or aspect scaling".into());
        }
        if fit == FisheyeFit::Circular && anisotropy != 1. {
            return Err(
                "circular fisheye requires anisotropy=1; use elliptical for unequal focal axes"
                    .into(),
            );
        }
        let [w, h] = size.map(|v| v as f64);
        // Use the actual rounded optical center that is exported in K. Fitting
        // about the unshifted sensor center can crop a nominal circular image
        // or leave invalid corners in a full-frame image.
        let cx = size[0] as f32 * (0.5 + offset[0]);
        let cy = size[1] as f32 * (0.5 + offset[1]);
        let a = f64::from(anisotropy);
        let dx = f64::from(cx).min(w - f64::from(cx));
        let dy = f64::from(cy).min(h - f64::from(cy));
        let radius = match fit {
            FisheyeFit::Diagonal => ((w - dx) / a).hypot((h - dy) * a),
            FisheyeFit::Circular | FisheyeFit::Elliptical => (dx / a).min(dy * a),
        };
        let focal = (radius / self.radius(self.max_angle as f64)) as f32;
        let c = CameraCalibration {
            schema_version: LensModel::AngularFisheye4
                .contract()
                .calibration_schema_version,
            lens_model: LensModel::AngularFisheye4,
            lens_model_version: LensModel::AngularFisheye4.contract().model_version,
            pixel_convention: PixelConvention::TopLeftCornerHalfPixelCenters,
            image_size: size,
            k: [
                [focal * anisotropy, 0., cx],
                [0., focal / anisotropy, cy],
                [0., 0., 1.],
            ],
            fisheye: Some(self),
        };
        c.validate()?;
        Ok(c)
    }
}

impl CameraCalibration {
    /// Unit direction in right/up/-Z view axes. Does not require an in-image pixel.
    pub fn ray(&self, pixel: [f32; 2]) -> Option<[f32; 3]> {
        let p = self.unproject(pixel, 1.)?;
        let n = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        (n > 0. && n.is_finite()).then(|| p.map(|x| x / n))
    }
    /// Metric payload convention of this camera's depth plane.
    pub fn depth(&self, point: [f32; 3]) -> f32 {
        if self.fisheye.is_some() {
            point.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            -point[2]
        }
    }
    pub fn depth_convention(&self) -> &'static str {
        if self.fisheye.is_some() {
            "ray_range_metres"
        } else {
            "camera_z_metres"
        }
    }
    /// Four vec4s shared by WGSL shaders. Zero model flag retains pinhole math.
    pub fn shader_parameters(&self) -> [[f32; 4]; 4] {
        [
            [self.k[0][0], self.k[1][1], self.k[0][2], self.k[1][2]],
            [
                self.k[0][1],
                self.image_size[0] as f32,
                self.image_size[1] as f32,
                f32::from(self.fisheye.is_some()),
            ],
            self.fisheye.map_or([0.; 4], |l| l.coefficients),
            [self.fisheye.map_or(0., |l| l.max_angle), 0., 0., 0.],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_sensor_fits_survive_aspect_ratio_offsets_and_anisotropy() {
        let lens = Fisheye {
            coefficients: [-0.03, 0.002, 0., 0.],
            max_angle: 1.8,
        };
        let r = lens.radius(f64::from(lens.max_angle));
        for size in [[512, 512], [640, 360], [360, 640], [513, 319]] {
            for offset in [[0., 0.], [-0.2, 0.15], [0.13, -0.2]] {
                for a in [0.8, 1., 1.25] {
                    for fit in [
                        FisheyeFit::Circular,
                        FisheyeFit::Elliptical,
                        FisheyeFit::Diagonal,
                    ] {
                        if fit == FisheyeFit::Circular && a != 1. {
                            assert!(lens.calibration(size, fit, offset, a).is_err());
                            continue;
                        }
                        let c = lens.calibration(size, fit, offset, a).unwrap();
                        if fit == FisheyeFit::Circular {
                            assert_eq!(c.k[0][0], c.k[1][1]);
                        }
                        if fit != FisheyeFit::Diagonal {
                            let [rx, ry] = [c.k[0][0], c.k[1][1]].map(|f| f64::from(f) * r);
                            let [cx, cy] = [c.k[0][2], c.k[1][2]].map(f64::from);
                            assert!(cx - rx >= -0.0001 && cy - ry >= -0.0001);
                            assert!(
                                cx + rx <= f64::from(size[0]) + 0.0001
                                    && cy + ry <= f64::from(size[1]) + 0.0001
                            );
                        } else {
                            for x in [0.5, size[0] as f32 - 0.5] {
                                for y in [0.5, size[1] as f32 - 0.5] {
                                    assert!(
                                        c.ray([x, y]).is_some(),
                                        "full-frame corner must be valid"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matches_opencv_fisheye_equations_in_the_forward_hemisphere() {
        // Independent OpenCV x-right/y-down/z-forward formulation, including
        // alpha=K01/fx. Zeroverse uses edge-origin half-integer pixel centers;
        // subtract 0.5 from the principal point for OpenCV integer centers.
        let lens = Fisheye {
            coefficients: [-0.031, 0.004, -0.0002, 0.00001],
            max_angle: 1.9,
        };
        let mut c = lens
            .calibration([641, 359], FisheyeFit::Diagonal, [0.1, -0.06], 1.04)
            .unwrap();
        c.k[0][1] = 1.3;
        for z in [0.1_f64, 1., 3.] {
            for x in [-3_f64, -0.2, 0., 0.4, 3.] {
                for y in [-2_f64, -0.1, 0., 0.9, 2.] {
                    let a = x / z;
                    let b = y / z;
                    let r = a.hypot(b);
                    let t = r.atan();
                    let td = t
                        * (1.
                            + f64::from(lens.coefficients[0]) * t.powi(2)
                            + f64::from(lens.coefficients[1]) * t.powi(4)
                            + f64::from(lens.coefficients[2]) * t.powi(6)
                            + f64::from(lens.coefficients[3]) * t.powi(8));
                    let s = if r == 0. { 1. } else { td / r };
                    let expected = [
                        (f64::from(c.k[0][0]) * a + f64::from(c.k[0][1]) * b) * s
                            + f64::from(c.k[0][2]),
                        f64::from(c.k[1][1]) * b * s + f64::from(c.k[1][2]),
                    ];
                    let got = c.project([x as f32, -y as f32, -z as f32]).unwrap();
                    assert!((f64::from(got[0]) - expected[0]).abs() < 0.0001);
                    assert!((f64::from(got[1]) - expected[1]).abs() < 0.0001);
                }
            }
        }
    }
    #[test]
    fn angular_roundtrip_including_rearward_rays_and_masks() {
        for degrees in [90_f32, 160., 180., 210.] {
            for coefficients in [
                [0.; 4],
                [0.04, -0.003, 0.0002, 0.00001],
                [-0.04, 0.01, 0., 0.],
            ] {
                let lens = Fisheye {
                    coefficients,
                    max_angle: degrees.to_radians() * 0.5,
                };
                for fit in [
                    FisheyeFit::Diagonal,
                    FisheyeFit::Circular,
                    FisheyeFit::Elliptical,
                ] {
                    let c = lens
                        .calibration(
                            [513, 319],
                            fit,
                            [0.07, -0.03],
                            if fit == FisheyeFit::Circular {
                                1.
                            } else {
                                1.05
                            },
                        )
                        .unwrap();
                    let mut count = 0;
                    for y in (0..319).step_by(7) {
                        for x in (0..513).step_by(7) {
                            let pixel = [x as f32 + 0.5, y as f32 + 0.5];
                            if let Some(p) = c.unproject(pixel, 3.7) {
                                let q = c.project(p).unwrap();
                                assert!(
                                    (pixel[0] - q[0]).abs() < 0.001
                                        && (pixel[1] - q[1]).abs() < 0.001
                                );
                                assert!((c.depth(p) - 3.7).abs() < 1e-5);
                                count += 1;
                            }
                        }
                    }
                    assert!(count > 100);
                }
            }
        }
    }
    #[test]
    fn pinhole_and_equidistant_are_distinct_and_folds_fail() {
        let lens = Fisheye {
            coefficients: [0.; 4],
            max_angle: 1.8,
        };
        let c = lens
            .calibration([512; 2], FisheyeFit::Circular, [0.; 2], 1.)
            .unwrap();
        let angle = 1.7_f32;
        let point = [angle.sin(), 0., -angle.cos()];
        assert!(point[2] > 0. && c.project(point).is_some());
        assert!(c.project([0.; 3]).is_none());
        assert!(c.ray([0.5, 0.5]).is_none());
        assert!(Fisheye {
            coefficients: [-0.5, 0., 0., 0.],
            ..lens
        }
        .validate()
        .is_err());
        assert!(Fisheye {
            coefficients: [f32::NAN, 0., 0., 0.],
            ..lens
        }
        .validate()
        .is_err());
        assert!(Fisheye {
            coefficients: [f32::MAX, 0., 0., 0.],
            ..lens
        }
        .validate()
        .is_err());
        assert_eq!(c.ray([256.; 2]), Some([0., 0., -1.]));
    }
}
