//! Versioned camera labels in pixel coordinates. K is row-major; image axes are
//! right/down, with the top-left pixel center at (0.5, 0.5). Projection helpers
//! accept Bevy view coordinates (right/up/-Z forward), matching world_from_view.
use serde::{Deserialize, Serialize};

mod fisheye;
mod model;
pub use fisheye::{Fisheye, FisheyeFit, FisheyeSensor};
pub use model::{DepthConvention, LensContract, LensModel};

/// Shared verbatim by Rust and Python tensor writers/readers.
pub const TENSOR_METADATA: &str = r#"{"schema_version":1,"lens_model":"pinhole","lens_model_version":1,"pixel_convention":"top_left_corner_half_pixel_centers","intrinsics":"row-major K; image x right, y down","world_from_view":"column-major; view x right, y up, -Z forward; metres","time":"normalized playback progress; legacy field retained","trajectory_progress":"trajectory parameter after playback easing","time_seconds":"explicit capture timeline; unspecified when time_seconds_valid is zero"}"#;

/// Mixed-lens tensors carry per-view model IDs and coefficients with explicit depth semantics.
pub const TENSOR_METADATA_V2: &str = r#"{"schema_version":2,"lens_model_versions":{"0":1,"1":1},"intrinsics":"row-major K; image x right, y down","image_size":"width,height; final output pixels","time":"normalized playback progress; legacy field retained","trajectory_progress":"trajectory parameter after playback easing","time_seconds":"explicit capture timeline; unspecified when time_seconds_valid is zero","models":"lens_model U8 (0=pinhole,1=angular_fisheye4); lens_parameters F32 [k1,k2,k3,k4,max_angle_radians]","pixel_convention":"top_left_corner_half_pixel_centers","depth":"pinhole: camera_z_metres; angular_fisheye4: ray_range_metres; zero is invalid","world_from_view":"column-major; view x right, y up, -Z forward; metres"}"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelConvention {
    TopLeftCornerHalfPixelCenters,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraCalibration {
    pub schema_version: u32,
    pub lens_model: LensModel,
    pub lens_model_version: u32,
    pub pixel_convention: PixelConvention,
    pub image_size: [u32; 2],
    /// [[fx, skew, cx], [0, fy, cy], [0, 0, 1]], row-major, in pixels.
    pub k: [[f32; 3]; 3],
    /// Angular lens parameters; absent from the v1 pinhole wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fisheye: Option<Fisheye>,
}

impl CameraCalibration {
    pub fn centered_pinhole(
        width: u32,
        height: u32,
        fovy: f32,
        aspect: f32,
    ) -> Result<Self, String> {
        if width == 0
            || height == 0
            || !fovy.is_finite()
            || !(0.0..std::f32::consts::PI).contains(&fovy)
            || !aspect.is_finite()
            || aspect <= 0.0
        {
            return Err("calibration requires nonzero image dimensions, a finite positive aspect and 0 < fovy < pi".into());
        }
        let tan = (fovy * 0.5).tan();
        let result = Self {
            fisheye: None,
            schema_version: LensModel::Pinhole.contract().calibration_schema_version,
            lens_model: LensModel::Pinhole,
            lens_model_version: LensModel::Pinhole.contract().model_version,
            pixel_convention: PixelConvention::TopLeftCornerHalfPixelCenters,
            image_size: [width, height],
            k: [
                [width as f32 / (2.0 * aspect * tan), 0.0, width as f32 * 0.5],
                [0.0, height as f32 / (2.0 * tan), height as f32 * 0.5],
                [0.0, 0.0, 1.0],
            ],
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), String> {
        let contract = self.lens_model.contract();
        let supported = match self.lens_model {
            LensModel::Pinhole => self.fisheye.is_none(),
            LensModel::AngularFisheye4 => self.fisheye.is_some(),
        };
        if !supported
            || self.schema_version != contract.calibration_schema_version
            || self.lens_model_version != contract.model_version
            || self.image_size.contains(&0)
            || self.k.iter().flatten().any(|v| !v.is_finite())
            || self.k[0][0] <= 0.0
            || self.k[1][1] <= 0.0
            || self.k[1][0] != 0.0
            || self.k[2] != [0.0, 0.0, 1.0]
        {
            return Err("invalid or unsupported camera calibration".into());
        }
        if let Some(lens) = self.fisheye {
            lens.validate()?;
        }
        Ok(())
    }

    /// Reproject a view-space point; visibility additionally requires clipping
    /// and a target first-surface depth test. This does not perform occlusion.
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        if let Some(lens) = self.fisheye {
            return lens.project(self, point);
        }
        let [x, y, z] = point;
        if point.iter().any(|v| !v.is_finite()) || z >= 0.0 {
            return None;
        }
        Some([
            (self.k[0][0] * x - self.k[0][1] * y) / -z + self.k[0][2],
            self.k[1][1] * -y / -z + self.k[1][2],
        ])
    }

    /// Pinhole depth is -view.z; fisheye depth is positive Euclidean ray range.
    /// The lens/schema determines the convention; neither is colorized.
    pub fn unproject(&self, pixel: [f32; 2], depth: f32) -> Option<[f32; 3]> {
        if !depth.is_finite() || depth <= 0.0 || pixel.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if let Some(lens) = self.fisheye {
            return lens.ray(self, pixel).map(|ray| ray.map(|v| v * depth));
        }
        let y = (pixel[1] - self.k[1][2]) / self.k[1][1];
        let x = (pixel[0] - self.k[0][2] - self.k[0][1] * y) / self.k[0][0];
        Some([x * depth, -y * depth, -depth])
    }

    pub fn in_image(&self, pixel: [f32; 2]) -> bool {
        (0.0..self.image_size[0] as f32).contains(&pixel[0])
            && (0.0..self.image_size[1] as f32).contains(&pixel[1])
            && self
                .fisheye
                .is_none_or(|lens| lens.ray(self, pixel).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portrait_landscape_and_off_center_pinhole_roundtrip() {
        for [w, h] in [[640, 400], [321, 719], [1, 1]] {
            for fov in [0.3, 1.2, 2.2, 145_f32.to_radians(), 150_f32.to_radians()] {
                let mut c =
                    CameraCalibration::centered_pinhole(w, h, fov, w as f32 / h as f32).unwrap();
                for offset in [0.0, 0.17] {
                    c.k[0][2] = w as f32 * (0.5 + offset);
                    c.k[1][2] = h as f32 * (0.5 - offset);
                    c.k[0][1] = offset * 2.0;
                    c.validate().unwrap();
                    for pixel in [[0.5, 0.5], [w as f32 - 0.5, h as f32 - 0.5]] {
                        let point = c.unproject(pixel, 3.7).unwrap();
                        let actual = c.project(point).unwrap();
                        assert!((actual[0] - pixel[0]).abs() < 0.001);
                        assert!((actual[1] - pixel[1]).abs() < 0.001);
                    }
                }
            }
        }
    }

    #[test]
    fn projection_sign_flow_units_and_frustum_edges_are_explicit() {
        let c = CameraCalibration::centered_pinhole(640, 400, std::f32::consts::FRAC_PI_2, 1.6)
            .unwrap();
        assert_eq!(c.project([0., 0., -2.]), Some([320., 200.]));
        // A camera translated right by 0.2 m moves a fixed surface left by fx * 0.2 / z.
        let target = c.project([-0.2, 0., -2.]).unwrap();
        assert!((target[0] - 300.).abs() < 0.001);
        assert!(c.project([0., 0., 1.]).is_none());
        assert!(c.in_image([0., 0.]));
        assert!(!c.in_image([640., 400.]));
        assert!(!c.in_image([f32::NAN, 0.]));
        for fov in [0., -1., f32::INFINITY, f32::NAN, std::f32::consts::PI] {
            assert!(CameraCalibration::centered_pinhole(640, 400, fov, 1.6).is_err());
        }
    }
}
