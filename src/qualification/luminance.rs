//! Streaming photometry for capture receipts. The display histogram counts optical
//! samples only; geometrically empty RGB pixels on a valid ray still count.
use bevy_zeroverse::calibration::CameraCalibration;
use serde::Serialize;

pub(super) struct Luminance {
    pixels: usize,
    sum: f64,
    squares: f64,
    dark: usize,
    clipped: usize,
    valid: usize,
    in_shadow: usize,
    channel_clipped: usize,
    levels: [usize; 256],
}

#[derive(Serialize)]
pub(super) struct DisplayLuminance {
    pub valid_sensor_pixels: usize,
    /// Rounded sRGB code for neutral gray with the same linear Rec.709 Y.
    pub srgb_gray_p10_p50_p90: [u8; 3],
    pub fraction_below_linear_y_0_01: f64,
    pub fraction_any_channel_above_0_99: f64,
}

impl Luminance {
    pub fn new() -> Self {
        Self {
            pixels: 0,
            sum: 0.,
            squares: 0.,
            dark: 0,
            clipped: 0,
            valid: 0,
            in_shadow: 0,
            channel_clipped: 0,
            levels: [0; 256],
        }
    }

    pub fn push(
        &mut self,
        p: [f32; 4],
        index: usize,
        calibration: &CameraCalibration,
        display: bool,
    ) {
        let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        self.pixels += 1;
        self.sum += f64::from(y);
        self.squares += f64::from(y).powi(2);
        self.dark += usize::from(y < 0.002);
        self.clipped += usize::from(y > 0.99);
        if !display {
            return;
        }
        if calibration.fisheye.is_some() {
            let width = calibration.image_size[0] as usize;
            let pixel = [(index % width) as f32 + 0.5, (index / width) as f32 + 0.5];
            if calibration.unproject(pixel, 1.).is_none() {
                return;
            }
        }
        self.valid += 1;
        self.in_shadow += usize::from(y < 0.01);
        self.channel_clipped += usize::from(p[..3].iter().any(|&v| v > 0.99));
        let code = (super::linear_to_srgb(y).clamp(0., 1.) * 255.).round() as usize;
        self.levels[code] += 1;
    }

    pub fn summary(&self) -> (f32, f32, f32, f32) {
        let n = self.pixels as f64;
        let mean = self.sum / n;
        (
            mean as f32,
            (self.squares / n - mean * mean).max(0.).sqrt() as f32,
            (self.dark as f64 / n) as f32,
            (self.clipped as f64 / n) as f32,
        )
    }

    pub fn display(&self) -> Option<DisplayLuminance> {
        if self.valid == 0 {
            return None;
        }
        let codes = [10, 50, 90].map(|p| {
            let rank = (p * self.valid).div_ceil(100);
            let mut count = 0;
            self.levels
                .iter()
                .position(|&n| {
                    count += n;
                    count >= rank
                })
                .unwrap() as u8
        });
        Some(DisplayLuminance {
            valid_sensor_pixels: self.valid,
            srgb_gray_p10_p50_p90: codes,
            fraction_below_linear_y_0_01: self.in_shadow as f64 / self.valid as f64,
            fraction_any_channel_above_0_99: self.channel_clipped as f64 / self.valid as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::calibration::{Fisheye, FisheyeFit};

    #[test]
    fn circular_mask_excludes_the_border_but_counts_black_on_valid_rays() {
        let cal = Fisheye {
            coefficients: [0.; 4],
            max_angle: 1.5,
        }
        .calibration([16, 16], FisheyeFit::Circular, [0., 0.], 1.)
        .unwrap();
        let mut light = Luminance::new();
        let mut count = 0;
        for i in 0..256 {
            let x = (i % 16) as f32 + 0.5 - 8.;
            let y = (i / 16) as f32 + 0.5 - 8.;
            let inside = x * x + y * y <= 64.;
            count += usize::from(inside);
            let v = if inside && i != 136 { 0.18 } else { 0. };
            light.push([v, v, v, 1.], i, &cal, true);
        }
        let stats = light.display().unwrap();
        assert_eq!(stats.valid_sensor_pixels, count);
        assert_eq!(stats.srgb_gray_p10_p50_p90, [118; 3]);
        assert_eq!(stats.fraction_below_linear_y_0_01, 1. / count as f64);
        assert_eq!(stats.fraction_any_channel_above_0_99, 0.);
    }
}
