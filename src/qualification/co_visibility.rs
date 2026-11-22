//! Exact co-visibility exports survive --no-raw; RGB previews are never numeric masks.
use anyhow::{Context, Result};
use bevy_zeroverse::{
    render::co_visibility::{mask_color, validate_plane},
    sample::View,
};
use serde::Serialize;
use std::path::Path;

/// Directed support within a semantic class in this source image. Counts are
/// additive across views; callers choose their own class or pixel weighting.
#[derive(Serialize)]
pub struct ClassSupport {
    pub valid_pixels: u64,
    pub shared_pixels: u64,
    pub peer_pixels: Vec<u64>,
}

impl ClassSupport {
    pub fn new(cameras: usize) -> Self {
        Self {
            valid_pixels: 0,
            shared_pixels: 0,
            peer_pixels: vec![0; cameras],
        }
    }
    pub fn observe(&mut self, pixel: &[u8; 16]) {
        let p: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
        if p[2] != 1. {
            return;
        }
        self.valid_pixels += 1;
        let mask = p[0] as u16;
        if mask == 0 {
            return;
        }
        self.shared_pixels += 1;
        for (i, count) in self.peer_pixels.iter_mut().enumerate() {
            *count += u64::from(mask & (1 << i) != 0);
        }
    }
}

#[derive(Serialize)]
pub struct VisibilityReport {
    pub valid_pixels: usize,
    pub shared_pixels: usize,
    pub peer_pixels: Vec<usize>,
    pub cardinality_pixels: Vec<usize>,
    pub shared_fraction_valid: f64,
    pub peer_fraction_valid: Vec<f64>,
}

pub fn save(
    view: &View,
    directory: &Path,
    view_index: usize,
    source: usize,
    count: usize,
    size: [u32; 2],
    raw: bool,
) -> Result<VisibilityReport> {
    let [width, height] = size;
    let pixels = width as usize * height as usize;
    validate_plane(&view.co_visibility, pixels, count, source).map_err(anyhow::Error::msg)?;
    let mut masks = Vec::with_capacity(pixels);
    let mut valid = Vec::with_capacity(pixels);
    let mut rgb = Vec::with_capacity(pixels * 3);
    let mut report = VisibilityReport {
        valid_pixels: 0,
        shared_pixels: 0,
        peer_pixels: vec![0; count],
        cardinality_pixels: vec![0; count],
        shared_fraction_valid: 0.0,
        peer_fraction_valid: vec![0.0; count],
    };
    for bytes in view.co_visibility.as_chunks::<16>().0 {
        let p: [f32; 4] = bytemuck::pod_read_unaligned(bytes);
        let mask = p[0] as u16;
        masks.push(mask);
        valid.push(p[2] as u8);
        rgb.extend(mask_color(mask, count));
        if p[2] == 1.0 {
            report.valid_pixels += 1;
            report.shared_pixels += usize::from(mask != 0);
            report.cardinality_pixels[mask.count_ones() as usize] += 1;
            for (peer, pixels) in report.peer_pixels.iter_mut().enumerate() {
                *pixels += usize::from(mask & (1 << peer) != 0);
            }
        }
    }
    let denominator = report.valid_pixels.max(1) as f64;
    report.shared_fraction_valid = report.shared_pixels as f64 / denominator;
    report.peer_fraction_valid = report
        .peer_pixels
        .iter()
        .map(|&n| n as f64 / denominator)
        .collect();
    let prefix = format!("view_{view_index:02}_co_visibility");
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(width, height, masks)
        .context("invalid mask dimensions")?
        .save(directory.join(format!("{prefix}_mask.png")))?;
    image::GrayImage::from_raw(width, height, valid)
        .context("invalid validity dimensions")?
        .save(directory.join(format!("{prefix}_valid.png")))?;
    image::RgbImage::from_raw(width, height, rgb)
        .context("invalid preview dimensions")?
        .save(directory.join(format!("{prefix}.png")))?;
    if raw {
        super::write_rgba32f(
            &directory.join(format!("{prefix}.rgba32f")),
            &view.co_visibility,
        )?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_support_keeps_unshared_surfaces_and_excludes_invalid_lens_pixels() {
        let mut support = ClassSupport::new(4);
        for (mask, valid) in [(0, 0), (0, 1), (0b1010, 1), (0b0010, 1)] {
            let p = [
                mask as f32,
                (mask as u16).count_ones() as f32,
                valid as f32,
                0.,
            ];
            support.observe(bytemuck::cast_ref(&p));
        }
        assert_eq!(support.valid_pixels, 3);
        assert_eq!(support.shared_pixels, 2);
        assert_eq!(support.peer_pixels, [0, 2, 0, 1]);
    }
}
