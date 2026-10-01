//! Exact co-visibility exports survive --no-raw; RGB previews are never numeric masks.
use anyhow::{Context, Result};
use bevy_zeroverse::{
    render::co_visibility::{mask_color, validate_plane},
    sample::View,
};
use serde::Serialize;
use std::{fs, path::Path};

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
        let little_endian: Vec<u8> = view
            .co_visibility
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| f32::from_ne_bytes(*b).to_le_bytes())
            .collect();
        fs::write(directory.join(format!("{prefix}.rgba32f")), little_endian)?;
    }
    Ok(report)
}
