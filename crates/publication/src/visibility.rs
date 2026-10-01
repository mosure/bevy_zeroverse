use anyhow::{ensure, Result};
use bevy_zeroverse_capture::{camera_color, mask_color};
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Stats {
    pub valid_pixels: u64,
    pub shared_pixels: u64,
    pub peer_pixels: Vec<u64>,
    pub cardinality_pixels: Vec<u64>,
    pub shared_fraction_valid: f64,
    pub peer_fraction_valid: Vec<f64>,
}
pub struct Plane {
    pub masks: Vec<u16>,
    pub stats: Stats,
}

pub fn validate_metadata(metadata: &Value, count: usize) -> Result<()> {
    ensure!(
        (1..=16).contains(&count)
            && metadata["schema_version"] == 1
            && metadata["camera_count"] == count,
        "invalid co-visibility metadata"
    );
    let legend = metadata["legend"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing camera legend"))?;
    ensure!(legend.len() == count, "wrong camera legend size");
    for (i, entry) in legend.iter().enumerate() {
        ensure!(
            *entry
                == serde_json::json!({"bit":i,"camera_index":i,"mask":1u32 << i,"rgb8":camera_color(i,count)}),
            "camera legend/bit ordering mismatch"
        );
    }
    Ok(())
}

pub fn decode(
    mask: &[u8],
    valid: &[u8],
    preview: Option<&RgbImage>,
    size: [u32; 2],
    source: usize,
    count: usize,
    recorded: &Stats,
) -> Result<Plane> {
    ensure!(
        (1..=16).contains(&count) && source < count,
        "invalid visibility camera"
    );
    ensure!(
        mask.len() >= 26 && &mask[..8] == b"\x89PNG\r\n\x1a\n" && mask[24..26] == [16, 0],
        "membership must be 16-bit grayscale PNG"
    );
    ensure!(
        valid.len() >= 26 && &valid[..8] == b"\x89PNG\r\n\x1a\n" && valid[24..26] == [8, 0],
        "validity must be 8-bit grayscale PNG"
    );
    let DynamicImage::ImageLuma16(masks) = image::load_from_memory(mask)? else {
        anyhow::bail!("lossy membership image");
    };
    let DynamicImage::ImageLuma8(validity) = image::load_from_memory(valid)? else {
        anyhow::bail!("invalid validity image");
    };
    ensure!(
        masks.dimensions() == (size[0], size[1]) && validity.dimensions() == masks.dimensions(),
        "visibility image size mismatch"
    );
    if let Some(p) = preview {
        ensure!(
            p.dimensions() == masks.dimensions(),
            "visibility preview size mismatch"
        );
    }
    let mut peers = vec![0u64; count];
    let mut cardinality = vec![0u64; count];
    let (mut n, mut shared) = (0u64, 0u64);
    for (i, (&m, &v)) in masks.as_raw().iter().zip(validity.as_raw()).enumerate() {
        ensure!(
            v <= 1 && (m as u32) < (1u32 << count) && m & (1 << source) == 0 && (v != 0 || m == 0),
            "invalid membership/source bit/background"
        );
        if v != 0 {
            n += 1;
            shared += u64::from(m != 0);
            cardinality[m.count_ones() as usize] += 1;
        }
        for (j, p) in peers.iter_mut().enumerate() {
            *p += u64::from(m & (1 << j) != 0);
        }
        if let Some(p) = preview {
            ensure!(
                p.as_raw()[i * 3..i * 3 + 3] == mask_color(m, count),
                "visibility preview differs from exact bits"
            );
        }
    }
    let denominator = n.max(1) as f64;
    let stats = Stats {
        valid_pixels: n,
        shared_pixels: shared,
        peer_fraction_valid: peers.iter().map(|&v| v as f64 / denominator).collect(),
        peer_pixels: peers,
        cardinality_pixels: cardinality,
        shared_fraction_valid: shared as f64 / denominator,
    };
    ensure!(
        stats.valid_pixels == recorded.valid_pixels
            && stats.shared_pixels == recorded.shared_pixels
            && stats.peer_pixels == recorded.peer_pixels
            && stats.cardinality_pixels == recorded.cardinality_pixels,
        "visibility counts disagree with exact masks"
    );
    ensure!(
        (stats.shared_fraction_valid - recorded.shared_fraction_valid).abs() < 1e-12
            && stats.peer_fraction_valid.len() == recorded.peer_fraction_valid.len()
            && stats
                .peer_fraction_valid
                .iter()
                .zip(&recorded.peer_fraction_valid)
                .all(|(a, b)| (a - b).abs() < 1e-12),
        "visibility denominators disagree"
    );
    Ok(Plane {
        masks: masks.into_raw(),
        stats,
    })
}
