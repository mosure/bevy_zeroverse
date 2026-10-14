//! Display-only transform of the validator's scalar PNG previews. The capture
//! archive and scalar training tensors are never rewritten by this module.
use anyhow::{ensure, Result};
use bevy_zeroverse_capture::depth::turbo_depth_rgb;
use image::{Rgb, RgbImage};

/// The validator's existing preview contract: round(clamp(axial metres/15)*255).
/// Zero denotes background (valid depth starts at the 0.1 m camera near plane).
const PREVIEW_FAR_METRES: f32 = 15.0;
pub const DESCRIPTION: &str = "Turbo depth: warm near, cool far; log distance on a fixed 0–15 m scale, clamped beyond 15 m. Colorized from the 8-bit axial-depth preview (15/255 m increments); black is background. Numeric training depth is unchanged.";

pub fn preview(source: &RgbImage) -> Result<RgbImage> {
    // One small lookup table per image, rather than log/pow per pixel. These
    // are display bytes, so undo the shared helper's linear-RGB encoding once.
    let lut: [Rgb<u8>; 256] = std::array::from_fn(|i| {
        if i == 0 {
            return Rgb([0; 3]);
        }
        Rgb(
            turbo_depth_rgb(i as f32 * PREVIEW_FAR_METRES / 255.0, PREVIEW_FAR_METRES).map(
                |linear| {
                    let srgb = if linear <= 0.0031308 {
                        12.92 * linear
                    } else {
                        1.055 * linear.powf(1.0 / 2.4) - 0.055
                    };
                    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
                },
            ),
        )
    });
    let mut out = source.clone();
    for pixel in out.pixels_mut() {
        ensure!(
            pixel[0] == pixel[1] && pixel[1] == pixel[2],
            "expected scalar depth/15m PNG; refusing to recolor an RGB depth preview"
        );
        *pixel = lut[usize::from(pixel[0])];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_preview_preserves_input_and_uses_one_fixed_scale() {
        let source = RgbImage::from_fn(256, 1, |x, _| Rgb([x as u8; 3]));
        let result = preview(&source).unwrap();
        assert_eq!(source.get_pixel(128, 0).0, [128; 3]);
        assert_eq!(result.get_pixel(0, 0).0, [0; 3]);
        let near = result.get_pixel(2, 0);
        let far = result.get_pixel(192, 0);
        assert!(near[0] > near[1] && near[0] > near[2]);
        assert!(far[2] > far[0] && far[2] > far[1]);
        // Turbo ends in dark purple, not saturated blue.
        assert_eq!(result.get_pixel(255, 0).0, [35, 23, 27]);
        for level in [2, 17, 64, 128, 255] {
            let single = preview(&RgbImage::from_pixel(1, 1, Rgb([level; 3]))).unwrap();
            assert_eq!(single.get_pixel(0, 0), result.get_pixel(level as u32, 0));
        }
        assert!(preview(&result).is_err());
    }

    /// Colorization can be reviewed on older measured captures without changing
    /// their provenance or weakening the current-generator publication gate.
    #[test]
    #[ignore = "requires the attested publication capture cache"]
    fn cached_publication_depth_preview_review() -> Result<()> {
        use crate::{config::Config, dataset::Capture, graphics, io, pipeline::Attestation};
        use std::{fs, path::Path};
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()?;
        let config = Config::load(&root)?;
        let captures = root.join(config.captures);
        let attestation: Attestation = io::read(&root.join("www/project/publication.json"))?;
        let before = io::files(&captures, &captures)?;
        for (name, hash) in &attestation.capture_sources {
            ensure!(
                before.get(name) == Some(hash),
                "capture source changed: {name}"
            );
        }
        let stage = root.join("out/depth_palette/publication_preview");
        fs::create_dir_all(&stage)?;
        let mut count = 0;
        let mut figure = Vec::new();
        for name in attestation
            .capture_sources
            .keys()
            .filter(|p| p.ends_with("/capture.json"))
        {
            let capture: Capture = io::read(&captures.join(name))?;
            let folder = captures.join(name).parent().unwrap().to_path_buf();
            for (index, view) in capture.views.iter().enumerate() {
                let source =
                    image::open(folder.join(format!("view_{index:02}_depth.png")))?.into_rgb8();
                let colored = preview(&source)?;
                let destination = stage.join(format!(
                    "s{}-t{}-c{}-depth.webp",
                    capture.seed, view.step_index, view.camera_index
                ));
                graphics::save_webp(colored.clone(), &destination)?;
                ensure!(
                    image::open(&destination)?.into_rgb8() == colored,
                    "lossless preview changed"
                );
                ensure!(
                    colored.pixels().any(|p| p[0] != p[1]),
                    "missing Turbo colors"
                );
                if index == 0 && figure.len() < 3 {
                    figure.push((
                        format!("Seed {} · camera 0", capture.seed),
                        vec![
                            (folder.join("view_00_color.png"), "RGB".into()),
                            (destination, "Turbo depth · 0–15 m".into()),
                            (folder.join("view_00_normal.png"), "Normals".into()),
                            (folder.join("view_00_semantic.png"), "Semantics".into()),
                        ],
                    ));
                }
                count += 1;
            }
        }
        ensure!(
            count
                == attestation.protocol.rendered_rooms
                    * attestation.protocol.cameras
                    * attestation.protocol.playback_steps,
            "incomplete capture set"
        );
        graphics::figure_rows(
            &figure,
            &stage.join("annotations.jpg"),
            480,
            attestation.protocol.height as f64 / attestation.protocol.width as f64,
        )?;
        ensure!(
            io::files(&captures, &captures)? == before,
            "source captures changed"
        );
        io::write(
            &stage.join("review.json"),
            &serde_json::json!({
                "kind":"display-only review of attested captures, not a new generator qualification",
                "source_generator_identity":attestation.generator_identity,
                "views":count,"depth_display":DESCRIPTION,"source_files_unchanged":true,
                "outputs":io::files(&stage, &stage)?.into_iter().filter(|(name,_)| name != "review.json").collect::<std::collections::BTreeMap<_,_>>()
            }),
        )?;
        println!(
            "{count} cached depth previews and annotation figure verified at {}",
            stage.display()
        );
        Ok(())
    }
}
