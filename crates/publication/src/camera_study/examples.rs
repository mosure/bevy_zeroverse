//! Fixed-room RGB comparisons come from the same native qualification as the
//! camera-count measurements. Preserve every view and the original RGB bytes.
use crate::{graphics, io};
use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};

fn noncamera_hash(mut manifest: Value) -> Result<String> {
    for key in [
        "cameras",
        "camera_settings",
        "camera_aspect_ratio",
        "camera_diagnostics",
    ] {
        manifest
            .as_object_mut()
            .context("invalid manifest")?
            .remove(key);
    }
    Ok(bevy_zeroverse_capture::provenance::sha(
        &serde_json::to_vec(&manifest)?,
    ))
}

pub fn collect(root: &Path, receipt: &Value) -> Result<Value> {
    let recipe = &receipt["recipe"];
    let cases: Vec<_> = recipe["cases"]
        .as_array()
        .context("native cases missing")?
        .iter()
        .filter(|c| c["cameras"] == 4)
        .collect();
    if cases.is_empty() {
        return Ok(Value::Null);
    }
    ensure!(
        cases.len() == 3
            && [0., 0.5, 1.].iter().all(|baseline| {
                cases
                    .iter()
                    .any(|case| case["camera"]["baseline"].as_f64() == Some(*baseline))
            })
            && cases
                .iter()
                .all(|case| case["image_size"] == json!([320, 200])),
        "baseline illustrations require the canonical three matched four-view cases"
    );
    let start = recipe["seed"].as_u64().context("native seed missing")?;
    let count = recipe["rooms_per_case"]
        .as_u64()
        .context("native room count missing")?
        .min(2);
    let mut rooms = Vec::new();
    for seed in start..start + count {
        let mut variants = Vec::new();
        let mut scene_hash = None;
        for case in &cases {
            let name = case["name"].as_str().context("case name missing")?;
            let folder = format!("{name}/seed_{seed:06}");
            let manifest: Value = io::read(&root.join(&folder).join("manifest.json"))?;
            let hash = noncamera_hash(manifest)?;
            ensure!(
                scene_hash.as_ref().is_none_or(|h| h == &hash),
                "baseline illustration changes noncamera scene fields"
            );
            scene_hash = Some(hash);
            let mut views = Vec::new();
            for camera in 0..4 {
                let source = format!("{folder}/view_{camera:02}_color.png");
                let bytes = fs::read(root.join(&source))?;
                let hash = bevy_zeroverse_capture::provenance::sha(&bytes);
                ensure!(receipt["artifacts"][&source] == hash, "unbound RGB example");
                views.push(json!({"camera_index":camera,"step_index":0,"source":source,"sha256":hash,"png_base64":STANDARD.encode(bytes)}));
            }
            variants.push(json!({"case":name,"baseline":case["camera"]["baseline"],"image_size":case["image_size"],"views":views}));
        }
        variants.sort_by(|a, b| {
            a["baseline"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["baseline"].as_f64().unwrap())
        });
        rooms.push(json!({"seed":seed,"noncamera_scene_sha256":scene_hash,"variants":variants}));
    }
    let examples = json!({"selection":"first two prespecified families, all four RGB views at t=0; no appearance filtering","rooms":rooms});
    verify(&examples, receipt)?;
    Ok(examples)
}

pub fn verify(examples: &Value, receipt: &Value) -> Result<()> {
    if examples.is_null() {
        return Ok(());
    }
    let mut sources = BTreeSet::new();
    for room in examples["rooms"]
        .as_array()
        .context("example rooms missing")?
    {
        for variant in room["variants"]
            .as_array()
            .context("example variants missing")?
        {
            let name = variant["case"].as_str().context("example case missing")?;
            let case = receipt["recipe"]["cases"]
                .as_array()
                .context("native case recipe missing")?
                .iter()
                .find(|case| case["name"] == name)
                .context("example case outside native recipe")?;
            ensure!(
                case["cameras"] == 4
                    && case["camera"]["baseline"] == variant["baseline"]
                    && case["image_size"] == variant["image_size"],
                "example baseline or dimensions differ from the native control"
            );
            let views = variant["views"]
                .as_array()
                .context("example views missing")?;
            ensure!(
                views.len() == 4,
                "RGB comparison must retain all four views"
            );
            for (camera, view) in views.iter().enumerate() {
                let source = view["source"].as_str().context("example source missing")?;
                crate::config::safe_relative(Path::new(source))?;
                let seed = room["seed"].as_u64().context("example seed missing")?;
                ensure!(
                    source == format!("{name}/seed_{seed:06}/view_{camera:02}_color.png"),
                    "example source does not match its camera identity"
                );
                ensure!(
                    sources.insert(source)
                        && view["camera_index"] == camera
                        && view["step_index"] == 0
                        && view["sha256"] == receipt["artifacts"][source],
                    "example source is duplicated, misaligned or outside the native receipt"
                );
                let bytes =
                    STANDARD.decode(view["png_base64"].as_str().context("example PNG missing")?)?;
                ensure!(
                    view["sha256"] == bevy_zeroverse_capture::provenance::sha(&bytes),
                    "example RGB bytes changed"
                );
                let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)?;
                ensure!(
                    variant["image_size"] == json!([image.width(), image.height()]),
                    "example image dimensions changed"
                );
            }
        }
    }
    Ok(())
}

pub fn build(report: &Value, stage: &Path, dest: &Path) -> Result<()> {
    let Some(rooms) = report["rendered_examples"]["rooms"].as_array() else {
        return Ok(());
    };
    for room in rooms {
        let variants = room["variants"]
            .as_array()
            .context("example variants missing")?;
        let width = 28 + 420 * variants.len() as u32;
        let mut svg = graphics::begin(width, 1256);
        graphics::text(
            &mut svg,
            28.,
            32.,
            22,
            &format!(
                "Same scene, different camera baseline · seed {}",
                room["seed"]
            ),
        );
        graphics::text(
            &mut svg,
            28.,
            59.,
            15,
            "Four complete RGB views at t=0; geometry, people, materials and lighting are identical.",
        );
        for (column, variant) in variants.iter().enumerate() {
            let x = 28. + 420. * column as f64;
            graphics::text(
                &mut svg,
                x,
                93.,
                18,
                &format!("Baseline {}", variant["baseline"]),
            );
            for (camera, view) in variant["views"].as_array().unwrap().iter().enumerate() {
                let y = 132. + 286. * camera as f64;
                graphics::text(&mut svg, x, y - 10., 14, &format!("Camera {camera}"));
                // Uniform scaling preserves the full original image extent.
                svg.push_str(&format!("<image x=\"{x}\" y=\"{y}\" width=\"400\" height=\"250\" href=\"data:image/png;base64,{}\"/>",view["png_base64"].as_str().unwrap()));
            }
        }
        svg.push_str("</g></svg>");
        let name = format!("camera-baseline-s{}", room["seed"]);
        graphics::save(&svg, &dest.join(format!("{name}.svg")))?;
        fs::copy(
            dest.join(format!("{name}.png")),
            stage.join(format!(
                "tex/generated/camera_count_baseline_s{}.png",
                room["seed"]
            )),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};
    use std::io::Cursor;

    #[test]
    fn baseline_example_allows_camera_diagnostics_but_requires_identical_content() {
        let a = json!({"objects":[{"size":[1.,2.,3.]}], "lighting":4., "camera_diagnostics":{"retries":1}});
        let mut b = a.clone();
        b["camera_diagnostics"]["retries"] = json!(8);
        assert_eq!(
            noncamera_hash(a.clone()).unwrap(),
            noncamera_hash(b.clone()).unwrap()
        );
        b["lighting"] = json!(5.);
        assert_ne!(
            noncamera_hash(a.clone()).unwrap(),
            noncamera_hash(b).unwrap()
        );
        let mut c = a.clone();
        c["objects"][0]["size"][0] = json!(2.);
        assert_ne!(noncamera_hash(a).unwrap(), noncamera_hash(c).unwrap());
    }

    #[test]
    fn illustration_binds_rgb_camera_identity_and_actual_baseline() {
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(2, 2))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let bytes = png.into_inner();
        let hash = bevy_zeroverse_capture::provenance::sha(&bytes);
        let mut artifacts = serde_json::Map::new();
        let mut views = Vec::new();
        for camera in 0..4 {
            let source = format!("wide/seed_000011/view_{camera:02}_color.png");
            artifacts.insert(source.clone(), json!(hash));
            views.push(json!({"camera_index":camera,"step_index":0,"source":source,"sha256":hash,"png_base64":STANDARD.encode(&bytes)}));
        }
        let receipt = json!({"recipe":{"cases":[{"name":"wide","cameras":4,"camera":{"baseline":1},"image_size":[2,2]}]},"artifacts":artifacts});
        let examples = json!({"rooms":[{"seed":11,"variants":[{"case":"wide","baseline":1,"image_size":[2,2],"views":views}]}]});
        verify(&examples, &receipt).unwrap();
        for (key, value) in [("baseline", json!(0)), ("image_size", json!([4, 2]))] {
            let mut changed = examples.clone();
            changed["rooms"][0]["variants"][0][key] = value;
            assert!(verify(&changed, &receipt).is_err());
        }
        let mut changed = examples.clone();
        changed["rooms"][0]["variants"][0]["views"][0]["png_base64"] = json!("AAAA");
        assert!(verify(&changed, &receipt).is_err());
        let mut changed = examples;
        changed["rooms"][0]["variants"][0]["views"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        assert!(verify(&changed, &receipt).is_err());
    }
}
