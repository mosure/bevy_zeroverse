//! Device-free production-map qualification. GPU/scene qualification is separate.
use super::{
    program::{self, MaterialRecipe},
    Surface,
};
use bevy::prelude::*;
use image::{Rgb, RgbImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

fn check_maps(r: &MaterialRecipe, style: u32, maps: &[Image; 3]) -> (String, [f64; 2]) {
    let mut hash = Sha256::new();
    let size = r.map_size(style) as usize;
    let expected = (size * size * 4 - 1) / 3 * 4;
    let metal = if r.surface == Surface::Chrome { 255 } else { 0 };
    let mut rough_sum = 0.;
    let mut rough_range = [255u8, 0];
    for (map, image) in maps.iter().enumerate() {
        assert_eq!(image.width(), size as u32);
        assert_eq!(image.texture_descriptor.mip_level_count, size.ilog2() + 1);
        assert_eq!(
            image.texture_descriptor.format,
            if map == 0 {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        );
        let data = image.data.as_ref().unwrap();
        assert_eq!(data.len(), expected);
        hash.update(data);
        for (i, px) in data.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(px[3], 255);
            if map == 1 {
                let normal =
                    Vec3::new(px[0] as f32, px[1] as f32, px[2] as f32) / 127.5 - Vec3::ONE;
                assert!((normal.length() - 1.).abs() < 0.014 && normal.z > 0.);
            }
            if map == 2 {
                assert_eq!(px[2], metal, "{:?} acquired invalid metalness", r.surface);
                assert!(px[1] >= 11);
                if i < size * size {
                    rough_sum += px[1] as f64 / 255.;
                    rough_range[0] = rough_range[0].min(px[1]);
                    rough_range[1] = rough_range[1].max(px[1]);
                }
            }
        }
    }
    let data = maps[2].data.as_ref().unwrap();
    let mean = rough_sum / (size * size) as f64;
    assert!(
        data[data.len() - 3] as f64 / 255. + 0.008 >= mean,
        "minification lost roughness"
    );
    (
        format!("{:x}", hash.finalize()),
        rough_range.map(|v| v as f64 / 255.),
    )
}

/// Four rows per seed: actual albedo, normal, roughness, and a fixed CPU GGX
/// directional-light diagnostic. The final row is not a Bevy scene rendering.
fn swatch(r: &MaterialRecipe, style: u32, maps: &[Image; 3], size: u32) -> RgbImage {
    let n = maps[0].width();
    let tint = r.texture_base_color(style).map(super::srgb_to_linear);
    let mut result = RgbImage::new(size, size * 4);
    let light = Vec3::new(-0.45, 0.3, 0.84).normalize();
    let view = Vec3::new(0., 0.50, 0.866).normalize();
    let half = (light + view).normalize();
    for y in 0..size {
        for x in 0..size {
            let i = ((y * n / size) * n + x * n / size) as usize * 4;
            let bytes = maps
                .each_ref()
                .map(|im| &im.data.as_ref().unwrap()[i..i + 4]);
            let color = Vec3::from_array(std::array::from_fn(|c| {
                super::srgb_to_linear(bytes[0][c] as f32 / 255.) * tint[c]
            }));
            let normal = (Vec3::new(bytes[1][0] as f32, bytes[1][1] as f32, bytes[1][2] as f32)
                / 127.5
                - Vec3::ONE)
                .normalize();
            let rough = (bytes[2][1] as f32 / 255.).max(0.05);
            let metal = bytes[2][2] as f32 / 255.;
            let nl = normal.dot(light).max(0.001);
            let nv = normal.dot(view).max(0.001);
            let nh = normal.dot(half).max(0.);
            let vh = view.dot(half).max(0.);
            let a2 = rough.powi(4);
            let d = a2 / (std::f32::consts::PI * (nh * nh * (a2 - 1.) + 1.).powi(2));
            let masking = |n: f32| 2. * n / (n + (a2 + (1. - a2) * n * n).sqrt());
            let f0 = Vec3::splat(0.04) * (1. - metal) + color * metal;
            let fresnel = f0 + (Vec3::ONE - f0) * (1. - vh).powi(5);
            let spec = fresnel * (d * masking(nl) * masking(nv) / (4. * nl * nv));
            let rgb = (color * ((1. - metal) * (0.18 + nl * 0.72)) + spec * nl * 1.2)
                .clamp(Vec3::ZERO, Vec3::ONE);
            let srgb = |v: Vec3| {
                Rgb(v
                    .to_array()
                    .map(|c| (super::linear_to_srgb(c) * 255.).round() as u8))
            };
            result.put_pixel(x, y, srgb(color));
            result.put_pixel(x, y + size, Rgb([bytes[1][0], bytes[1][1], bytes[1][2]]));
            result.put_pixel(x, y + size * 2, Rgb([bytes[2][1]; 3]));
            result.put_pixel(x, y + size * 3, srgb(rgb));
        }
    }
    result
}

#[test]
#[ignore = "bounded CPU map study; writes out/material_diversity without a device"]
fn surface_space_qualification() {
    let dir = Path::new("out/material_diversity");
    std::fs::create_dir_all(dir).unwrap();
    let categories = [
        ("timber-floor", Surface::Floor, 0),
        ("carpet-floor", Surface::Floor, 1),
        ("mineral-floor", Surface::Floor, 2),
        ("wall", Surface::Paint, 0),
        ("accent-wall", Surface::Accent, 0),
        ("wood", Surface::Wood, 0),
        ("ceramic", Surface::Ceramic, 0),
        ("concrete", Surface::Concrete, 0),
        ("fabric", Surface::Fabric, 0),
        ("leather", Surface::Leather, 0),
        ("chrome", Surface::Chrome, 0),
        ("coated-metal", Surface::Metal, 0),
        ("plastic", Surface::Plastic, 0),
        ("rubber", Surface::Rubber, 0),
    ];
    let mut groups = Vec::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut html=String::from("<!doctype html><meta charset=utf-8><title>Material qualification</title><style>body{background:#111827;color:#edf2f7;font:16px system-ui;max-width:1150px;margin:35px auto}img{width:100%;border-radius:8px}section{margin:35px 0}code{color:#c4b5fd}p{line-height:1.6}</style><h1>Production material maps</h1><p>Eight consecutive seeds per category. Rows: albedo, tangent normal, perceptual roughness, fixed CPU GGX light diagnostic. The diagnostic has no scene lighting, reflections, shadows, clearcoat or anisotropy; it is not a Bevy capture. Actual production resolutions and all mips are checked. Local only.</p>");
    for (name, surface, style) in categories {
        let mut sheet = RgbImage::new(128 * 8, 128 * 4);
        let mut records = Vec::new();
        let mut hashes = BTreeSet::new();
        for seed in 0..64 {
            let r = program::sample_with_floor(seed, style).unwrap()[surface as usize].clone();
            assert_eq!(
                r,
                program::sample_with_floor(seed, style).unwrap()[surface as usize]
            );
            r.validate().unwrap();
            let serialized = serde_json::to_vec(&r).unwrap();
            assert_eq!(
                r,
                serde_json::from_slice::<MaterialRecipe>(&serialized).unwrap()
            );
            if let Some(f) = &r.flooring {
                *counts.entry(format!("{name}/{:?}", f.layout)).or_default() += 1;
            }
            if seed < 8 {
                let start = Instant::now();
                let maps = r.maps(style);
                let ms = start.elapsed().as_secs_f64() * 1000.;
                let (sha, roughness) = check_maps(&r, style, &maps);
                hashes.insert(sha.clone());
                if seed < 2 {
                    assert_eq!(sha, check_maps(&r, style, &r.maps(style)).0);
                }
                let cell = swatch(&r, style, &maps, 128);
                image::imageops::replace(&mut sheet, &cell, seed as i64 * 128, 0);
                records.push(json!({"seed":seed,"recipe":r,"maps_sha256":sha,"maps_ms":ms,"roughness_range":roughness}));
            }
        }
        sheet.save(dir.join(format!("{name}.png"))).unwrap();
        html.push_str(&format!("<section><h2>{name}</h2><p>Left to right: seeds 0–7.</p><img src='{name}.png'></section>"));
        groups.push(json!({"category":name,"recipes_validated":64,"map_triplets":8,"distinct_map_hashes":hashes.len(),"samples":records}));
    }
    let mut installations = Vec::new();
    for layout in 0..7 {
        let (r, style) = (0..256)
            .flat_map(|seed| [0, 2].map(move |style| (seed, style)))
            .find_map(|(seed, style)| {
                let r = program::sample_with_floor(seed, style).unwrap()[Surface::Floor as usize]
                    .clone();
                (r.flooring.as_ref().unwrap().layout as u32 == layout).then_some((r, style))
            })
            .unwrap();
        let name = format!("{:?}", r.flooring.as_ref().unwrap().layout);
        let maps = r.maps(style);
        check_maps(&r, style, &maps);
        swatch(&r, style, &maps, 256)
            .save(dir.join(format!("installation-{name}.png")))
            .unwrap();
        installations.push(json!({"name":name,"recipe":r,"style":style}));
        html.push_str(&format!("<section><h2>{name} installation</h2><img style='width:256px' src='installation-{name}.png'></section>"));
    }
    std::fs::write(dir.join("index.html"), html).unwrap();
    std::fs::write(dir.join("report.json"),serde_json::to_vec_pretty(&json!({
        "schema":1,"provenance":crate::provenance::capture_provenance(),
        "scope":"CPU production maps, every mip, fixed CPU GGX diagnostic; no GPU, scene-render or training measurements",
        "recipes_validated":14*64,"map_triplets":14*8+7,"replayed_triplets":14*2,
        "new_material_roles":0,"maximum_additional_used_role_map_triplets":1,"maximum_resolution":512,
        "groups":groups,"floor_installations":counts,"installation_gallery":installations,
        "limits":["exact hashes establish replay, not perceptual diversity","no estimate of effective dataset size or downstream transfer","GPU parity and concurrent training throughput not tested"]
    })).unwrap()).unwrap();
}
