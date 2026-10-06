//! Independent pre-row-scheduling atlas oracle.
use super::super::*;

fn original_texture_maps(
    surface: Surface,
    floor_style: u32,
    seed: u64,
    roughness: f32,
    recipe: Option<&program::MaterialRecipe>,
    prepare: bool,
) -> TextureMaps {
    // Older serialized recipes omit the new programs. Resolve their defaults
    // once per map, rather than initializing an RNG for each of 65,536 texels.
    let initialized = recipe.and_then(|r| {
        let textile = matches!(surface, Surface::Fabric | Surface::FabricAlt)
            || surface == Surface::Floor && floor_style == 1;
        if textile && r.textile.is_none() || surface == Surface::Leather && r.leather.is_none() {
            let mut r = r.clone();
            if textile {
                r.textile = Some(textile::TextileRecipe::sample(r.seed));
            }
            if surface == Surface::Leather {
                r.leather = Some(leather::LeatherRecipe::sample(r.seed));
            }
            Some(r)
        } else {
            None
        }
    });
    let recipe = initialized.as_ref().or(recipe);
    let prepared = prepare
        .then(|| recipe.and_then(|r| r.prepare_texels(floor_style)))
        .flatten();
    let n = recipe.map_or(256, |r| r.map_size(floor_style)) as usize;
    let mut heights = vec![0.0; n * n];
    let mut colors = Vec::with_capacity(n * n * 4);
    let mut data = Vec::with_capacity(n * n * 4);
    for y in 0..n {
        for x in 0..n {
            let u = x as f32 / n as f32;
            let v = y as f32 / n as f32;
            let fallback = || {
                let noise = hash(x as u32, y as u32, seed);
                let wood_surface = matches!(surface, Surface::Wood | Surface::WoodEdge)
                    || (surface == Surface::Floor && floor_style == 0);
                let fibre = if wood_surface {
                    let warp = periodic_noise(u, v, 3, 3, seed.wrapping_add(17)) * 0.035;
                    0.65 * periodic_noise(u + warp, v, 80, 4, seed)
                        + 0.35 * periodic_noise(u + warp, v, 29, 2, seed.wrapping_add(61))
                } else {
                    0.5
                };
                let (shade, h) = match surface {
                    Surface::Wood | Surface::WoodEdge => {
                        (0.81 + fibre * 0.15 + noise * 0.016, fibre * 0.001)
                    }
                    Surface::Floor if floor_style == 0 => {
                        let strip = (u * 10.0).floor() as u32;
                        let stagger = if strip.is_multiple_of(2) { 0.0 } else { 0.5 };
                        let seam =
                            (u * 10.0).fract() < 0.022 || (v * 2.0 + stagger).fract() < 0.008;
                        let shade = if seam {
                            0.36
                        } else {
                            0.75 + hash(strip, (v * 2.0 + stagger).floor() as u32, seed) * 0.19
                                + (fibre - 0.5) * 0.09
                        };
                        (shade, if seam { -0.025 } else { fibre * 0.002 })
                    }
                    Surface::Floor if floor_style == 2 => {
                        let seam = (u * 4.0).fract() < 0.014 || (v * 4.0).fract() < 0.014;
                        (
                            if seam { 0.6 } else { 0.89 + noise * 0.09 },
                            if seam { -0.06 } else { noise * 0.003 },
                        )
                    }
                    Surface::Fabric | Surface::FabricAlt | Surface::Floor => {
                        let weave = ((x / 2 + y / 2) % 2) as f32;
                        (
                            0.76 + noise * 0.17 + weave * 0.06,
                            weave * 0.012 + noise * 0.009,
                        )
                    }
                    Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => {
                        let vein = (u - 0.5).abs() < 0.015
                            || ((v + (u - 0.5).abs() * 0.7) * 14.0).fract() < 0.028;
                        (
                            if vein {
                                1.0
                            } else {
                                if surface == Surface::LeafVariegated {
                                    0.53 + 0.24 * (v * 78.0 + (u * 30.0).sin()).sin().abs()
                                        + 0.21 * ((u - 0.5).abs() * 2.0).powi(5)
                                } else {
                                    0.75 + 0.15 * (v * std::f32::consts::PI).sin() + noise * 0.07
                                }
                            },
                            if vein { 0.013 } else { 0.0 },
                        )
                    }
                    Surface::Bark => {
                        let ridges = periodic_noise(u, v, 26, 3, seed);
                        (0.66 + ridges * 0.28 + noise * 0.06, ridges * 0.045)
                    }
                    Surface::Terracotta => (0.85 + noise * 0.12, noise * 0.02),
                    Surface::Soil => (0.50 + noise * 0.45, noise * 0.12),
                    Surface::Concrete => (0.84 + noise * 0.13, noise * 0.022),
                    Surface::Ceiling => {
                        (0.94 + noise * 0.04, if noise < 0.1 { -0.018 } else { 0.0 })
                    }
                    _ => (0.95 + noise * 0.04, noise * 0.008),
                };
                (shade, h * 0.001, roughness + (noise - 0.5) * 0.06)
            };
            let texel = if let Some(recipe) = recipe {
                recipe.texel_prepared(u, v, floor_style, prepared.as_ref())
            } else {
                let (shade, h, roughness) = fallback();
                program::Texel {
                    color: [shade; 3],
                    height: h,
                    roughness,
                    occlusion: 1.,
                }
            };
            heights[y * n + x] = texel.height;
            let rgb = texel.color.map(|c| (c.clamp(0., 1.) * 255.) as u8);
            colors.extend([rgb[0], rgb[1], rgb[2], 255]);
            data.extend([
                (texel.occlusion * 255.).round() as u8,
                (texel.roughness.clamp(0.05, 1.0) * 255.0) as u8,
                if surface == Surface::Chrome { 255 } else { 0 },
                255,
            ]);
        }
    }
    let mut normals = Vec::with_capacity(n * n * 4);
    let period = recipe.map_or(Vec2::ONE, |r| {
        r.leaf
            .as_ref()
            .map_or_else(|| r.period_uv(), |l| Vec2::from_array(l.reference_size_m))
    });
    let slope = Vec2::splat(n as f32 * 0.5) / period;
    for y in 0..n {
        for x in 0..n {
            let atlas = recipe.is_some_and(|r| r.leaf.is_some());
            let [left, right, bottom, top] = if atlas {
                [
                    x.saturating_sub(1),
                    (x + 1).min(n - 1),
                    y.saturating_sub(1),
                    (y + 1).min(n - 1),
                ]
            } else {
                [(x + n - 1) % n, (x + 1) % n, (y + n - 1) % n, (y + 1) % n]
            };
            let dx = heights[y * n + right] - heights[y * n + left];
            let dy = heights[top * n + x] - heights[bottom * n + x];
            let slope = if atlas {
                Vec2::new(
                    n as f32 / (right - left) as f32,
                    n as f32 / (top - bottom) as f32,
                ) / period
            } else {
                slope
            };
            // Tangent space follows increasing mesh U/V. Both slopes oppose the
            // height gradient; flipping only Y would invert relief in one axis.
            let normal = Vec3::new(-dx * slope.x, -dy * slope.y, 1.0).normalize();
            normals.extend(filter::encode(normal));
        }
    }
    (colors, normals, data)
}

fn assert_images_equal(reference: [Image; 3], actual: [Image; 3], label: &str) -> usize {
    let mut bytes = 0;
    for (channel, (reference, actual)) in reference.into_iter().zip(actual).enumerate() {
        assert_eq!(
            reference.texture_descriptor, actual.texture_descriptor,
            "{label}/{channel}"
        );
        assert_eq!(
            reference.asset_usage, actual.asset_usage,
            "{label}/{channel}"
        );
        assert_eq!(
            format!("{:?}", reference.sampler),
            format!("{:?}", actual.sampler),
            "{label}/{channel}"
        );
        bytes += reference.data.as_ref().unwrap().len();
        assert_eq!(
            reference.data, actual.data,
            "{label}/{channel}: every base/mip byte"
        );
    }
    bytes
}

#[test]
fn row_bands_match_original_atlas_bytes_and_every_mip() {
    let mut count = 0;
    let mut bytes = 0;
    let recipes = program::sample(42_430_575);
    for (surface, style) in [
        (Surface::Paint, 0),
        (Surface::Wood, 0),
        (Surface::Floor, 0),
        (Surface::Floor, 1),
        (Surface::Floor, 2),
        (Surface::Concrete, 0),
        (Surface::Fabric, 0),
        (Surface::Leather, 0),
        (Surface::Leaf, 0),
        (Surface::Chrome, 0),
        (Surface::Ceramic, 0),
    ] {
        let recipe = &recipes[surface as usize];
        let reference = original_texture_maps(
            surface,
            style,
            recipe.seed,
            recipe.roughness,
            Some(recipe),
            true,
        );
        let actual = texture_maps(surface, style, recipe.seed, recipe.roughness, Some(recipe));
        bytes += assert_images_equal(
            mapped_images(reference, recipe.map_size(style), Some(recipe)),
            mapped_images(actual, recipe.map_size(style), Some(recipe)),
            &format!("{surface:?}/style{style}"),
        );
        count += 1;
    }
    // Missing older wardrobe programs still initialize once per atlas. Knit
    // dispatch and clamped leaf-edge derivatives retain their original behavior.
    for surface in [Surface::Fabric, Surface::Leather] {
        let mut recipe = recipes[surface as usize].clone();
        recipe.textile = None;
        recipe.leather = None;
        let reference = original_texture_maps(
            surface,
            0,
            recipe.seed,
            recipe.roughness,
            Some(&recipe),
            true,
        );
        let actual = texture_maps(surface, 0, recipe.seed, recipe.roughness, Some(&recipe));
        bytes += assert_images_equal(
            mapped_images(reference, recipe.map_size(0), Some(&recipe)),
            mapped_images(actual, recipe.map_size(0), Some(&recipe)),
            &format!("missing/{surface:?}"),
        );
        count += 1;
    }
    let mut knit = recipes[Surface::Fabric as usize].clone();
    knit.textile
        .get_or_insert_with(|| textile::TextileRecipe::sample(knit.seed))
        .knit = true;
    bytes += assert_images_equal(
        mapped_images(
            original_texture_maps(
                knit.surface,
                0,
                knit.seed,
                knit.roughness,
                Some(&knit),
                true,
            ),
            knit.map_size(0),
            Some(&knit),
        ),
        knit.maps(0),
        "knit",
    );
    count += 1;
    for (surface, style) in [
        (Surface::Wood, 0),
        (Surface::Floor, 2),
        (Surface::LeafVariegated, 0),
        (Surface::Chrome, 0),
    ] {
        bytes += assert_images_equal(
            mapped_images(
                original_texture_maps(surface, style, 7, 0.6, None, true),
                256,
                None,
            ),
            mapped_images(texture_maps(surface, style, 7, 0.6, None), 256, None),
            &format!("fallback/{surface:?}/style{style}"),
        );
        count += 1;
    }
    eprintln!(
        "{}",
        serde_json::json!({"scope":"copied original serial atlas; all three base/mip planes/descriptors/samplers","triplets":count,"images":count*3,"bytes":bytes})
    );
}

#[test]
fn bounded_bands_cover_disjoint_rows_before_neighbor_reads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for size in [1, 17, 255, 256, 257, 512] {
        let mut heights = vec![f32::NAN; size * size];
        let mut colors = vec![0; size * size * 4];
        let mut data = vec![0; size * size * 4];
        let calls = AtomicUsize::new(0);
        super::texels(
            size,
            true,
            &mut heights,
            &mut colors,
            &mut data,
            |first, heights, colors, data| {
                calls.fetch_add(1, Ordering::Relaxed);
                for (i, height) in heights.iter_mut().enumerate() {
                    let global = first * size + i;
                    *height = global as f32;
                    colors[i * 4..i * 4 + 4].copy_from_slice(&(global as u32).to_le_bytes());
                    data[i * 4..i * 4 + 4].copy_from_slice(&(!(global as u32)).to_le_bytes());
                }
            },
        );
        #[cfg(not(target_arch = "wasm32"))]
        let expected = if size >= 256 { 4 } else { 1 };
        #[cfg(target_arch = "wasm32")]
        let expected = 1;
        assert_eq!(calls.load(Ordering::Relaxed), expected);
        assert!(calls.load(Ordering::Relaxed) <= 4);
        for (i, height) in heights.iter().enumerate() {
            assert_eq!(*height, i as f32);
            assert_eq!(&colors[i * 4..i * 4 + 4], &(i as u32).to_le_bytes());
            assert_eq!(&data[i * 4..i * 4 + 4], &(!(i as u32)).to_le_bytes());
        }
        let mut normals = vec![0; size * size * 4];
        let normal_calls = AtomicUsize::new(0);
        super::normals(size, true, &mut normals, |first, normals| {
            normal_calls.fetch_add(1, Ordering::Relaxed);
            for (i, normal) in normals.chunks_mut(4).enumerate() {
                let global = first * size + i;
                let neighbor = ((global / size + 1) % size) * size + global % size;
                normal.copy_from_slice(&heights[neighbor].to_bits().to_le_bytes());
            }
        });
        assert_eq!(normal_calls.load(Ordering::Relaxed), expected);
        for (i, normal) in normals.chunks(4).enumerate() {
            let neighbor = ((i / size + 1) % size) * size + i % size;
            assert_eq!(normal, &heights[neighbor].to_bits().to_le_bytes());
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn nested_map_scopes_join_without_changing_row_order() {
    let results = super::super::super::preparation::workers::pool().scope(|scope| {
        for identity in 0..2 {
            scope.spawn(async move {
                let size = 257;
                let mut height = vec![0.; size * size];
                let mut color = vec![0; size * size * 4];
                let mut data = vec![0; size * size * 4];
                super::texels(
                    size,
                    true,
                    &mut height,
                    &mut color,
                    &mut data,
                    |first, height, _, _| {
                        for (i, value) in height.iter_mut().enumerate() {
                            *value = (identity * size * size + first * size + i) as f32;
                        }
                    },
                );
                (identity, height)
            });
        }
    });
    for (identity, (actual_identity, height)) in results.into_iter().enumerate() {
        assert_eq!(actual_identity, identity);
        assert!(height
            .iter()
            .enumerate()
            .all(|(i, value)| *value == (identity * 257 * 257 + i) as f32));
    }
}

#[test]
#[ignore = "diagnostic material recipe CPU profile; root coordinates compilation/timing"]
fn material_recipe_cpu_profile() {
    use crate::scene::procedural_indoor::{
        architecture, humans, layout::IndoorLayout, objects, preparation::SceneGeometry,
    };
    use std::{collections::BTreeSet, hint::black_box, time::Instant};

    fn profile(
        scene: &IndoorManifest,
        recipe: &program::MaterialRecipe,
        structure: usize,
        style: u32,
        seed: u64,
        roughness: f32,
    ) -> serde_json::Value {
        let started = Instant::now();
        // The copied pre-row-scheduling routine retains current prepared
        // constants/caches but executes every base texel and normal serially.
        let maps =
            original_texture_maps(recipe.surface, style, seed, roughness, Some(recipe), true);
        let texture_seconds = started.elapsed().as_secs_f64();
        let size = recipe.map_size(style);
        let started = Instant::now();
        let images = mapped_images(maps, size, Some(recipe));
        let mip_seconds = started.elapsed().as_secs_f64();
        let bytes: usize = images
            .iter()
            .map(|image| image.data.as_ref().unwrap().len())
            .sum();
        black_box(images);
        serde_json::json!({
            "room_seed": scene.seed,
            "surface": recipe.surface,
            "structure": structure,
            "floor_style": style,
            "kind": if structure == 0 { "base" } else { "structural_variant" },
            "recipe_seed": recipe.seed,
            "size": size,
            "texture_eval_ms": texture_seconds * 1000.,
            "pbr_mips_ms": mip_seconds * 1000.,
            "image_bytes": bytes,
            "textile_knit": recipe.textile.as_ref().is_some_and(|textile| textile.knit),
            "paint_application": recipe.coating.as_ref().is_some_and(|coating| coating.application.is_some()),
            "mineral": recipe.mineral.is_some(),
            "casting": recipe.mineral.as_ref().is_some_and(|mineral| mineral.casting.is_some()),
            "leaf": recipe.leaf.is_some(),
        })
    }

    let mut rows = Vec::new();
    for seed in [200, 202, 207, 43_084_482] {
        let humans = if seed == 43_084_482 { 0. } else { 0.25 };
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, humans)
                .unwrap();
        // Use the actual assembly-selected base and finish consumers. Geometry
        // construction and selection are outside all per-map timing boundaries.
        let selection = {
            let geometry = SceneGeometry {
                architecture: architecture::architecture(&scene),
                objects: scene.objects.iter().map(objects::build_object).collect(),
                humans: scene.humans.iter().map(humans::build_human).collect(),
            };
            geometry.material_selection(&scene)
        };
        let recipes = &scene.program.as_ref().unwrap().materials;
        let mut checked = BTreeSet::new();
        for definition in definitions(&scene) {
            let (surface, _, roughness, _, _) = definition;
            // Same atlas applicability as prepare_map. Separate screen/board/
            // paper raster programs are deliberately outside this recipe profile.
            let atlas = matches!(
                surface,
                Surface::Wood
                    | Surface::WoodEdge
                    | Surface::Floor
                    | Surface::Paint
                    | Surface::Accent
                    | Surface::Ceiling
                    | Surface::Fabric
                    | Surface::FabricAlt
                    | Surface::Bark
                    | Surface::Terracotta
                    | Surface::Soil
                    | Surface::Concrete
                    | Surface::Ceramic
                    | Surface::Leaf
                    | Surface::LeafLight
                    | Surface::LeafVariegated
                    | Surface::Metal
                    | Surface::Chrome
                    | Surface::Plastic
                    | Surface::Rubber
                    | Surface::Paper
                    | Surface::Art
                    | Surface::Leather
            );
            if !atlas || !selection.needs_maps(&scene, surface) {
                continue;
            }
            let row = profile(
                &scene,
                &recipes[surface as usize],
                0,
                scene.floor_style,
                scene.material_seed(),
                roughness,
            );
            eprintln!("{row}");
            rows.push(row);
            checked.insert((surface, 0));
        }
        for &(surface, slot) in &selection.finishes {
            let structure = slot % variants::structure_count(surface);
            if structure == 0 || !checked.insert((surface, structure)) {
                continue;
            }
            let recipe = recipes[surface as usize].variant(structure);
            let row = profile(
                &scene,
                &recipe,
                structure,
                scene.floor_style,
                recipe.seed,
                recipe.roughness,
            );
            eprintln!("{row}");
            rows.push(row);
        }
        if !scene.humans.is_empty() {
            // Exact room-shared knit recipe from human::knit. Furniture Fabric
            // recipes remain woven and do not represent this separate atlas.
            let mut knit = program::sample(scene.material_seed().wrapping_add(0x4b4e4954))
                [Surface::Fabric as usize]
                .clone();
            knit.layers = None;
            let textile = knit.textile.as_mut().unwrap();
            textile.knit = true;
            textile.yarn_tint = [[1.; 3]; 2];
            textile.lustre *= 0.35;
            knit.period_m = 0.0015 * textile.yarns[1] as f32;
            knit.relief_m = 0.00010 + textile.crimp * 0.0002;
            knit.roughness = 0.83;
            let mut row = profile(&scene, &knit, 0, 0, knit.seed, knit.roughness);
            row["kind"] = serde_json::json!("human_shared_knit");
            eprintln!("{row}");
            rows.push(row);

            // Separately time the existing shared skin/hair path including its
            // mips and staged asset insertion. Store setup/drop are excluded.
            let image_assets = Assets::<Image>::default();
            let material_assets = Assets::<StandardMaterial>::default();
            let mut images =
                crate::scene::procedural_indoor::preparation::StagedAssets::new(&image_assets);
            let mut materials =
                crate::scene::procedural_indoor::preparation::StagedAssets::new(&material_assets);
            let started = Instant::now();
            let handles = human::maps(
                scene.material_seed(),
                StandardMaterial::default(),
                &mut images,
                &mut materials,
            );
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
            black_box(handles);
            eprintln!(
                "{}",
                serde_json::json!({
                    "room_seed": scene.seed,
                    "kind": "human_shared_skin_hair",
                    "size": 256,
                    "images": 6,
                    "combined_eval_mips_ms": elapsed_ms,
                    "scope": "human::maps with neutral cloth template; skin/hair fields+mips+staged insertion; excludes store setup/drop, workers, GPU/capture",
                })
            );
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "scope": "diagnostic only; exact current prepared serial atlas evaluation including setup/height/base color/normal/ORM; PBR mips/descriptors separately; excludes workers/geometry/selection/image destruction/GPU/capture",
            "room_seeds": [200,202,207,43084482],
            "maps": rows.len(),
            "texture_eval_ms": rows.iter().map(|row| row["texture_eval_ms"].as_f64().unwrap()).sum::<f64>(),
            "pbr_mips_ms": rows.iter().map(|row| row["pbr_mips_ms"].as_f64().unwrap()).sum::<f64>(),
        })
    );
}
