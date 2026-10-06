//! Full atlas replay at actual geometry-selected base/finish recipes, including
//! all mip bytes and image metadata. The reference disables only map preparation.
use super::super::{humans, layout::IndoorLayout, preparation::SceneGeometry};
use super::*;
use std::collections::BTreeSet;

fn same_images(reference: [Image; 3], actual: [Image; 3], seed: u64, label: &str) -> usize {
    let mut bytes = 0;
    for (channel, (a, b)) in reference.into_iter().zip(actual).enumerate() {
        assert_eq!(
            a.texture_descriptor, b.texture_descriptor,
            "seed {seed} {label} channel {channel}"
        );
        assert_eq!(
            a.asset_usage, b.asset_usage,
            "seed {seed} {label} channel {channel}"
        );
        assert_eq!(
            format!("{:?}", a.sampler),
            format!("{:?}", b.sampler),
            "seed {seed} {label} channel {channel}"
        );
        bytes += a.data.as_ref().unwrap().len();
        assert_eq!(
            a.data, b.data,
            "seed {seed} {label} channel {channel} base or mip bytes"
        );
    }
    bytes
}

fn scalar_images(
    recipe: &program::MaterialRecipe,
    style: u32,
    seed: u64,
    roughness: f32,
) -> [Image; 3] {
    mapped_images(
        texture_maps_with_preparation(recipe.surface, style, seed, roughness, Some(recipe), false),
        recipe.map_size(style),
        Some(recipe),
    )
}

#[test]
fn consumed_scene_recipes_replay_every_scalar_map_byte_and_sampler() {
    let mut triplets = 0;
    let mut bytes = 0;
    for seed in [43_084_482, 202, 207] {
        let human_density = if seed == 43_084_482 { 0. } else { 0.25 };
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 5, human_density)
                .unwrap();
        if seed == 43_084_482 {
            assert!(
                scene.humans.is_empty(),
                "match the sparse RGB replay fixture"
            );
        }
        let geometry = SceneGeometry {
            architecture: super::super::architecture::architecture(&scene),
            objects: scene
                .objects
                .iter()
                .map(super::super::objects::build_object)
                .collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        let recipes = &scene.program.as_ref().unwrap().materials;
        let mut checked = BTreeSet::new();
        let mut room_triplets = 0;
        // The base path is the same one used by IndoorMaterials construction;
        // prepare_map returns None for surfaces rendered by separate programs.
        for definition in definitions(&scene) {
            let (surface, _, roughness, _, _) = definition;
            if !selection.needs_maps(&scene, surface) {
                continue;
            }
            if let Some(actual) = prepare_map(&scene, &definition) {
                let recipe = &recipes[surface as usize];
                bytes += same_images(
                    scalar_images(recipe, scene.floor_style, scene.material_seed(), roughness),
                    actual,
                    seed,
                    &format!("base/{surface:?}"),
                );
                checked.insert((surface, 0));
                room_triplets += 1;
            }
        }
        // Nonzero groups replace all parent maps. Compare each actually used
        // structure once, matching the production deduplication of finish slots.
        for &(surface, slot) in &selection.finishes {
            let group = slot % variants::structure_count(surface);
            if group == 0 || !checked.insert((surface, group)) {
                continue;
            }
            let recipe = recipes[surface as usize].variant(group);
            bytes += same_images(
                scalar_images(&recipe, scene.floor_style, recipe.seed, recipe.roughness),
                recipe.maps(scene.floor_style),
                seed,
                &format!("finish/{surface:?}/{group}"),
            );
            room_triplets += 1;
        }
        assert!(room_triplets > 0);
        triplets += room_triplets;
        eprintln!(
            "{}",
            serde_json::json!({"seed":seed,"humans":scene.humans.len(),"checked_map_triplets":room_triplets,"checked_recipes":checked})
        );
    }
    eprintln!(
        "{}",
        serde_json::json!({"scope":"scalar versus prepared complete consumed atlases, all mip bytes/descriptors/samplers","room_seeds":[43084482,202,207],"checked_map_triplets":triplets,"checked_images":triplets*3,"checked_bytes":bytes})
    );
}
