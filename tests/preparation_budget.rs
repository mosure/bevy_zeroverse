//! Process-isolated qualification of the shared preparation pool. Nested mesh,
//! material and room tasks must complete even with a single worker, and the
//! worker budget must not change generated geometry or CPU material maps.
#![cfg(not(target_arch = "wasm32"))]

use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    scene::{
        procedural_indoor::{
            gi::IndoorGiSettings,
            preparation::{configure_preparation_threads, prepare_indoor, StagedAssets},
        },
        ZeroverseSceneSettings,
    },
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    process::Command,
};

fn digest(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> u64 {
    let mut hash = DefaultHasher::new();
    for part in parts {
        part.as_ref().hash(&mut hash);
    }
    hash.finish()
}

#[test]
fn preparation_worker_budgets_preserve_scene_assets() {
    if let Ok(threads) = std::env::var("ZERO_TEST_PREPARATION_THREADS") {
        let threads = threads.parse::<usize>().unwrap();
        configure_preparation_threads(threads).unwrap();
        bevy_zeroverse::headless::setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
        let config = BevyZeroverseConfig {
            headless: true,
            indoor_gpu_materials: false,
            ..default()
        };
        let settings = ZeroverseSceneSettings {
            num_cameras: 3,
            ..default()
        };
        let mut fingerprints = Vec::new();
        for seed in [400, 401] {
            let mut images = Assets::<Image>::default();
            let mut materials = Assets::<StandardMaterial>::default();
            let mut meshes = Assets::<Mesh>::default();
            let prepared = bevy::tasks::block_on(prepare_indoor(
                seed,
                &config,
                &settings,
                IndoorGiSettings::default(),
                StagedAssets::new(&images),
                StagedAssets::new(&materials),
                StagedAssets::new(&meshes),
            ))
            .unwrap();
            let manifest = serde_json::to_vec(&prepared.manifest).unwrap();
            prepared.images.commit(&mut images);
            prepared.materials.commit(&mut materials);
            prepared.meshes.commit(&mut meshes);
            let mut mesh_hashes: Vec<_> = meshes
                .iter()
                .map(|(_, mesh)| {
                    let mut parts: Vec<_> = mesh
                        .attributes()
                        .map(|(_, values)| values.get_bytes().to_vec())
                        .collect();
                    let indices: Vec<_> = mesh
                        .indices()
                        .map(|indices| indices.iter().map(|i| i as u32).collect())
                        .unwrap_or_default();
                    parts.push(bytemuck::cast_slice(&indices).to_vec());
                    digest(parts)
                })
                .collect();
            let mut image_hashes: Vec<_> = images
                .iter()
                .map(|(_, image)| {
                    digest([
                        format!("{:?}", image.texture_descriptor).into_bytes(),
                        image.data.clone().unwrap_or_default(),
                    ])
                })
                .collect();
            mesh_hashes.sort_unstable();
            image_hashes.sort_unstable();
            assert!(!mesh_hashes.is_empty() && !image_hashes.is_empty());
            fingerprints.push((
                digest([manifest]),
                mesh_hashes,
                image_hashes,
                materials.len(),
            ));
        }
        configure_preparation_threads(threads).unwrap();
        assert!(configure_preparation_threads(if threads == 1 { 4 } else { 1 }).is_err());
        std::fs::write(
            std::env::var("ZERO_TEST_PREPARATION_OUTPUT").unwrap(),
            serde_json::to_vec(&fingerprints).unwrap(),
        )
        .unwrap();
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let mut outputs = Vec::new();
    for threads in [1, 4] {
        let path = temp.path().join(format!("threads-{threads}.json"));
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "preparation_worker_budgets_preserve_scene_assets",
            ])
            .env("ZERO_TEST_PREPARATION_THREADS", threads.to_string())
            .env("ZERO_TEST_PREPARATION_OUTPUT", &path)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "preparation failed with {threads} workers"
                );
                break;
            }
            if std::time::Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("nested preparation stalled with {threads} workers");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        outputs.push(std::fs::read(path).unwrap());
    }
    assert_eq!(outputs[0], outputs[1], "worker budget changed scene assets");
}
