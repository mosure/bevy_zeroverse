use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use bevy::{gltf::Gltf, mesh::VertexAttributeValues, prelude::*};
use noise::{NoiseFn, Perlin};
use rand::{seq::IteratorRandom, Rng};

#[cfg(not(target_family = "wasm"))]
use itertools::Itertools;

use crate::{app::BevyZeroverseConfig, asset::WaitForAssets, scene::RegenerateSceneEvent};

#[cfg(not(target_family = "wasm"))]
use crate::asset::asset_root;

pub type MeshCategory = String;

#[derive(Default, Debug)]
pub struct ZeroverseMesh {
    pub category: MeshCategory,
    pub handle: Handle<Mesh>,
    pub material: Option<Handle<StandardMaterial>>,
    pub gltf: Handle<Gltf>,
    // pub path: String,
    // pub subset: String,
}

#[derive(Resource, Default, Debug)]
pub struct ZeroverseMeshes {
    pub meshes: HashMap<MeshCategory, Vec<ZeroverseMesh>>,
    pub normalized: HashSet<Handle<Mesh>>,
    pub original_sizes: HashMap<Handle<Mesh>, Vec3>,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct NormalizeMeshesSet;

#[derive(Event, Message)]
pub struct ShuffleMeshesEvent;

#[derive(Event, Message)]
pub struct MeshesLoadedEvent;

pub struct ZeroverseMeshPlugin;

impl Plugin for ZeroverseMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<MeshesLoadedEvent>();
        app.add_message::<ShuffleMeshesEvent>();

        app.init_resource::<MeshLoaderSettings>();
        app.init_resource::<MeshRoots>();
        app.init_resource::<ZeroverseMeshes>();

        app.register_type::<MeshLoaderSettings>();

        app.init_resource::<crate::asset::CatalogTask<HashMap<MeshCategory, Vec<PathBuf>>>>();
        app.add_systems(
            First,
            (discover_meshes, reload_meshes)
                .chain()
                .after(crate::asset::AssetDemandSet),
        );
        app.add_systems(
            Update,
            (mesh_exchange, normalize_meshes.in_set(NormalizeMeshesSet)),
        );
    }
}

#[derive(Resource, Reflect, Debug)]
#[reflect(Resource)]
pub struct MeshLoaderSettings {
    pub category_batch_size: usize,
}

impl Default for MeshLoaderSettings {
    fn default() -> Self {
        Self {
            category_batch_size: 25,
        }
    }
}

#[derive(Resource, Default, Debug)]
pub struct MeshRoots {
    pub categories: HashMap<MeshCategory, Vec<PathBuf>>,
}

fn discover_meshes(
    demand: Res<crate::asset::SceneAssetDemand>,
    mut found: ResMut<MeshRoots>,
    mut task: ResMut<crate::asset::CatalogTask<HashMap<MeshCategory, Vec<PathBuf>>>>,
    mut wait: ResMut<WaitForAssets>,
    mut shuffle: MessageWriter<ShuffleMeshesEvent>,
    mut complete: Local<bool>,
    mut active: Local<std::collections::BTreeSet<String>>,
) {
    if let Some(job) = task.0.as_mut() {
        if let Some(categories) = bevy::tasks::block_on(bevy::tasks::poll_once(job)) {
            found.categories = categories;
            *complete = true;
            task.0 = None;
            wait.pending_catalogs -= 1;
        }
    }
    if !demand.mesh_categories.is_empty() && !*complete && task.0.is_none() {
        wait.pending_catalogs += 1;
        task.0 = Some(bevy::tasks::IoTaskPool::get().spawn(async { find_meshes() }));
    }
    if *complete && *active != demand.mesh_categories {
        *active = demand.mesh_categories.clone();
        shuffle.write(ShuffleMeshesEvent);
    }
}

fn find_meshes() -> HashMap<MeshCategory, Vec<PathBuf>> {
    #[cfg(target_family = "wasm")]
    {
        HashMap::from([(
            "chair".into(),
            vec![PathBuf::from("models/subset/chair/0.glb")],
        )])
    }

    // TODO: add manifest file caching to improve load times
    #[cfg(not(target_family = "wasm"))]
    {
        let asset_server_path = asset_root();
        let pattern: String = format!("{}/models/**/*.glb", asset_server_path.to_string_lossy());

        let mut categories = glob::glob(&pattern)
            .expect("failed to read glob pattern")
            .filter_map(Result::ok)
            .filter_map(|path| {
                let category = path.parent()?.file_name()?.to_str()?.to_string();
                path.strip_prefix(&asset_server_path)
                    .ok()
                    .map(|p| (category, p.to_path_buf()))
            })
            .into_group_map();

        for paths in categories.values_mut() {
            paths.sort();
        }
        for (category, paths) in &categories {
            info!("found {} meshes for category `{}`", paths.len(), category);
        }
        categories
    }
}

fn load_meshes(
    demand: Res<crate::asset::SceneAssetDemand>,
    asset_server: Res<AssetServer>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
    mut load_event: MessageWriter<MeshesLoadedEvent>,
    mesh_loader_settings: Res<MeshLoaderSettings>,
    found_meshes: Res<MeshRoots>,
    mut wait_for: ResMut<WaitForAssets>,
) {
    let mut rng = rand::rng();

    for (category, paths) in &found_meshes.categories {
        if !demand.mesh_categories.contains(category) {
            continue;
        }
        let selected_paths = paths
            .iter()
            .choose_multiple(&mut rng, mesh_loader_settings.category_batch_size);

        for path in &selected_paths {
            let path_str = path.to_string_lossy();

            let gltf_handle: Handle<Gltf> = asset_server.load(path_str.to_string());
            let glb_path = format!("{path_str}#Mesh0/Primitive0");

            let mesh_handle = asset_server.load(glb_path);

            let zeroverse_mesh = ZeroverseMesh {
                category: category.clone(),
                handle: mesh_handle.clone(),
                material: None,
                gltf: gltf_handle,
            };

            wait_for.handles.push(mesh_handle.untyped());
            zeroverse_meshes
                .meshes
                .entry(category.clone())
                .or_default()
                .push(zeroverse_mesh);
        }

        info!(
            "loaded {} meshes for category `{}`",
            selected_paths.len(),
            category
        );
    }

    info!(
        "loaded total of {} categories",
        zeroverse_meshes.meshes.len()
    );

    load_event.write(MeshesLoadedEvent);
}

#[allow(clippy::too_many_arguments)]
fn reload_meshes(
    demand: Res<crate::asset::SceneAssetDemand>,
    asset_server: Res<AssetServer>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
    mut shuffle_events: MessageReader<ShuffleMeshesEvent>,
    load_event: MessageWriter<MeshesLoadedEvent>,
    mesh_loader_settings: Res<MeshLoaderSettings>,
    found_meshes: Res<MeshRoots>,
    wait_for: ResMut<WaitForAssets>,
) {
    if shuffle_events.is_empty() {
        return;
    }
    shuffle_events.clear();

    zeroverse_meshes.meshes.clear();
    zeroverse_meshes.normalized.clear();
    zeroverse_meshes.original_sizes.clear();

    load_meshes(
        demand,
        asset_server,
        zeroverse_meshes,
        load_event,
        mesh_loader_settings,
        found_meshes,
        wait_for,
    );
}

fn mesh_exchange(
    args: Res<BevyZeroverseConfig>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    mut shuffle_events: MessageWriter<ShuffleMeshesEvent>,
    mut scene_counter: Local<u32>,
) {
    if args.regenerate_scene_mesh_shuffle_period == 0 {
        return;
    }

    for _ in regenerate_events.read() {
        *scene_counter += 1;
    }

    if *scene_counter >= args.regenerate_scene_mesh_shuffle_period {
        *scene_counter = 0;

        shuffle_events.write(ShuffleMeshesEvent);
    }
}

fn normalize_meshes(
    mut meshes: ResMut<Assets<Mesh>>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
) {
    let handles: Vec<Handle<Mesh>> = zeroverse_meshes
        .meshes
        .values()
        .flatten()
        .map(|mesh| mesh.handle.clone())
        .collect();

    for handle in handles {
        if zeroverse_meshes.normalized.contains(&handle) {
            continue;
        }

        if let Some(mut mesh_asset) = meshes.get_mut(&handle) {
            match normalize_mesh_to_unit_cube(&mut mesh_asset) {
                Some(size) => {
                    zeroverse_meshes.original_sizes.insert(handle.clone(), size);
                    zeroverse_meshes.normalized.insert(handle);
                }
                None => {
                    debug!(
                        "unable to normalize mesh {:?}; missing positions or zero extents",
                        handle
                    );
                }
            }
        }
    }
}

pub fn normalize_mesh_to_unit_cube(mesh: &mut Mesh) -> Option<Vec3> {
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        let (min, max) = mesh_bounds_from_positions(positions);

        let size = max - min;
        let max_extent = size.max_element();

        if max_extent <= f32::EPSILON {
            return None;
        }

        let center_x = (max.x + min.x) * 0.5;
        let center_z = (max.z + min.z) * 0.5;
        let floor = min.y;

        for position in positions.iter_mut() {
            position[0] = (position[0] - center_x) / max_extent;
            position[1] = (position[1] - floor) / max_extent;
            position[2] = (position[2] - center_z) / max_extent;
        }

        return Some(size);
    }

    None
}

pub fn mesh_bounds(mesh: &Mesh) -> Option<(Vec3, Vec3)> {
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    {
        return Some(mesh_bounds_from_positions(positions));
    }

    None
}

fn mesh_bounds_from_positions(positions: &[[f32; 3]]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for position in positions.iter() {
        min.x = min.x.min(position[0]);
        min.y = min.y.min(position[1]);
        min.z = min.z.min(position[2]);

        max.x = max.x.max(position[0]);
        max.y = max.y.max(position[1]);
        max.z = max.z.max(position[2]);
    }

    (min, max)
}

pub fn displace_vertices_with_noise(mesh: &mut Mesh, frequency: f32, scale: f32) {
    let mut rng = rand::rng();

    let perlin_x = Perlin::new(rng.random());
    let perlin_y = Perlin::new(rng.random());
    let perlin_z = Perlin::new(rng.random());

    let mut positions_attr = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap()
        .to_vec();

    for position in positions_attr.iter_mut() {
        let coords = [
            (position[0] * frequency) as f64,
            (position[1] * frequency) as f64,
            (position[2] * frequency) as f64,
        ];

        let n_x = perlin_x.get(coords) * scale as f64;
        position[0] += n_x as f32;

        let n_y = perlin_y.get(coords) * scale as f64;
        position[1] += n_y as f32;

        let n_z = perlin_z.get(coords) * scale as f64;
        position[2] += n_z as f32;
    }

    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions_attr),
    );
}
