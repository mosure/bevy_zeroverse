use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use bevy::{
    prelude::*,
    render::mesh::VertexAttributeValues,
};
use itertools::Itertools;
use noise::{NoiseFn, Perlin};
use rand::Rng;

use crate::{
    app::BevyZeroverseConfig,
    asset::WaitForAssets,
    scene::RegenerateSceneEvent,
};


pub type MeshCategory = String;

#[derive(Default, Debug)]
pub struct ZeroverseMesh {
    pub category: MeshCategory,
    pub handle: Handle<Mesh>,
    // pub path: String,
    // pub subset: String,
}

#[derive(Resource, Default, Debug)]
pub struct ZeroverseMeshes {
    pub meshes: HashMap<MeshCategory, Vec<ZeroverseMesh>>,
}


#[derive(Event)]
pub struct ShuffleMeshesEvent;

#[derive(Event)]
pub struct MeshesLoadedEvent;


pub struct ZeroverseMeshPlugin;

impl Plugin for ZeroverseMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<MeshesLoadedEvent>();
        app.add_event::<ShuffleMeshesEvent>();

        app.init_resource::<MeshLoaderSettings>();
        app.init_resource::<MeshRoots>();
        app.init_resource::<ZeroverseMeshes>();

        app.register_type::<MeshLoaderSettings>();

        app.add_systems(PreStartup, find_meshes);
        app.add_systems(Startup, load_meshes);
        app.add_systems(Update, mesh_exchange);
        app.add_systems(PostUpdate, reload_meshes);
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


fn strip_extended_length_prefix(path: &Path) -> PathBuf {
    if cfg!(windows) {
        let prefix = r"\\?\";
        if let Some(path_str) = path.to_str() {
            if let Some(stripped) = path_str.strip_prefix(prefix) {
                return PathBuf::from(stripped);
            }
        }
    }
    path.to_path_buf()
}


fn find_meshes(
    mut found_meshes: ResMut<MeshRoots>,
) {
    #[cfg(target_family = "wasm")]
    {
        found_meshes.roots = vec![
            PathBuf::from("meshes/subset/chair/0.glb"),
        ];
        return;
    }

    // TODO: add manifest file caching to improve load times
    #[cfg(not(target_family = "wasm"))]
    {
        let cwd = match std::env::var("BEVY_ASSET_ROOT") {
            Ok(asset_root) => {
                info!("BEVY_ASSET_ROOT: `{}`", asset_root);
                let abs_path = PathBuf::from(asset_root).canonicalize().expect("failed to canonicalize asset root");

                strip_extended_length_prefix(&abs_path)
            }
            Err(_) => {
                std::env::current_dir().expect("failed to get current working directory")
            }
        };

        let asset_server_path = cwd.join("./assets");
        let pattern: String = format!("{}/**/*.glb", asset_server_path.to_string_lossy());

        found_meshes.categories = glob::glob(&pattern)
            .expect("failed to read glob pattern")
            .filter_map(Result::ok)
            .filter_map(|path| {
                let category = path.parent()?
                    .file_name()?
                    .to_str()?
                    .to_string();
                path.strip_prefix(&asset_server_path).ok()
                    .map(|p| (category, p.to_path_buf()))
            })
            .into_group_map();

        for (category, paths) in found_meshes.categories.iter() {
            info!("found {} meshes for category `{}`", paths.len(), category);
        }
    }
}


fn load_meshes(
    asset_server: Res<AssetServer>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
    mut load_event: EventWriter<MeshesLoadedEvent>,
    mesh_loader_settings: Res<MeshLoaderSettings>,
    found_meshes: Res<MeshRoots>,
    mut wait_for: ResMut<WaitForAssets>,
) {
    use rand::seq::IteratorRandom;
    let rng = &mut rand::thread_rng();

    for (category, paths) in &found_meshes.categories {
        let selected_paths = paths.iter()
            .choose_multiple(rng, mesh_loader_settings.category_batch_size);

        for path in &selected_paths {
            let path_str = path.to_string_lossy();
            let glb_path = format!("{}#Mesh0/Primitive0", path_str);
            let mesh_handle = asset_server.load(glb_path);
            let zeroverse_mesh = ZeroverseMesh {
                category: category.clone(),
                handle: mesh_handle.clone(),
            };

            wait_for.handles.push(mesh_handle.untyped());
            zeroverse_meshes.meshes
                .entry(category.clone())
                .or_default()
                .push(zeroverse_mesh);
        }

        info!("loaded {} meshes for category `{}`", selected_paths.len(), category);
    }

    info!("loaded total of {} meshes", zeroverse_meshes.meshes.len());

    load_event.send(MeshesLoadedEvent);
}


fn reload_meshes(
    asset_server: Res<AssetServer>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
    mut shuffle_events: EventReader<ShuffleMeshesEvent>,
    load_event: EventWriter<MeshesLoadedEvent>,
    mesh_loader_settings: Res<MeshLoaderSettings>,
    found_meshes: Res<MeshRoots>,
    wait_for: ResMut<WaitForAssets>,
) {
    if shuffle_events.is_empty() {
        return;
    }
    shuffle_events.clear();

    zeroverse_meshes.meshes.clear();

    load_meshes(
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
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    mut shuffle_events: EventWriter<ShuffleMeshesEvent>,
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

        shuffle_events.send(ShuffleMeshesEvent);
    }
}



pub fn displace_vertices_with_noise(mesh: &mut Mesh, frequency: f32, scale: f32) {
    let rng = &mut rand::thread_rng();

    let perlin_x = Perlin::new(rng.gen());
    let perlin_y = Perlin::new(rng.gen());
    let perlin_z = Perlin::new(rng.gen());

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

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(positions_attr));
}
