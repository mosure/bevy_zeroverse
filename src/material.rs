use std::path::PathBuf;

use bevy::prelude::*;
use rand::seq::IteratorRandom;


#[derive(Resource, Default, Debug)]
pub struct ZeroverseMaterials {
    // TODO: support material metadata (e.g. material name, category, split)
    pub materials: Vec<Handle<StandardMaterial>>,
}


pub struct ZeroverseMaterialPlugin;

impl Plugin for ZeroverseMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseMaterials>();

        app.add_systems(PreStartup, load_materials);
    }
}

#[cfg(target_family = "wasm")]
fn get_material_roots() -> Vec<PathBuf> {
    vec![
        PathBuf::from("materials/subset/Ceramic/0557_brick_uneven_stones"),
        PathBuf::from("materials/subset/Ground/acg_rocks_023"),
        PathBuf::from("materials/subset/Marble/st_marble_038"),
        PathBuf::from("materials/subset/Wood/acg_planks_003"),
    ]
}

#[cfg(not(target_family = "wasm"))]
fn get_material_roots() -> Vec<PathBuf> {
    // TODO: use asset_server scanning: https://github.com/bevyengine/bevy/issues/2291

    let cwd = std::env::current_dir().expect("failed to get current working directory");
    info!("current working directory: {}", cwd.to_string_lossy());

    let asset_server_path = cwd.join("./assets");
    let pattern = format!("{}/**/**/basecolor.jpg", asset_server_path.to_string_lossy());

    glob::glob(&pattern)
        .expect("failed to read glob pattern")
        .filter_map(Result::ok)
        .filter_map(|path| {
            path.parent()
                .and_then(|parent| parent.strip_prefix(&asset_server_path).ok())
                .map(std::path::Path::to_path_buf)
        })
        .choose_multiple(&mut rand::thread_rng(), 100)
}

// TODO: support batched loading to avoid GPU RAM exhaustion
fn load_materials(
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
) {
    let roots = get_material_roots();

    for root in roots {
        let basecolor_path = root.join("basecolor.jpg");
        let basecolor_handle = asset_server.load(basecolor_path);

        let metallic_roughness_path = root.join("metallic_roughness.jpg");
        let metallic_roughness_handle = asset_server.load(metallic_roughness_path);

        let normal_map_path = root.join("normal.jpg");
        let normal_map_handle = asset_server.load(normal_map_path);

        let depth_map_path = root.join("displacement.jpg");
        let depth_map_handle = asset_server.load(depth_map_path);

        let material = materials.add(StandardMaterial {
            base_color_texture: basecolor_handle.into(),
            metallic_roughness_texture: metallic_roughness_handle.into(),
            normal_map_texture: normal_map_handle.into(),
            depth_map: depth_map_handle.into(),
            ..Default::default()
        });

        zeroverse_materials.materials.push(material);
    }

    info!("loaded {} materials", zeroverse_materials.materials.len());
}
