use std::path::PathBuf;
use rand::Rng;

use bevy::prelude::*;


#[derive(Resource, Default, Debug)]
pub struct ZeroverseMaterials {
    // TODO: support material metadata (e.g. material name, category, split)
    pub materials: Vec<Handle<StandardMaterial>>,
}


#[derive(Event)]
pub struct ShuffleMaterialsEvent;

#[derive(Event)]
pub struct MaterialsLoadedEvent;


pub struct ZeroverseMaterialPlugin;

impl Plugin for ZeroverseMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<MaterialsLoadedEvent>();
        app.add_event::<ShuffleMaterialsEvent>();

        app.init_resource::<ZeroverseMaterials>();

        app.add_systems(PreStartup, load_materials);
        app.add_systems(PostUpdate, reload_materials);
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
    use rand::seq::IteratorRandom;

    // TODO: use asset_server scanning: https://github.com/bevyengine/bevy/issues/2291

    let cwd = std::env::current_dir().expect("failed to get current working directory");
    info!("current working directory: {}", cwd.to_string_lossy());

    let asset_server_path = cwd.join("./assets");
    let pattern = format!("{}/**/**/basecolor.jpg", asset_server_path.to_string_lossy());

    let available: Vec<PathBuf> = glob::glob(&pattern)
        .expect("failed to read glob pattern")
        .filter_map(Result::ok)
        .filter_map(|path| {
            path.parent()
                .and_then(|parent| parent.strip_prefix(&asset_server_path).ok())
                .map(std::path::Path::to_path_buf)
        })
        .collect::<Vec<_>>();

    info!("found {} materials", available.len());

    available.into_iter()
        .choose_multiple(&mut rand::thread_rng(), 100)
}


// TODO: support batched loading to avoid GPU RAM exhaustion
fn load_materials(
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut load_event: EventWriter<MaterialsLoadedEvent>,
) {
    let rng = &mut rand::thread_rng();

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
            double_sided: true,
            cull_mode: None,
            // specular_transmission: (rng.gen_range(0.0..1.0) as f32).powf(2.0),
            // ior: rng.gen_range(1.0..2.0),
            perceptual_roughness: rng.gen_range(0.3..0.7),
            reflectance: rng.gen_range(0.3..0.7),
            ..Default::default()
        });

        zeroverse_materials.materials.push(material);
    }

    info!("loaded {} materials", zeroverse_materials.materials.len());

    load_event.send(MaterialsLoadedEvent);
}


fn reload_materials(
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut shuffle_events: EventReader<ShuffleMaterialsEvent>,
    load_event: EventWriter<MaterialsLoadedEvent>,
) {
    if shuffle_events.is_empty() {
        return;
    }

    shuffle_events.clear();

    for material in zeroverse_materials.materials.iter() {
        materials.remove(material);
    }
    zeroverse_materials.materials.clear();

    load_materials(
        asset_server,
        materials,
        zeroverse_materials,
        load_event,
    );
}
