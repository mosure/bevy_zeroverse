use std::path::PathBuf;
use rand::Rng;

use bevy::prelude::*;

use crate::{
    app::BevyZeroverseConfig,
    scene::RegenerateSceneEvent,
};

#[cfg(not(target_family = "wasm"))]
use crate::util::strip_extended_length_prefix;


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

        app.init_resource::<MaterialLoaderSettings>();
        app.init_resource::<MaterialRoots>();
        app.init_resource::<ZeroverseMaterials>();

        app.register_type::<MaterialLoaderSettings>();

        app.add_systems(PreStartup, find_materials);
        app.add_systems(Startup, load_materials);
        app.add_systems(Update, material_exchange);
        app.add_systems(PostUpdate, reload_materials);
    }
}

#[derive(Resource, Reflect, Debug)]
#[reflect(Resource)]
pub struct MaterialLoaderSettings {
    pub batch_size: usize,
}

impl Default for MaterialLoaderSettings {
    fn default() -> Self {
        Self {
            batch_size: 100,
        }
    }
}


#[derive(Resource, Default, Debug)]
pub struct MaterialRoots {
    pub roots: Vec<PathBuf>,
}


fn find_materials(
    mut found_materials: ResMut<MaterialRoots>,
) {
    #[cfg(target_family = "wasm")]
    {
        found_materials.roots = vec![
            PathBuf::from("materials/subset/Ceramic/0557_brick_uneven_stones"),
            PathBuf::from("materials/subset/Fabric/acg_fabric_009"),
            PathBuf::from("materials/subset/Ground/acg_rocks_023"),
            PathBuf::from("materials/subset/Marble/st_marble_038"),
            PathBuf::from("materials/subset/Terracotta/acg_painted_bricks_002"),
            PathBuf::from("materials/subset/Wood/acg_planks_003"),
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

        info!("current working directory: {}", cwd.to_string_lossy());

        let asset_server_path = cwd.join("./assets");
        let pattern = format!("{}/**/**/basecolor.jpg", asset_server_path.to_string_lossy());

        found_materials.roots = glob::glob(&pattern)
            .expect("failed to read glob pattern")
            .filter_map(Result::ok)
            .filter_map(|path| {
                path.parent()
                    .and_then(|parent| parent.strip_prefix(&asset_server_path).ok())
                    .map(std::path::Path::to_path_buf)
            })
            .collect();

        info!("found {} materials", found_materials.roots.len());
    }
}


fn load_materials(
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut load_event: EventWriter<MaterialsLoadedEvent>,
    material_loader_settings: Res<MaterialLoaderSettings>,
    found_materials: Res<MaterialRoots>,
) {
    use rand::seq::IteratorRandom;
    let rng = &mut rand::thread_rng();

    let roots = found_materials.roots.iter()
        .choose_multiple(rng, material_loader_settings.batch_size);

    for root in roots {
        let basecolor_path = root.join("basecolor.jpg");
        let basecolor_handle = asset_server.load(basecolor_path);

        let metallic_roughness_path = root.join("metallic_roughness.jpg");
        let metallic_roughness_handle = asset_server.load(metallic_roughness_path);

        let normal_map_path = root.join("normal.jpg");
        let normal_map_handle = asset_server.load(normal_map_path);

        let depth_map_path = root.join("height.jpg");
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
            reflectance: (rng.gen_range(0.0..0.8) as f32).powf(1.8),
            ..Default::default()
        });

        zeroverse_materials.materials.push(material);
    }

    info!("loaded {} materials", zeroverse_materials.materials.len());

    load_event.write(MaterialsLoadedEvent);
}


fn reload_materials(
    asset_server: Res<AssetServer>,
    materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut shuffle_events: EventReader<ShuffleMaterialsEvent>,
    load_event: EventWriter<MaterialsLoadedEvent>,
    material_loader_settings: Res<MaterialLoaderSettings>,
    found_materials: Res<MaterialRoots>,
) {
    if shuffle_events.is_empty() {
        return;
    }
    shuffle_events.clear();

    zeroverse_materials.materials.clear();

    load_materials(
        asset_server,
        materials,
        zeroverse_materials,
        load_event,
        material_loader_settings,
        found_materials,
    );
}


fn material_exchange(
    args: Res<BevyZeroverseConfig>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    mut shuffle_events: EventWriter<ShuffleMaterialsEvent>,
    mut scene_counter: Local<u32>,
) {
    if args.regenerate_scene_material_shuffle_period == 0 {
        return;
    }

    for _ in regenerate_events.read() {
        *scene_counter += 1;
    }

    if *scene_counter >= args.regenerate_scene_material_shuffle_period {
        *scene_counter = 0;

        shuffle_events.write(ShuffleMaterialsEvent);
    }
}
