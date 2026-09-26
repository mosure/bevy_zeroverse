use rand::{seq::IteratorRandom, Rng};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::{app::BevyZeroverseConfig, scene::RegenerateSceneEvent};

#[cfg(not(target_family = "wasm"))]
use crate::asset::asset_root;

#[derive(Resource, Default, Debug)]
pub struct ZeroverseMaterials {
    // TODO: support material metadata (e.g. material name, category, split)
    pub materials: Vec<Handle<StandardMaterial>>,
}

/// Palette entries keep paths until a primitive actually chooses them. A 25-entry
/// sampling pool no longer decodes/uploads 100 textures for a four-part object.
#[derive(Resource, Default)]
pub struct MaterialTextureCatalog {
    roots: std::collections::HashMap<AssetId<StandardMaterial>, PathBuf>,
}
impl MaterialTextureCatalog {
    pub fn activate(
        &mut self,
        handle: &Handle<StandardMaterial>,
        server: &AssetServer,
        materials: &mut Assets<StandardMaterial>,
        wait: &mut crate::asset::WaitForAssets,
    ) {
        let Some(root) = self.roots.remove(&handle.id()) else {
            return;
        };
        let Some(mut material) = materials.get_mut(handle) else {
            return;
        };
        let mut load = |name: &str, srgb: bool| {
            let handle: Handle<Image> = server
                .load_builder()
                .with_settings(move |settings: &mut bevy::image::ImageLoaderSettings| {
                    settings.is_srgb = srgb;
                    settings.sampler = bevy::image::ImageSampler::Descriptor(
                        bevy::image::ImageSamplerDescriptor {
                            address_mode_u: bevy::image::ImageAddressMode::Repeat,
                            address_mode_v: bevy::image::ImageAddressMode::Repeat,
                            mag_filter: bevy::image::ImageFilterMode::Linear,
                            min_filter: bevy::image::ImageFilterMode::Linear,
                            mipmap_filter: bevy::image::ImageFilterMode::Linear,
                            ..default()
                        },
                    );
                })
                .load(root.join(name));
            wait.handles.push(handle.clone().untyped());
            Some(handle)
        };
        material.base_color_texture = load("basecolor.jpg", true);
        material.metallic_roughness_texture = load("metallic_roughness.jpg", false);
        material.normal_map_texture = load("normal.jpg", false);
        material.depth_map = load("height.jpg", false);
    }
}

#[derive(Event, Message)]
pub struct ShuffleMaterialsEvent;

#[derive(Event, Message)]
pub struct MaterialsLoadedEvent;

pub struct ZeroverseMaterialPlugin;

impl Plugin for ZeroverseMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<MaterialsLoadedEvent>();
        app.add_message::<ShuffleMaterialsEvent>();

        app.init_resource::<MaterialLoaderSettings>();
        app.init_resource::<MaterialRoots>();
        app.init_resource::<ZeroverseMaterials>();
        app.init_resource::<MaterialTextureCatalog>();

        app.register_type::<MaterialLoaderSettings>();

        app.init_resource::<crate::asset::CatalogTask<Vec<PathBuf>>>();
        app.add_systems(
            First,
            (discover_materials, reload_materials)
                .chain()
                .after(crate::asset::AssetDemandSet),
        );
        app.add_systems(Update, material_exchange);
    }
}

#[derive(Resource, Reflect, Debug)]
#[reflect(Resource)]
pub struct MaterialLoaderSettings {
    pub batch_size: usize,
}

impl Default for MaterialLoaderSettings {
    fn default() -> Self {
        Self { batch_size: 25 }
    }
}

#[derive(Resource, Default, Debug)]
pub struct MaterialRoots {
    pub roots: Vec<PathBuf>,
}

fn discover_materials(
    demand: Res<crate::asset::SceneAssetDemand>,
    mut found: ResMut<MaterialRoots>,
    mut task: ResMut<crate::asset::CatalogTask<Vec<PathBuf>>>,
    mut wait: ResMut<crate::asset::WaitForAssets>,
    mut shuffle: MessageWriter<ShuffleMaterialsEvent>,
    mut complete: Local<bool>,
    mut active: Local<bool>,
) {
    if let Some(job) = task.0.as_mut() {
        if let Some(roots) = bevy::tasks::block_on(bevy::tasks::poll_once(job)) {
            found.roots = roots;
            *complete = true;
            task.0 = None;
            wait.pending_catalogs -= 1;
        }
    }
    if demand.materials && !*complete && task.0.is_none() {
        wait.pending_catalogs += 1;
        task.0 = Some(bevy::tasks::IoTaskPool::get().spawn(async { find_materials() }));
    }
    if demand.materials && *complete && !*active {
        shuffle.write(ShuffleMaterialsEvent);
        *active = true;
    } else if !demand.materials {
        *active = false;
    }
}

fn find_materials() -> Vec<PathBuf> {
    #[cfg(target_family = "wasm")]
    {
        vec![
            PathBuf::from("materials/subset/Ceramic/0557_brick_uneven_stones"),
            PathBuf::from("materials/subset/Fabric/acg_fabric_009"),
            PathBuf::from("materials/subset/Ground/acg_rocks_023"),
            PathBuf::from("materials/subset/Marble/st_marble_038"),
            PathBuf::from("materials/subset/Terracotta/acg_painted_bricks_002"),
            PathBuf::from("materials/subset/Wood/acg_planks_003"),
        ]
    }

    // TODO: add manifest file caching to improve load times
    #[cfg(not(target_family = "wasm"))]
    {
        let asset_server_path = asset_root();
        let pattern = format!(
            "{}/materials/**/basecolor.jpg",
            asset_server_path.to_string_lossy()
        );

        let mut roots: Vec<_> = glob::glob(&pattern)
            .expect("failed to read glob pattern")
            .filter_map(Result::ok)
            .filter_map(|path| {
                path.parent()
                    .and_then(|parent| parent.strip_prefix(&asset_server_path).ok())
                    .map(std::path::Path::to_path_buf)
            })
            .collect();

        roots.sort();
        info!("found {} materials", roots.len());
        roots
    }
}

#[allow(clippy::too_many_arguments)]
fn load_materials(
    args: Option<Res<BevyZeroverseConfig>>,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut load_event: MessageWriter<MaterialsLoadedEvent>,
    material_loader_settings: Res<MaterialLoaderSettings>,
    found_materials: Res<MaterialRoots>,
    mut textures: ResMut<MaterialTextureCatalog>,
    mut wait: ResMut<crate::asset::WaitForAssets>,
) {
    if args.as_ref().is_some_and(|args| {
        args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor && !args.material_grid
    }) {
        load_event.write(MaterialsLoadedEvent);
        return;
    }
    let mut rng = rand::rng();

    let roots = found_materials
        .roots
        .iter()
        .choose_multiple(&mut rng, material_loader_settings.batch_size);

    for root in roots {
        let material = materials.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            // specular_transmission: (rng.random_range(0.0..1.0) as f32).powf(2.0),
            // ior: rng.random_range(1.0..2.0),
            perceptual_roughness: rng.random_range(0.3..0.7),
            reflectance: (rng.random_range(0.0..0.8) as f32).powf(1.8),
            ..Default::default()
        });

        textures.roots.insert(material.id(), root.clone());
        if args.as_ref().is_some_and(|args| args.material_grid) {
            textures.activate(&material, &asset_server, &mut materials, &mut wait);
        }
        zeroverse_materials.materials.push(material);
    }

    // Basic legacy scenes also work without a downloaded texture catalog.
    if zeroverse_materials.materials.is_empty() {
        zeroverse_materials
            .materials
            .push(materials.add(StandardMaterial::default()));
    }

    info!("loaded {} materials", zeroverse_materials.materials.len());

    load_event.write(MaterialsLoadedEvent);
}

#[allow(clippy::too_many_arguments)]
fn reload_materials(
    args: Option<Res<BevyZeroverseConfig>>,
    asset_server: Res<AssetServer>,
    materials: ResMut<Assets<StandardMaterial>>,
    mut zeroverse_materials: ResMut<ZeroverseMaterials>,
    mut shuffle_events: MessageReader<ShuffleMaterialsEvent>,
    load_event: MessageWriter<MaterialsLoadedEvent>,
    material_loader_settings: Res<MaterialLoaderSettings>,
    found_materials: Res<MaterialRoots>,
    mut textures: ResMut<MaterialTextureCatalog>,
    wait: ResMut<crate::asset::WaitForAssets>,
) {
    if shuffle_events.is_empty() {
        return;
    }
    shuffle_events.clear();

    zeroverse_materials.materials.clear();
    textures.roots.clear();

    load_materials(
        args,
        asset_server,
        materials,
        zeroverse_materials,
        load_event,
        material_loader_settings,
        found_materials,
        textures,
        wait,
    );
}

fn material_exchange(
    args: Res<BevyZeroverseConfig>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    mut shuffle_events: MessageWriter<ShuffleMaterialsEvent>,
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
