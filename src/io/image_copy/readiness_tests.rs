use super::*;
use crate::{scene::procedural_indoor::preparation::residency::FutureAssets, scene::SceneAabbNode};
use bevy::render::{ExtractSchedule, MainWorld};

/// Exercise the real Extract parameter against independently ticking worlds.
/// Mutations do not manually advance the main tick: Extract's SystemState must
/// fetch the resources' main-world ticks correctly between schedule runs.
struct Extraction {
    main: World,
    render: World,
    schedule: Schedule,
    scratch: Option<MainWorld>,
    scene: Entity,
}

impl Extraction {
    fn new() -> Self {
        let mut main = World::new();
        main.init_resource::<Assets<Mesh>>();
        main.init_resource::<Assets<Image>>();
        main.init_resource::<Assets<StandardMaterial>>();
        let scene = main.spawn(SceneAabbNode).id();
        let mut render = World::new();
        render.init_resource::<ExpectedAssets>();
        let mut schedule = Schedule::new(ExtractSchedule);
        schedule.add_systems(extract);
        Self {
            main,
            render,
            schedule,
            scratch: Some(MainWorld::default()),
            scene,
        }
    }

    fn check(&mut self, rebuilt: bool) {
        let previous_key = self.render.resource::<ExpectedAssets>().key;
        let previous_tick = self
            .render
            .get_resource_change_ticks::<ExpectedAssets>()
            .unwrap()
            .changed;
        let mut main_world = self.scratch.take().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.render.insert_resource(main_world);
        self.schedule.run(&mut self.render);
        let mut main_world = self.render.remove_resource::<MainWorld>().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.scratch = Some(main_world);

        // Independent literal original gather, not the cached helper or key.
        let scene = self
            .main
            .query_filtered::<Entity, With<SceneAabbNode>>()
            .single(&self.main)
            .ok();
        let future = self.main.get_resource::<FutureAssets>();
        let mut meshes = Vec::new();
        for (id, mesh) in self.main.resource::<Assets<Mesh>>().iter() {
            if mesh.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)
                && future.is_none_or(|f| !f.meshes.contains(&id))
            {
                meshes.push(id);
            }
        }
        let mut images = Vec::new();
        for (id, image) in self.main.resource::<Assets<Image>>().iter() {
            if image.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)
                && future.is_none_or(|f| !f.images.contains(&id))
            {
                images.push(id);
            }
        }
        let mut materials = Vec::new();
        for id in self.main.resource::<Assets<StandardMaterial>>().ids() {
            let id = id.untyped();
            if future.is_none_or(|f| !f.materials.contains(&id)) {
                materials.push(id);
            }
        }
        let expected = self.render.resource::<ExpectedAssets>();
        assert_eq!(expected.scene, scene);
        assert_eq!(expected.meshes, meshes);
        assert_eq!(expected.images, images);
        assert_eq!(expected.materials, materials);
        let key = expected.key.unwrap();
        assert_eq!(key.scene, scene);
        assert_eq!(
            key.meshes,
            self.main
                .get_resource_change_ticks::<Assets<Mesh>>()
                .unwrap()
                .changed
        );
        assert_eq!(
            key.images,
            self.main
                .get_resource_change_ticks::<Assets<Image>>()
                .unwrap()
                .changed
        );
        assert_eq!(
            key.materials,
            self.main
                .get_resource_change_ticks::<Assets<StandardMaterial>>()
                .unwrap()
                .changed
        );
        assert_eq!(
            key.future,
            self.main
                .get_resource_change_ticks::<FutureAssets>()
                .map(|ticks| ticks.changed)
        );
        if previous_key.is_some() {
            assert_eq!(previous_key != expected.key, rebuilt);
            assert_eq!(
                previous_tick
                    != self
                        .render
                        .get_resource_change_ticks::<ExpectedAssets>()
                        .unwrap()
                        .changed,
                rebuilt,
                "a cache hit must not mutably touch the required ID lists"
            );
        } else {
            assert!(
                rebuilt,
                "the first extraction must gather even empty assets"
            );
        }
    }

    fn add_assets(&mut self) -> (Handle<Mesh>, Handle<Image>, Handle<StandardMaterial>) {
        (
            self.main
                .resource_mut::<Assets<Mesh>>()
                .add(Cuboid::default()),
            self.main
                .resource_mut::<Assets<Image>>()
                .add(Image::default()),
            self.main
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial::default()),
        )
    }
}

#[test]
fn extraction_cache_tracks_actual_main_resource_ticks_and_asset_usage() {
    let mut extraction = Extraction::new();
    extraction.check(true);
    // A render-world tick ahead of the main world must neither force a gather
    // nor hide a later change to a main-world asset resource.
    for _ in 0..100 {
        extraction.render.increment_change_tick();
    }
    extraction.check(false);
    let (mesh, image, material) = extraction.add_assets();
    extraction.check(true);
    extraction.check(false);
    extraction
        .main
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&mesh)
        .unwrap()
        .asset_usage = RenderAssetUsages::MAIN_WORLD;
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<Image>>()
        .get_mut(&image)
        .unwrap()
        .asset_usage = RenderAssetUsages::MAIN_WORLD;
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&material)
        .unwrap()
        .perceptual_roughness = 0.25;
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&mesh)
        .unwrap()
        .asset_usage = RenderAssetUsages::all();
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<Image>>()
        .get_mut(&image)
        .unwrap()
        .asset_usage = RenderAssetUsages::all();
    extraction.check(true);
    extraction.main.resource_mut::<Assets<Mesh>>().remove(&mesh);
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<Image>>()
        .remove(&image);
    extraction.check(true);
    extraction
        .main
        .resource_mut::<Assets<StandardMaterial>>()
        .remove(&material);
    extraction.check(true);
    extraction.check(false);
}

#[test]
fn extraction_cache_rebuilds_after_resource_replacement() {
    let mut extraction = Extraction::new();
    let _handles = extraction.add_assets();
    extraction.check(true);
    extraction.main.insert_resource(Assets::<Mesh>::default());
    extraction.check(true);
    extraction.main.insert_resource(Assets::<Image>::default());
    extraction.check(true);
    extraction
        .main
        .insert_resource(Assets::<StandardMaterial>::default());
    extraction.check(true);
    extraction.check(false);
}

#[test]
fn extraction_cache_tracks_future_presence_promotion_and_scene_identity() {
    let mut extraction = Extraction::new();
    let _current = extraction.add_assets();
    let (mesh, image, material) = extraction.add_assets();
    extraction.check(true);
    extraction.main.insert_resource(FutureAssets {
        meshes: [mesh.id()].into_iter().collect(),
        images: [image.id()].into_iter().collect(),
        materials: [material.id().untyped()].into_iter().collect(),
        ..default()
    });
    extraction.check(true);
    extraction.check(false);
    extraction
        .main
        .resource_mut::<FutureAssets>()
        .images
        .clear();
    extraction.check(true);
    extraction
        .main
        .resource_mut::<FutureAssets>()
        .materials
        .clear();
    extraction.check(true);
    extraction
        .main
        .resource_mut::<FutureAssets>()
        .meshes
        .clear();
    extraction.check(true);
    extraction.main.resource_mut::<FutureAssets>().clear();
    extraction.main.despawn(extraction.scene);
    extraction.scene = extraction.main.spawn(SceneAabbNode).id();
    extraction.check(true);
    extraction.main.insert_resource(FutureAssets {
        meshes: [mesh.id()].into_iter().collect(),
        ..default()
    });
    extraction.check(true);
    extraction.main.remove_resource::<FutureAssets>();
    extraction.check(true);
    extraction.main.despawn(extraction.scene);
    extraction.check(true);
    extraction.check(false);
    extraction.scene = extraction.main.spawn(SceneAabbNode).id();
    extraction.check(true);
    extraction.check(false);
}
