use super::*;
use bevy::{light::cluster::ObjectsInClusterCpu, render::MainWorld};

struct Extraction {
    main: World,
    render: World,
    camera: Entity,
    view: Entity,
    light: Entity,
    schedule: Schedule,
}

impl Extraction {
    fn new() -> Self {
        let mut main = World::new();
        main.insert_resource(GlobalClusterSettings {
            supports_storage_buffers: true,
            clustered_decals_are_usable: true,
            gpu_clustering: None,
            max_uniform_buffer_clusterable_objects: 204,
            view_cluster_bindings_max_indices: 16384,
        });
        let mut render = World::new();
        let view = render.spawn_empty().id();
        let rendered_light = render.spawn_empty().id();
        let light = main
            .spawn((PointLight::default(), RenderEntity::from(rendered_light)))
            .id();
        let mut cluster = ObjectsInClusterCpu::default();
        cluster.add_point_light(light);
        let camera = main
            .spawn((
                Camera::default(),
                RenderEntity::from(view),
                Clusters {
                    clusterable_objects: ClusterableObjects::Cpu(vec![cluster]),
                    ..default()
                },
            ))
            .id();
        let mut schedule = Schedule::new(ExtractSchedule);
        schedule.add_systems(extract_key);
        Self {
            main,
            render,
            camera,
            view,
            light,
            schedule,
        }
    }

    fn extract(&mut self) -> Option<Vec<Record>> {
        let mut main = MainWorld::default();
        std::mem::swap(&mut self.main, &mut *main);
        self.render.insert_resource(main);
        self.schedule.run(&mut self.render);
        let mut main = self.render.remove_resource::<MainWorld>().unwrap();
        std::mem::swap(&mut self.main, &mut *main);
        self.render
            .get::<InputKey>(self.view)
            .map(|key| key.0.clone())
    }
}

#[test]
fn extraction_tracks_membership_classification_and_entity_remapping() {
    let mut h = Extraction::new();
    let initial = h.extract().unwrap();
    // Lighting uniforms update independently; the index buffer remains valid.
    h.main.get_mut::<PointLight>(h.light).unwrap().intensity *= 2.;
    assert_eq!(h.extract().unwrap(), initial);
    let new_render = h.render.spawn_empty().id();
    h.main
        .entity_mut(h.light)
        .insert(RenderEntity::from(new_render));
    let remapped = h.extract().unwrap();
    assert_ne!(remapped, initial);
    h.main
        .entity_mut(h.light)
        .remove::<PointLight>()
        .insert(SpotLight::default());
    assert_ne!(h.extract().unwrap(), remapped);
    // A stale membership must not mask Bevy's missing-object diagnostics.
    h.main.despawn(h.light);
    assert!(h.extract().is_none());
}

#[test]
fn disabled_camera_and_gpu_clustering_invalidate_retained_inputs() {
    let mut h = Extraction::new();
    assert!(h.extract().is_some());
    h.main.get_mut::<Camera>(h.camera).unwrap().is_active = false;
    assert!(h.extract().is_none());
    h.main.get_mut::<Camera>(h.camera).unwrap().is_active = true;
    assert!(h.extract().is_some());
    h.main
        .get_mut::<Clusters>(h.camera)
        .unwrap()
        .clusterable_objects = ClusterableObjects::Gpu;
    assert!(h.extract().is_none());
}

#[test]
fn probe_readiness_reindexing_and_removal_change_the_resolved_key() {
    let mut probes = MainEntityHashMap::default();
    let main = MainEntity::from(Entity::PLACEHOLDER);
    let absent = probe_indices(Some(&probes));
    probes.insert(main, 0);
    let ready = probe_indices(Some(&probes));
    assert_ne!(ready, absent);
    probes.insert(main, 1);
    assert_ne!(probe_indices(Some(&probes)), ready);
    probes.clear();
    assert_eq!(probe_indices(Some(&probes)), absent);
}
