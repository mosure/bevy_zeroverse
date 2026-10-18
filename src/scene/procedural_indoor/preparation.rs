//! Prepare immutable render assets away from the ECS schedule. Handles are reserved
//! from the live stores, so cancellation and scene replacement retain Bevy ownership.
mod cooperative;
pub(crate) use cooperative::cooperate;
mod geometry;
mod group_plan;
mod timings;
pub(crate) use geometry::SceneGeometry;
pub use pipeline::prepare_indoor;
pub use timings::{PreparationTimings, PreparationWall, TransportTimings};
pub(crate) mod pipeline;
pub(crate) mod residency;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod wait;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod workers;
use super::{
    architecture, gi, humans,
    layout::IndoorManifest,
    materials::{self, IndoorMaterials, MaterialSelection, Surface},
    objects, IndoorQuality,
};
use crate::{
    annotation::obb::{ObbClass, ObbTracked},
    ovoxel::{OvoxelExcluded, OvoxelTracked},
    render::semantic::SemanticLabel,
};
use bevy::{
    asset::{Asset, AssetHandleProvider, AssetId},
    camera::primitives::Aabb,
    light::NotShadowCaster,
    prelude::*,
};
pub use pipeline::IndoorPrefetch;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
pub use workers::configure_preparation_threads;

pub trait AssetStore<A: Asset> {
    fn add(&mut self, asset: A) -> Handle<A>;
    fn get(&self, handle: &Handle<A>) -> Option<&A>;
}
impl<A: Asset> AssetStore<A> for Assets<A> {
    fn add(&mut self, asset: A) -> Handle<A> {
        Assets::add(self, asset)
    }
    fn get(&self, handle: &Handle<A>) -> Option<&A> {
        Assets::get(self, handle)
    }
}
pub struct StagedAssets<A: Asset> {
    provider: AssetHandleProvider,
    entries: HashMap<AssetId<A>, (Handle<A>, A)>,
    // Keep speculative assets alive until their scene is promoted or canceled.
    resident: Vec<Handle<A>>,
}
impl<A: Asset> StagedAssets<A> {
    pub fn new(assets: &Assets<A>) -> Self {
        Self {
            provider: assets.get_handle_provider(),
            entries: HashMap::new(),
            resident: Vec::new(),
        }
    }
    pub fn commit(mut self, assets: &mut Assets<A>) {
        self.upload(assets);
    }
    fn upload(&mut self, assets: &mut Assets<A>) {
        for (id, (handle, asset)) in self.entries.drain() {
            assets
                .insert(id, asset)
                .expect("staged asset uses the live allocator");
            self.resident.push(handle);
        }
    }
}
impl<A: Asset> AssetStore<A> for StagedAssets<A> {
    fn add(&mut self, asset: A) -> Handle<A> {
        let handle = self.provider.reserve_handle().typed::<A>();
        self.entries.insert(handle.id(), (handle.clone(), asset));
        handle
    }
    fn get(&self, handle: &Handle<A>) -> Option<&A> {
        self.entries.get(&handle.id()).map(|(_, asset)| asset)
    }
}
impl StagedAssets<Mesh> {
    fn defer_geometry(
        &self,
        geometry: super::geometry::Geometry,
        jobs: &mut Vec<(Handle<Mesh>, super::geometry::Geometry)>,
    ) -> Handle<Mesh> {
        let handle = self.provider.reserve_handle().typed::<Mesh>();
        jobs.push((handle.clone(), geometry));
        handle
    }

    async fn realize_geometry(&mut self, jobs: Vec<(Handle<Mesh>, super::geometry::Geometry)>) {
        // MikkTSpace tangent construction is substantial for furnished rooms.
        // Reserve handles in scene order, then convert independent parts on a
        // bounded native pool. No mesh/material quality or batching is changed.
        #[cfg(not(target_arch = "wasm32"))]
        let ready = {
            // Start expensive parts first to avoid a tail consisting of one
            // large body/ceiling mesh. Handles were reserved in manifest order.
            let mut jobs: Vec<_> = jobs.into_iter().enumerate().collect();
            jobs.sort_by_key(|(_, (_, geometry))| std::cmp::Reverse(geometry.indices.len()));
            let mut tasks: Vec<_> = jobs
                .into_iter()
                .map(|(order, (handle, geometry))| {
                    (
                        order,
                        workers::pool().spawn(async move { (handle, geometry.into_mesh()) }),
                    )
                })
                .collect();
            // Preserve asset insertion as well as reserved-handle/group order.
            tasks.sort_by_key(|(order, _)| *order);
            let mut ready = Vec::with_capacity(tasks.len());
            // These jobs own their geometry. Awaiting them yields the worker
            // instead of nesting a blocking scope that can steal unrelated work.
            // Dropping preparation also drops/cancels every outstanding task.
            for (_, task) in tasks {
                ready.push(task.await);
            }
            ready
        };
        #[cfg(target_arch = "wasm32")]
        let ready = {
            let mut ready = Vec::with_capacity(jobs.len());
            for (handle, geometry) in jobs {
                cooperate().await;
                ready.push((handle, geometry.into_mesh()));
            }
            ready
        };
        for (handle, mesh) in ready {
            self.entries.insert(handle.id(), (handle, mesh));
        }
    }
}

struct Part {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    label: SemanticLabel,
    surface: Option<super::materials::Surface>,
    human_surface: Option<humans::HumanSurface>,
    ovoxel_excluded: bool,
    motion: Option<super::motion::AnimatedPart>,
    bounds: Option<(Vec3, Vec3)>,
}
struct Group {
    name: String,
    transform: Transform,
    bounds: (Vec3, Vec3),
    object: Option<(usize, super::layout::ObjectKind)>,
    human: Option<(usize, Vec<Vec3>, Quat)>,
    annotation: Option<SemanticLabel>,
    parts: Vec<Part>,
}

pub struct PreparedIndoor {
    pub timings: PreparationTimings,
    pub manifest: IndoorManifest,
    pub material_set: IndoorMaterials,
    pub images: StagedAssets<Image>,
    pub materials: StagedAssets<StandardMaterial>,
    pub meshes: StagedAssets<Mesh>,
    groups: Vec<Group>,
    pub probes: Option<(Handle<Image>, Transform, gi::BakeStatistics)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub gpu_request: Option<gi::gpu::GpuBakeRequest>,
    #[cfg(not(target_arch = "wasm32"))]
    pub cpu_bake: Option<bevy::tasks::Task<gi::ProbeData>>,
}
impl PreparedIndoor {
    pub async fn build(
        manifest: IndoorManifest,
        quality: IndoorQuality,
        settings: gi::IndoorGiSettings,
        moving_humans: Vec<usize>,
        images: StagedAssets<Image>,
        materials: StagedAssets<StandardMaterial>,
        meshes: StagedAssets<Mesh>,
    ) -> Self {
        Self::build_with_materials(
            manifest,
            quality,
            settings,
            moving_humans,
            images,
            materials,
            meshes,
            false,
        )
        .await
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn build_with_materials(
        mut manifest: IndoorManifest,
        quality: IndoorQuality,
        settings: gi::IndoorGiSettings,
        moving_humans: Vec<usize>,
        mut images: StagedAssets<Image>,
        mut materials: StagedAssets<StandardMaterial>,
        mut meshes: StagedAssets<Mesh>,
        gpu_materials: bool,
    ) -> Self {
        let build_started = bevy::platform::time::Instant::now();
        let started = bevy::platform::time::Instant::now();
        let (mut geometry, geometry_jobs) = SceneGeometry::build(&manifest).await;
        let mut geometry_seconds = started.elapsed().as_secs_f64();
        let motion_seconds = if manifest.indoor_motion.is_some() {
            let started = bevy::platform::time::Instant::now();
            super::motion::prepare(&mut manifest, &mut geometry);
            started.elapsed().as_secs_f64()
        } else {
            0.
        };
        let mut wall = PreparationWall {
            assembly_seconds: geometry_seconds,
            motion_planning_seconds: motion_seconds,
            ..default()
        };
        let started = bevy::platform::time::Instant::now();
        let selection = geometry.material_selection(&manifest);
        wall.material_selection_seconds = started.elapsed().as_secs_f64();
        let parallel_started = bevy::platform::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        let (
            mut material_set,
            transport_geometry,
            materials_seconds,
            mut transport_seconds,
            group_plans,
            meshes_seconds,
            planning_seconds,
            material_transport_seconds,
        ) = {
            enum Stage {
                TransportMeshes {
                    transport: gi::GeometryTransport,
                    groups: Vec<group_plan::GroupPlan>,
                    transport_seconds: f64,
                    transport_completed: f64,
                    planning_seconds: f64,
                    meshes_seconds: f64,
                },
                Materials(IndoorMaterials, f64, f64),
            }
            let manifest = &manifest;
            let moving_humans = &moving_humans;
            let meshes = &mut meshes;
            let parallel_started = &parallel_started;
            let mut ready = workers::pool()
                .scope(|scope| {
                    // Only ordered triangle emission borrows the assemblies.
                    // Tree building and mesh/tangent realization then own disjoint
                    // data and share the bounded pool. Nothing is published until
                    // both have joined; neither geometry nor scenes are cloned.
                    scope.spawn(async move {
                        let started = bevy::platform::time::Instant::now();
                        let input = gi::GeometryTransport::emit(manifest, moving_humans, &geometry);
                        let emission_seconds = started.elapsed().as_secs_f64();
                        enum Independent {
                            Trees(Box<gi::GeometryTransport>, f64, f64),
                            Meshes(Vec<group_plan::GroupPlan>, f64, f64),
                        }
                        let mut independent = workers::pool()
                            .scope(|scope| {
                                scope.spawn(async move {
                                    let started = bevy::platform::time::Instant::now();
                                    let transport = input.finish(
                                        quality.diffuse_gi() && settings.enabled && settings.gpu,
                                    );
                                    Independent::Trees(
                                        Box::new(transport),
                                        emission_seconds + started.elapsed().as_secs_f64(),
                                        parallel_started.elapsed().as_secs_f64(),
                                    )
                                });
                                scope.spawn(async move {
                                    let started = bevy::platform::time::Instant::now();
                                    let (groups, jobs) =
                                        group_plan::plan(manifest, geometry, meshes).await;
                                    let planning_seconds = started.elapsed().as_secs_f64();
                                    let started = bevy::platform::time::Instant::now();
                                    meshes.realize_geometry(jobs).await;
                                    Independent::Meshes(
                                        groups,
                                        planning_seconds,
                                        started.elapsed().as_secs_f64(),
                                    )
                                });
                            })
                            .into_iter();
                        let Some(Independent::Trees(
                            transport,
                            transport_seconds,
                            transport_completed,
                        )) = independent.next()
                        else {
                            unreachable!("scoped tree work retains submission order");
                        };
                        let Some(Independent::Meshes(groups, planning_seconds, meshes_seconds)) =
                            independent.next()
                        else {
                            unreachable!("scoped mesh work retains submission order");
                        };
                        Stage::TransportMeshes {
                            transport: *transport,
                            groups,
                            transport_seconds,
                            transport_completed,
                            planning_seconds,
                            meshes_seconds,
                        }
                    });
                    scope.spawn(async {
                        let started = bevy::platform::time::Instant::now();
                        let materials = IndoorMaterials::build_async_with_selection(
                            manifest,
                            quality,
                            &mut images,
                            &mut materials,
                            Some(&selection),
                            gpu_materials,
                        )
                        .await;
                        Stage::Materials(
                            materials,
                            started.elapsed().as_secs_f64(),
                            parallel_started.elapsed().as_secs_f64(),
                        )
                    });
                })
                .into_iter();
            let Some(Stage::TransportMeshes {
                transport,
                groups,
                transport_seconds,
                transport_completed,
                planning_seconds,
                meshes_seconds,
            }) = ready.next()
            else {
                unreachable!("scoped transport retains submission order");
            };
            let Some(Stage::Materials(materials, materials_seconds, materials_completed)) =
                ready.next()
            else {
                unreachable!("scoped materials retain submission order");
            };
            (
                materials,
                transport,
                materials_seconds,
                transport_seconds,
                groups,
                meshes_seconds,
                planning_seconds,
                materials_completed.max(transport_completed),
            )
        };
        #[cfg(target_arch = "wasm32")]
        let (mut material_set, transport_geometry, materials_seconds, mut transport_seconds) = {
            let started = bevy::platform::time::Instant::now();
            let material_set = IndoorMaterials::build_async_with_selection(
                &manifest,
                quality,
                &mut images,
                &mut materials,
                Some(&selection),
                gpu_materials,
            )
            .await;
            let materials_seconds = started.elapsed().as_secs_f64();
            let started = bevy::platform::time::Instant::now();
            let transport =
                gi::GeometryTransport::build(&manifest, &moving_humans, &geometry, false);
            (
                material_set,
                transport,
                materials_seconds,
                started.elapsed().as_secs_f64(),
            )
        };
        #[cfg(not(target_arch = "wasm32"))]
        let material_transport_mesh_seconds = parallel_started.elapsed().as_secs_f64();
        #[cfg(not(target_arch = "wasm32"))]
        {
            geometry_seconds += planning_seconds;
        }
        #[cfg(target_arch = "wasm32")]
        let material_transport_seconds = parallel_started.elapsed().as_secs_f64();
        wall.parallel_assets_seconds = parallel_started.elapsed().as_secs_f64();
        material_set.environment.rotation = Quat::from_rotation_y(manifest.world_yaw);
        #[cfg(not(target_arch = "wasm32"))]
        let mut probes = None;
        #[cfg(target_arch = "wasm32")]
        let probes = None;
        #[cfg(target_arch = "wasm32")]
        let _ = settings;
        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_request = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut cpu_bake = None;
        // Ordered binding is a small linear remap after both scoped jobs finish.
        // Reuse the same immutable trees for HDR reflections and native GI.
        let started = bevy::platform::time::Instant::now();
        let transport_started = bevy::platform::time::Instant::now();
        let transport_work = transport_geometry.timings.clone();
        let transport = transport_geometry.bind(
            &manifest,
            &material_set,
            &materials,
            &images,
            &moving_humans,
        );
        let binding_seconds = transport_started.elapsed().as_secs_f64();
        transport_seconds += binding_seconds;
        wall.transport_binding_seconds = binding_seconds;
        let environment_started = bevy::platform::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        material_set.build_environment(&manifest, &transport, &mut images);
        #[cfg(target_arch = "wasm32")]
        material_set
            .build_environment_async(&manifest, &transport, &mut images)
            .await;
        let environment_seconds = environment_started.elapsed().as_secs_f64();
        wall.environment_seconds = environment_seconds;
        let gi_started = bevy::platform::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_gi_setup_seconds = 0.0;
        #[cfg(target_arch = "wasm32")]
        let gpu_gi_setup_seconds = 0.0;
        #[cfg(not(target_arch = "wasm32"))]
        if quality.diffuse_gi() && settings.enabled {
            #[cfg(not(target_arch = "wasm32"))]
            if settings.gpu {
                let gpu_started = bevy::platform::time::Instant::now();
                let (request, transform, statistics) =
                    gi::gpu::prepare(&transport, settings.bake, manifest.seed, &mut images);
                gpu_gi_setup_seconds = gpu_started.elapsed().as_secs_f64();
                probes = Some((request.image.clone(), transform, statistics));
                gpu_request = Some(request);
            }
            #[cfg(not(target_arch = "wasm32"))]
            if !settings.gpu {
                cpu_bake = Some(super::lighting::prepare(
                    transport,
                    settings.bake,
                    manifest.seed,
                ));
            }
        }
        wall.gi_request_seconds = gi_started.elapsed().as_secs_f64();
        let gi_setup_seconds =
            started.elapsed().as_secs_f64() + transport_seconds - binding_seconds;
        #[cfg(target_arch = "wasm32")]
        let (group_plans, mesh_jobs, planning_seconds) = {
            let started = bevy::platform::time::Instant::now();
            let planned = group_plan::plan(&manifest, geometry, &meshes).await;
            let planning_seconds = started.elapsed().as_secs_f64();
            geometry_seconds += planning_seconds;
            (planned.0, planned.1, planning_seconds)
        };
        // Material identities resolve only after map synthesis; preserve the
        // original fixture/shell/object/human ordering and human material calls.
        let started = bevy::platform::time::Instant::now();
        let groups = group_plans
            .into_iter()
            .map(|plan| plan.bind(&manifest, &material_set, &mut materials))
            .collect();
        wall.group_binding_seconds = started.elapsed().as_secs_f64();
        geometry_seconds += wall.group_binding_seconds;
        #[cfg(target_arch = "wasm32")]
        let meshes_seconds = {
            let started = bevy::platform::time::Instant::now();
            meshes.realize_geometry(mesh_jobs).await;
            started.elapsed().as_secs_f64()
        };
        #[cfg(target_arch = "wasm32")]
        let material_transport_mesh_seconds =
            material_transport_seconds + planning_seconds + meshes_seconds;
        #[cfg(target_arch = "wasm32")]
        {
            wall.serial_meshes_seconds = planning_seconds + meshes_seconds;
        }
        wall.finish(build_started.elapsed().as_secs_f64());
        Self {
            timings: PreparationTimings {
                wall,
                geometry_jobs,
                materials_seconds,
                material_transport_seconds,
                material_transport_mesh_seconds,
                gi_setup_seconds,
                transport_seconds,
                transport_work,
                environment_seconds,
                gpu_gi_setup_seconds,
                geometry_seconds,
                meshes_seconds,
                ..Default::default()
            },
            manifest,
            material_set,
            images,
            materials,
            meshes,
            groups,
            probes,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_request,
            #[cfg(not(target_arch = "wasm32"))]
            cpu_bake,
        }
    }
    pub fn spawn_groups(&mut self, parent: Entity, commands: &mut Commands) {
        self.material_set.spawn_reflections(parent, commands);
        let mut annotation_order = 0u64;
        for group in self.groups.drain(..) {
            let mut root = commands.spawn((
                Name::new(group.name),
                crate::render::semantic::PreserveSemanticParts,
                group.transform,
                Visibility::default(),
                ChildOf(parent),
            ));
            if let Some(label) = group.annotation {
                root.insert((
                    ObbTracked,
                    ObbClass(label.as_str().into()),
                    label,
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            if let Some((id, kind)) = group.object {
                let object = &self.manifest.objects[id];
                if object.neighbor || !self.manifest.in_primary_room(object.position, 0.0) {
                    root.insert(OvoxelExcluded);
                }
                root.insert((
                    objects::IndoorInstance { id, kind },
                    SemanticLabel::from_label(kind.class_name()).expect("indoor object class"),
                    ObbTracked,
                    ObbClass(kind.class_name().into()),
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            if let Some((id, local_joints, local_head_rotation)) = group.human {
                let person = self
                    .manifest
                    .humans
                    .iter()
                    .find(|person| person.id == id)
                    .expect("prepared human instance belongs to manifest");
                if person.neighbor || !self.manifest.in_primary_room(person.position, 0.0) {
                    root.insert(OvoxelExcluded);
                }
                root.insert((
                    humans::IndoorHumanInstance {
                        id,
                        local_joints,
                        local_head_rotation,
                    },
                    SemanticLabel::Person,
                    ObbTracked,
                    ObbClass("person".into()),
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            if group.parts.iter().any(|p| p.motion.is_some()) {
                root.insert((
                    super::motion::AnimatedBounds,
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            let root = root.id();
            for part in group.parts {
                let mut child = commands.spawn((
                    crate::render::ground_truth::AnnotationOrder(annotation_order),
                    Mesh3d(part.mesh),
                    MeshMaterial3d(part.material),
                    part.label,
                    OvoxelTracked,
                    ChildOf(root),
                ));
                annotation_order += 1;
                if let Some(bounds) = part.bounds {
                    child.insert(Aabb::from_min_max(bounds.0, bounds.1));
                }
                if let Some(motion) = part.motion {
                    child.insert(motion);
                }
                if part.ovoxel_excluded {
                    child.insert(OvoxelExcluded);
                }
                if let Some(surface) = part.surface {
                    child.insert((
                        Name::new(format!("{surface:?}")),
                        objects::IndoorSurface(surface),
                    ));
                    if matches!(
                        surface,
                        super::materials::Surface::Glass
                            | super::materials::Surface::GlassInterior
                            | super::materials::Surface::ContainerGlass
                            | super::materials::Surface::Liquid
                            | super::materials::Surface::Light
                    ) {
                        child.insert(NotShadowCaster);
                    }
                }
                if let Some(surface) = part.human_surface {
                    if surface == humans::HumanSurface::Lens {
                        child.insert(NotShadowCaster);
                    }
                    child.insert((
                        Name::new(format!("person/{surface:?}")),
                        humans::IndoorHumanSurface(surface),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn concurrent_builders_preserve_ordered_geometry_and_surface_assignments() {
        fn geometry(a: &super::super::geometry::Geometry, b: &super::super::geometry::Geometry) {
            assert_eq!(a.positions, b.positions);
            assert_eq!(a.normals, b.normals);
            assert_eq!(a.uvs, b.uvs);
            assert_eq!(a.indices, b.indices);
        }
        fn assembly(a: &objects::Assembly, b: &objects::Assembly) {
            assert_eq!(a.parts.len(), b.parts.len());
            for (key, part) in &a.parts {
                geometry(part, &b.parts[key]);
            }
        }
        for seed in [0, 7, 207] {
            let scene =
                IndoorManifest::generate_with_humans(seed, default(), 0.65, 0, 0.5).unwrap();
            let (parallel, jobs) = bevy::tasks::block_on(SceneGeometry::build(&scene));
            assert_eq!(jobs.architecture.count, 1);
            assert_eq!(jobs.objects.count, scene.objects.len());
            assert_eq!(jobs.humans.count, scene.humans.len());
            assembly(&parallel.architecture, &architecture::architecture(&scene));
            assert_eq!(parallel.objects.len(), scene.objects.len());
            for (object, built) in scene.objects.iter().zip(&parallel.objects) {
                assembly(built, &objects::build_object(object));
            }
            assert_eq!(parallel.humans.len(), scene.humans.len());
            for (person, built) in scene.humans.iter().zip(&parallel.humans) {
                let serial = humans::build_human(person);
                assert_eq!(built.local_joints, serial.local_joints);
                assert_eq!(built.parts.len(), serial.parts.len());
                for (key, part) in &built.parts {
                    geometry(part, &serial.parts[key]);
                }
            }
        }
    }
    #[test]
    fn asynchronous_mesh_jobs_preserve_handles_and_exact_attributes() {
        use super::super::geometry::Geometry;
        let mut live = Assets::<Mesh>::default();
        let mut staged = StagedAssets::new(&live);
        let mut jobs = Vec::new();
        let mut expected = Vec::new();
        // Deliberately submit small, large, medium parts. Dispatch may change;
        // bindings, indices and tangent frames must retain their original IDs.
        for repeats in [1, 19, 4] {
            let mut geometry = Geometry::default();
            for i in 0..repeats {
                geometry.cuboid(
                    Vec3::splat(0.2),
                    0.01,
                    Transform::from_xyz(i as f32 * 0.31, 0., 0.),
                );
            }
            let mesh = geometry.clone().into_mesh();
            let handle = staged.defer_geometry(geometry, &mut jobs);
            expected.push((handle, mesh));
        }
        bevy::tasks::block_on(staged.realize_geometry(jobs));
        for (handle, expected) in &expected {
            let actual = staged.get(handle).unwrap();
            assert_eq!(actual.indices(), expected.indices());
            for (id, values) in expected.attributes() {
                assert_eq!(
                    actual.attribute(*id).unwrap().get_bytes(),
                    values.get_bytes()
                );
            }
        }
        staged.commit(&mut live);
        assert_eq!(live.len(), expected.len());
        for (handle, _) in expected {
            assert!(live.get(&handle).is_some());
        }
    }

    #[derive(Asset, bevy::reflect::TypePath)]
    struct Value(u32);
    #[test]
    fn speculative_upload_promotes_without_replacing_or_copying_assets() {
        let mut live = Assets::<Value>::default();
        let mut staged = StagedAssets::new(&live);
        let handle = staged.add(Value(17));
        staged.upload(&mut live);
        assert!(staged.entries.is_empty());
        assert_eq!(staged.resident.len(), 1);
        assert_eq!(live.get(&handle).unwrap().0, 17);
        // A second stage/commit must not replace the uploaded object: preserve
        // even a main-world modification made after staging.
        live.get_mut(&handle).unwrap().0 = 23;
        staged.upload(&mut live);
        staged.commit(&mut live);
        assert_eq!(live.len(), 1);
        assert_eq!(live.get(&handle).unwrap().0, 23);
    }
    #[test]
    fn reserved_assets_share_allocator_and_cancel_without_insertion() {
        let mut live = Assets::<Value>::default();
        let existing = live.add(Value(3));
        let mut canceled = StagedAssets::new(&live);
        let canceled_handle = canceled.add(Value(4));
        drop(canceled);
        drop(canceled_handle);
        let mut staged = StagedAssets::new(&live);
        let prepared = staged.add(Value(9));
        assert_ne!(prepared.id(), existing.id());
        assert!(live.get(&prepared).is_none());
        assert_eq!(staged.get(&prepared).unwrap().0, 9);
        staged.commit(&mut live);
        assert_eq!(live.get(&prepared).unwrap().0, 9);
        assert_eq!(live.get(&existing).unwrap().0, 3);
        assert_eq!(live.len(), 2);
    }
}
