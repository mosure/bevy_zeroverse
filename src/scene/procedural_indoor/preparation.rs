//! Prepare immutable render assets away from the ECS schedule. Handles are reserved
//! from the live stores, so cancellation and scene replacement retain Bevy ownership.
mod group_plan;
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
            workers::pool().scope(|scope| {
                for (handle, geometry) in jobs {
                    scope.spawn(async move { (handle, geometry.into_mesh()) });
                }
            })
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
}
struct Group {
    name: String,
    transform: Transform,
    bounds: (Vec3, Vec3),
    object: Option<(usize, super::layout::ObjectKind)>,
    human: Option<(usize, Vec<Vec3>)>,
    annotation: Option<SemanticLabel>,
    parts: Vec<Part>,
}
/// Wall durations for CPU preparation stages (seconds). Material synthesis and
/// geometry-only transport preparation overlap on the bounded native pool.
/// Materials include their bounded worker-pool join. GPU baking, model loading,
/// deferred ECS commands and render uploads are measured by the caller's total.
#[derive(Resource, Debug, Default, Clone, serde::Serialize)]
pub struct PreparationTimings {
    pub layout_seconds: f64,
    pub cameras_seconds: f64,
    pub materials_seconds: f64,
    /// Union wall time until maps and geometry-only transport finish. The
    /// independent mesh tail and final join belong to the complete union below.
    /// Individual stage durations must not be summed as exclusive CPU work.
    pub material_transport_seconds: f64,
    /// Complete union wall time for map, geometry-only transport and mesh
    /// preparation. Native mesh realization follows transport while maps can
    /// still be running; this includes planning, the mesh tail and scoped join.
    /// Browser stages remain serial and record the sum of those stage windows.
    pub material_transport_mesh_seconds: f64,
    pub gi_setup_seconds: f64,
    /// Components of gi_setup_seconds, for CPU bottleneck attribution.
    pub transport_seconds: f64,
    pub environment_seconds: f64,
    pub gpu_gi_setup_seconds: f64,
    pub geometry_seconds: f64,
    pub meshes_seconds: f64,
    pub asset_insertion_seconds: f64,
}

/// One construction of each assembly, shared by diffuse transport and rendering.
/// Ownership passes into render meshes after the baker has copied its triangles.
pub(crate) struct SceneGeometry {
    pub architecture: objects::Assembly,
    pub objects: Vec<objects::Assembly>,
    pub humans: Vec<humans::HumanAssembly>,
}
impl SceneGeometry {
    /// Geometry, rather than probabilistic object kinds, decides which texture
    /// programs are needed. Finish parent handles remain available even when a
    /// nonzero structure replaces every map and no part uses the base slot.
    pub(crate) fn material_selection(&self, manifest: &IndoorManifest) -> MaterialSelection {
        let keys = self
            .architecture
            .parts
            .iter()
            .chain(self.objects.iter().flat_map(|assembly| &assembly.parts))
            .filter(|(_, geometry)| !geometry.indices.is_empty())
            .map(|(key, _)| key);
        let mut selection = MaterialSelection {
            surfaces: std::collections::BTreeSet::new(),
            finishes: std::collections::BTreeSet::new(),
            direct_surfaces: std::collections::BTreeSet::new(),
        };
        for (surface, label) in keys {
            selection.surfaces.insert(*surface);
            if let Some(slot) = label
                .rsplit_once("#finish")
                .and_then(|(_, slot)| slot.parse::<usize>().ok())
            {
                if materials::variants::supports(*surface) && slot < materials::variants::COUNT {
                    selection.finishes.insert((*surface, slot));
                } else {
                    selection.direct_surfaces.insert(*surface);
                }
            } else {
                selection.direct_surfaces.insert(*surface);
            }
        }
        if !manifest.humans.is_empty() {
            // Human material selection also uses neutral cloth, shoe leather
            // and sole rubber templates, separate from assembly surface keys.
            selection
                .surfaces
                .extend([Surface::Fabric, Surface::Leather, Surface::Rubber]);
            selection
                .direct_surfaces
                .extend([Surface::Fabric, Surface::Leather, Surface::Rubber]);
        }
        for person in &manifest.humans {
            for surface in [humans::HumanSurface::Top, humans::HumanSurface::Trousers] {
                if surface == humans::HumanSurface::Top
                    && person.outfit.knitted()
                    && !person.outfit.open_front()
                {
                    continue;
                }
                let finish = humans::cloth_finish(person, surface);
                selection.surfaces.insert(finish.0);
                selection.finishes.insert(finish);
            }
        }
        selection
    }

    async fn build(scene: &IndoorManifest) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            enum Part {
                Architecture(objects::Assembly),
                Object(objects::Assembly),
                Human(humans::HumanAssembly),
            }
            // All jobs are spawned directly by the scope: Bevy preserves this
            // order independently of completion order. Each builder owns its RNG.
            let parts = workers::pool().scope(|scope| {
                scope.spawn(async move { Part::Architecture(architecture::architecture(scene)) });
                for object in &scene.objects {
                    scope.spawn(async move { Part::Object(objects::build_object(object)) });
                }
                for person in &scene.humans {
                    scope.spawn(async move { Part::Human(humans::build_human(person)) });
                }
            });
            let mut result = Self {
                architecture: default(),
                objects: Vec::with_capacity(scene.objects.len()),
                humans: Vec::with_capacity(scene.humans.len()),
            };
            for part in parts {
                match part {
                    Part::Architecture(a) => result.architecture = a,
                    Part::Object(a) => result.objects.push(a),
                    Part::Human(a) => result.humans.push(a),
                }
            }
            result
        }
        #[cfg(target_arch = "wasm32")]
        {
            let architecture = architecture::architecture(scene);
            let mut objects = Vec::with_capacity(scene.objects.len());
            for object in &scene.objects {
                cooperate().await;
                objects.push(objects::build_object(object));
            }
            let mut humans = Vec::with_capacity(scene.humans.len());
            for person in &scene.humans {
                cooperate().await;
                humans.push(humans::build_human(person));
            }
            Self {
                architecture,
                objects,
                humans,
            }
        }
    }
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
        mut images: StagedAssets<Image>,
        mut materials: StagedAssets<StandardMaterial>,
        mut meshes: StagedAssets<Mesh>,
    ) -> Self {
        let started = bevy::platform::time::Instant::now();
        let geometry = SceneGeometry::build(&manifest).await;
        let mut geometry_seconds = started.elapsed().as_secs_f64();
        let selection = geometry.material_selection(&manifest);
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
                    // Transport finishes its shared borrow before geometry moves
                    // into render meshes. No scene clone or extra executor is
                    // needed, and staged handles remain unpublished until join.
                    scope.spawn(async move {
                        let started = bevy::platform::time::Instant::now();
                        let transport = gi::GeometryTransport::build(
                            manifest,
                            moving_humans,
                            &geometry,
                            quality.diffuse_gi() && settings.enabled && settings.gpu,
                        );
                        let transport_seconds = started.elapsed().as_secs_f64();
                        let transport_completed = parallel_started.elapsed().as_secs_f64();
                        let started = bevy::platform::time::Instant::now();
                        let (groups, jobs) = group_plan::plan(manifest, geometry, meshes).await;
                        let planning_seconds = started.elapsed().as_secs_f64();
                        let started = bevy::platform::time::Instant::now();
                        meshes.realize_geometry(jobs).await;
                        Stage::TransportMeshes {
                            transport,
                            groups,
                            transport_seconds,
                            transport_completed,
                            planning_seconds,
                            meshes_seconds: started.elapsed().as_secs_f64(),
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
        let transport = transport_geometry.bind(
            &manifest,
            &material_set,
            &materials,
            &images,
            &moving_humans,
        );
        let binding_seconds = transport_started.elapsed().as_secs_f64();
        transport_seconds += binding_seconds;
        let environment_started = bevy::platform::time::Instant::now();
        material_set.build_environment(&manifest, &transport, &mut images);
        let environment_seconds = environment_started.elapsed().as_secs_f64();
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
        geometry_seconds += started.elapsed().as_secs_f64();
        #[cfg(target_arch = "wasm32")]
        let meshes_seconds = {
            let started = bevy::platform::time::Instant::now();
            meshes.realize_geometry(mesh_jobs).await;
            started.elapsed().as_secs_f64()
        };
        #[cfg(target_arch = "wasm32")]
        let material_transport_mesh_seconds =
            material_transport_seconds + planning_seconds + meshes_seconds;
        Self {
            timings: PreparationTimings {
                materials_seconds,
                material_transport_seconds,
                material_transport_mesh_seconds,
                gi_setup_seconds,
                transport_seconds,
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
        for group in self.groups.drain(..) {
            let mut root = commands.spawn((
                Name::new(group.name),
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
            if let Some((id, local_joints)) = group.human {
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
                    humans::IndoorHumanInstance { id, local_joints },
                    SemanticLabel::Person,
                    ObbTracked,
                    ObbClass("person".into()),
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            let root = root.id();
            for part in group.parts {
                let mut child = commands.spawn((
                    Mesh3d(part.mesh),
                    MeshMaterial3d(part.material),
                    part.label,
                    OvoxelTracked,
                    ChildOf(root),
                ));
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

/// A browser task must return to the event loop, not merely yield to the same
/// microtask queue. Native jobs already run on the compute pool.
pub(crate) async fn cooperate() {
    #[cfg(target_arch = "wasm32")]
    {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        use wasm_bindgen::{closure::Closure, JsCast};
        let ready = Arc::new(AtomicBool::new(false));
        let mut scheduled = false;
        futures_lite::future::poll_fn(move |cx| {
            if ready.load(Ordering::Acquire) {
                return std::task::Poll::Ready(());
            }
            if !scheduled {
                scheduled = true;
                let ready = ready.clone();
                let wake = cx.waker().clone();
                let callback = Closure::once_into_js(move || {
                    ready.store(true, Ordering::Release);
                    wake.wake();
                });
                web_sys::window()
                    .expect("browser window")
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        callback.unchecked_ref(),
                        0,
                    )
                    .expect("schedule browser preparation");
            }
            std::task::Poll::Pending
        })
        .await;
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
            let parallel = bevy::tasks::block_on(SceneGeometry::build(&scene));
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
