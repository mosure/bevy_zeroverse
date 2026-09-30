//! Prepare immutable render assets away from the ECS schedule. Handles are reserved
//! from the live stores, so cancellation and scene replacement retain Bevy ownership.
use super::{
    architecture, gi, humans, layout::IndoorManifest, materials::IndoorMaterials, objects,
    IndoorQuality,
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
}
impl<A: Asset> StagedAssets<A> {
    pub fn new(assets: &Assets<A>) -> Self {
        Self {
            provider: assets.get_handle_provider(),
            entries: HashMap::new(),
        }
    }
    pub fn commit(self, assets: &mut Assets<A>) {
        for (id, (_handle, asset)) in self.entries {
            assets
                .insert(id, asset)
                .expect("staged asset uses the live allocator");
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
            static POOL: std::sync::OnceLock<bevy::tasks::TaskPool> = std::sync::OnceLock::new();
            let pool = POOL.get_or_init(|| {
                bevy::tasks::TaskPoolBuilder::new()
                    .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
                    .thread_name("indoor-mesh".into())
                    .build()
            });
            pool.scope(|scope| {
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
/// Wall durations for non-overlapping CPU preparation stages (seconds).
/// Materials include their bounded worker-pool join. GPU baking, model loading,
/// deferred ECS commands and render uploads are measured by the caller's total.
#[derive(Resource, Debug, Default, Clone, serde::Serialize)]
pub struct PreparationTimings {
    pub layout_seconds: f64,
    pub cameras_seconds: f64,
    pub materials_seconds: f64,
    pub gi_setup_seconds: f64,
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
    async fn build(scene: &IndoorManifest) -> Self {
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
        let finishes = geometry
            .architecture
            .parts
            .keys()
            .chain(
                geometry
                    .objects
                    .iter()
                    .flat_map(|assembly| assembly.parts.keys()),
            )
            .filter_map(|(surface, label)| {
                label
                    .rsplit_once("#finish")
                    .and_then(|(_, slot)| slot.parse::<usize>().ok())
                    .map(|slot| (*surface, slot))
            })
            .collect();
        let started = bevy::platform::time::Instant::now();
        let mut material_set = IndoorMaterials::build_async_with_finishes(
            &manifest,
            quality,
            &mut images,
            &mut materials,
            Some(&finishes),
        )
        .await;
        let materials_seconds = started.elapsed().as_secs_f64();
        let started = bevy::platform::time::Instant::now();
        material_set.environment.rotation = Quat::from_rotation_y(manifest.world_yaw);
        #[cfg(not(target_arch = "wasm32"))]
        let mut probes = None;
        #[cfg(target_arch = "wasm32")]
        let probes = None;
        #[cfg(target_arch = "wasm32")]
        let _ = (settings, moving_humans);
        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_request = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut cpu_bake = None;
        #[cfg(not(target_arch = "wasm32"))]
        if quality.diffuse_gi() && settings.enabled {
            // A static irradiance volume must not retain a moving person's old
            // occlusion. These candidates still cast live direct shadows; a
            // rejected candidate also remains excluded until the next bake.
            let transport = gi::BakeScene::from_geometry(
                &manifest,
                &material_set,
                &materials,
                &images,
                &moving_humans,
                &geometry,
            );
            #[cfg(not(target_arch = "wasm32"))]
            if settings.gpu {
                let (request, transform, statistics) =
                    gi::gpu::prepare(&transport, settings.bake, manifest.seed, &mut images);
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
        let gi_setup_seconds = started.elapsed().as_secs_f64();
        let started = bevy::platform::time::Instant::now();
        let mut groups = Vec::new();
        let mut mesh_jobs = Vec::new();
        let mut add = |assembly: objects::Assembly,
                       name: String,
                       transform: Transform,
                       object,
                       annotation| {
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|((surface, label), geometry)| Part {
                    mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                    material: material_set.for_part(surface, &label),
                    label: SemanticLabel::from_label(objects::part_label(&label))
                        .expect("indoor semantic vocabulary"),
                    surface: Some(surface),
                    human_surface: None,
                    ovoxel_excluded: label.ends_with("#exterior"),
                })
                .collect();
            groups.push(Group {
                name,
                transform,
                bounds,
                object,
                human: None,
                annotation,
                parts,
            });
        };
        let mut shell = objects::Assembly::default();
        let mut fixtures = std::collections::BTreeMap::<String, objects::Assembly>::new();
        for ((surface, label), geometry) in geometry.architecture.parts {
            if label.starts_with("lamp#") {
                fixtures
                    .entry(label.clone())
                    .or_default()
                    .parts
                    .insert((surface, label), geometry);
            } else {
                shell.parts.insert((surface, label), geometry);
            }
        }
        for (name, fixture) in fixtures {
            add(
                fixture,
                format!("ceiling_{name}"),
                Transform::IDENTITY,
                None,
                Some(SemanticLabel::Lamp),
            );
        }
        add(
            shell,
            "architecture".into(),
            Transform::IDENTITY,
            None,
            None,
        );
        for (object, assembly) in manifest.objects.iter().zip(geometry.objects) {
            cooperate().await;
            add(
                assembly,
                format!("{:?}_{}", object.kind, object.id),
                object.transform(),
                Some((object.id, object.kind)),
                None,
            );
        }
        for (person, assembly) in manifest.humans.iter().zip(geometry.humans) {
            cooperate().await;
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|(surface, geometry)| {
                    let material =
                        humans::person_material(person, surface, &material_set, &materials);
                    Part {
                        mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                        material: materials.add(material),
                        label: SemanticLabel::Person,
                        surface: None,
                        human_surface: Some(surface),
                        ovoxel_excluded: false,
                    }
                })
                .collect();
            groups.push(Group {
                name: format!("person_{}", person.id),
                transform: person.transform(),
                bounds,
                object: None,
                human: Some((person.id, assembly.local_joints)),
                annotation: None,
                parts,
            });
        }
        geometry_seconds += started.elapsed().as_secs_f64();
        let started = bevy::platform::time::Instant::now();
        meshes.realize_geometry(mesh_jobs).await;
        let meshes_seconds = started.elapsed().as_secs_f64();
        Self {
            timings: PreparationTimings {
                materials_seconds,
                gi_setup_seconds,
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
    #[derive(Asset, bevy::reflect::TypePath)]
    struct Value(u32);
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
