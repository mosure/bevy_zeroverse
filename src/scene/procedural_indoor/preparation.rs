//! Prepare immutable render assets away from the ECS schedule. Handles are reserved
//! from the live stores, so cancellation and scene replacement retain Bevy ownership.
use super::{
    architecture, gi, humans, layout::IndoorManifest, materials::IndoorMaterials, objects,
    IndoorQuality,
};
use crate::{
    annotation::obb::{ObbClass, ObbTracked},
    ovoxel::OvoxelTracked,
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
struct Part {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    label: SemanticLabel,
    surface: Option<super::materials::Surface>,
    human_surface: Option<humans::HumanSurface>,
}
struct Group {
    name: String,
    transform: Transform,
    bounds: (Vec3, Vec3),
    object: Option<(usize, super::layout::ObjectKind)>,
    human: Option<(usize, Vec<Vec3>)>,
    parts: Vec<Part>,
}
pub struct PreparedIndoor {
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
        mut images: StagedAssets<Image>,
        mut materials: StagedAssets<StandardMaterial>,
        mut meshes: StagedAssets<Mesh>,
    ) -> Self {
        let mut material_set =
            IndoorMaterials::build_async(&manifest, quality, &mut images, &mut materials).await;
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
        #[cfg(not(target_arch = "wasm32"))]
        if quality.diffuse_gi() && settings.enabled {
            let transport =
                gi::BakeScene::from_manifest(&manifest, &material_set, &materials, &images);
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
        let mut groups = Vec::new();
        let mut add = |assembly: objects::Assembly, name: String, transform: Transform, object| {
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|((surface, label), geometry)| Part {
                    mesh: meshes.add(geometry.into_mesh()),
                    material: material_set.for_part(surface, &label),
                    label: SemanticLabel::from_label(objects::part_label(&label))
                        .expect("indoor semantic vocabulary"),
                    surface: Some(surface),
                    human_surface: None,
                })
                .collect();
            groups.push(Group {
                name,
                transform,
                bounds,
                object,
                human: None,
                parts,
            });
        };
        add(
            architecture::architecture(&manifest),
            "architecture".into(),
            Transform::IDENTITY,
            None,
        );
        for object in &manifest.objects {
            cooperate().await;
            add(
                objects::build_object(object),
                format!("{:?}_{}", object.kind, object.id),
                object.transform(),
                Some((object.id, object.kind)),
            );
        }
        for person in &manifest.humans {
            cooperate().await;
            let assembly = humans::build_human(person);
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|(surface, geometry)| {
                    let material =
                        humans::person_material(person, surface, &material_set, &materials);
                    Part {
                        mesh: meshes.add(geometry.into_mesh()),
                        material: materials.add(material),
                        label: SemanticLabel::Person,
                        surface: None,
                        human_surface: Some(surface),
                    }
                })
                .collect();
            groups.push(Group {
                name: format!("person_{}", person.id),
                transform: person.transform(),
                bounds,
                object: None,
                human: Some((person.id, assembly.local_joints)),
                parts,
            });
        }
        Self {
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
            if let Some((id, kind)) = group.object {
                root.insert((
                    objects::IndoorInstance { id, kind },
                    ObbTracked,
                    ObbClass(kind.class_name().into()),
                    Aabb::from_min_max(group.bounds.0, group.bounds.1),
                ));
            }
            if let Some((id, local_joints)) = group.human {
                root.insert((
                    humans::IndoorHumanInstance { id, local_joints },
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
                if let Some(surface) = part.surface {
                    child.insert((
                        Name::new(format!("{surface:?}")),
                        objects::IndoorSurface(surface),
                    ));
                    if matches!(
                        surface,
                        super::materials::Surface::Glass
                            | super::materials::Surface::GlassInterior
                            | super::materials::Surface::Light
                    ) {
                        child.insert(NotShadowCaster);
                    }
                }
                if let Some(surface) = part.human_surface {
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
