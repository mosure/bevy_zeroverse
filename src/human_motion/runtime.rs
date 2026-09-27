//! One lazily created inference worker/device and bounded model/embedding caches.
use super::{
    planning,
    skin::{self, MotionFrame, PreparedActor, SkinVertices},
    *,
};
use crate::{
    app::{BevyZeroverseConfig, OvoxelMode},
    camera::Playback,
    ovoxel::OvoxelTracked,
    render::semantic::SemanticLabel,
    scene::{
        procedural_indoor::{
            humans::{IndoorHumanInstance, IndoorHumanSurface},
            layout::IndoorManifest,
        },
        ZeroverseSceneRoot,
    },
};
use bevy::{
    camera::primitives::Aabb,
    mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    render::{
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
        RenderApp,
    },
};
use burn::backend::wgpu::{init_device, WgpuSetup};
use burn_ardy::Ardy;
use burn_human_inference::gpu::WgpuBackend;
use burn_human_motion::{MotionClip, TextEmbedding};
use burn_llama::TextEncoder;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

pub struct HumanMotionPlugin;
impl Plugin for HumanMotionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::sample::CaptureFailure>();
        app.init_resource::<HumanMotionReport>()
            .init_resource::<Runtime>()
            .init_resource::<HumanMotionClips>()
            .add_systems(
                Update,
                (
                    manage,
                    playback.after(crate::camera::update_camera_trajectory),
                )
                    .chain(),
            );
    }
    fn finish(&self, app: &mut App) {
        // Cloning Bevy's setup is cheap; Burn device initialization happens only
        // in the first nonempty inference job, never in plugin startup.
        let setup = app.get_sub_app(RenderApp).and_then(|render| {
            let w = render.world();
            let instance = w.get_resource::<RenderInstance>()?;
            let adapter = w.get_resource::<RenderAdapter>()?;
            let device = w.get_resource::<RenderDevice>()?;
            let queue = w.get_resource::<RenderQueue>()?;
            Some(WgpuSetup {
                instance: (****instance).clone(),
                adapter: (****adapter).clone(),
                device: device.wgpu_device().clone(),
                queue: (****queue).clone(),
                backend: adapter.get_info().backend,
            })
        });
        app.world_mut().resource_mut::<Runtime>().setup = setup;
    }
}

/// Validated source clips can be exported alongside a dataset for exact replay.
#[derive(Resource, Default)]
pub struct HumanMotionClips(pub Vec<(usize, MotionClip)>);
struct Models {
    ardy: Ardy<WgpuBackend>,
    text: TextEncoder<WgpuBackend>,
    embeddings: VecDeque<TextEmbedding>,
}
#[derive(Default)]
struct Shared {
    device: Option<burn::backend::wgpu::WgpuDevice>,
    models: Option<Models>,
    busy: bool,
    stage: String,
    result: Option<(u64, Result<Outcome, String>)>,
    loads: usize,
    batches: usize,
    hits: usize,
}
struct Outcome {
    actors: Vec<PreparedActor>,
    rejected: Vec<MotionRejection>,
}
struct Job {
    scene: IndoorManifest,
    config: HumanMotionConfig,
    plans: Vec<MotionPlan>,
    epoch: u64,
}
#[derive(Resource, Default)]
struct Runtime {
    setup: Option<WgpuSetup>,
    shared: Arc<Mutex<Shared>>,
    epoch: Arc<AtomicU64>,
    key: Option<(Entity, u64, String)>,
    pending: Option<Job>,
    own_failure: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>>,
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }
}

fn fail(world: &mut World, error: String) {
    world.resource_mut::<Runtime>().own_failure = Some(error.clone());
    world.resource_mut::<crate::sample::CaptureFailure>().0 = Some(error);
}

fn manage(world: &mut World) {
    let scene = world.get_resource::<IndoorManifest>();
    let seed = scene.map(|s| s.seed);
    let applied = world
        .query_filtered::<(Entity, &SceneMotionPolicy), With<ZeroverseSceneRoot>>()
        .iter(world)
        .next()
        .and_then(|(entity, policy)| policy.0.clone().map(|json| (entity, json)));
    let key = applied
        .zip(seed)
        .map(|((root, json), seed)| (root, seed, json));
    let changed = world.resource::<Runtime>().key != key;
    if changed {
        let mut runtime = world.resource_mut::<Runtime>();
        runtime.key = key.clone();
        runtime.pending = None;
        let epoch = runtime.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let own_failure = runtime.own_failure.take();
        if own_failure.is_some()
            && world.resource::<crate::sample::CaptureFailure>().0 == own_failure
        {
            world.resource_mut::<crate::sample::CaptureFailure>().0 = None;
        }
        world.resource_mut::<HumanMotionClips>().0.clear();
        world.insert_resource(HumanMotionReport::default());
        if let Some((_, _, json)) = &key {
            let result = HumanMotionConfig::parse(json).and_then(|config| {
                let scene = world.resource::<IndoorManifest>().clone();
                let (plans, rejected) = planning::plan(&scene, &config)?;
                Ok((
                    Job {
                        scene,
                        config,
                        plans,
                        epoch,
                    },
                    rejected,
                ))
            });
            match result {
                Ok((job, rejected)) => {
                    let failed = job.config.strict && !rejected.is_empty();
                    let pending = !job.plans.is_empty() && !failed;
                    world.insert_resource(HumanMotionReport {
                        scene_seed: job.scene.seed,
                        pending,
                        stage: if failed {
                            "Rejected"
                        } else if pending {
                            "Queued"
                        } else {
                            "Ready"
                        }
                        .into(),
                        requested: job.plans.clone(),
                        rejected,
                        ..Default::default()
                    });
                    if failed {
                        fail(
                            world,
                            "strict motion policy could not plan every selected actor".into(),
                        );
                    }
                    if pending {
                        world.resource_mut::<Runtime>().pending = Some(job);
                    }
                }
                Err(error) => fail(world, error),
            }
        }
    }
    let shared = world.resource::<Runtime>().shared.clone();
    {
        let state = shared.lock().unwrap();
        let mut report = world.resource_mut::<HumanMotionReport>();
        if report.pending && report.stage != state.stage {
            report.stage.clone_from(&state.stage);
        }
        report.model_loads = state.loads;
        report.generated_batches = state.batches;
        report.embedding_hits = state.hits;
        if state.loads > 0 && report.model_artifacts.is_empty() {
            for artifact in [
                burn_ardy::pretrained::DEFAULT,
                burn_llama::pretrained::DEFAULT,
            ] {
                report
                    .model_artifacts
                    .insert(artifact.bundle.into(), artifact.sha256.into());
            }
        }
    }
    let result = shared.lock().unwrap().result.take();
    if let Some((epoch, result)) = result {
        if epoch == world.resource::<Runtime>().epoch.load(Ordering::Relaxed) && key.is_some() {
            match result {
                Ok(outcome) => {
                    for actor in outcome.actors {
                        world
                            .resource_mut::<HumanMotionReport>()
                            .accepted
                            .push(actor.plan.clone());
                        world
                            .resource_mut::<HumanMotionClips>()
                            .0
                            .push((actor.plan.actor_id, actor.clip.clone()));
                        install_actor(world, actor);
                    }
                    world
                        .resource_mut::<HumanMotionReport>()
                        .rejected
                        .extend(outcome.rejected);
                }
                Err(error) => {
                    error!("human motion: {error}");
                    fail(world, error);
                }
            }
            let state = shared.lock().unwrap();
            let mut report = world.resource_mut::<HumanMotionReport>();
            report.pending = false;
            report.stage = "Ready".into();
            report.model_loads = state.loads;
            report.generated_batches = state.batches;
            report.embedding_hits = state.hits;
            info!(
                "human motion: {} accepted, {} retained static",
                report.accepted.len(),
                report.rejected.len()
            );
        }
    }
    if !shared.lock().unwrap().busy {
        let mut runtime = world.resource_mut::<Runtime>();
        if let Some(job) = runtime.pending.take() {
            if let Some(setup) = runtime.setup.clone() {
                shared.lock().unwrap().busy = true;
                let epoch = runtime.epoch.clone();
                let shared_job = shared.clone();
                #[cfg(not(target_arch = "wasm32"))]
                let stamp = job.epoch;
                let task = move || async move {
                    let mut models = shared_job.lock().unwrap().models.take();
                    let result = run_job(&job, setup, &mut models, &shared_job, &epoch).await;
                    let mut state = shared_job.lock().unwrap();
                    state.models = models;
                    state.busy = false;
                    state.result = Some((job.epoch, result));
                };
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let worker = runtime.worker.get_or_insert_with(|| {
                        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
                        std::thread::Builder::new()
                            .name("zeroverse-motion".into())
                            .spawn(move || {
                                for job in rx {
                                    job();
                                }
                            })
                            .expect("motion worker");
                        tx
                    });
                    let failed = shared.clone();
                    worker
                        .send(Box::new(move || {
                            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                pollster::block_on(task())
                            }))
                            .is_err()
                            {
                                let mut state = failed.lock().unwrap();
                                state.busy = false;
                                state.result = Some((
                                    stamp,
                                    Err("motion GPU worker panicked; inspect device diagnostics"
                                        .into()),
                                ));
                            }
                        }))
                        .expect("motion worker alive");
                }
                #[cfg(target_arch = "wasm32")]
                wasm_bindgen_futures::spawn_local(task());
            } else {
                world.resource_mut::<HumanMotionReport>().pending = false;
                fail(
                    world,
                    "human motion requires a Bevy WGPU/WebGPU render device".into(),
                );
            }
        }
    }
}

async fn run_job(
    job: &Job,
    setup: WgpuSetup,
    models: &mut Option<Models>,
    state: &Arc<Mutex<Shared>>,
    epoch: &AtomicU64,
) -> Result<Outcome, String> {
    if models.is_none() {
        info!("loading cached ARDY and Llama motion models on Bevy's device");
        // Retain the registered Burn handle even if downloading a model fails.
        // Retrying a policy must not grow the backend's global device registry.
        let existing_device = state.lock().unwrap().device.clone();
        let device = existing_device.unwrap_or_else(|| {
            let device = init_device(setup, Default::default());
            state.lock().unwrap().device = Some(device.clone());
            device
        });
        let root = job
            .config
            .model_root
            .as_deref()
            .unwrap_or(burn_ardy::pretrained::CDN_ROOT);
        let ardy = Ardy::load_pretrained_from(root, &device, |i, n| {
            state.lock().unwrap().stage = format!("Loading motion model {i}/{n}");
        })
        .await
        .map_err(|e| format!("ARDY load: {e}"))?;
        let text = TextEncoder::load_pretrained_from(root, &device, |i, n| {
            state.lock().unwrap().stage = format!("Loading text model {i}/{n}");
        })
        .await
        .map_err(|e| format!("text encoder load: {e}"))?;
        *models = Some(Models {
            ardy,
            text,
            embeddings: VecDeque::new(),
        });
        state.lock().unwrap().loads += 1;
    }
    let models = models.as_mut().unwrap();
    let mut outcome = Outcome {
        actors: Vec::new(),
        rejected: Vec::new(),
    };
    let mut remaining = job.plans.clone();
    for attempt in 0..job.config.max_attempts {
        let round = std::mem::take(&mut remaining);
        if round.is_empty() {
            break;
        }
        for plans in round.chunks(job.config.batch_size) {
            if epoch.load(Ordering::Relaxed) != job.epoch {
                return Err("superseded motion generation".into());
            }
            let mut embeddings = Vec::new();
            for plan in plans {
                // Only text features are reusable by prompt. Always generate
                // the clip below with this scene/actor's request seed.
                let embedding = if let Some(i) = models
                    .embeddings
                    .iter()
                    .position(|e| e.prompt == plan.request.prompt)
                {
                    state.lock().unwrap().hits += 1;
                    models.embeddings.remove(i).unwrap()
                } else {
                    models
                        .text
                        .encode(&plan.request.prompt, |i, n| {
                            state.lock().unwrap().stage = format!("Encoding motion prompt {i}/{n}");
                        })
                        .await
                        .map_err(|e| format!("prompt encoding: {e}"))?
                };
                embeddings.push(embedding.clone());
                models.embeddings.push_back(embedding);
                while models.embeddings.len() > 128 {
                    models.embeddings.pop_front();
                }
            }
            let (requests, transforms): (Vec<_>, Vec<_>) = plans
                .iter()
                .map(|p| {
                    let person = job
                        .scene
                        .humans
                        .iter()
                        .find(|h| h.id == p.actor_id)
                        .unwrap();
                    planning::canonical_request(p, person.yaw + std::f32::consts::PI)
                })
                .unzip();
            let clips = models
                .ardy
                .generate_batch(&requests, &embeddings, |_| {
                    state.lock().unwrap().stage = "Generating motion batch".into();
                    epoch.load(Ordering::Relaxed) == job.epoch
                })
                .await
                .map_err(|e| e.to_string())?;
            state.lock().unwrap().batches += 1;
            for ((plan, clip), transform) in plans.iter().zip(clips).zip(transforms) {
                state.lock().unwrap().stage =
                    format!("Checking motion for person {}", plan.actor_id);
                let clip = planning::scene_clip(clip, transform);
                if epoch.load(Ordering::Relaxed) != job.epoch {
                    return Err("superseded motion generation".into());
                }
                let person = job
                    .scene
                    .humans
                    .iter()
                    .find(|h| h.id == plan.actor_id)
                    .unwrap();
                match skin::prepare(&job.scene, person, plan.clone(), clip).and_then(|actor| {
                    if outcome.actors.iter().any(|other| {
                        other
                            .frames
                            .iter()
                            .zip(&actor.frames)
                            .any(|(a, b)| super::validation::bounds_overlap(a.bounds, b.bounds))
                    }) {
                        Err("moving people have overlapping swept bounds".into())
                    } else {
                        Ok(actor)
                    }
                }) {
                    Ok(actor) => outcome.actors.push(actor),
                    Err(reason) => {
                        if attempt + 1 < job.config.max_attempts {
                            let mut retry = plan.clone();
                            retry.request.seed =
                                retry.request.seed.wrapping_add(0x9e3779b97f4a7c15);
                            remaining.push(retry);
                            continue;
                        }
                        if job.config.strict {
                            return Err(format!("actor {} rejected: {reason}", plan.actor_id));
                        }
                        outcome.rejected.push(MotionRejection {
                            actor_id: plan.actor_id,
                            reason,
                        });
                    }
                }
                burn_human_inference::cooperative::yield_to_browser().await;
            }
        }
    }
    Ok(outcome)
}

#[derive(Component)]
struct Actor {
    frames: Vec<MotionFrame>,
    rig: Arc<skin::MotionRig>,
    inverse_bind: Vec<Mat4>,
    bones: Vec<Entity>,
    parts: Vec<Entity>,
    last: Option<(u32, bool)>,
}
#[derive(Component)]
struct MotionPart {
    skin: SkinVertices,
    skinned: SkinnedMesh,
}

fn install_actor(world: &mut World, actor: PreparedActor) {
    let Some(root) = world
        .query::<(Entity, &IndoorHumanInstance)>()
        .iter(world)
        .find(|(_, h)| h.id == actor.plan.actor_id)
        .map(|(e, _)| e)
    else {
        return;
    };
    let children: Vec<_> = world
        .get::<Children>(root)
        .map(|c| c.iter().collect())
        .unwrap_or_default();
    let mut materials = std::collections::BTreeMap::new();
    for &child in &children {
        if let (Some(surface), Some(material)) = (
            world.get::<IndoorHumanSurface>(child),
            world.get::<MeshMaterial3d<StandardMaterial>>(child),
        ) {
            materials.insert(surface.0, material.0.clone());
        }
    }
    for child in children {
        world.despawn(child);
    }
    let inverse_bind = world
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .add(SkinnedMeshInverseBindposes::from(
            actor.inverse_bind.clone(),
        ));
    let bones: Vec<_> = actor.frames[0]
        .bones
        .iter()
        .map(|&transform| world.spawn((transform, ChildOf(root))).id())
        .collect();
    let mut parts = Vec::new();
    for (surface, (geometry, mut skin)) in actor.parts {
        let Some(material) = materials.get(&surface) else {
            continue;
        };
        let mut mesh = geometry.into_mesh();
        if let Some(bevy::mesh::VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        {
            skin.tangents.clone_from(tangents);
        }
        skin.attributes(&mut mesh);
        let handle = world.resource_mut::<Assets<Mesh>>().add(mesh);
        let skinned = SkinnedMesh {
            inverse_bindposes: inverse_bind.clone(),
            joints: bones.clone(),
        };
        let mut part = world.spawn((
            Name::new(format!("person/{surface:?}")),
            Mesh3d(handle),
            MeshMaterial3d(material.clone()),
            SemanticLabel::Person,
            IndoorHumanSurface(surface),
            OvoxelTracked,
            ChildOf(root),
            MotionPart { skin, skinned },
            bevy::camera::visibility::NoFrustumCulling,
        ));
        if surface == crate::scene::procedural_indoor::humans::HumanSurface::Lens {
            part.insert(bevy::light::NotShadowCaster);
        }
        parts.push(part.id());
    }
    world.entity_mut(root).insert((
        Transform::IDENTITY,
        Actor {
            frames: actor.frames,
            rig: actor.rig,
            inverse_bind: actor.inverse_bind,
            bones,
            parts,
            last: None,
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn playback(
    mut commands: Commands,
    time: Res<Playback>,
    config: Res<BevyZeroverseConfig>,
    mut actors: Query<(&mut Actor, &mut IndoorHumanInstance, &mut Aabb)>,
    mut parts: Query<(&MotionPart, &mut Mesh3d)>,
    mut transforms: Query<&mut Transform>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let t = time.mode.map_progress(time.progress).clamp(0.0, 1.0);
    let baked = config.image_copiers || config.ovoxel_mode != OvoxelMode::Disabled;
    for (mut actor, mut human, mut aabb) in &mut actors {
        if actor.last == Some((t.to_bits(), baked)) {
            continue;
        }
        let f = t * (actor.frames.len() - 1) as f32;
        let i = f.floor() as usize;
        let frame = actor.frames[i].interpolate(
            &actor.frames[(i + 1).min(actor.frames.len() - 1)],
            f.fract(),
            &actor.rig,
        );
        let mode_changed = actor.last.is_none_or(|(_, old)| old != baked);
        let matrices: Vec<_> = frame
            .bones
            .iter()
            .zip(&actor.inverse_bind)
            .filter(|_| baked)
            .map(|(b, i)| b.to_matrix() * *i)
            .collect();
        for (&entity, transform) in actor.bones.iter().zip(&frame.bones) {
            if let Ok(mut t) = transforms.get_mut(entity) {
                *t = *transform;
            }
        }
        for &entity in &actor.parts {
            let Ok((part, mut mesh)) = parts.get_mut(entity) else {
                continue;
            };
            if baked || mode_changed {
                if let Some(mut asset) = meshes.get_mut(&mesh.0) {
                    let vertices = if baked {
                        part.skin.deform(&matrices)
                    } else {
                        skin::DeformedVertices {
                            positions: part.skin.positions.iter().map(|p| p.to_array()).collect(),
                            normals: part.skin.normals.iter().map(|p| p.to_array()).collect(),
                            tangents: part.skin.tangents.clone(),
                        }
                    };
                    if baked {
                        asset.remove_attribute(Mesh::ATTRIBUTE_JOINT_INDEX);
                        asset.remove_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT);
                    } else {
                        part.skin.attributes(&mut asset);
                    }
                    asset.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices.positions);
                    asset.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertices.normals);
                    if !vertices.tangents.is_empty() {
                        asset.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vertices.tangents);
                    }
                    mesh.set_changed(); // invalidate CPU O-voxel and ground-truth caches.
                }
                if baked {
                    commands.entity(entity).remove::<SkinnedMesh>();
                } else {
                    commands.entity(entity).insert(part.skinned.clone());
                }
            }
        }
        human.local_joints = frame.joints;
        *aabb = Aabb::from_min_max(
            frame.bounds.0 - Vec3::splat(0.04),
            frame.bounds.1 + Vec3::splat(0.04),
        );
        actor.last = Some((t.to_bits(), baked));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ZeroverseSceneType;

    #[test]
    fn regeneration_queues_fresh_noise_for_an_unchanged_prompt_and_person() {
        use crate::scene::procedural_indoor::layout::IndoorLayout;
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.35, 0, 0.7).unwrap();
        let mut config = HumanMotionConfig {
            fraction: 1.0,
            locomotion_fraction: 1.0,
            ..default()
        };
        planning::prepare_scene(&mut scene, &config).unwrap();
        let (plans, _) = planning::plan(&scene, &config).unwrap();
        let plan = plans.into_iter().next().unwrap();
        config.fraction = 0.0;
        config.trajectories = vec![HumanTrajectory {
            actor_id: plan.actor_id,
            prompt: plan.request.prompt,
            support_chair: plan.support_chair,
            waypoints: plan.request.waypoints,
        }];
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<Assets<Mesh>>()
            .insert_resource(scene)
            .add_plugins(HumanMotionPlugin);
        app.world_mut().spawn((
            ZeroverseSceneRoot,
            SceneMotionPolicy(Some(serde_json::to_string(&config).unwrap())),
        ));
        app.finish();
        app.cleanup();
        // Leave jobs queued, exercising the real scene-change path without
        // needing a renderer, worker, model load or GPU in the unit suite.
        app.world()
            .resource::<Runtime>()
            .shared
            .lock()
            .unwrap()
            .busy = true;
        let mut seeds = Vec::new();
        for scene_seed in [0, 1, 0] {
            app.world_mut().resource_mut::<IndoorManifest>().seed = scene_seed;
            app.update();
            let runtime = app.world().resource::<Runtime>();
            let job = runtime.pending.as_ref().unwrap();
            assert_eq!(job.scene.seed, scene_seed);
            assert_eq!(job.plans[0].request.prompt, config.trajectories[0].prompt);
            seeds.push(job.plans[0].request.seed);
            assert_eq!(app.world().resource::<HumanMotionReport>().model_loads, 0);
            #[cfg(not(target_arch = "wasm32"))]
            assert!(runtime.worker.is_none());
        }
        assert_ne!(seeds[0], seeds[1]);
        assert_eq!(seeds[0], seeds[2]);
    }

    #[test]
    fn absent_policy_never_initializes_device_models_or_worker() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<Assets<Mesh>>()
            .add_plugins(HumanMotionPlugin);
        app.finish();
        app.cleanup();
        for _ in 0..5 {
            app.update();
        }
        let runtime = app.world().resource::<Runtime>();
        assert!(runtime.setup.is_none());
        assert!(runtime.shared.lock().unwrap().device.is_none());
        assert!(runtime.shared.lock().unwrap().models.is_none());
        assert!(!runtime.shared.lock().unwrap().busy);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(runtime.worker.is_none());
        assert_eq!(app.world().resource::<HumanMotionReport>().model_loads, 0);
    }

    #[test]
    fn empty_policy_in_a_real_manifest_does_not_start_inference() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<Assets<Mesh>>()
            .add_plugins(HumanMotionPlugin);
        {
            let mut config = app.world_mut().resource_mut::<BevyZeroverseConfig>();
            config.scene_type = ZeroverseSceneType::ProceduralIndoor;
            config.human_motion = Some(r#"{"fraction":0}"#.into());
        }
        app.insert_resource(
            IndoorManifest::generate_with_humans(
                0,
                crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
                0.4,
                0,
                0.0,
            )
            .unwrap(),
        );
        app.world_mut().spawn((
            ZeroverseSceneRoot,
            SceneMotionPolicy(Some(r#"{"fraction":0}"#.into())),
        ));
        app.finish();
        app.cleanup();
        app.update();
        let runtime = app.world().resource::<Runtime>();
        assert!(runtime.shared.lock().unwrap().models.is_none());
        assert!(!runtime.shared.lock().unwrap().busy);
        let epoch = runtime.epoch.load(Ordering::Relaxed);
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .human_motion = Some(r#"{"fraction":1}"#.into());
        app.update();
        assert_eq!(
            app.world()
                .resource::<Runtime>()
                .epoch
                .load(Ordering::Relaxed),
            epoch,
            "editing an unapplied policy must not restart motion generation"
        );
        assert!(!app.world().resource::<HumanMotionReport>().pending);
        assert!(app
            .world()
            .resource::<crate::sample::CaptureFailure>()
            .0
            .is_none());
    }

    #[test]
    fn sampling_is_order_independent_and_marks_geometry_and_pose_dirty() {
        let mut app = App::new();
        app.init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<Assets<Mesh>>()
            .add_systems(Update, playback);
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .image_copiers = true;
        app.world_mut().resource_mut::<Playback>().mode = crate::camera::PlaybackMode::Still;
        let bone = app.world_mut().spawn(Transform::IDENTITY).id();
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::all(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 1.0, 0.0]]);
        let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
        let part = app
            .world_mut()
            .spawn((
                Mesh3d(handle.clone()),
                MotionPart {
                    skin: SkinVertices {
                        positions: vec![Vec3::Y],
                        normals: vec![Vec3::Y],
                        tangents: vec![[1.0, 0.0, 0.0, 1.0]],
                        indices: vec![[0; 4]],
                        weights: vec![[1.0, 0.0, 0.0, 0.0]],
                    },
                    skinned: SkinnedMesh {
                        inverse_bindposes: Handle::default(),
                        joints: vec![bone],
                    },
                },
            ))
            .id();
        let frame = |x| MotionFrame {
            bones: vec![Transform::from_xyz(x, 0.0, 0.0)],
            joints: vec![Vec3::new(x, 1.0, 0.0); 21],
            bounds: (Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 2.0, 1.0)),
        };
        let root = app
            .world_mut()
            .spawn((
                Actor {
                    frames: vec![frame(0.0), frame(2.0)],
                    rig: Arc::new(skin::MotionRig {
                        parents: vec![None],
                        annotation_indices: [0; 21],
                        stature: 1.75,
                        local_bounds: Vec::new(),
                    }),
                    inverse_bind: vec![Mat4::IDENTITY],
                    bones: vec![bone],
                    parts: vec![part],
                    last: None,
                },
                IndoorHumanInstance {
                    id: 7,
                    local_joints: Vec::new(),
                },
                Aabb::default(),
            ))
            .id();
        for t in [0.0, 1.0, 0.25, 0.75, 0.25] {
            app.world_mut().resource_mut::<Playback>().progress = t;
            app.update();
            let mesh = app.world().resource::<Assets<Mesh>>().get(&handle).unwrap();
            let bevy::mesh::VertexAttributeValues::Float32x3(p) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
            else {
                panic!()
            };
            assert_eq!(p[0], [2.0 * t, 1.0, 0.0]);
            assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_none());
            assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT).is_none());
            assert_eq!(
                app.world()
                    .get::<IndoorHumanInstance>(root)
                    .unwrap()
                    .local_joints[0],
                Vec3::new(2.0 * t, 0.0, 0.0)
            );
            assert!(app.world().get::<SkinnedMesh>(part).is_none());
        }
        // Switching render paths must remove/recreate both the joint attributes
        // and the SkinnedMesh component, including O-voxel without readback.
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .image_copiers = false;
        app.update();
        assert!(app.world().get::<SkinnedMesh>(part).is_some());
        let mesh = app.world().resource::<Assets<Mesh>>().get(&handle).unwrap();
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT).is_some());
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .ovoxel_mode = OvoxelMode::CpuAsync;
        app.update();
        assert!(app.world().get::<SkinnedMesh>(part).is_none());
        let mesh = app.world().resource::<Assets<Mesh>>().get(&handle).unwrap();
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_none());
        let bevy::mesh::VertexAttributeValues::Float32x3(p) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!()
        };
        assert_eq!(p[0], [0.5, 1.0, 0.0]);
    }

    #[test]
    fn regeneration_discards_stale_completion_without_loading_models() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<Assets<Mesh>>()
            .add_plugins(HumanMotionPlugin);
        {
            let mut config = app.world_mut().resource_mut::<BevyZeroverseConfig>();
            config.scene_type = ZeroverseSceneType::ProceduralIndoor;
            config.human_motion = Some(r#"{"fraction":0}"#.into());
        }
        let scene = IndoorManifest::generate_with_humans(
            0,
            crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
            0.35,
            0,
            0.7,
        )
        .unwrap();
        app.insert_resource(scene);
        let root = app
            .world_mut()
            .spawn((
                ZeroverseSceneRoot,
                SceneMotionPolicy(Some(r#"{"fraction":0}"#.into())),
            ))
            .id();
        app.finish();
        app.cleanup();
        app.update();
        let old_epoch = app
            .world()
            .resource::<Runtime>()
            .epoch
            .load(Ordering::Relaxed);
        app.world()
            .resource::<Runtime>()
            .shared
            .lock()
            .unwrap()
            .result = Some((old_epoch, Err("stale failure must be discarded".into())));
        app.world_mut().despawn(root);
        let root = app
            .world_mut()
            .spawn((
                ZeroverseSceneRoot,
                SceneMotionPolicy(Some(r#"{"fraction":0}"#.into())),
            ))
            .id();
        app.update();
        assert!(
            app.world()
                .resource::<Runtime>()
                .epoch
                .load(Ordering::Relaxed)
                > old_epoch
        );
        assert!(app
            .world()
            .resource::<crate::sample::CaptureFailure>()
            .0
            .is_none());
        assert_eq!(app.world().resource::<HumanMotionReport>().model_loads, 0);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(app.world().resource::<Runtime>().worker.is_none());
        app.world_mut()
            .get_mut::<SceneMotionPolicy>(root)
            .unwrap()
            .0 = Some("invalid".into());
        app.update();
        assert!(app
            .world()
            .resource::<crate::sample::CaptureFailure>()
            .0
            .is_some());
        app.world_mut()
            .get_mut::<SceneMotionPolicy>(root)
            .unwrap()
            .0 = Some(r#"{"fraction":0}"#.into());
        app.update();
        assert!(app
            .world()
            .resource::<crate::sample::CaptureFailure>()
            .0
            .is_none());
        app.world_mut()
            .resource_mut::<crate::sample::CaptureFailure>()
            .0 = Some("unrelated render failure".into());
        app.world_mut()
            .get_mut::<SceneMotionPolicy>(root)
            .unwrap()
            .0 = None;
        app.update();
        assert_eq!(
            app.world()
                .resource::<crate::sample::CaptureFailure>()
                .0
                .as_deref(),
            Some("unrelated render failure")
        );
    }
}
