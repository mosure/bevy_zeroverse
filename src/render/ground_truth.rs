//! Full precision, single-pass geometric ground truth on native and WebGPU.
//!
//! Two RGBA32Float attachments contain world XYZ/linear view-Z depth and encoded
//! geometric view normal/semantic ID respectively. No HDR intermediate, lighting,
//! texture filtering, blending, antialiasing, or tonemapping touches these values.
//! The 32 bytes/pixel MRT total fits the standard attachment-byte limit.
use super::{semantic::SemanticLabel, DisabledPbrMaterial};
use bevy::{
    asset::{load_internal_asset, uuid_handle, AssetEvent, AssetId, RenderAssetUsages},
    camera::visibility::RenderLayers,
    mesh::{skinning::SkinnedMesh, PrimitiveTopology, VertexAttributeValues, VertexBufferLayout},
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        view::ExtractedView,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    num::NonZeroU64,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
};

pub mod flow;
mod skin;

const SHADER: Handle<Shader> = uuid_handle!("57c3d933-c7a7-41bc-b590-8f5d40a91758");

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct GroundTruthLabel;

/// Set `frame_id` alongside the RGB capture stamp before render extraction.
/// `rendered_frame()` reports encoding of this pass, not GPU/map completion.
#[derive(Component, Clone, ExtractComponent)]
pub struct GroundTruthCamera {
    pub world_depth: Handle<Image>,
    pub normal_semantic: Handle<Image>,
    pub frame_id: u64,
    pub near: f32,
    pub far: f32,
    pub layers: RenderLayers,
    /// Optional forward-flow output. Its pixels belong to the preceding capture.
    pub flow: Option<Handle<Image>>,
    /// Change this at every sequence boundary; warm-up frames never advance history.
    pub flow_sequence: u64,
    pub co_visibility: Option<super::co_visibility::CoVisibilityOutput>,
    depth: Handle<Image>,
    status: Arc<CameraStatus>,
}

#[derive(Default)]
struct CameraStatus {
    valid: AtomicBool,
    frame: AtomicU64,
    generation: AtomicU64,
    failure: Mutex<Option<String>>,
}

impl GroundTruthCamera {
    pub fn new(images: &mut Assets<Image>, size: UVec2) -> Self {
        assert!(size.x > 0 && size.y > 0);
        let mut image = |label: &'static str, format, copy_src| {
            images.add(Image {
                data: None,
                texture_descriptor: TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT
                        | if copy_src {
                            TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING
                        } else {
                            TextureUsages::empty()
                        },
                    view_formats: &[],
                },
                asset_usage: RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                ..default()
            })
        };
        Self {
            world_depth: image(
                "ground_truth_world_depth_f32",
                TextureFormat::Rgba32Float,
                true,
            ),
            normal_semantic: image(
                "ground_truth_normal_semantic_f32",
                TextureFormat::Rgba32Float,
                true,
            ),
            depth: image(
                "ground_truth_depth_test_f32",
                TextureFormat::Depth32Float,
                false,
            ),
            frame_id: 0,
            near: 0.1,
            far: 50.0,
            layers: RenderLayers::default(),
            flow: None,
            flow_sequence: 0,
            co_visibility: None,
            status: Arc::default(),
        }
    }

    pub fn enable_flow(&mut self, images: &mut Assets<Image>) -> Handle<Image> {
        let mut target = images.get(&self.world_depth).unwrap().clone();
        target.texture_descriptor.label = Some("forward_flow_f32");
        let handle = images.add(target);
        self.flow = Some(handle.clone());
        handle
    }

    pub fn rendered_frame(&self) -> Option<u64> {
        let frame = self.geometry_frame()?;
        if self
            .co_visibility
            .as_ref()
            .is_some_and(|c| c.rendered_frame() != Some(frame))
        {
            return None;
        }
        Some(frame)
    }

    pub fn enable_co_visibility(
        &mut self,
        images: &mut Assets<Image>,
        size: UVec2,
    ) -> Handle<Image> {
        let output = super::co_visibility::CoVisibilityOutput::new(images, size);
        let image = output.image.clone();
        self.co_visibility = Some(output);
        image
    }

    pub(crate) fn geometry_frame(&self) -> Option<u64> {
        self.status
            .valid
            .load(Ordering::Acquire)
            .then(|| self.status.frame.load(Ordering::Acquire))
    }

    pub fn generation(&self) -> u64 {
        self.status.generation.load(Ordering::Acquire)
    }

    pub fn failure(&self) -> Option<String> {
        self.status.failure.lock().unwrap().clone()
    }

    pub(crate) fn fail(&self, message: String) {
        *self.status.failure.lock().unwrap() = Some(message);
    }
}

/// Current residency/work and cumulative upload counters, shared with main world.
#[derive(Default, Debug, Clone, Serialize)]
pub struct GroundTruthStats {
    pub generation: u64,
    pub mesh_instances: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub draw_batches: usize,
    pub geometry_bytes: usize,
    pub geometry_uploads: u64,
    pub instance_updates: u64,
    pub rendered_views: u64,
}

#[derive(Resource, Default, Clone)]
pub struct GroundTruthDiagnostics(Arc<Mutex<GroundTruthStats>>);

impl GroundTruthDiagnostics {
    pub fn snapshot(&self) -> GroundTruthStats {
        self.0.lock().unwrap().clone()
    }
}

pub struct GroundTruthPlugin;

impl Plugin for GroundTruthPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "ground_truth.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            flow::SHADER,
            "ground_truth/flow.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(ExtractComponentPlugin::<GroundTruthCamera>::default());
        let diagnostics = GroundTruthDiagnostics::default();
        app.insert_resource(diagnostics.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(diagnostics);
        render_app.init_resource::<ExtractedGeometry>();
        render_app.init_resource::<GeometryCache>();
        render_app.init_resource::<GpuGeometry>();
        render_app.init_resource::<flow::FlowHistory>();
        render_app.add_systems(ExtractSchedule, extract_geometry);
        render_app.add_systems(
            Render,
            (
                initialize_pipeline,
                prepare_geometry,
                prepare_cameras,
                flow::prepare,
            )
                .chain()
                .in_set(RenderSystems::PrepareResources),
        );
        render_app.init_resource::<GroundTruthPipelines>();
        render_app.init_resource::<flow::FlowPipeline>();
        render_app.add_systems(
            RenderGraph,
            render_ground_truth
                .after(bevy::core_pipeline::schedule::camera_driver)
                .in_set(GroundTruthLabel)
                .in_set(RenderGraphSystems::Render),
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    world_from_local: [[f32; 4]; 4],
    normal_from_local: [[f32; 4]; 4],
    semantic: u32,
    padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    clip_from_world: [[f32; 4]; 4],
    view_from_world: [[f32; 4]; 4],
    limits: [f32; 4],
}

#[derive(Clone)]
struct Batch {
    indices: Range<u32>,
    /// 0: double-sided, 1: back-face culling, 2: front-face culling.
    cull: usize,
    layers: RenderLayers,
}

#[derive(Resource, Default)]
struct ExtractedGeometry {
    generation: u64,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    batches: Vec<Batch>,
    instances: Vec<Instance>,
    failure: Option<String>,
    topology: Vec<ObjectTopology>,
}

#[derive(Clone)]
struct ObjectTopology {
    entity: Entity,
    mesh: AssetId<Mesh>,
    vertices: Range<usize>,
    indices_hash: u64,
}

#[derive(Resource, Default)]
struct GeometryCache {
    flow_enabled: bool,
    skin_stamp: u64,
    skinned_entities: Vec<Entity>,
    keys: Vec<(Entity, AssetId<Mesh>, usize, RenderLayers)>,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn extract_geometry(
    cameras: Extract<Query<&GroundTruthCamera>>,
    objects: Extract<
        Query<
            (
                Entity,
                &Mesh3d,
                &GlobalTransform,
                Option<&InheritedVisibility>,
                Option<&SemanticLabel>,
                Option<&RenderLayers>,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Option<&DisabledPbrMaterial>,
                Option<&SkinnedMesh>,
            ),
            Without<super::RenderOnlyOverlay>,
        >,
    >,
    meshes: Extract<Res<Assets<Mesh>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    inverse_bindposes: Extract<Res<Assets<bevy::mesh::skinning::SkinnedMeshInverseBindposes>>>,
    joints: Extract<Query<Ref<GlobalTransform>>>,
    mut mesh_events: Extract<MessageReader<AssetEvent<Mesh>>>,
    mut cache: ResMut<GeometryCache>,
    mut geometry: ResMut<ExtractedGeometry>,
) {
    let events: Vec<_> = mesh_events.read().copied().collect();
    if cameras.is_empty() {
        if !geometry.vertices.is_empty() {
            *geometry = ExtractedGeometry {
                generation: geometry.generation.wrapping_add(1),
                ..default()
            };
        }
        cache.keys.clear();
        return;
    }
    let flow_enabled = cameras.iter().any(|c| c.flow.is_some());
    let retry_failed_geometry = geometry.failure.is_some();
    geometry.failure = None;
    let mut objects: Vec<_> = objects
        .iter()
        .filter(|(_, _, _, visible, ..)| visible.is_none_or(|v| v.get()))
        .collect();
    objects.sort_by_key(|(entity, ..)| entity.to_bits());
    let keys: Vec<_> = objects
        .iter()
        .map(|(entity, mesh, _, _, _, layers, material, disabled, _)| {
            let cull = material
                .and_then(|m| materials.get(&m.0))
                .map(|m| m.cull_mode)
                .or_else(|| disabled.map(|m| m.cull_mode))
                .flatten();
            let cull = match cull {
                None => 0,
                Some(Face::Back) => 1,
                Some(Face::Front) => 2,
            };
            (
                *entity,
                mesh.id(),
                cull,
                layers.cloned().unwrap_or_default(),
            )
        })
        .collect();
    let skin_stamp = cameras.iter().map(|c| c.frame_id).max().unwrap_or(0);
    let skinned_entities: Vec<_> = objects
        .iter()
        .filter(|o| o.8.is_some())
        .map(|o| o.0)
        .collect();
    // A new capture refreshes skin data even when only inverse binds or the
    // SkinnedMesh component changed. Warm-up joint updates remain inexpensive.
    let skin_changed = skinned_entities != cache.skinned_entities
        || (!skinned_entities.is_empty() && skin_stamp != cache.skin_stamp)
        || objects.iter().any(|(_, _, _, _, _, _, _, _, skin)| {
            skin.is_some_and(|s| {
                s.joints
                    .iter()
                    .any(|e| joints.get(*e).is_ok_and(|t| t.is_changed()))
            })
        });
    cache.skin_stamp = skin_stamp;
    cache.skinned_entities = skinned_entities;
    let changed = skin_changed || retry_failed_geometry || keys != cache.keys || flow_enabled != cache.flow_enabled || events.iter().any(|event| matches!(event, AssetEvent::Modified { id } | AssetEvent::Removed { id } if keys.iter().any(|(_, mesh, ..)| mesh == id)));
    if changed {
        geometry.generation = geometry.generation.wrapping_add(1);
        geometry.vertices.clear();
        geometry.indices.clear();
        geometry.batches.clear();
        geometry.topology.clear();
        let mut batch_indices = BTreeMap::<(usize, RenderLayers), Vec<u32>>::new();
        for (instance, ((entity, mesh_handle, _, _, _, _, _, _, skin), (_, _, cull, layers))) in
            objects.iter().zip(&keys).enumerate()
        {
            let Some(mesh) = meshes.get(mesh_handle.id()) else {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} is unavailable in the CPU asset world"
                ));
                break;
            };
            if mesh.morph_targets().is_some() {
                geometry.failure = Some(format!(
                    "ground truth requires baked geometry; morph deformation on {entity:?} is unsupported"
                ));
                break;
            }
            if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} is not a triangle list"
                ));
                break;
            }
            let (
                Some(VertexAttributeValues::Float32x3(positions)),
                Some(VertexAttributeValues::Float32x3(normals)),
            ) = (
                mesh.attribute(Mesh::ATTRIBUTE_POSITION),
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
            )
            else {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} lacks Float32 positions/normals"
                ));
                break;
            };
            if positions.len() != normals.len()
                || positions.len() > u32::MAX as usize - geometry.vertices.len()
            {
                geometry.failure = Some(
                    "ground truth mesh has inconsistent or excessive vertex attributes".into(),
                );
                break;
            }
            if positions.iter().any(|p| !Vec3::from_array(*p).is_finite())
                || normals.iter().any(|n| {
                    !Vec3::from_array(*n).is_finite()
                        || Vec3::from_array(*n).length_squared() < 1e-12
                })
                || mesh.indices().is_some_and(|indices| {
                    indices.len() % 3 != 0 || indices.iter().any(|index| index >= positions.len())
                })
                || (mesh.indices().is_none() && positions.len() % 3 != 0)
            {
                geometry.failure =
                    Some(format!("ground truth mesh {entity:?} has invalid geometry: positions={}, normals={}, bad_position={:?}, bad_normal={:?}, indices={:?}",
                        positions.len(), normals.len(),
                        positions.iter().position(|p| !Vec3::from_array(*p).is_finite()),
                        normals.iter().position(|n| !Vec3::from_array(*n).is_finite() || Vec3::from_array(*n).length_squared() < 1e-12),
                        mesh.indices().map(|i| i.len())));
                break;
            }
            let baked;
            let (positions, normals) = if let Some(skin) = skin {
                match skin::bake(mesh, skin, &inverse_bindposes, &joints) {
                    Ok(value) => baked = value,
                    Err(error) => {
                        geometry.failure = Some(format!("ground truth skin {entity:?}: {error}"));
                        break;
                    }
                }
                (&baked.0, &baked.1)
            } else {
                (positions, normals)
            };
            let offset = geometry.vertices.len() as u32;
            if flow_enabled {
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                if let Some(indices) = mesh.indices() {
                    for index in indices.iter() {
                        index.hash(&mut hash);
                    }
                }
                geometry.topology.push(ObjectTopology {
                    entity: *entity,
                    mesh: mesh_handle.id(),
                    vertices: offset as usize..offset as usize + positions.len(),
                    indices_hash: hash.finish(),
                });
            }
            geometry
                .vertices
                .extend(
                    positions
                        .iter()
                        .zip(normals)
                        .map(|(position, normal)| Vertex {
                            position: *position,
                            normal: *normal,
                            instance: instance as u32,
                        }),
                );
            let indices = batch_indices.entry((*cull, layers.clone())).or_default();
            if let Some(source) = mesh.indices() {
                indices.extend(source.iter().map(|i| offset + i as u32));
            } else {
                indices.extend(offset..offset + positions.len() as u32);
            }
        }
        for ((cull, layers), indices) in batch_indices {
            let start = geometry.indices.len() as u32;
            geometry.indices.extend(indices);
            let end = geometry.indices.len() as u32;
            geometry.batches.push(Batch {
                indices: start..end,
                cull,
                layers,
            });
        }
        cache.keys = keys;
        cache.flow_enabled = flow_enabled;
    }
    geometry.instances.clear();
    for (entity, _, transform, _, semantic, _, _, _, skin) in objects {
        // Bevy's skin matrices already transform rest vertices to world space.
        let world_from_local = if skin.is_some() {
            Mat4::IDENTITY
        } else {
            transform.to_matrix()
        };
        let normal_from_local = world_from_local.inverse().transpose();
        if !world_from_local.is_finite() || !normal_from_local.is_finite() {
            geometry.failure = Some(format!(
                "ground truth entity {entity:?} has a singular/nonfinite transform"
            ));
            break;
        }
        geometry.instances.push(Instance {
            world_from_local: world_from_local.to_cols_array_2d(),
            normal_from_local: normal_from_local.to_cols_array_2d(),
            semantic: semantic.map_or(0, semantic_id),
            padding: [0; 3],
        });
    }
    for camera in &cameras {
        *camera.status.failure.lock().unwrap() = geometry.failure.clone();
        if geometry.failure.is_some() {
            camera.status.valid.store(false, Ordering::Release);
        }
    }
}

/// Stable NYU/Hypersim vocabulary IDs; 0 is unlabeled/background, 1..40 are classes.
pub fn semantic_id(label: &SemanticLabel) -> u32 {
    label.clone() as u32 + 1
}

pub fn semantic_label(id: u32) -> Option<SemanticLabel> {
    const NAMES: [&str; 40] = [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "blinds",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "floormat",
        "clothes",
        "ceiling",
        "books",
        "refrigerator",
        "television",
        "paper",
        "towel",
        "shower_curtain",
        "box",
        "whiteboard",
        "person",
        "nightstand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "bag",
        "other_structure",
        "other_furniture",
        "other_prop",
    ];
    id.checked_sub(1)
        .and_then(|i| NAMES.get(i as usize))
        .and_then(|name| SemanticLabel::from_label(name))
}

struct GroundTruthPipeline {
    camera_layout: BindGroupLayout,
    instance_layout: BindGroupLayout,
    pipelines: [CachedRenderPipelineId; 3],
}

#[derive(Resource, Default)]
struct GroundTruthPipelines(Option<GroundTruthPipeline>);

fn initialize_pipeline(
    cameras: Query<&GroundTruthCamera>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<GroundTruthPipelines>,
) {
    if !cameras.is_empty() && pipelines.0.is_none() {
        pipelines.0 = Some(GroundTruthPipeline::new(&cache));
    }
}

impl GroundTruthPipeline {
    fn new(cache: &PipelineCache) -> Self {
        let camera_descriptor = BindGroupLayoutDescriptor::new(
            "ground_truth_camera",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX_FRAGMENT,
                uniform_buffer_sized(
                    false,
                    NonZeroU64::new(std::mem::size_of::<CameraUniform>() as u64),
                ),
            ),
        );
        let instance_descriptor = BindGroupLayoutDescriptor::new(
            "ground_truth_instances",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                storage_buffer_read_only_sized(
                    false,
                    NonZeroU64::new(std::mem::size_of::<Instance>() as u64),
                ),
            ),
        );
        let camera_layout = cache.get_bind_group_layout(&camera_descriptor);
        let instance_layout = cache.get_bind_group_layout(&instance_descriptor);
        let pipelines = [None, Some(Face::Back), Some(Face::Front)].map(|cull_mode| {
            cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("ground_truth_rgba32float_mrt".into()),
                layout: vec![camera_descriptor.clone(), instance_descriptor.clone()],
                vertex: VertexState {
                    shader: SHADER,
                    entry_point: Some("vertex".into()),
                    buffers: vec![VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                            VertexAttribute {
                                format: VertexFormat::Uint32,
                                offset: 24,
                                shader_location: 2,
                            },
                        ],
                    }],
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: SHADER,
                    entry_point: Some("fragment".into()),
                    targets: vec![
                        Some(ColorTargetState {
                            format: TextureFormat::Rgba32Float,
                            blend: None,
                            write_mask: ColorWrites::ALL
                        });
                        2
                    ],
                    ..default()
                }),
                primitive: PrimitiveState {
                    cull_mode,
                    ..default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: default(),
                    bias: default(),
                }),
                ..default()
            })
        });
        Self {
            camera_layout,
            instance_layout,
            pipelines,
        }
    }
}

#[derive(Resource, Default)]
struct GpuGeometry {
    generation: u64,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    instances: Option<Buffer>,
    instance_bind_group: Option<BindGroup>,
    instance_bytes: Vec<u8>,
}

fn prepare_geometry(
    geometry: Res<ExtractedGeometry>,
    pipelines: Res<GroundTruthPipelines>,
    mut gpu: ResMut<GpuGeometry>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    diagnostics: Res<GroundTruthDiagnostics>,
) {
    if geometry.vertices.is_empty() || geometry.failure.is_some() {
        *gpu = GpuGeometry::default();
        let mut stats = diagnostics.0.lock().unwrap();
        stats.mesh_instances = 0;
        stats.vertices = 0;
        stats.triangles = 0;
        stats.draw_batches = 0;
        stats.geometry_bytes = 0;
        return;
    }
    let Some(pipeline) = pipelines.0.as_ref() else {
        return;
    };
    let mut stats = diagnostics.0.lock().unwrap();
    if gpu.vertices.is_none() || gpu.generation != geometry.generation {
        gpu.vertices = Some(super::upload_buffer(
            &device,
            &queue,
            &BufferInitDescriptor {
                label: Some("ground_truth_vertices"),
                contents: bytemuck::cast_slice(&geometry.vertices),
                usage: BufferUsages::VERTEX,
            },
        ));
        gpu.indices = Some(super::upload_buffer(
            &device,
            &queue,
            &BufferInitDescriptor {
                label: Some("ground_truth_indices"),
                contents: bytemuck::cast_slice(&geometry.indices),
                usage: BufferUsages::INDEX,
            },
        ));
        gpu.instances = None;
        gpu.generation = geometry.generation;
        stats.generation = geometry.generation;
        stats.mesh_instances = geometry.instances.len();
        stats.vertices = geometry.vertices.len();
        stats.triangles = geometry.indices.len() / 3;
        stats.draw_batches = geometry.batches.len();
        stats.geometry_bytes =
            geometry.vertices.len() * std::mem::size_of::<Vertex>() + geometry.indices.len() * 4;
        stats.geometry_uploads += 1;
    }
    let bytes = bytemuck::cast_slice(&geometry.instances);
    if gpu.instances.is_none() || gpu.instance_bytes.len() != bytes.len() {
        let buffer = super::upload_buffer(
            &device,
            &queue,
            &BufferInitDescriptor {
                label: Some("ground_truth_instances"),
                contents: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        );
        gpu.instance_bind_group = Some(device.create_bind_group(
            "ground_truth_instances",
            &pipeline.instance_layout,
            &BindGroupEntries::single(buffer.as_entire_binding()),
        ));
        gpu.instances = Some(buffer);
        gpu.instance_bytes = bytes.to_vec();
        stats.instance_updates += 1;
    } else if gpu.instance_bytes != bytes {
        queue.write_buffer(gpu.instances.as_ref().unwrap(), 0, bytes);
        gpu.instance_bytes = bytes.to_vec();
        stats.instance_updates += 1;
    }
}

#[derive(Component)]
struct PreparedCamera {
    buffer: Buffer,
    bind_group: BindGroup,
}

fn prepare_cameras(
    mut commands: Commands,
    cameras: Query<(
        Entity,
        &GroundTruthCamera,
        &ExtractedView,
        Option<&PreparedCamera>,
    )>,
    pipelines: Res<GroundTruthPipelines>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(pipeline) = pipelines.0.as_ref() else {
        return;
    };
    for (entity, camera, view, prepared) in &cameras {
        let view_from_world = view.world_from_view.to_matrix().inverse();
        let uniform = CameraUniform {
            clip_from_world: view
                .clip_from_world
                .unwrap_or(view.clip_from_view * view_from_world)
                .to_cols_array_2d(),
            view_from_world: view_from_world.to_cols_array_2d(),
            limits: [camera.near, camera.far, 0.0, 0.0],
        };
        if let Some(prepared) = prepared {
            queue.write_buffer(&prepared.buffer, 0, bytemuck::bytes_of(&uniform));
        } else {
            let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("ground_truth_camera"),
                contents: bytemuck::bytes_of(&uniform),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(
                "ground_truth_camera",
                &pipeline.camera_layout,
                &BindGroupEntries::single(buffer.as_entire_binding()),
            );
            commands
                .entity(entity)
                .insert(PreparedCamera { buffer, bind_group });
        }
    }
}

fn render_ground_truth(
    world: &World,
    mut context: RenderContext,
    cameras: Query<(&GroundTruthCamera, &PreparedCamera)>,
) {
    let geometry = world.resource::<ExtractedGeometry>();
    let gpu = world.resource::<GpuGeometry>();
    if geometry.failure.is_some() {
        return;
    }
    let Some(pipeline) = world.resource::<GroundTruthPipelines>().0.as_ref() else {
        return;
    };
    let cache = world.resource::<PipelineCache>();
    let images = world.resource::<RenderAssets<GpuImage>>();
    // An empty scene is a valid background-only capture. Clear its targets
    // and complete the request without retaining or binding geometry buffers.
    // A nonempty scene still waits for its uploads and render pipelines.
    let draw_data = if geometry.vertices.is_empty() {
        None
    } else {
        let (Some(vertices), Some(indices), Some(instances)) =
            (&gpu.vertices, &gpu.indices, &gpu.instance_bind_group)
        else {
            return;
        };
        let Some(pipelines) = pipeline
            .pipelines
            .iter()
            .map(|id| cache.get_render_pipeline(*id))
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        Some((vertices, indices, instances, pipelines))
    };
    // One stable diagnostic path covers all requested views. Idle frames
    // create neither render work nor empty GPU timestamp measurements.
    let active: Vec<_> = cameras
        .iter()
        .filter_map(|(camera, prepared)| {
            if camera.frame_id == 0 || camera.geometry_frame() == Some(camera.frame_id) {
                return None;
            }
            Some((
                camera,
                prepared,
                images.get(&camera.world_depth)?,
                images.get(&camera.normal_semantic)?,
                images.get(&camera.depth)?,
            ))
        })
        .collect();
    if active.is_empty() {
        return;
    }
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(context.command_encoder(), "indoor_ground_truth");
    for (camera, prepared, world_depth, normal_semantic, depth) in active {
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ground_truth_float32_mrt"),
                    color_attachments: &[
                        Some(RenderPassColorAttachment {
                            view: &world_depth.texture_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: StoreOp::Store,
                            },
                        }),
                        Some(RenderPassColorAttachment {
                            view: &normal_semantic.texture_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth.texture_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            if let Some((vertices, indices, instances, pipelines)) = &draw_data {
                pass.set_vertex_buffer(0, *vertices.slice(..));
                pass.set_index_buffer(*indices.slice(..), IndexFormat::Uint32);
                pass.set_bind_group(0, &prepared.bind_group, &[]);
                pass.set_bind_group(1, *instances, &[]);
                for batch in &geometry.batches {
                    if !camera.layers.intersects(&batch.layers) {
                        continue;
                    }
                    pass.set_pipeline(pipelines[batch.cull]);
                    pass.draw_indexed(batch.indices.clone(), 0, 0..1);
                }
            }
        }
        if camera.flow.is_some() && !flow::render(world, &mut context, camera, &depth.texture_view)
        {
            continue;
        }
        camera
            .status
            .generation
            .store(geometry.generation, Ordering::Release);
        camera
            .status
            .frame
            .store(camera.frame_id, Ordering::Release);
        camera.status.valid.store(true, Ordering::Release);
        world
            .resource::<GroundTruthDiagnostics>()
            .0
            .lock()
            .unwrap()
            .rendered_views += 1;
    }
    time_span.end(context.command_encoder());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_ids_round_trip_and_match_existing_palette() {
        for id in 1..=40 {
            let label = semantic_label(id).unwrap();
            assert_eq!(semantic_id(&label), id);
            assert_eq!(SemanticLabel::from_label(label.as_str()).unwrap(), label);
        }
        assert!(semantic_label(0).is_none());
        assert!(semantic_label(41).is_none());
    }

    #[test]
    fn ground_truth_targets_have_no_half_precision_or_color_conversion() {
        let mut images = Assets::<Image>::default();
        let camera = GroundTruthCamera::new(&mut images, UVec2::new(641, 479));
        for handle in [&camera.world_depth, &camera.normal_semantic] {
            let image = images.get(handle).unwrap();
            assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba32Float);
            assert_eq!(image.texture_descriptor.sample_count, 1);
            assert!(image
                .texture_descriptor
                .usage
                .contains(TextureUsages::COPY_SRC));
            assert!(image.data.is_none());
        }
        assert_eq!(std::mem::size_of::<Instance>(), 144);
        assert_eq!(std::mem::size_of::<CameraUniform>(), 144);
        assert_eq!(std::mem::size_of::<Vertex>(), 28);
        assert_eq!(camera.rendered_frame(), None);
    }
}
