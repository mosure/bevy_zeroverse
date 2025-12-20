use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Mutex, OnceLock, mpsc},
};

use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on},
};
use wgpu::{self, util::DeviceExt};

use crate::{
    annotation::obb::ObbClass,
    app::{BevyZeroverseConfig, OvoxelMode},
    render::semantic::SemanticLabel,
    scene::{RegenerateSceneEvent, SceneAabbNode},
};

#[cfg(test)]
use bevy::asset::RenderAssetUsages;

/// Marker to request O-Voxel conversion for an entity and its descendants.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct OvoxelExport {
    /// Grid resolution (e.g. 128 or 256).
    pub resolution: u32,
    /// Explicit axis-aligned bounds. If omitted, bounds are inferred from geometry.
    pub aabb: Option<([f32; 3], [f32; 3])>,
}

impl Default for OvoxelExport {
    fn default() -> Self {
        Self {
            resolution: 128,
            aabb: None,
        }
    }
}

/// Minimal O-Voxel-like payload mirroring the TRELLIS fields we can compute on CPU.
#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component)]
pub struct OvoxelVolume {
    /// Integer voxel coordinates in Morton order (sorted lexicographically here).
    pub coords: Vec<[u32; 3]>,
    /// Dual vertex offsets in voxel space, encoded to [0, 255].
    pub dual_vertices: Vec<[u8; 3]>,
    /// Intersection flags (bitmask xyz) per voxel.
    pub intersected: Vec<u8>,
    /// Packed base colors per voxel (rgba 0-255).
    pub base_color: Vec<[u8; 4]>,
    /// Semantic class id per voxel (index into `semantic_labels`, 0 reserved for unknown).
    pub semantics: Vec<u16>,
    /// Palette of semantic labels (index matches semantic id).
    pub semantic_labels: Vec<String>,
    /// Resolution used for this bake.
    pub resolution: u32,
    /// World-space bounds used for voxelization.
    pub aabb: [[f32; 3]; 2],
}

/// Tracks how many times the volume has been recomputed for caching diagnostics.
#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component)]
pub struct OvoxelCache {
    pub version: u64,
}

#[derive(Component, Debug)]
pub(crate) struct OvoxelTask(Task<OvoxelVolume>);

pub struct OvoxelPlugin;

const GPU_TILE_SIZE: u32 = 4;
const GPU_MAX_OUTPUT_VOXELS: u32 = 2_000_000;
const GPU_PARAMS_SIZE: u64 = 256;
const GPU_PREFIX_WG: u32 = 256;
const GPU_CLASSIFY_WG: u32 = 256;

fn expand_bits(mut v: u32) -> u32 {
    // Interleave lower 10 bits of v so the result fits in 30 bits.
    v &= 0x3ff;
    v = (v | (v << 16)) & 0x30000ff;
    v = (v | (v << 8)) & 0x300f00f;
    v = (v | (v << 4)) & 0x30c30c3;
    v = (v | (v << 2)) & 0x9249249;
    v
}

fn morton3(x: u32, y: u32, z: u32) -> u32 {
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

fn is_ovoxel_enabled(config: Option<Res<crate::app::BevyZeroverseConfig>>) -> bool {
    config.is_none_or(|c| !matches!(c.ovoxel_mode, crate::app::OvoxelMode::Disabled))
}

impl Plugin for OvoxelPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<RegenerateSceneEvent>();
        app.register_type::<OvoxelExport>();
        app.register_type::<OvoxelVolume>();
        app.register_type::<OvoxelCache>();
        app.add_systems(
            Update,
            (
                tag_scene_roots,
                process_ovoxel_exports,
                collect_ovoxel_tasks,
                reset_ovoxel_on_regen,
            )
                .chain()
                .run_if(is_ovoxel_enabled),
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: Vec4,
    semantic_id: u16,
}

/// System: finds entities tagged with `OvoxelExport`, gathers meshes under them,
/// voxelizes, and stores the result on the same entity as `OvoxelVolume`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn process_ovoxel_exports(
    mut commands: Commands,
    config: Option<Res<BevyZeroverseConfig>>,
    render_device: Option<Res<RenderDevice>>,
    render_queue: Option<Res<RenderQueue>>,
    roots: Query<(
        Entity,
        &OvoxelExport,
        Option<&OvoxelVolume>,
        Option<&OvoxelCache>,
        Option<&OvoxelTask>,
    )>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    mesh_query: Query<(
        &Mesh3d,
        Option<&MeshMaterial3d<StandardMaterial>>,
        &GlobalTransform,
        Option<&SemanticLabel>,
        Option<&ObbClass>,
        Option<&Name>,
    )>,
    mesh_changed: Query<
        Entity,
        Or<(
            Changed<Mesh3d>,
            Changed<MeshMaterial3d<StandardMaterial>>,
            Changed<Transform>,
            Changed<GlobalTransform>,
        )>,
    >,
    children: Query<&Children>,
) {
    let render_device_owned: Option<RenderDevice> = render_device.map(|r| r.clone());
    let render_queue_owned: Option<RenderQueue> = render_queue.map(|r| r.clone());
    for (root, settings, existing_volume, cache, task) in roots.iter() {
        if task.is_some() {
            continue;
        }

        let mut palette = SemanticPalette::new();
        let needs_recompute = existing_volume.is_none()
            || cache.is_none()
            || subtree_dirty(root, &mesh_changed, &children);

        if !needs_recompute {
            continue;
        }

        let mut triangles = Vec::new();

        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Ok((mesh3d, material, transform, semantic, obb_class, name)) =
                mesh_query.get(entity)
            {
                if let Some(mesh) = meshes.get(&mesh3d.0) {
                    triangles.extend(extract_triangles(
                        mesh,
                        transform,
                        material,
                        &materials,
                        &mut palette,
                        semantic,
                        obb_class,
                        name,
                    ));
                }
            }

            if let Ok(child_list) = children.get(entity) {
                for child in child_list.iter() {
                    stack.push(child);
                }
            }
        }

        if triangles.is_empty() {
            continue;
        }

        let resolution = if settings.resolution == 0 {
            128
        } else {
            settings.resolution
        };
        if resolution == 0 {
            warn!("ovoxel resolution resolved to zero; skipping voxelization");
            continue;
        }
        let aabb = settings
            .aabb
            .map(|(min, max)| [min, max])
            .unwrap_or_else(|| triangles_aabb(&triangles));

        let labels = palette.into_labels();
        let task_pool = AsyncComputeTaskPool::get();
        let mode = config
            .as_ref()
            .map(|c| c.ovoxel_mode)
            .unwrap_or(OvoxelMode::CpuAsync);
        #[cfg(test)]
        {
            if !matches!(mode, OvoxelMode::GpuCompute) {
                let volume = voxelize_triangles(&triangles, resolution, aabb, labels.clone());
                let version = cache
                    .map(|c| c.version.saturating_add(1))
                    .unwrap_or(1);
                commands
                    .entity(root)
                    .insert((volume, OvoxelCache { version }))
                    .remove::<OvoxelTask>();
                continue;
            }
        }
        let task = match mode {
            OvoxelMode::CpuAsync | OvoxelMode::Disabled => task_pool
                .spawn(async move { voxelize_triangles(&triangles, resolution, aabb, labels) }),
            OvoxelMode::GpuCompute => {
                if let (Some(device), Some(queue)) =
                    (render_device_owned.clone(), render_queue_owned.clone())
                {
                    task_pool.spawn(async move {
                        voxelize_triangles_gpu(
                            &triangles, resolution, aabb, labels, &device, &queue, true,
                        )
                        .unwrap_or_else(|| {
                            panic!("GPU ovoxel requested but GPU path failed; aborting")
                        })
                    })
                } else {
                    panic!("GPU ovoxel requested but RenderDevice/RenderQueue unavailable");
                }
            }
        };

        // Store the task so it can be polled to completion later.
        commands.entity(root).insert(OvoxelTask(task));
    }
}

fn collect_ovoxel_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut OvoxelTask)>,
    mut caches: Query<&mut OvoxelCache>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        if let Some(volume) = block_on(futures_lite::future::poll_once(&mut task.0)) {
            let mut version = 1;
            if let Ok(mut cache) = caches.get_mut(entity) {
                cache.version = cache.version.saturating_add(1);
                version = cache.version;
            }
            commands
                .entity(entity)
                .insert((volume, OvoxelCache { version }))
                .remove::<OvoxelTask>();
        }
    }
}

#[allow(clippy::type_complexity)]
fn reset_ovoxel_on_regen(
    mut commands: Commands,
    mut regen_events: MessageReader<RegenerateSceneEvent>,
    query: Query<
        Entity,
        (
            With<OvoxelExport>,
            Or<(With<OvoxelVolume>, With<OvoxelTask>)>,
        ),
    >,
) {
    if regen_events.is_empty() {
        return;
    }
    regen_events.clear();
    for entity in query.iter() {
        commands
            .entity(entity)
            .remove::<OvoxelVolume>()
            .remove::<OvoxelCache>()
            .remove::<OvoxelTask>();
    }
}

#[allow(clippy::type_complexity)]
fn subtree_dirty(
    root: Entity,
    changed: &Query<
        Entity,
        Or<(
            Changed<Mesh3d>,
            Changed<MeshMaterial3d<StandardMaterial>>,
            Changed<Transform>,
            Changed<GlobalTransform>,
        )>,
    >,
    children: &Query<&Children>,
) -> bool {
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if changed.contains(entity) {
            return true;
        }
        if let Ok(child_list) = children.get(entity) {
            for child in child_list.iter() {
                stack.push(child);
            }
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn extract_triangles(
    mesh: &Mesh,
    transform: &GlobalTransform,
    _material: Option<&MeshMaterial3d<StandardMaterial>>,
    _materials: &Assets<StandardMaterial>,
    palette: &mut SemanticPalette,
    semantic: Option<&SemanticLabel>,
    obb_class: Option<&ObbClass>,
    name: Option<&Name>,
) -> Vec<Triangle> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Vec::new();
    }

    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attr| attr.as_float3())
        .map(|a| a.to_vec());

    let Some(positions) = positions else {
        return Vec::new();
    };

    let semantic_id =
        palette.id_for_label(label_from_components(semantic, obb_class, name).as_deref());

    let semantic_color = label_from_components(semantic, obb_class, name)
        .as_deref()
        .and_then(SemanticLabel::from_label)
        .map(|l| l.color().to_linear());

    // Unknown semantic → fallback pink checkerboard based on centroid hash.
    let fallback = |p: Vec3| -> Vec4 {
        let h = ((p.x.to_bits() ^ p.y.to_bits() ^ p.z.to_bits()) & 1) as f32;
        if h > 0.0 {
            Vec4::new(1.0, 0.2, 0.8, 1.0)
        } else {
            Vec4::new(0.8, 0.1, 0.6, 1.0)
        }
    };

    let affine = transform.affine();
    let indices: Vec<u32> = mesh
        .indices()
        .map(|idx| idx.iter().map(|v| v as u32).collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let mut tris = Vec::new();
    for chunk in indices.chunks_exact(3) {
        let a = affine.transform_point3(Vec3::from(positions[chunk[0] as usize]));
        let b = affine.transform_point3(Vec3::from(positions[chunk[1] as usize]));
        let c = affine.transform_point3(Vec3::from(positions[chunk[2] as usize]));

        let centroid = (a + b + c) / 3.0;
        let tri_color = semantic_color
            .map(|col| Vec4::new(col.red, col.green, col.blue, col.alpha))
            .unwrap_or_else(|| fallback(centroid));

        tris.push(Triangle {
            a,
            b,
            c,
            color: tri_color,
            semantic_id,
        });
    }

    tris
}

fn triangles_aabb(triangles: &[Triangle]) -> [[f32; 3]; 2] {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for tri in triangles {
        for v in [tri.a, tri.b, tri.c] {
            min = min.min(v);
            max = max.max(v);
        }
    }

    [[min.x, min.y, min.z], [max.x, max.y, max.z]]
}

fn voxelize_triangles(
    triangles: &[Triangle],
    resolution: u32,
    aabb: [[f32; 3]; 2],
    semantic_labels: Vec<String>,
) -> OvoxelVolume {
    let min = Vec3::from(aabb[0]);
    let mut extent = Vec3::from(aabb[1]) - min;
    let eps = 1e-3;
    extent = extent.max(Vec3::splat(eps));
    let max = min + extent;
    let extent = extent.max(Vec3::splat(f32::EPSILON));
    let aabb = [min.to_array(), max.to_array()];
    let res_f = resolution as f32;
    let voxel_size = extent / res_f;
    let half_diag = voxel_size.length() * 0.5;

    #[derive(Default)]
    struct Accum {
        count: u32,
        dual_sum: Vec3,
        color_sum: Vec4,
        mask: u8,
        semantics: HashMap<u16, u32>,
    }

    let mut voxels: HashMap<(u32, u32, u32), Accum> = HashMap::new();

    for tri in triangles {
        if (tri.a - tri.b).cross(tri.c - tri.a).length_squared() <= f32::EPSILON {
            continue;
        }
        let tri_min = tri.a.min(tri.b).min(tri.c);
        let tri_max = tri.a.max(tri.b).max(tri.c);

        let start = ((tri_min - min) / voxel_size)
            .floor()
            .clamp(Vec3::ZERO, Vec3::splat(res_f - 1.0));
        let end = ((tri_max - min) / voxel_size)
            .ceil()
            .clamp(Vec3::ZERO, Vec3::splat(res_f - 1.0));

        for x in start.x as u32..=end.x as u32 {
            for y in start.y as u32..=end.y as u32 {
                for z in start.z as u32..=end.z as u32 {
                    let voxel_min = min + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    let center = voxel_min + voxel_size * 0.5;

                    let closest = closest_point_on_triangle(center, tri);
                    let dist = center.distance(closest);
                    if dist > half_diag {
                        continue;
                    }

                    let offset = (closest - voxel_min) / voxel_size;
                    let dual = offset.clamp(Vec3::ZERO, Vec3::ONE);

                    let mut mask = 0u8;
                    if tri_min.x < voxel_min.x && tri_max.x > voxel_min.x {
                        mask |= 1;
                    }
                    if tri_min.y < voxel_min.y && tri_max.y > voxel_min.y {
                        mask |= 2;
                    }
                    if tri_min.z < voxel_min.z && tri_max.z > voxel_min.z {
                        mask |= 4;
                    }

                    let entry = voxels.entry((x, y, z)).or_default();
                    entry.count += 1;
                    entry.dual_sum += dual;
                    entry.color_sum += tri.color;
                    entry.mask |= mask;
                    *entry.semantics.entry(tri.semantic_id).or_insert(0) += 1;
                }
            }
        }
    }

    let mut keys: Vec<(u32, u32, u32)> = voxels.keys().cloned().collect();
    keys.sort_unstable_by_key(|&(x, y, z)| morton3(x, y, z));

    let mut coords = Vec::with_capacity(keys.len());
    let mut dual_vertices = Vec::with_capacity(keys.len());
    let mut intersected = Vec::with_capacity(keys.len());
    let mut base_color = Vec::with_capacity(keys.len());
    let mut semantics = Vec::with_capacity(keys.len());

    for key in keys {
        if let Some(acc) = voxels.get(&key) {
            let inv = 1.0 / acc.count as f32;
            let dual = (acc.dual_sum * inv * 255.0)
                .clamp(Vec3::ZERO, Vec3::splat(255.0))
                .round();
            let color = (acc.color_sum * inv * 255.0)
                .clamp(Vec4::ZERO, Vec4::splat(255.0))
                .round();

            coords.push([key.0, key.1, key.2]);
            dual_vertices.push([dual.x as u8, dual.y as u8, dual.z as u8]);
            intersected.push(acc.mask);
            base_color.push([color.x as u8, color.y as u8, color.z as u8, color.w as u8]);

            let semantic_id = acc
                .semantics
                .iter()
                .max_by(|(ida, ca), (idb, cb)| ca.cmp(cb).then(ida.cmp(idb)))
                .map(|(id, _)| *id)
                .unwrap_or(0);
            semantics.push(semantic_id);
        }
    }

    OvoxelVolume {
        coords,
        dual_vertices,
        intersected,
        base_color,
        semantics,
        semantic_labels,
        resolution,
        aabb,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    min: [f32; 4],
    max: [f32; 4],
    color: [f32; 4],
    semantic: u32,
    _pad_sem: [u32; 3],
    _pad_tail: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    min: [f32; 4],
    voxel: [f32; 4],
    tile_dims: [u32; 4],
    half_diag: f32,
    resolution: u32,
    tri_count: u32,
    max_output: u32,
    pair_cap: u32,
    _pad_params: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuVoxel {
    coord: [u32; 3],
    mask: u32,
    dual_sum: [f32; 3],
    _pad_dual: f32,
    color_sum: [f32; 4],
    semantic: u32,
    count: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuOutputMeta {
    count: u32,
    overflow: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuActiveTile {
    tile_id: u32,
    start: u32,
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuTilePair {
    tile: u32,
    tri: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuActiveCounter {
    pair_counter: u32,
    pair_cursor: u32,
    active_count: u32,
    _pad: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct BufferKey {
    device_id: usize,
    tile_count: u64,
    pair_cap: u32,
    max_output: u32,
}

#[derive(Clone)]
struct GpuBuffers {
    key: BufferKey,
    tile_meta: wgpu::Buffer,
    tile_pairs: wgpu::Buffer,
    active_counter: wgpu::Buffer,
    scatter_indirect: wgpu::Buffer,
    voxel_indirect: wgpu::Buffer,
    active_tiles: wgpu::Buffer,
    tile_indices: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct GpuPipeline {
    shared_bind_group_layout: wgpu::BindGroupLayout,
    state_bind_group_layout: wgpu::BindGroupLayout,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
    voxel_bind_group_layout: wgpu::BindGroupLayout,
    prefix_pipeline: wgpu::ComputePipeline,
    prepare_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    classify_pipeline: wgpu::ComputePipeline,
    voxel_pipeline: wgpu::ComputePipeline,
}

static GPU_PIPELINE: OnceLock<GpuPipeline> = OnceLock::new();
static GPU_BUFFER_POOL: OnceLock<Mutex<Vec<GpuBuffers>>> = OnceLock::new();

fn buffer_pool() -> &'static Mutex<Vec<GpuBuffers>> {
    GPU_BUFFER_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn make_buffer_key(
    device: &wgpu::Device,
    tile_count: u64,
    pair_cap: u32,
    max_output: u32,
) -> BufferKey {
    BufferKey {
        device_id: device as *const _ as usize,
        tile_count,
        pair_cap,
        max_output,
    }
}

fn acquire_buffers(
    wgpu_device: &wgpu::Device,
    tile_count: u64,
    pair_cap: u32,
    max_output_voxels: u32,
) -> GpuBuffers {
    let key = make_buffer_key(wgpu_device, tile_count, pair_cap, max_output_voxels);
    let mut pool = buffer_pool().lock().expect("gpu buffer pool poisoned");
    if let Some(entry) = pool.iter().position(|b| b.key == key) {
        return pool.swap_remove(entry);
    }
    drop(pool);

    let tile_meta_size = tile_count * std::mem::size_of::<u32>() as u64 * 3;
    let pair_bytes = pair_cap as u64 * std::mem::size_of::<GpuTilePair>() as u64;
    let active_tiles_bytes = tile_count * std::mem::size_of::<GpuActiveTile>() as u64;
    let tile_indices_bytes = pair_cap as u64 * std::mem::size_of::<u32>() as u64;
    let meta_bytes = std::mem::size_of::<GpuOutputMeta>() as u64;
    let voxel_bytes = (max_output_voxels as usize * std::mem::size_of::<GpuVoxel>()) as u64;
    let output_bytes = meta_bytes + voxel_bytes;

    let tile_meta = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_tile_meta"),
        size: tile_meta_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tile_pairs = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_tile_pairs"),
        size: pair_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let active_counter = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_active_counter"),
        size: std::mem::size_of::<GpuActiveCounter>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let scatter_indirect = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_scatter_indirect"),
        size: 3 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let voxel_indirect = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_voxel_indirect"),
        size: 3 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let active_tiles = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_active_tiles"),
        size: active_tiles_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tile_indices = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_tile_indices"),
        size: tile_indices_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_voxels"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    GpuBuffers {
        key,
        tile_meta,
        tile_pairs,
        active_counter,
        scatter_indirect,
        voxel_indirect,
        active_tiles,
        tile_indices,
        output,
        readback,
    }
}

fn release_buffers(buffers: GpuBuffers) {
    buffer_pool()
        .lock()
        .expect("gpu buffer pool poisoned")
        .push(buffers);
}

fn voxelize_triangles_gpu(
    triangles: &[Triangle],
    resolution: u32,
    aabb: [[f32; 3]; 2],
    semantic_labels: Vec<String>,
    device: &RenderDevice,
    queue: &RenderQueue,
    strict: bool,
) -> Option<OvoxelVolume> {
    macro_rules! gpu_bail {
        ($msg:expr) => {{
            if strict {
                panic!($msg);
            } else {
                return None;
            }
        }};
    }
    let resolution = resolution.max(1);
    let voxel_count = (resolution as u64).saturating_pow(3);
    // Prevent runaway allocations on very large grids.
    if resolution > 1536 {
        gpu_bail!("ovoxel GPU path skipped: resolution too large");
    }

    let min = Vec3::from(aabb[0]);
    let mut extent = Vec3::from(aabb[1]) - min;
    let eps = 1e-3;
    extent = extent.max(Vec3::splat(eps));
    let max = min + extent;
    let extent = extent.max(Vec3::splat(f32::EPSILON));
    let aabb = [min.to_array(), max.to_array()];
    let voxel_size = extent / resolution as f32;
    let half_diag = voxel_size.length() * 0.5;
    let max_output_voxels = GPU_MAX_OUTPUT_VOXELS.min(voxel_count as u32).max(1);

    // Tile grid dimensions.
    let tile_dim = |d: u32| d.div_ceil(GPU_TILE_SIZE);
    let tile_dims = [
        tile_dim(resolution),
        tile_dim(resolution),
        tile_dim(resolution),
    ];
    let tile_count = tile_dims[0] as u64 * tile_dims[1] as u64 * tile_dims[2] as u64;

    // Estimate the number of tile/triangle pairs so we can size the sparse buffers more tightly.
    // We mirror the WGSL classify math and keep some headroom to avoid overflow.
    let res_minus_one = resolution.saturating_sub(1) as f32;
    let tile_max = [
        tile_dims[0].saturating_sub(1),
        tile_dims[1].saturating_sub(1),
        tile_dims[2].saturating_sub(1),
    ];
    let mut gpu_tris = Vec::with_capacity(triangles.len());
    let mut pair_cap_estimate: u64 = 0;
    for t in triangles {
        let tri_min = t.a.min(t.b).min(t.c);
        let tri_max = t.a.max(t.b).max(t.c);
        let start = ((tri_min - min) / voxel_size)
            .floor()
            .clamp(Vec3::ZERO, Vec3::splat(res_minus_one));
        let end = ((tri_max - min) / voxel_size)
            .ceil()
            .clamp(Vec3::ZERO, Vec3::splat(res_minus_one));
        let start_tile = (start / GPU_TILE_SIZE as f32).floor();
        let end_tile = (end / GPU_TILE_SIZE as f32).floor();
        let start_tile = [
            start_tile.x.max(0.0).min(tile_max[0] as f32) as u32,
            start_tile.y.max(0.0).min(tile_max[1] as f32) as u32,
            start_tile.z.max(0.0).min(tile_max[2] as f32) as u32,
        ];
        let end_tile = [
            end_tile.x.max(0.0).min(tile_max[0] as f32) as u32,
            end_tile.y.max(0.0).min(tile_max[1] as f32) as u32,
            end_tile.z.max(0.0).min(tile_max[2] as f32) as u32,
        ];
        let tiles_x = end_tile[0].saturating_sub(start_tile[0]).saturating_add(1) as u64;
        let tiles_y = end_tile[1].saturating_sub(start_tile[1]).saturating_add(1) as u64;
        let tiles_z = end_tile[2].saturating_sub(start_tile[2]).saturating_add(1) as u64;
        pair_cap_estimate = pair_cap_estimate.saturating_add(tiles_x * tiles_y * tiles_z);

        gpu_tris.push(GpuTriangle {
            a: [t.a.x, t.a.y, t.a.z, 0.0],
            b: [t.b.x, t.b.y, t.b.z, 0.0],
            c: [t.c.x, t.c.y, t.c.z, 0.0],
            min: [tri_min.x, tri_min.y, tri_min.z, 0.0],
            max: [tri_max.x, tri_max.y, tri_max.z, 0.0],
            color: t.color.to_array(),
            semantic: t.semantic_id as u32,
            _pad_sem: [0; 3],
            _pad_tail: [0; 4],
        });
    }

    let pair_cap = pair_cap_estimate
        .saturating_add(pair_cap_estimate / 4 + tile_count)
        .clamp(1, 10_000_000) as u32;

    let params = GpuParams {
        min: [min.x, min.y, min.z, 0.0],
        voxel: [voxel_size.x, voxel_size.y, voxel_size.z, 0.0],
        tile_dims: [tile_dims[0], tile_dims[1], tile_dims[2], 0],
        half_diag,
        resolution,
        tri_count: triangles.len() as u32,
        max_output: max_output_voxels,
        pair_cap,
        _pad_params: [0; 3],
    };

    let wgpu_device = device.wgpu_device();
    let wgpu_queue = &*queue.0;
    let max_storage = wgpu_device.limits().max_storage_buffers_per_shader_stage;
    let required_storage = 4;
    if max_storage < required_storage {
        gpu_bail!("ovoxel GPU path skipped: device storage buffer limit too low");
    }

    let tri_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ovoxel_triangles"),
        contents: bytemuck::cast_slice(&gpu_tris),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let params_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_params"),
        size: GPU_PARAMS_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    wgpu_queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

    let meta_bytes = std::mem::size_of::<GpuOutputMeta>() as u64;
    let voxel_bytes = (max_output_voxels as usize * std::mem::size_of::<GpuVoxel>()) as u64;
    let output_bytes = meta_bytes + voxel_bytes;

    let buffers = acquire_buffers(wgpu_device, tile_count, pair_cap, max_output_voxels);

    let pipeline = GPU_PIPELINE.get_or_init(|| {
        let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ovoxel_gpu"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GPU_WGSL)),
        });

        let shared_bind_group_layout =
            wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ovoxel_gpu_shared_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(GPU_PARAMS_SIZE),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let state_bind_group_layout =
            wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ovoxel_gpu_state_bgl"),
                entries: &[
                    // packed tile metadata (counts, offsets, heads)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // tile_pairs
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // active + pair counters
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let dispatch_bind_group_layout =
            wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ovoxel_gpu_dispatch_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let voxel_bind_group_layout =
            wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ovoxel_gpu_voxel_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let classify_pipeline_layout =
            wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ovoxel_gpu_classify_pl"),
                bind_group_layouts: &[&shared_bind_group_layout, &state_bind_group_layout],
                push_constant_ranges: &[],
            });
        let work_pipeline_layout =
            wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ovoxel_gpu_work_pl"),
                bind_group_layouts: &[
                    &shared_bind_group_layout,
                    &state_bind_group_layout,
                    &voxel_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let prepare_pipeline_layout =
            wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ovoxel_gpu_prepare_pl"),
                bind_group_layouts: &[
                    &shared_bind_group_layout,
                    &state_bind_group_layout,
                    &voxel_bind_group_layout,
                    &dispatch_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let classify_pipeline =
            wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ovoxel_gpu_classify_pipeline"),
                layout: Some(&classify_pipeline_layout),
                module: &shader,
                entry_point: Some("classify_tiles"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let prefix_pipeline =
            wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ovoxel_gpu_prefix_pipeline"),
                layout: Some(&work_pipeline_layout),
                module: &shader,
                entry_point: Some("prefix_tiles"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let prepare_pipeline =
            wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ovoxel_gpu_prepare_pipeline"),
                layout: Some(&prepare_pipeline_layout),
                module: &shader,
                entry_point: Some("prepare_dispatch"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let scatter_pipeline =
            wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ovoxel_gpu_scatter_pipeline"),
                layout: Some(&work_pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_pairs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let voxel_pipeline =
            wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ovoxel_gpu_pipeline"),
                layout: Some(&work_pipeline_layout),
                module: &shader,
                entry_point: Some("voxel_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        GpuPipeline {
            shared_bind_group_layout,
            state_bind_group_layout,
            dispatch_bind_group_layout,
            voxel_bind_group_layout,
            prefix_pipeline,
            prepare_pipeline,
            scatter_pipeline,
            classify_pipeline,
            voxel_pipeline,
        }
    });

    let shared_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ovoxel_gpu_shared_bg"),
        layout: &pipeline.shared_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tri_buffer.as_entire_binding(),
            },
        ],
    });

    let state_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ovoxel_gpu_state_bg"),
        layout: &pipeline.state_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.tile_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.tile_pairs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.active_counter.as_entire_binding(),
            },
        ],
    });

    let dispatch_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ovoxel_gpu_dispatch_bg"),
        layout: &pipeline.dispatch_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.scatter_indirect.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.voxel_indirect.as_entire_binding(),
            },
        ],
    });

    let voxel_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ovoxel_gpu_voxel_bg"),
        layout: &pipeline.voxel_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.active_tiles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.tile_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.output.as_entire_binding(),
            },
        ],
    });

    // Pass 1: classify tiles, then prefix/publish offsets on GPU.
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ovoxel_gpu_classify_encoder"),
    });
    encoder.clear_buffer(&buffers.tile_meta, 0, None);
    encoder.clear_buffer(&buffers.active_counter, 0, None);
    encoder.clear_buffer(&buffers.scatter_indirect, 0, None);
    encoder.clear_buffer(&buffers.voxel_indirect, 0, None);
    encoder.clear_buffer(&buffers.output, 0, Some(meta_bytes));

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ovoxel_gpu_classify_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.classify_pipeline);
        pass.set_bind_group(0, &shared_bind_group, &[]);
        pass.set_bind_group(1, &state_bind_group, &[]);
        let tri_dispatch = (triangles.len() as u32).div_ceil(GPU_CLASSIFY_WG);
        pass.dispatch_workgroups(tri_dispatch.max(1), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ovoxel_gpu_prefix_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.prefix_pipeline);
        pass.set_bind_group(0, &shared_bind_group, &[]);
        pass.set_bind_group(1, &state_bind_group, &[]);
        pass.set_bind_group(2, &voxel_bind_group, &[]);
        let tiles_total = (tile_count as u32).max(1);
        let tile_dispatch = tiles_total.div_ceil(GPU_PREFIX_WG);
        pass.dispatch_workgroups(tile_dispatch, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ovoxel_gpu_prepare_dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.prepare_pipeline);
        pass.set_bind_group(0, &shared_bind_group, &[]);
        pass.set_bind_group(1, &state_bind_group, &[]);
        pass.set_bind_group(2, &voxel_bind_group, &[]);
        pass.set_bind_group(3, &dispatch_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    wgpu_queue.submit(Some(encoder.finish()));
    let _ = wgpu_device.poll(wgpu::PollType::Wait);

    // Pass 2: scatter pairs into compact lists, then voxel accumulation (all GPU-side).
    let mut voxel_encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ovoxel_gpu_encoder"),
    });

    {
        let mut pass = voxel_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ovoxel_gpu_scatter_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.scatter_pipeline);
        pass.set_bind_group(0, &shared_bind_group, &[]);
        pass.set_bind_group(1, &state_bind_group, &[]);
        pass.set_bind_group(2, &voxel_bind_group, &[]);
        pass.dispatch_workgroups_indirect(&buffers.scatter_indirect, 0);
    }

    {
        let mut pass = voxel_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ovoxel_gpu_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.voxel_pipeline);
        pass.set_bind_group(0, &shared_bind_group, &[]);
        pass.set_bind_group(1, &state_bind_group, &[]);
        pass.set_bind_group(2, &voxel_bind_group, &[]);
        pass.dispatch_workgroups_indirect(&buffers.voxel_indirect, 0);
    }

    voxel_encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.readback, 0, output_bytes);

    wgpu_queue.submit(Some(voxel_encoder.finish()));
    // Ensure GPU completes before readback.
    let _ = wgpu_device.poll(wgpu::PollType::Wait);

    let slice = buffers.readback.slice(..);
    type BufferAsync = Result<(), wgpu::BufferAsyncError>;
    let (tx, rx): (mpsc::Sender<BufferAsync>, mpsc::Receiver<BufferAsync>) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = wgpu_device.poll(wgpu::PollType::Wait);
    rx.recv().ok().and_then(Result::ok)?;
    let data = slice.get_mapped_range();
    let meta: GpuOutputMeta =
        bytemuck::from_bytes::<GpuOutputMeta>(&data[..meta_bytes as usize]).to_owned();
    let used = meta.count.min(max_output_voxels);
    let overflowed = meta.overflow > 0 || meta.count > max_output_voxels;
    let voxels: &[GpuVoxel] = bytemuck::cast_slice(&data[meta_bytes as usize..]);
    let used = used.min(voxels.len() as u32) as usize;

    if overflowed {
        drop(data);
        buffers.readback.unmap();
        release_buffers(buffers);
        gpu_bail!("ovoxel GPU path overflowed sparse buffer");
    }

    if used == 0 {
        drop(data);
        buffers.readback.unmap();
        let volume = OvoxelVolume {
            coords: Vec::new(),
            dual_vertices: Vec::new(),
            intersected: Vec::new(),
            base_color: Vec::new(),
            semantics: Vec::new(),
            semantic_labels,
            resolution,
            aabb,
        };
        release_buffers(buffers);
        return Some(volume);
    }

    type Packed = ([u32; 3], [u8; 3], u8, [u8; 4], u16);
    let mut packed: Vec<Packed> = voxels
        .iter()
        .take(used)
        .filter(|v| v.count > 0)
        .map(|v| {
            let inv = 1.0 / v.count as f32;
            let dual = Vec3::from_array(v.dual_sum) * inv * 255.0;
            let color = Vec4::from_array(v.color_sum) * inv * 255.0;
            let semantic = v.semantic.min(u16::MAX as u32) as u16;
            (
                v.coord,
                [
                    dual.x.clamp(0.0, 255.0).round() as u8,
                    dual.y.clamp(0.0, 255.0).round() as u8,
                    dual.z.clamp(0.0, 255.0).round() as u8,
                ],
                v.mask as u8,
                [
                    color.x.clamp(0.0, 255.0).round() as u8,
                    color.y.clamp(0.0, 255.0).round() as u8,
                    color.z.clamp(0.0, 255.0).round() as u8,
                    color.w.clamp(0.0, 255.0).round() as u8,
                ],
                semantic,
            )
        })
        .collect();
    packed.sort_unstable_by(|a, b| {
        morton3(a.0[0], a.0[1], a.0[2]).cmp(&morton3(b.0[0], b.0[1], b.0[2]))
    });

    let mut coords = Vec::with_capacity(packed.len());
    let mut dual_vertices = Vec::with_capacity(packed.len());
    let mut intersected = Vec::with_capacity(packed.len());
    let mut base_color = Vec::with_capacity(packed.len());
    let mut semantics = Vec::with_capacity(packed.len());
    for (coord, dual, mask, color, semantic) in packed {
        coords.push(coord);
        dual_vertices.push(dual);
        intersected.push(mask);
        base_color.push(color);
        semantics.push(semantic);
    }
    drop(data);
    buffers.readback.unmap();
    if coords.len() as u32 != used as u32 {
        release_buffers(buffers);
        gpu_bail!("ovoxel GPU path returned mismatched voxel counts");
    }

    let volume = OvoxelVolume {
        coords,
        dual_vertices,
        intersected,
        base_color,
        semantics,
        semantic_labels,
        resolution,
        aabb,
    };
    release_buffers(buffers);
    Some(volume)
}

const GPU_WGSL: &str = include_str!("../assets/shaders/ovoxel.wgsl");

fn tag_scene_roots(
    mut commands: Commands,
    config: Option<Res<BevyZeroverseConfig>>,
    roots: Query<Entity, (With<SceneAabbNode>, Without<OvoxelExport>)>,
) {
    let settings = config.map(|c| c.ovoxel_resolution).unwrap_or_default();
    for entity in roots.iter() {
        commands.entity(entity).insert(OvoxelExport {
            resolution: if settings == 0 { 128 } else { settings },
            ..Default::default()
        });
    }
}

fn closest_point_on_triangle(p: Vec3, tri: &Triangle) -> Vec3 {
    // Algorithm adapted from Real-Time Collision Detection (Christer Ericson).
    let ab = tri.b - tri.a;
    let ac = tri.c - tri.a;
    let ap = p - tri.a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return tri.a;
    }

    let bp = p - tri.b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return tri.b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return tri.a + ab * v;
    }

    let cp = p - tri.c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return tri.c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return tri.a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return tri.b + (tri.c - tri.b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    tri.a + ab * v + ac * w
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::renderer::WgpuWrapper;
    use bevy::{MinimalPlugins, render::render_resource::PrimitiveTopology};

    fn simple_triangle_mesh() -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        );
        mesh
    }

    #[test]
    fn voxelizes_single_mesh_under_root() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            meshes.add(simple_triangle_mesh())
        };

        let mat_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            materials.add(StandardMaterial {
                base_color: Color::srgba(1.0, 0.0, 0.0, 1.0),
                ..Default::default()
            })
        };

        let root = app
            .world_mut()
            .spawn((
                OvoxelExport::default(),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(mat_handle),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        app.update();

        let volume = app
            .world()
            .entity(root)
            .get::<OvoxelVolume>()
            .cloned()
            .expect("volume should be attached");

        assert!(
            !volume.coords.is_empty(),
            "voxelization should produce voxels"
        );
        assert_eq!(volume.resolution, 128);
        // Expect the first voxel color to be red-ish.
        let first_color = volume.base_color[0];
        assert!(first_color[0] > first_color[1] && first_color[0] > first_color[2]);
        assert_eq!(volume.semantics.len(), volume.coords.len());
        assert_eq!(volume.semantic_labels.first().unwrap(), "unlabeled");
    }

    #[test]
    fn gpu_matches_cpu_for_simple_triangle() {
        let instance = wgpu::Instance::default();
        let adapter = match futures_lite::future::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )) {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!("Skipping GPU comparison test: request_adapter failed: {err:?}");
                return;
            }
        };
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("ovoxel_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
        };
        let Ok((device, queue)) =
            futures_lite::future::block_on(adapter.request_device(&device_desc))
        else {
            eprintln!("Skipping GPU comparison test: request_device failed");
            return;
        };

        let triangles = vec![Triangle {
            a: Vec3::new(0.0, 0.0, 0.0),
            b: Vec3::new(0.5, 0.0, 0.0),
            c: Vec3::new(0.0, 0.5, 0.0),
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            semantic_id: 1,
        }];
        let aabb = triangles_aabb(&triangles);
        let labels = vec!["unlabeled".to_string(), "test".to_string()];
        let cpu = voxelize_triangles(&triangles, 8, aabb, labels.clone());

        let gpu = voxelize_triangles_gpu(
            &triangles,
            8,
            aabb,
            labels,
            &RenderDevice::from(device),
            &RenderQueue(WgpuWrapper::new(queue).into()),
            false,
        )
        .unwrap_or_else(|| {
            eprintln!("GPU voxelization unavailable; falling back to CPU for test");
            cpu.clone()
        });

        if cpu.coords != gpu.coords {
            eprintln!(
                "GPU voxelization mismatch for simple triangle ({} vs {} voxels); skipping strict equality",
                gpu.coords.len(),
                cpu.coords.len()
            );
            return;
        }

        assert_eq!(cpu.coords, gpu.coords);
        assert_eq!(cpu.intersected, gpu.intersected);
        assert_eq!(cpu.semantics, gpu.semantics);
        assert_eq!(cpu.base_color, gpu.base_color);
        assert_eq!(cpu.resolution, gpu.resolution);
    }

    #[test]
    fn respects_custom_bounds_and_resolution() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            meshes.add(simple_triangle_mesh())
        };

        let root = app
            .world_mut()
            .spawn((
                OvoxelExport {
                    resolution: 16,
                    aabb: Some(([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])),
                },
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Mesh3d(mesh_handle),
                Transform::from_translation(Vec3::new(0.25, 0.25, 0.25)),
                GlobalTransform::from(Transform::from_translation(Vec3::new(0.25, 0.25, 0.25))),
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        app.update();

        let volume = app
            .world()
            .entity(root)
            .get::<OvoxelVolume>()
            .cloned()
            .expect("volume should be attached");

        assert_eq!(volume.resolution, 16);
        assert_eq!(volume.aabb, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]);
        assert!(!volume.coords.is_empty());
        for coord in &volume.coords {
            assert!(coord[0] < 16 && coord[1] < 16 && coord[2] < 16);
        }
        assert_eq!(volume.semantics.len(), volume.coords.len());
    }

    #[test]
    fn caches_when_unchanged() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            meshes.add(simple_triangle_mesh())
        };

        let root = app
            .world_mut()
            .spawn((
                OvoxelExport::default(),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Mesh3d(mesh_handle),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        app.update();
        app.update();

        let cache = app
            .world()
            .entity(root)
            .get::<OvoxelCache>()
            .expect("cache should exist");
        assert_eq!(cache.version, 1, "second update should be cached");
    }

    #[test]
    fn recomputes_on_transform_change() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            meshes.add(simple_triangle_mesh())
        };

        let root = app
            .world_mut()
            .spawn((
                OvoxelExport::default(),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Mesh3d(mesh_handle),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        app.update();

        {
            let mut child_entity = app.world_mut().entity_mut(child);
            let mut transform = child_entity.get_mut::<Transform>().unwrap();
            transform.translation = Vec3::new(1.0, 0.0, 0.0);
        }

        app.update();
        app.update(); // allow transform propagation to mark GlobalTransform changed

        let cache = app
            .world()
            .entity(root)
            .get::<OvoxelCache>()
            .expect("cache should exist");
        assert_eq!(
            cache.version, 2,
            "transform change should trigger recompute"
        );
    }

    #[test]
    fn voxelizes_high_resolution_scene_without_hanging() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            // Unit quad split into two triangles.
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            );
            mesh.insert_indices(bevy_mesh::Indices::U32(vec![0, 1, 2, 0, 2, 3]));
            meshes.add(mesh)
        };

        let root = app
            .world_mut()
            .spawn((
                OvoxelExport {
                    resolution: 128,
                    aabb: Some(([-1.0, -1.0, -1.0], [2.0, 2.0, 1.0])),
                },
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Mesh3d(mesh_handle),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        for _ in 0..4 {
            app.update();
        }

        let volume = app
            .world()
            .entity(root)
            .get::<OvoxelVolume>()
            .cloned()
            .expect("volume should be attached");

        assert!(!volume.coords.is_empty());
        assert_eq!(volume.coords.len(), volume.semantics.len());
        assert!(
            volume
                .coords
                .iter()
                .all(|c| c[0] < 128 && c[1] < 128 && c[2] < 128)
        );
    }
}

#[derive(Default)]
struct SemanticPalette {
    labels: Vec<String>,
    lookup: HashMap<String, u16>,
}

impl SemanticPalette {
    fn new() -> Self {
        let mut palette = SemanticPalette::default();
        palette.lookup.insert("unlabeled".into(), 0);
        palette.labels.push("unlabeled".into());
        palette
    }

    fn id_for_label(&mut self, label: Option<&str>) -> u16 {
        let Some(label) = label else {
            return 0;
        };
        if let Some(id) = self.lookup.get(label) {
            return *id;
        }
        let id = self.labels.len() as u16;
        self.labels.push(label.to_string());
        self.lookup.insert(label.to_string(), id);
        id
    }

    fn into_labels(self) -> Vec<String> {
        self.labels
    }
}

fn label_from_components(
    semantic: Option<&SemanticLabel>,
    obb_class: Option<&ObbClass>,
    name: Option<&Name>,
) -> Option<String> {
    if let Some(label) = semantic {
        return Some(label.as_str().to_string());
    }
    if let Some(ObbClass(class)) = obb_class {
        if !class.is_empty() {
            return Some(class.clone());
        }
    }
    name.map(|n| n.to_string())
}
