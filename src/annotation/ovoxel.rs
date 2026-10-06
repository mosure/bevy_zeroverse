pub mod contract;
mod scope;
pub use scope::{OvoxelExcluded, OvoxelRegion};

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{mpsc, Arc, Mutex, OnceLock},
};

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    ecs::change_detection::Tick,
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
    tasks::{block_on, AsyncComputeTaskPool, Task},
    transform::TransformSystems,
};
use bevy_burn_human::{BurnHumanInput, BurnHumanMeshMode, BurnHumanRenderMode};
use wgpu::{self, util::DeviceExt};

use crate::{
    annotation::obb::ObbClass,
    app::{BevyZeroverseConfig, OvoxelMode},
    render::semantic::SemanticLabel,
    sample::{CaptureFailure, CaptureReadiness, SamplerState, StartupDelay},
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

/// Marker for entities whose meshes should be included in O-Voxel exports.
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OvoxelTracked;

/// Minimal O-Voxel-like payload mirroring the TRELLIS fields we can compute on CPU.
#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component)]
pub struct OvoxelVolume {
    /// Unique integer voxel coordinates in lexicographic xyz order.
    pub coords: Vec<[u32; 3]>,
    /// Dual vertex offsets in voxel space, encoded to [0, 255].
    pub dual_vertices: Vec<[u8; 3]>,
    /// Triangle crossings of canonical +X/+Y/+Z edges from each voxel minimum corner.
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
pub(crate) struct OvoxelTask(Task<Result<(OvoxelVolume, OvoxelStatistics), String>>);

#[derive(Component, Debug, Clone, serde::Serialize)]
pub struct OvoxelStatistics {
    pub input_triangles: usize,
    pub clipped_triangles: usize,
    pub preparation_seconds: f64,
    pub elapsed_seconds: f64,
    pub backend: &'static str,
}

#[derive(Component, Debug)]
pub(crate) struct OvoxelFailure(String);

pub struct OvoxelPlugin;

pub const OVOXEL_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2c8cc5c9-a774-4e4c-b3a7-219894c3d2f2");

const GPU_TILE_SIZE: u32 = 4;
pub const GPU_DEFAULT_MAX_OUTPUT_VOXELS: u32 = 16_000_000;
const GPU_BUFFER_POOL_LIMIT: usize = 4;
const GPU_PARAMS_SIZE: u64 = 256;
const GPU_PREFIX_WG: u32 = 256;
const GPU_CLASSIFY_WG: u32 = 256;

fn is_ovoxel_enabled(config: Option<Res<crate::app::BevyZeroverseConfig>>) -> bool {
    config.is_none_or(|c| !matches!(c.ovoxel_mode, crate::app::OvoxelMode::Disabled))
}

fn gpu_shader_source(shaders: &Assets<Shader>) -> Option<Arc<str>> {
    shaders
        .get(&OVOXEL_SHADER_HANDLE)
        .map(|shader| Arc::<str>::from(shader.source.as_str()))
}

impl Plugin for OvoxelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Assets<Shader>>();
        load_internal_asset!(app, OVOXEL_SHADER_HANDLE, "ovoxel.wgsl", Shader::from_wgsl);

        app.add_message::<RegenerateSceneEvent>();
        app.register_type::<OvoxelExport>();
        app.register_type::<OvoxelTracked>();
        app.register_type::<OvoxelExcluded>();
        app.register_type::<OvoxelRegion>();
        app.register_type::<OvoxelVolume>();
        app.register_type::<OvoxelCache>();
        app.add_systems(
            PostUpdate,
            (
                tag_scene_roots,
                process_ovoxel_exports,
                collect_ovoxel_tasks,
                reset_ovoxel_on_regen,
            )
                .chain()
                .after(TransformSystems::Propagate)
                .run_if(is_ovoxel_enabled),
        );
        app.add_systems(PreUpdate, sync_burn_human_render_mode);
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

// Render-mode switches replace material handles, and cameras/lights move under
// the same root. Neither changes this semantic geometry representation.
type GeometryChanged = (
    With<Mesh3d>,
    Or<(
        Changed<Mesh3d>,
        Changed<Transform>,
        Changed<GlobalTransform>,
        Changed<SemanticLabel>,
        Changed<ObbClass>,
        Changed<Name>,
    )>,
);

impl Triangle {
    fn is_degenerate(&self) -> bool {
        // Compare squared area with a squared tolerance. EPSILON alone used to
        // discard centimetre-scale clothing and trim triangles.
        (self.a - self.b).cross(self.c - self.a).length_squared() <= f32::EPSILON * f32::EPSILON
    }
}

/// System: finds entities tagged with `OvoxelExport`, gathers meshes under
/// `OvoxelTracked` subtrees, voxelizes, and stores the result on the same entity
/// as `OvoxelVolume`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn process_ovoxel_exports(
    mut commands: Commands,
    config: Option<Res<BevyZeroverseConfig>>,
    gpu: (
        Option<Res<RenderDevice>>,
        Option<Res<RenderQueue>>,
        Local<Option<(Tick, Tick, Arc<GpuContext>)>>,
    ),
    sampler_state: Option<Res<SamplerState>>,
    startup_delay: Option<Res<StartupDelay>>,
    readiness: Option<Res<CaptureReadiness>>,
    roots: Query<(
        Entity,
        Ref<OvoxelExport>,
        Option<&OvoxelVolume>,
        Option<&OvoxelCache>,
        Option<&OvoxelTask>,
        Option<&OvoxelFailure>,
        Option<Ref<OvoxelRegion>>,
        Option<&GlobalTransform>,
    )>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    shaders: Res<Assets<Shader>>,
    mesh_query: Query<(
        &Mesh3d,
        Option<&MeshMaterial3d<StandardMaterial>>,
        &GlobalTransform,
        Option<&SemanticLabel>,
        Option<&ObbClass>,
        Option<&Name>,
    )>,
    mesh_changed: Query<Entity, GeometryChanged>,
    tracked: Query<(), With<OvoxelTracked>>,
    excluded: Query<(), With<OvoxelExcluded>>,
    children: Query<&Children>,
) {
    if readiness.as_ref().is_some_and(|r| !r.scene_ready())
        || config
            .as_ref()
            .is_some_and(|c| c.validate_ovoxel().is_err())
    {
        return;
    }
    if let (Some(state), Some(startup)) = (sampler_state.as_ref(), startup_delay.as_ref()) {
        if !state.enabled || !startup.done || state.warmup_frames > 0 || state.frames > 0 {
            return;
        }
    }

    let (render_device, render_queue, mut gpu_context) = gpu;
    // RenderDevice equality and stack addresses are not cross-Instance identities.
    // Own one cache per App/resource generation, including its matching queue.
    let gpu_context_owned = if config
        .as_ref()
        .is_some_and(|c| c.ovoxel_mode == OvoxelMode::GpuCompute)
    {
        match (render_device.as_ref(), render_queue.as_ref()) {
            (Some(device), Some(queue)) => {
                let generation = (device.last_changed(), queue.last_changed());
                if gpu_context
                    .as_ref()
                    .is_none_or(|(d, q, _)| (*d, *q) != generation)
                {
                    *gpu_context = Some((
                        generation.0,
                        generation.1,
                        Arc::new(GpuContext::new((**device).clone(), (**queue).clone())),
                    ));
                }
                gpu_context.as_ref().map(|(_, _, cache)| cache.clone())
            }
            _ => {
                *gpu_context = None;
                None
            }
        }
    } else {
        None
    };
    for (root, settings, existing_volume, cache, task, failed, region, transform) in roots.iter() {
        if task.is_some() || failed.is_some() {
            continue;
        }

        let needs_recompute = existing_volume.is_none()
            || cache.is_none()
            || settings.is_changed()
            || region.as_ref().is_some_and(|r| r.is_changed())
            || subtree_dirty(root, &mesh_changed, &children, &tracked, &excluded);

        if !needs_recompute {
            continue;
        }

        let preparation_started = bevy::platform::time::Instant::now();
        let mut palette = SemanticPalette::new();
        let mut triangles = Vec::new();

        let mut stack = vec![(root, false)];
        while let Some((entity, parent_tracked)) = stack.pop() {
            if excluded.contains(entity) {
                continue;
            }
            let is_tracked = tracked.contains(entity);
            let include = parent_tracked || is_tracked;

            if include {
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
            }

            if let Ok(child_list) = children.get(entity) {
                for child in child_list.iter() {
                    stack.push((child, include));
                }
            }
        }

        let input_triangles = triangles.len();
        let world_from_local =
            transform.map_or(bevy::math::Affine3A::IDENTITY, GlobalTransform::affine);
        if let Some(region) = &region {
            triangles = region.clip(triangles, world_from_local);
        }
        if triangles.is_empty() {
            commands
                .entity(root)
                .insert(OvoxelFailure("O-voxel scope contains no triangles".into()));
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
            .unwrap_or_else(|| {
                region.as_ref().map_or_else(
                    || triangles_aabb(&triangles),
                    |r| r.world_bounds(world_from_local),
                )
            });

        let labels = palette.into_labels();
        let task_pool = AsyncComputeTaskPool::get();
        let mode = config
            .as_ref()
            .map(|c| c.ovoxel_mode)
            .unwrap_or(OvoxelMode::CpuAsync);
        let max_output_voxels = config
            .as_ref()
            .map(|c| c.ovoxel_max_output_voxels)
            .unwrap_or(GPU_DEFAULT_MAX_OUTPUT_VOXELS);

        if matches!(mode, OvoxelMode::Disabled) {
            continue;
        }

        let shader = gpu_shader_source(&shaders);
        let gpu_context = gpu_context_owned.clone();
        let preparation_seconds = preparation_started.elapsed().as_secs_f64();
        let task = task_pool.spawn(async move {
            let started = bevy::platform::time::Instant::now();
            let clipped_triangles = triangles.len();
            let volume = match mode {
                OvoxelMode::CpuAsync => voxelize_triangles_bounded(&triangles, resolution, aabb, labels, max_output_voxels)?,
                OvoxelMode::GpuCompute => {
                    let (Some(gpu_context), Some(shader)) = (gpu_context, shader) else {
                        return Err("GPU O-voxel requires render device, queue and shader".into());
                    };
                    voxelize_triangles_gpu(&triangles, resolution, aabb, labels, shader, &gpu_context, max_output_voxels, false)
                        .ok_or_else(|| "GPU O-voxel failed: capacity exceeded or unsupported device; no partial volume exported".to_string())?
                }
                OvoxelMode::Disabled => unreachable!(),
            };
            Ok((volume, OvoxelStatistics { input_triangles, clipped_triangles, preparation_seconds,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                backend: if mode == OvoxelMode::GpuCompute { "gpu_compute" } else { "cpu_async" },
            }))
        });

        // Store the task so it can be polled to completion later.
        commands.entity(root).insert(OvoxelTask(task));
    }
}

fn collect_ovoxel_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut OvoxelTask)>,
    mut caches: Query<&mut OvoxelCache>,
    failures: Query<&OvoxelFailure>,
    mut capture_failure: Option<ResMut<CaptureFailure>>,
) {
    if let Some(failed) = failures.iter().next() {
        if let Some(failure) = capture_failure.as_mut() {
            failure.0 = Some(failed.0.clone());
        }
    }
    for (entity, mut task) in tasks.iter_mut() {
        if let Some(result) = block_on(futures_lite::future::poll_once(&mut task.0)) {
            let (volume, statistics) = match result {
                Ok(result) => result,
                Err(error) => {
                    if let Some(failure) = capture_failure.as_mut() {
                        failure.0 = Some(error.clone());
                    }
                    commands
                        .entity(entity)
                        .insert(OvoxelFailure(error))
                        .remove::<OvoxelTask>();
                    continue;
                }
            };
            let mut version = 1;
            if let Ok(mut cache) = caches.get_mut(entity) {
                cache.version = cache.version.saturating_add(1);
                version = cache.version;
            }
            commands
                .entity(entity)
                .insert((volume, statistics, OvoxelCache { version }))
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
            Or<(With<OvoxelVolume>, With<OvoxelTask>, With<OvoxelFailure>)>,
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
            .remove::<OvoxelStatistics>()
            .remove::<OvoxelFailure>()
            .remove::<OvoxelTask>();
    }
}

#[allow(clippy::type_complexity)]
fn subtree_dirty(
    root: Entity,
    changed: &Query<Entity, GeometryChanged>,
    children: &Query<&Children>,
    tracked: &Query<(), With<OvoxelTracked>>,
    excluded: &Query<(), With<OvoxelExcluded>>,
) -> bool {
    let mut stack = vec![(root, false)];
    while let Some((entity, parent_tracked)) = stack.pop() {
        if excluded.contains(entity) {
            continue;
        }
        let include = parent_tracked || tracked.contains(entity);
        if include && changed.contains(entity) {
            return true;
        }
        if let Ok(child_list) = children.get(entity) {
            for child in child_list.iter() {
                stack.push((child, include));
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
        .and_then(|attr| attr.as_float3());

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
    let count = mesh.indices().map_or(positions.len(), |i| i.len());
    let index = |i| match mesh.indices() {
        Some(bevy::mesh::Indices::U16(v)) => v[i] as usize,
        Some(bevy::mesh::Indices::U32(v)) => v[i] as usize,
        None => i,
    };
    let mut tris = Vec::with_capacity(count / 3);
    for first in (0..count.saturating_sub(2)).step_by(3) {
        let chunk = [index(first), index(first + 1), index(first + 2)];
        let a = affine.transform_point3(Vec3::from(positions[chunk[0]]));
        let b = affine.transform_point3(Vec3::from(positions[chunk[1]]));
        let c = affine.transform_point3(Vec3::from(positions[chunk[2]]));

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

// Two-sided Moller-Trumbore on a finite canonical grid edge. Coplanar edges
// are not crossings. Barycentric tolerance keeps shared triangle edges closed.
fn segment_crosses_triangle(origin: Vec3, delta: Vec3, tri: &Triangle) -> bool {
    let e1 = tri.b - tri.a;
    let e2 = tri.c - tri.a;
    let p = delta.cross(e2);
    let det = e1.dot(p);
    if det.abs() <= 1e-7 * e1.length() * e2.length() * delta.length() {
        return false;
    }
    let offset = origin - tri.a;
    let u = offset.dot(p) / det;
    let q = offset.cross(e1);
    let v = delta.dot(q) / det;
    let t = e2.dot(q) / det;
    u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 && (-1e-5..=1.00001).contains(&t)
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

fn nondegenerate_grid_aabb(mut aabb: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    // Retain supplied corners bit-for-bit unless an axis needs padding. The
    // round trip min + (max - min) can change max, separating voxel coordinates
    // from the scene annotation's otherwise identical reconstruction bounds.
    for axis in 0..3 {
        if aabb[1][axis] - aabb[0][axis] < 1e-3 {
            aabb[1][axis] = aabb[0][axis] + 1e-3;
        }
    }
    aabb
}

#[cfg(test)]
fn voxelize_triangles(
    triangles: &[Triangle],
    resolution: u32,
    aabb: [[f32; 3]; 2],
    semantic_labels: Vec<String>,
) -> OvoxelVolume {
    voxelize_triangles_bounded(triangles, resolution, aabb, semantic_labels, u32::MAX).unwrap()
}

fn voxelize_triangles_bounded(
    triangles: &[Triangle],
    resolution: u32,
    aabb: [[f32; 3]; 2],
    semantic_labels: Vec<String>,
    max_output: u32,
) -> Result<OvoxelVolume, String> {
    let aabb = nondegenerate_grid_aabb(aabb);
    let min = Vec3::from(aabb[0]);
    let extent = Vec3::from(aabb[1]) - min;
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
        if tri.is_degenerate() {
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
                    for axis in 0..3 {
                        let mut delta = Vec3::ZERO;
                        delta[axis] = voxel_size[axis];
                        if segment_crosses_triangle(voxel_min, delta, tri) {
                            mask |= 1 << axis;
                        }
                    }

                    if voxels.len() >= max_output as usize && !voxels.contains_key(&(x, y, z)) {
                        return Err(format!("CPU O-voxel capacity exceeded ({max_output}); no partial annotation returned"));
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
    // Keep coords lexicographically ordered so downstream consumers can skip resorting.
    keys.sort_unstable();

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

    Ok(OvoxelVolume {
        coords,
        dual_vertices,
        intersected,
        base_color,
        semantics,
        semantic_labels,
        resolution,
        aabb,
    })
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
    // Scalar start coordinates share the semantic vec4; end is vec4-aligned.
    // Computing these once also avoids CPU/GPU rounding differences at cells.
    start: [u32; 3],
    end: [u32; 4],
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
    max_dispatch: u32,
    _pad_params: [u32; 2],
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

/// A cache-owned identity remains alive in pooled buffers. Unlike a Device
/// wrapper address or context-local wgpu ID, it cannot alias another renderer.
#[derive(Clone)]
struct DeviceKey(Arc<()>);

impl PartialEq for DeviceKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for DeviceKey {}

#[derive(Clone, PartialEq, Eq)]
struct BufferKey {
    device_id: DeviceKey,
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
    meta_readback: wgpu::Buffer,
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

pub(crate) struct GpuContext {
    device: RenderDevice,
    queue: RenderQueue,
    key: DeviceKey,
    pipeline: Mutex<Option<(Arc<str>, Arc<GpuPipeline>)>>,
}

impl GpuContext {
    fn new(device: RenderDevice, queue: RenderQueue) -> Self {
        Self {
            device,
            queue,
            key: DeviceKey(Arc::new(())),
            pipeline: Mutex::new(None),
        }
    }

    fn pipeline(&self, source: Arc<str>) -> Arc<GpuPipeline> {
        let mut cache = self
            .pipeline
            .lock()
            .expect("ovoxel GPU pipeline cache poisoned");
        if let Some((cached_source, pipeline)) = cache.as_ref() {
            if cached_source == &source {
                return pipeline.clone();
            }
        }
        let pipeline = Arc::new(create_gpu_pipeline(self.device.wgpu_device(), &source));
        *cache = Some((source, pipeline.clone()));
        pipeline
    }
}

static GPU_BUFFER_POOL: OnceLock<Mutex<Vec<GpuBuffers>>> = OnceLock::new();

fn create_gpu_pipeline(device: &wgpu::Device, shader_source: &str) -> GpuPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ovoxel_gpu"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });

    let shared_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let classify_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ovoxel_gpu_classify_pl"),
        bind_group_layouts: &[
            Some(&shared_bind_group_layout),
            Some(&state_bind_group_layout),
        ],
        immediate_size: 0,
    });
    let work_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ovoxel_gpu_work_pl"),
        bind_group_layouts: &[
            Some(&shared_bind_group_layout),
            Some(&state_bind_group_layout),
            Some(&voxel_bind_group_layout),
        ],
        immediate_size: 0,
    });
    let prepare_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ovoxel_gpu_prepare_pl"),
        bind_group_layouts: &[
            Some(&shared_bind_group_layout),
            Some(&state_bind_group_layout),
            Some(&voxel_bind_group_layout),
            Some(&dispatch_bind_group_layout),
        ],
        immediate_size: 0,
    });

    let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ovoxel_gpu_classify_pipeline"),
        layout: Some(&classify_pipeline_layout),
        module: &shader,
        entry_point: Some("classify_tiles"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let prefix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ovoxel_gpu_prefix_pipeline"),
        layout: Some(&work_pipeline_layout),
        module: &shader,
        entry_point: Some("prefix_tiles"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ovoxel_gpu_prepare_pipeline"),
        layout: Some(&prepare_pipeline_layout),
        module: &shader,
        entry_point: Some("prepare_dispatch"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ovoxel_gpu_scatter_pipeline"),
        layout: Some(&work_pipeline_layout),
        module: &shader,
        entry_point: Some("scatter_pairs"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let voxel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
}

fn buffer_pool() -> &'static Mutex<Vec<GpuBuffers>> {
    GPU_BUFFER_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn make_buffer_key(
    device_id: &DeviceKey,
    tile_count: u64,
    pair_cap: u32,
    max_output: u32,
) -> BufferKey {
    BufferKey {
        device_id: device_id.clone(),
        tile_count,
        pair_cap,
        max_output,
    }
}

fn acquire_buffers(
    wgpu_device: &wgpu::Device,
    device_id: &DeviceKey,
    tile_count: u64,
    pair_cap: u32,
    max_output_voxels: u32,
) -> GpuBuffers {
    let key = make_buffer_key(device_id, tile_count, pair_cap, max_output_voxels);
    let mut pool = buffer_pool().lock().expect("gpu buffer pool poisoned");
    // Prefer exact match; otherwise reuse a superset (bigger buffers) for the same device/tile grid.
    if let Some(entry) = pool.iter().position(|b| b.key == key) {
        return pool.swap_remove(entry);
    }
    if let Some(entry) = pool.iter().position(|b| {
        b.key.device_id == key.device_id
            && b.key.tile_count == key.tile_count
            && b.key.max_output >= key.max_output
            && b.key.pair_cap >= key.pair_cap
    }) {
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
    // output_bytes would be meta + voxel bytes here if we needed the full span.
    // meta_bytes is reused below for readback sizing.

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
    let meta_readback = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ovoxel_meta_readback"),
        size: meta_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        meta_readback,
        readback,
    }
}

fn release_buffers(buffers: GpuBuffers) {
    let mut pool = buffer_pool().lock().expect("gpu buffer pool poisoned");
    pool.push(buffers);
    if pool.len() > GPU_BUFFER_POOL_LIMIT {
        let to_remove = pool.len().saturating_sub(GPU_BUFFER_POOL_LIMIT);
        pool.drain(0..to_remove);
    }
}

/// Clears pooled GPU buffers; useful to avoid holding VRAM across long runs/tests.
pub fn clear_gpu_buffer_pool() {
    buffer_pool()
        .lock()
        .expect("gpu buffer pool poisoned")
        .clear();
}

#[cfg(test)]
fn gpu_buffer_pool_len() -> usize {
    buffer_pool()
        .lock()
        .expect("gpu buffer pool poisoned")
        .len()
}

#[allow(clippy::too_many_arguments)]
fn voxelize_triangles_gpu(
    triangles: &[Triangle],
    resolution: u32,
    aabb: [[f32; 3]; 2],
    semantic_labels: Vec<String>,
    shader_source: Arc<str>,
    context: &GpuContext,
    max_output_voxels: u32,
    strict: bool,
) -> Option<OvoxelVolume> {
    macro_rules! gpu_bail {
        ($msg:expr) => {{
            if strict {
                panic!("{}", $msg);
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

    let aabb = nondegenerate_grid_aabb(aabb);
    let min = Vec3::from(aabb[0]);
    let extent = Vec3::from(aabb[1]) - min;
    let voxel_size = extent / resolution as f32;
    let half_diag = voxel_size.length() * 0.5;
    let max_output_voxels = max_output_voxels.max(1).min(voxel_count as u32);

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
        if t.is_degenerate() {
            continue;
        }
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
            start: [start.x as u32, start.y as u32, start.z as u32],
            end: [end.x as u32, end.y as u32, end.z as u32, 0],
        });
    }

    if gpu_tris.is_empty() {
        return Some(OvoxelVolume {
            semantic_labels,
            resolution,
            aabb,
            ..default()
        });
    }

    let pair_cap = pair_cap_estimate
        .saturating_add(pair_cap_estimate / 4 + tile_count)
        .clamp(1, 10_000_000) as u32;
    // Each pair can cover up to a whole 4^3 tile, not just one output cell.
    let max_output_voxels = max_output_voxels.min(pair_cap.saturating_mul(GPU_TILE_SIZE.pow(3)));

    let wgpu_device = context.device.wgpu_device();
    let limits = wgpu_device.limits();
    let binding_limit = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    let required_sizes = [
        std::mem::size_of_val(gpu_tris.as_slice()) as u64,
        tile_count * 12, // metadata
        tile_count * std::mem::size_of::<GpuActiveTile>() as u64,
        u64::from(pair_cap) * std::mem::size_of::<GpuTilePair>() as u64,
        std::mem::size_of::<GpuOutputMeta>() as u64
            + u64::from(max_output_voxels) * std::mem::size_of::<GpuVoxel>() as u64,
    ];
    if required_sizes.iter().any(|&size| size > binding_limit)
        || gpu_tris.len().div_ceil(GPU_CLASSIFY_WG as usize)
            > limits.max_compute_workgroups_per_dimension as usize
        || tile_count.div_ceil(u64::from(GPU_PREFIX_WG))
            > u64::from(limits.max_compute_workgroups_per_dimension)
    {
        gpu_bail!("O-voxel exceeds device buffer/dispatch limits; reduce resolution or output capacity, or use CPU");
    }
    let params = GpuParams {
        min: [min.x, min.y, min.z, 0.0],
        voxel: [voxel_size.x, voxel_size.y, voxel_size.z, 0.0],
        tile_dims: [tile_dims[0], tile_dims[1], tile_dims[2], 0],
        half_diag,
        resolution,
        tri_count: gpu_tris.len() as u32,
        max_output: max_output_voxels,
        pair_cap,
        max_dispatch: limits.max_compute_workgroups_per_dimension,
        _pad_params: [0; 2],
    };

    let wgpu_queue = &*context.queue.0;
    let max_storage = wgpu_device.limits().max_storage_buffers_per_shader_stage;
    // The prepare layout declares nine storage buffers even when an
    // individual entry point uses fewer. wgpu validates the complete layout.
    let required_storage = 9;
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

    let buffers = acquire_buffers(
        wgpu_device,
        &context.key,
        tile_count,
        pair_cap,
        max_output_voxels,
    );

    let pipeline = context.pipeline(shader_source);

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
        let tri_dispatch = params.tri_count.div_ceil(GPU_CLASSIFY_WG);
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
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());

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

    // Copy out just the metadata first to discover how many voxels we need to read back.
    voxel_encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.meta_readback, 0, meta_bytes);

    wgpu_queue.submit(Some(voxel_encoder.finish()));
    // Ensure GPU completes before readback.
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());

    // Map metadata slice.
    let meta_slice = buffers.meta_readback.slice(0..meta_bytes);
    type BufferAsync = Result<(), wgpu::BufferAsyncError>;
    let (tx_meta, rx_meta): (mpsc::Sender<BufferAsync>, mpsc::Receiver<BufferAsync>) =
        mpsc::channel();
    meta_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_meta.send(res);
    });
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());
    rx_meta.recv().ok().and_then(Result::ok)?;
    let meta_view = meta_slice.get_mapped_range();
    debug_assert_eq!(meta_view.len() as u64, meta_bytes);
    let meta: GpuOutputMeta = bytemuck::from_bytes::<GpuOutputMeta>(&meta_view).to_owned();
    let used = meta.count.min(max_output_voxels);
    let overflowed = meta.overflow > 0 || meta.count > max_output_voxels;
    static LOG_COUNTS: OnceLock<bool> = OnceLock::new();
    if *LOG_COUNTS.get_or_init(|| std::env::var("OVOXEL_LOG_COUNTS").is_ok()) {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!(
                "ovoxel gpu used={} overflow={} cap={}",
                used, meta.overflow, max_output_voxels
            );
        });
    }
    drop(meta_view);
    buffers.meta_readback.unmap();

    // A truncated sparse volume is not a valid annotation. Flags cover output,
    // tile-pair and per-cell semantic-vote capacity, including classify overflow.
    if overflowed {
        release_buffers(buffers);
        gpu_bail!(format!(
            "ovoxel GPU capacity exceeded (flags {}, cells {}, cap {}); no partial annotation returned",
            meta.overflow, meta.count, max_output_voxels
        ));
    }

    if used == 0 {
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

    // Copy only the voxel payload actually produced.
    let used_bytes = (used as u64).saturating_mul(std::mem::size_of::<GpuVoxel>() as u64);
    let voxel_range = meta_bytes..(meta_bytes + used_bytes);
    let mut copy_encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ovoxel_gpu_readback_copy_encoder"),
    });
    copy_encoder.copy_buffer_to_buffer(
        &buffers.output,
        meta_bytes,
        &buffers.readback,
        meta_bytes,
        used_bytes,
    );
    wgpu_queue.submit(Some(copy_encoder.finish()));
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());

    let slice = buffers.readback.slice(voxel_range.clone());
    let (tx, rx): (mpsc::Sender<BufferAsync>, mpsc::Receiver<BufferAsync>) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok().and_then(Result::ok)?;
    let data = slice.get_mapped_range();
    debug_assert_eq!(data.len() as u64, used_bytes);
    let voxels: &[GpuVoxel] = bytemuck::cast_slice(&data);
    let used = used.min(voxels.len() as u32) as usize;

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
    // Keep coords lexicographically ordered so CPU-side chunking can avoid extra sorts.
    packed.sort_unstable_by_key(|a| a.0);

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

// Cross-product barycentrics avoid the cancellation in differences of dot
// products for long, thin trim/floor triangles. Outside the face, compare the
// three finite edges directly; the result always lies on actual geometry.
fn closest_point_on_triangle(p: Vec3, tri: &Triangle) -> Vec3 {
    let ab = tri.b - tri.a;
    let ac = tri.c - tri.a;
    let ap = p - tri.a;
    let normal = ab.cross(ac);
    let area2 = normal.length_squared();
    if area2 > 0.0 {
        let u = ap.cross(ac).dot(normal);
        let v = ab.cross(ap).dot(normal);
        if u >= 0.0 && v >= 0.0 && u + v <= area2 {
            return p - normal * (ap.dot(normal) / area2);
        }
    }
    let closest_edge = |a: Vec3, b: Vec3| {
        let edge = b - a;
        a + edge
            * ((p - a).dot(edge) / edge.length_squared().max(f32::MIN_POSITIVE)).clamp(0.0, 1.0)
    };
    let mut best = closest_edge(tri.a, tri.b);
    for q in [closest_edge(tri.a, tri.c), closest_edge(tri.b, tri.c)] {
        if p.distance_squared(q) < p.distance_squared(best) {
            best = q;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::renderer::WgpuWrapper;
    use bevy::{render::render_resource::PrimitiveTopology, MinimalPlugins};

    #[test]
    fn grid_aabb_preserves_nondegenerate_corners_exactly() {
        let bounds = [[-2., -0.2, -10.], [2., 4., 0.1]];
        assert_eq!(nondegenerate_grid_aabb(bounds), bounds);
        let volume = voxelize_triangles(&[], 16, bounds, vec!["unlabeled".into()]);
        assert_eq!(volume.aabb, bounds);
        let planar = [[1., 0., -3.], [2., 0., 4.]];
        assert_eq!(
            nondegenerate_grid_aabb(planar),
            [[1., 0., -3.], [2., 0.001, 4.]]
        );
    }

    use std::{
        sync::{Mutex, OnceLock},
        thread,
    };

    static GPU_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn gpu_test_lock() -> &'static Mutex<()> {
        GPU_TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

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

    fn wait_for_volume(app: &mut App, root: Entity) -> OvoxelVolume {
        wait_for_volume_with_limit(app, root, 300)
    }

    fn wait_for_volume_with_limit(app: &mut App, root: Entity, limit: usize) -> OvoxelVolume {
        for _ in 0..limit {
            app.update();
            if let Some(v) = app.world().entity(root).get::<OvoxelVolume>() {
                return v.clone();
            }
            thread::yield_now();
        }
        panic!("volume should be attached after {limit} updates");
    }

    fn wait_for_cache_version(app: &mut App, root: Entity, min_version: u64) -> OvoxelCache {
        for _ in 0..300 {
            app.update();
            if let Some(cache) = app.world().entity(root).get::<OvoxelCache>() {
                if cache.version >= min_version {
                    return cache.clone();
                }
            }
            thread::yield_now();
        }
        panic!("cache version did not reach {min_version}");
    }

    #[test]
    fn grid_edge_flags_describe_crossings_not_triangle_bounds() {
        let floor = Triangle {
            a: Vec3::new(-1., 0.5, -1.),
            b: Vec3::new(2., 0.5, -1.),
            c: Vec3::new(-1., 0.5, 2.),
            color: Vec4::ONE,
            semantic_id: 1,
        };
        assert!(segment_crosses_triangle(Vec3::ZERO, Vec3::Y, &floor));
        assert!(!segment_crosses_triangle(Vec3::ZERO, Vec3::X, &floor));
        assert!(!segment_crosses_triangle(Vec3::ZERO, Vec3::Z, &floor));
        assert!(!segment_crosses_triangle(
            Vec3::new(2., 0., 2.),
            Vec3::Y,
            &floor
        ));
        assert!(!segment_crosses_triangle(
            Vec3::new(0., 0.5, 0.),
            Vec3::X,
            &floor
        ));
        let small = voxelize_triangles(
            &[floor],
            8,
            [[0.; 3], [1.; 3]],
            vec!["unlabeled".into(), "floor".into()],
        );
        assert!(small.intersected.contains(&2));
        assert!(small
            .intersected
            .iter()
            .all(|&flags| flags == 0 || flags == 2));
        assert!(voxelize_triangles_bounded(
            &[floor],
            8,
            [[0.; 3], [1.; 3]],
            vec!["unlabeled".into(), "floor".into()],
            1
        )
        .is_err());
    }

    #[test]
    fn closest_points_on_thin_trim_are_finite_and_stay_on_the_face() {
        for width in [0.0001, 0.001, 0.01] {
            let triangle = Triangle {
                a: Vec3::ZERO,
                b: Vec3::new(10., 0., 0.),
                c: Vec3::new(10., width, 0.),
                color: Vec4::ONE,
                semantic_id: 1,
            };
            let p = Vec3::new(7., width * 0.3, 0.2);
            let q = closest_point_on_triangle(p, &triangle);
            assert!(q.distance(Vec3::new(p.x, p.y, 0.)) < 1e-6);
            let outside = closest_point_on_triangle(Vec3::new(5., -1., 0.2), &triangle);
            assert!(outside.distance(Vec3::new(5., 0., 0.)) < 1e-6);
        }
    }

    #[test]
    fn excluded_subtrees_override_inherited_tracking_and_do_not_dirty_cache() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(OvoxelPlugin);
        app.add_systems(Update, crate::scene::create_scene_aabb);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(simple_triangle_mesh());
        let root = app
            .world_mut()
            .spawn((
                OvoxelExport {
                    resolution: 16,
                    aabb: None,
                },
                OvoxelTracked,
                SceneAabbNode,
                OvoxelRegion {
                    min: Vec3::splat(-1.),
                    max: Vec3::splat(2.),
                },
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut().spawn((
            Mesh3d(mesh.clone()),
            GlobalTransform::IDENTITY,
            SemanticLabel::Floor,
            ChildOf(root),
        ));
        let excluded = app.world_mut().spawn((OvoxelExcluded, ChildOf(root))).id();
        let outside = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                GlobalTransform::from_translation(Vec3::splat(100.)),
                SemanticLabel::Person,
                OvoxelTracked,
                ChildOf(excluded),
            ))
            .id();
        let volume = wait_for_volume(&mut app, root);
        assert_eq!(volume.aabb, [[-1.; 3], [2.; 3]]);
        let annotation = app.world().get::<crate::scene::SceneAabb>(root).unwrap();
        assert_eq!(
            volume.aabb,
            [annotation.min.to_array(), annotation.max.to_array()]
        );
        assert!(!volume.semantic_labels.iter().any(|l| l == "person"));
        for i in 0..8 {
            app.world_mut()
                .entity_mut(outside)
                .insert(GlobalTransform::from_translation(Vec3::splat(
                    100. + i as f32,
                )));
            app.update();
        }
        assert_eq!(app.world().get::<OvoxelCache>(root).unwrap().version, 1);
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
                OvoxelTracked,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        let volume = wait_for_volume(&mut app, root);

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
    fn gpu_matches_cpu_occupancy_votes_and_capacity_failures() {
        let _guard = gpu_test_lock().lock().expect("gpu test lock poisoned");
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
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 9,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::default(),
        };
        let Ok((device, queue)) =
            futures_lite::future::block_on(adapter.request_device(&device_desc))
        else {
            eprintln!("Skipping GPU comparison test: request_device failed");
            return;
        };

        let triangle = Triangle {
            a: Vec3::new(0.17, 0.19, 0.38),
            b: Vec3::new(0.73, 0.19, 0.38),
            c: Vec3::new(0.17, 0.81, 0.38),
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            semantic_id: 1,
        };
        let aabb = [[0.0; 3], [1.0; 3]];
        let labels = vec!["unlabeled".to_string(), "test".to_string()];

        let shader_source = {
            let mut shader_app = App::new();
            shader_app.init_resource::<Assets<Shader>>();
            load_internal_asset!(
                shader_app,
                OVOXEL_SHADER_HANDLE,
                "ovoxel.wgsl",
                Shader::from_wgsl
            );
            let shaders = shader_app
                .world()
                .get_resource::<Assets<Shader>>()
                .expect("Assets<Shader> missing in shader test app");
            gpu_shader_source(shaders)
                .expect("ovoxel shader asset should be available for GPU test")
        };

        let device = RenderDevice::from(device);
        let queue = RenderQueue(WgpuWrapper::new(queue).into());
        let context = GpuContext::new(device, queue);
        let run_gpu = |triangles: &[Triangle], cap| {
            voxelize_triangles_gpu(
                triangles,
                8,
                aabb,
                labels.clone(),
                shader_source.clone(),
                &context,
                cap,
                false,
            )
        };
        let other = Triangle {
            semantic_id: 2000,
            ..triangle
        };
        let tiny = Triangle {
            a: Vec3::splat(0.67),
            b: Vec3::new(0.68, 0.67, 0.67),
            c: Vec3::new(0.67, 0.68, 0.67),
            ..triangle
        };
        let degenerate = Triangle {
            a: triangle.b,
            ..triangle
        };
        for triangles in [
            vec![triangle],
            vec![triangle, other], // deterministic tie: larger semantic ID
            vec![other, triangle, triangle], // majority beats first hit
            vec![tiny],            // centimetre detail must survive the area test
            vec![triangle, degenerate],
            vec![degenerate],
        ] {
            let cpu = voxelize_triangles(&triangles, 8, aabb, labels.clone());
            for _ in 0..3 {
                let gpu = run_gpu(&triangles, GPU_DEFAULT_MAX_OUTPUT_VOXELS)
                    .expect("GPU path must succeed once the device has been created");
                assert_eq!(cpu.coords, gpu.coords);
                assert_eq!(cpu.intersected, gpu.intersected);
                assert_eq!(cpu.semantics, gpu.semantics);
                assert_eq!(cpu.base_color, gpu.base_color);
                assert_eq!(cpu.resolution, gpu.resolution);
            }
        }
        assert!(!voxelize_triangles(&[tiny], 8, aabb, labels.clone())
            .coords
            .is_empty());
        assert!(
            run_gpu(&[triangle], 1).is_none(),
            "output overflow must not return partial data"
        );
        let crowded: Vec<_> = (0..33)
            .map(|semantic_id| Triangle {
                semantic_id,
                ..triangle
            })
            .collect();
        assert!(
            run_gpu(&crowded, GPU_DEFAULT_MAX_OUTPUT_VOXELS).is_none(),
            "vote overflow must not return arbitrary labels"
        );
    }

    #[test]
    fn gpu_buffer_keys_identify_owned_contexts_not_wrapper_addresses() {
        let first = DeviceKey(Arc::new(()));
        let clone = first.clone();
        let other = DeviceKey(Arc::new(()));
        let first_key = make_buffer_key(&first, 64, 128, 256);
        let cloned_key = make_buffer_key(&clone, 64, 128, 256);
        let other_key = make_buffer_key(&other, 64, 128, 256);
        assert!(
            first_key == cloned_key,
            "clones retain the same renderer generation"
        );
        assert!(
            first_key != other_key,
            "identical sizes never cross renderer contexts"
        );
        drop(first);
        drop(clone);
        assert!(
            first_key != make_buffer_key(&DeviceKey(Arc::new(())), 64, 128, 256),
            "pooled ownership prevents token address reuse after its App drops"
        );
    }

    #[test]
    fn gpu_voxelization_reuses_only_its_renderer_context_and_shader() {
        let _guard = gpu_test_lock().lock().expect("gpu test lock poisoned");
        clear_gpu_buffer_pool();
        let make_context = || {
            let instance = wgpu::Instance::default();
            let adapter = futures_lite::future::block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                },
            ))
            .ok()?;
            let (device, queue) =
                futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("ovoxel_independent_renderer_test"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 9,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    trace: wgpu::Trace::default(),
                }))
                .ok()?;
            Some(GpuContext::new(
                RenderDevice::from(device),
                RenderQueue(WgpuWrapper::new(queue).into()),
            ))
        };
        let (Some(first), Some(second)) = (make_context(), make_context()) else {
            eprintln!("Skipping independent renderer test: GPU device unavailable");
            return;
        };
        let shader_source: Arc<str> = Arc::from(include_str!("ovoxel.wgsl"));
        let triangle = Triangle {
            a: Vec3::new(0.17, 0.19, 0.38),
            b: Vec3::new(0.73, 0.19, 0.38),
            c: Vec3::new(0.17, 0.81, 0.38),
            // Qualify resource ownership with exact binary colors away from
            // byte-quantization ties (CPU ties-even versus WGSL floor(x + 0.5)).
            color: Vec4::new(0.25, 0.625, 0.75, 1.0),
            semantic_id: 1,
        };
        let aabb = [[0.; 3], [1.; 3]];
        let labels = vec!["unlabeled".into(), "test".into()];
        let cpu = voxelize_triangles(&[triangle], 8, aabb, labels.clone());
        // Returning to the first renderer after the second catches global cache
        // reuse in either direction, including pooled buffers of identical size.
        for context in [&first, &second, &first, &second] {
            let gpu = voxelize_triangles_gpu(
                &[triangle],
                8,
                aabb,
                labels.clone(),
                shader_source.clone(),
                context,
                GPU_DEFAULT_MAX_OUTPUT_VOXELS,
                true,
            )
            .unwrap();
            assert_eq!(gpu.coords, cpu.coords);
            assert_eq!(gpu.base_color, cpu.base_color);
            assert_eq!(gpu.semantics, cpu.semantics);
            assert_eq!(gpu.intersected, cpu.intersected);
            assert_eq!(gpu.resolution, cpu.resolution);
            assert_eq!(gpu.aabb, cpu.aabb);
        }
        let original = first.pipeline(shader_source.clone());
        assert!(Arc::ptr_eq(
            &original,
            &first.pipeline(shader_source.clone())
        ));
        assert!(!Arc::ptr_eq(
            &original,
            &second.pipeline(shader_source.clone())
        ));
        let changed_source: Arc<str> =
            Arc::from(format!("{}\n// shader cache revision\n", shader_source));
        let changed = first.pipeline(changed_source.clone());
        assert!(
            !Arc::ptr_eq(&original, &changed),
            "source changes invalidate only this cache"
        );
        assert!(Arc::ptr_eq(&changed, &first.pipeline(changed_source)));
        clear_gpu_buffer_pool();
    }

    #[test]
    fn gpu_buffer_pool_trims() {
        let _guard = gpu_test_lock().lock().expect("gpu test lock poisoned");
        clear_gpu_buffer_pool();

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
                eprintln!("Skipping buffer pool trim test: request_adapter failed: {err:?}");
                return;
            }
        };
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("ovoxel_pool_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 9,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::default(),
        };
        let Ok((device, _queue)) =
            futures_lite::future::block_on(adapter.request_device(&device_desc))
        else {
            eprintln!("Skipping buffer pool trim test: request_device failed");
            return;
        };

        let device_key = DeviceKey(Arc::new(()));
        // Push more buffers than the cap and ensure we trim back down.
        for i in 0..(GPU_BUFFER_POOL_LIMIT as u64 + 2) {
            let buffers = acquire_buffers(&device, &device_key, 1 + i, 16, 16);
            release_buffers(buffers);
        }
        assert!(
            gpu_buffer_pool_len() <= GPU_BUFFER_POOL_LIMIT,
            "pool should trim to limit after releases"
        );

        clear_gpu_buffer_pool();
    }

    #[test]
    fn gpu_buffer_pool_reuses_supersets() {
        let _guard = gpu_test_lock().lock().expect("gpu test lock poisoned");
        clear_gpu_buffer_pool();

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
                eprintln!("Skipping buffer pool reuse test: request_adapter failed: {err:?}");
                return;
            }
        };
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("ovoxel_pool_reuse_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 9,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::default(),
        };
        let Ok((device, _queue)) =
            futures_lite::future::block_on(adapter.request_device(&device_desc))
        else {
            eprintln!("Skipping buffer pool reuse test: request_device failed");
            return;
        };

        let device_key = DeviceKey(Arc::new(()));
        let big = acquire_buffers(&device, &device_key, 64, 128, 256);
        release_buffers(big);
        assert_eq!(gpu_buffer_pool_len(), 1);

        // Smaller pair_cap with same max_output should reuse the stored buffers.
        let small = acquire_buffers(&device, &device_key, 64, 64, 256);
        assert_eq!(gpu_buffer_pool_len(), 0, "should pop from pool");
        release_buffers(small);
        assert_eq!(
            gpu_buffer_pool_len(),
            1,
            "after release pool holds one entry"
        );

        clear_gpu_buffer_pool();
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
                OvoxelTracked,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        let volume = wait_for_volume(&mut app, root);

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
                OvoxelTracked,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        wait_for_volume(&mut app, root);
        let cache = wait_for_cache_version(&mut app, root, 1);
        // Extra ticks should not bump the cache if nothing changed.
        app.update();
        app.update();
        let cache_after = app
            .world()
            .entity(root)
            .get::<OvoxelCache>()
            .expect("cache should exist after idle updates");
        assert_eq!(
            cache_after.version, cache.version,
            "idle updates should be cached"
        );

        let camera = app
            .world_mut()
            .spawn((Transform::IDENTITY, ChildOf(root)))
            .id();
        let untracked_mesh = app
            .world_mut()
            .spawn((Mesh3d::default(), Transform::IDENTITY, ChildOf(root)))
            .id();
        for i in 1..8 {
            // Annotation modes change PBR handles; camera motion and untracked
            // debug geometry must also leave the semantic geometry bake alone.
            app.world_mut()
                .entity_mut(child)
                .insert(MeshMaterial3d::<StandardMaterial>::default());
            for entity in [camera, untracked_mesh] {
                app.world_mut()
                    .entity_mut(entity)
                    .insert(Transform::from_xyz(i as f32, 0., 0.));
            }
            app.update();
            assert!(app.world().get::<OvoxelTask>(root).is_none());
            assert_eq!(
                app.world().get::<OvoxelCache>(root).unwrap().version,
                cache.version
            );
        }
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
                OvoxelTracked,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        wait_for_volume(&mut app, root);

        {
            let mut child_entity = app.world_mut().entity_mut(child);
            let mut transform = child_entity.get_mut::<Transform>().unwrap();
            transform.translation = Vec3::new(1.0, 0.0, 0.0);
        }

        // allow transform propagation to mark GlobalTransform changed
        app.update();
        let cache = wait_for_cache_version(&mut app, root, 2);
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
                OvoxelTracked,
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        let volume = wait_for_volume_with_limit(&mut app, root, 500);

        assert!(!volume.coords.is_empty());
        assert_eq!(volume.coords.len(), volume.semantics.len());
        assert!(volume
            .coords
            .iter()
            .all(|c| c[0] < 128 && c[1] < 128 && c[2] < 128));
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

fn sync_burn_human_render_mode(
    mut commands: Commands,
    config: Option<Res<BevyZeroverseConfig>>,
    parents: Query<&ChildOf>,
    tracked: Query<(), With<OvoxelTracked>>,
    humans: Query<(Entity, Option<&BurnHumanRenderMode>), With<BurnHumanInput>>,
) {
    let enabled = config.is_some_and(|c| !matches!(c.ovoxel_mode, OvoxelMode::Disabled));
    let target = if enabled {
        BurnHumanMeshMode::BakedMesh
    } else {
        BurnHumanMeshMode::SkinnedMesh
    };

    for (entity, current) in humans.iter() {
        if !has_ovoxel_scope(entity, &parents, &tracked) {
            continue;
        }

        let needs_update = current.is_none_or(|mode| mode.0 != target);
        if needs_update {
            commands.entity(entity).insert(BurnHumanRenderMode(target));
        }
    }
}

fn has_ovoxel_scope(
    mut entity: Entity,
    parents: &Query<&ChildOf>,
    tracked: &Query<(), With<OvoxelTracked>>,
) -> bool {
    loop {
        if tracked.contains(entity) {
            return true;
        }

        let Ok(parent) = parents.get(entity) else {
            return false;
        };
        entity = parent.parent();
    }
}
