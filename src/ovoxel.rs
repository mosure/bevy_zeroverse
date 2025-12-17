use std::collections::HashMap;

use bevy::{prelude::*, render::render_resource::PrimitiveTopology};

use crate::{
    annotation::obb::ObbClass, app::BevyZeroverseConfig, render::semantic::SemanticLabel,
    scene::SceneAabbNode,
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

pub struct OvoxelPlugin;

impl Plugin for OvoxelPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<OvoxelExport>();
        app.register_type::<OvoxelVolume>();
        app.register_type::<OvoxelCache>();

        app.add_systems(Update, (tag_scene_roots, process_ovoxel_exports));
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
#[allow(clippy::type_complexity)]
pub fn process_ovoxel_exports(
    mut commands: Commands,
    roots: Query<(
        Entity,
        &OvoxelExport,
        Option<&OvoxelVolume>,
        Option<&OvoxelCache>,
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
    for (root, settings, existing_volume, cache) in roots.iter() {
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

        let aabb = settings
            .aabb
            .map(|(min, max)| [min, max])
            .unwrap_or_else(|| triangles_aabb(&triangles));

        let volume =
            voxelize_triangles(&triangles, settings.resolution, aabb, palette.into_labels());
        let version = cache.map(|c| c.version + 1).unwrap_or(1);
        commands
            .entity(root)
            .insert((volume, OvoxelCache { version }));
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
    material: Option<&MeshMaterial3d<StandardMaterial>>,
    materials: &Assets<StandardMaterial>,
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

    let color = {
        let c = material
            .and_then(|mat| materials.get(&mat.0))
            .map(|mat| mat.base_color)
            .unwrap_or(Color::WHITE)
            .to_linear();
        Vec4::new(c.red, c.green, c.blue, c.alpha)
    };

    let semantic_id =
        palette.id_for_label(label_from_components(semantic, obb_class, name).as_deref());

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

        tris.push(Triangle {
            a,
            b,
            c,
            color,
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
    let max = Vec3::from(aabb[1]);
    let extent = (max - min).max(Vec3::splat(f32::EPSILON));
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
    use bevy::{render::render_resource::PrimitiveTopology, MinimalPlugins};

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
