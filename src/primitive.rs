use bevy::{
    asset::LoadState,
    gltf::{Gltf, GltfMesh},
    mesh::VertexAttributeValues,
    light::{NotShadowCaster, TransmittedShadowReceiver},
    math::{
        primitives::{
            Capsule3d, Cone, ConicalFrustum, Cuboid, Cylinder, Sphere, Tetrahedron, Torus,
        },
        sampling::ShapeSample,
    },
    pbr::wireframe::{Wireframe, WireframeColor},
    prelude::*,
    render::render_resource::Face,
};
use itertools::izip;
use rand::seq::IndexedRandom;
use rand::Rng;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::{
    annotation::obb::ObbTracked,
    manifold::ManifoldOperations,
    material::ZeroverseMaterials,
    mesh::{
        displace_vertices_with_noise, mesh_bounds, normalize_mesh_to_unit_cube, MeshCategory,
        NormalizeMeshesSet, ZeroverseMeshes,
    },
};

pub struct ZeroversePrimitivePlugin;
impl Plugin for ZeroversePrimitivePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ZeroversePrimitiveSettings>();

        #[cfg(not(target_family = "wasm"))]
        // note: web does not handle `POLYGON_MODE_LINE`, so we skip wireframes
        app.add_plugins(bevy::pbr::wireframe::WireframePlugin::default());

        app.add_systems(Update, process_primitives.after(NormalizeMeshesSet));
    }
}

#[derive(Clone, Debug, EnumIter, Reflect)]
pub enum ZeroversePrimitives {
    Capsule,
    Cone,
    ConicalFrustum,
    Cuboid,
    Cylinder,
    Mesh(MeshCategory),
    Plane,
    Sphere,
    Tetrahedron,
    Torus,
}

// TODO: support scale and rotation pdfs via https://github.com/villor/bevy_lookup_curve
#[derive(Clone, Component, Debug, Reflect)]
#[require(Transform, Visibility)]
pub struct ZeroversePrimitiveSettings {
    pub components: CountSampler,
    pub available_materials: Option<Vec<Handle<StandardMaterial>>>,
    pub available_operations: Vec<ManifoldOperations>,
    pub available_types: Vec<ZeroversePrimitives>,
    pub wireframe_probability: f32,
    pub noise_probability: f32,
    pub smooth_normals_probability: f32,
    pub noise_frequency_lower_bound: f32,
    pub noise_frequency_upper_bound: f32,
    pub noise_scale_lower_bound: f32,
    pub noise_scale_upper_bound: f32,
    pub rotation_sampler: RotationSampler,
    pub position_sampler: PositionSampler,
    pub scale_sampler: ScaleSampler,
    pub invert_normals: bool,
    pub track_obb: bool,
    #[reflect(ignore)]
    pub cull_mode: Option<Face>,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub preferred_mesh: Option<Handle<Mesh>>,
    /// When true, scale.y is treated as the target height and x/z are treated as aspect multipliers.
    pub height_preserve_scale: bool,
    /// When true, scale to exactly match the sampled bounds on all axes (per-axis fit).
    pub fit_to_sampled_box: bool,
}

impl Default for ZeroversePrimitiveSettings {
    fn default() -> ZeroversePrimitiveSettings {
        ZeroversePrimitiveSettings {
            components: CountSampler::Bounded(4, 6),
            available_materials: None,
            available_operations: ManifoldOperations::iter().collect(),
            available_types: vec![
                ZeroversePrimitives::Capsule,
                ZeroversePrimitives::Cone,
                ZeroversePrimitives::ConicalFrustum,
                ZeroversePrimitives::Cuboid,
                ZeroversePrimitives::Cylinder,
                ZeroversePrimitives::Sphere,
                ZeroversePrimitives::Tetrahedron,
                ZeroversePrimitives::Torus,
            ],
            wireframe_probability: 0.0,
            noise_probability: 0.3,
            smooth_normals_probability: 1.0,
            noise_frequency_lower_bound: 0.5,
            noise_frequency_upper_bound: 2.0,
            noise_scale_lower_bound: 0.0,
            noise_scale_upper_bound: 0.6,
            rotation_sampler: RotationSampler::Random,
            position_sampler: PositionSampler::default(),
            scale_sampler: ScaleSampler::Bounded(Vec3::splat(0.05), Vec3::splat(1.0)),
            invert_normals: false,
            track_obb: true,
            cull_mode: None,
            cast_shadows: true,
            receive_shadows: true, // TODO: fix shadows + camera trajectories clipping
            preferred_mesh: None,
            height_preserve_scale: false,
            fit_to_sampled_box: true,
        }
    }
}

#[derive(Clone, Debug, Reflect)]
pub enum CountSampler {
    Bounded(usize, usize),
    Exact(usize),
}

impl CountSampler {
    pub fn sample(&self) -> usize {
        let mut rng = rand::rng();

        match *self {
            CountSampler::Bounded(lower_bound, upper_bound) => {
                rng.random_range(lower_bound..upper_bound)
            }
            CountSampler::Exact(count) => count,
        }
    }
}

#[derive(Clone, Debug, Reflect)]
pub enum RotationSampler {
    Bounded { max: Vec3, min: Vec3 },
    Exact(Quat),
    Identity,
    Random,
}

impl RotationSampler {
    pub fn sample(&self) -> Quat {
        let mut rng = rand::rng();

        match *self {
            RotationSampler::Bounded { min, max } => {
                if min == max {
                    Quat::IDENTITY
                } else {
                    Quat::from_scaled_axis(Vec3::new(
                        if min.x != max.x {
                            rng.random_range(min.x..max.x)
                        } else {
                            0.0
                        },
                        if min.y != max.y {
                            rng.random_range(min.y..max.y)
                        } else {
                            0.0
                        },
                        if min.z != max.z {
                            rng.random_range(min.z..max.z)
                        } else {
                            0.0
                        },
                    ))
                }
            }
            RotationSampler::Exact(rotation) => rotation,
            RotationSampler::Identity => Quat::IDENTITY,
            RotationSampler::Random => Quat::from_rng(&mut rng),
        }
    }
}

#[derive(Clone, Debug, Reflect)]
pub enum ScaleSampler {
    Bounded(Vec3, Vec3),
    Exact(Vec3),
}

impl ScaleSampler {
    pub fn sample(&self) -> Vec3 {
        let mut rng = rand::rng();

        match *self {
            ScaleSampler::Bounded(lower_bound, upper_bound) => Vec3::new(
                rng.random_range(lower_bound.x..=upper_bound.x),
                rng.random_range(lower_bound.y..=upper_bound.y),
                rng.random_range(lower_bound.z..=upper_bound.z),
            ),
            ScaleSampler::Exact(scale) => scale,
        }
    }
}

#[derive(Clone, Debug, Reflect)]
pub enum PositionSampler {
    Band { size: Vec3 },
    Capsule { radius: f32, length: f32 },
    Cube { extents: Vec3 },
    Cylinder { radius: f32, height: f32 },
    Exact { position: Vec3 },
    Origin,
    Sphere { radius: f32 },
}

impl Default for PositionSampler {
    fn default() -> PositionSampler {
        PositionSampler::Cube {
            extents: Vec3::splat(0.5),
        }
    }
}

impl PositionSampler {
    pub fn is_valid(&self) -> bool {
        match *self {
            PositionSampler::Band { size } => size.x > 0.0 && size.y > 0.0 && size.z > 0.0,
            PositionSampler::Capsule { radius, length } => radius > 0.0 && length > 0.0,
            PositionSampler::Cube { extents } => {
                extents.x > 0.0 && extents.y > 0.0 && extents.z > 0.0
            }
            PositionSampler::Cylinder { radius, height } => radius > 0.0 && height > 0.0,
            PositionSampler::Exact { position: _ } => true,
            PositionSampler::Origin => true,
            PositionSampler::Sphere { radius } => radius > 0.0,
        }
    }

    pub fn sample(&self) -> Vec3 {
        let mut rng = rand::rng();

        match *self {
            PositionSampler::Band { size } => {
                let face = rng.random_range(0..4);

                let (x, z) = match face {
                    0 => (-size.x / 2.0, rng.random_range(-size.z / 2.0..size.z / 2.0)),
                    1 => (size.x / 2.0, rng.random_range(-size.z / 2.0..size.z / 2.0)),
                    2 => (rng.random_range(-size.x / 2.0..size.x / 2.0), -size.z / 2.0),
                    3 => (rng.random_range(-size.x / 2.0..size.x / 2.0), size.z / 2.0),
                    _ => unreachable!(),
                };

                let y = rng.random_range(-size.y / 2.0..size.y / 2.0);
                Vec3::new(x, y, z)
            }
            PositionSampler::Capsule { radius, length } => {
                Capsule3d::new(radius, length).sample_interior(&mut rng)
            }
            PositionSampler::Cube { extents } => {
                Cuboid::from_size(extents).sample_interior(&mut rng)
            }
            PositionSampler::Cylinder { radius, height } => {
                Cylinder::new(radius, height).sample_interior(&mut rng)
            }
            PositionSampler::Exact { position } => position,
            PositionSampler::Origin => Vec3::ZERO,
            PositionSampler::Sphere { radius } => Sphere::new(radius).sample_interior(&mut rng),
        }
    }
}

#[derive(Clone, Component, Debug, Reflect)]
pub struct ZeroversePrimitive;

fn build_primitive(
    commands: &mut ChildSpawnerCommands,
    settings: &ZeroversePrimitiveSettings,
    meshes: &mut ResMut<Assets<Mesh>>,
    standard_materials: &mut ResMut<Assets<StandardMaterial>>,
    zeroverse_materials: &Res<ZeroverseMaterials>,
    zeroverse_meshes: &mut ResMut<ZeroverseMeshes>,
    gltfs: &Assets<Gltf>,
    gltf_meshes: &Assets<GltfMesh>,
    asset_server: &AssetServer,
) {
    let mut rng = rand::rng();

    let components = settings.components.sample();

    let primitive_types = choose_multiple_with_replacement(&settings.available_types, components);

    let scales = (0..components)
        .map(|_| settings.scale_sampler.sample())
        .collect::<Vec<_>>();

    let positions = (0..components)
        .map(|_| settings.position_sampler.sample())
        .collect::<Vec<_>>();

    let rotations = (0..components)
        .map(|_| settings.rotation_sampler.sample())
        .collect::<Vec<_>>();

    izip!(primitive_types, scales, positions, rotations,).for_each(
        |(primitive_type, scale, position, rotation)| {
            let mut material = zeroverse_materials
                .materials
                .choose(&mut rng)
                .cloned()
                .unwrap_or(standard_materials.add(StandardMaterial::default()));

            if settings.cull_mode.is_some() {
                let mut new_material = standard_materials.get(&material).unwrap().clone();

                new_material.double_sided = false;
                new_material.cull_mode = settings.cull_mode;

                material = standard_materials.add(new_material);
            }

            let mut chosen_mesh_meta: Option<(Handle<Mesh>, Vec3)> = None;

            let mut mesh = match primitive_type {
                ZeroversePrimitives::Capsule => Capsule3d::default()
                    .mesh()
                    .latitudes(rng.random_range(4..64))
                    .longitudes(rng.random_range(4..64))
                    .rings(rng.random_range(4..32))
                    .build(),
                ZeroversePrimitives::Cone => Cone {
                    height: rng.random_range(0.3..1.5),
                    radius: rng.random_range(0.3..1.5),
                }
                .mesh()
                .resolution(rng.random_range(3..64))
                .build(),
                ZeroversePrimitives::ConicalFrustum => ConicalFrustum {
                    radius_bottom: rng.random_range(0.3..1.5),
                    radius_top: rng.random_range(0.3..1.5),
                    height: rng.random_range(0.3..1.5),
                }
                .mesh()
                .resolution(rng.random_range(3..64))
                .build(),
                ZeroversePrimitives::Cuboid => Cuboid::default().mesh().build(),
                ZeroversePrimitives::Cylinder => Cylinder::default()
                    .mesh()
                    .resolution(rng.random_range(4..64))
                    .segments(rng.random_range(3..64))
                    .build(),
                ZeroversePrimitives::Plane => Plane3d::new(Vec3::Y, Vec2::ONE)
                    .mesh()
                    .subdivisions(rng.random_range(0..16))
                    .build(),
                ZeroversePrimitives::Sphere => Sphere::default()
                    .mesh()
                    .uv(rng.random_range(24..64), rng.random_range(12..32)),
                ZeroversePrimitives::Tetrahedron => Tetrahedron::default().mesh().build(),
                ZeroversePrimitives::Torus => {
                    let inner_radius = rng.random_range(0.01..1.0);
                    let outer_radius = inner_radius + rng.random_range(0.01..1.0);

                    Torus::new(inner_radius, outer_radius)
                        .mesh()
                        .major_resolution(rng.random_range(3..64))
                        .minor_resolution(rng.random_range(3..64))
                        .build()
                }
                ZeroversePrimitives::Mesh(ref category) => {
                    let mesh_choice = settings
                        .preferred_mesh
                        .as_ref()
                        .and_then(|preferred| {
                            zeroverse_meshes
                                .meshes
                                .get(category)
                                .and_then(|mesh_vec| {
                                    mesh_vec.iter().find(|m| m.handle == *preferred)
                                })
                        })
                        .or_else(|| {
                            zeroverse_meshes
                                .meshes
                                .get(category)
                                .and_then(|mesh_vec| mesh_vec.choose(&mut rng))
                        });

                    let mesh_choice = mesh_choice.map(|mesh| {
                        (
                            mesh.handle.clone(),
                            mesh.material.clone(),
                            mesh.gltf.clone(),
                        )
                    });

                    match mesh_choice {
                        Some((mesh_handle, mut chosen_material, mesh_gltf)) => {
                            if chosen_material.is_none() {
                                chosen_material = resolve_mesh_material(
                                    &mesh_handle,
                                    &mesh_gltf,
                                    gltfs,
                                    gltf_meshes,
                                );
                            }

                            if let Some(mesh_material) = chosen_material.as_ref() {
                                match asset_server.get_load_state(mesh_material) {
                                    Some(LoadState::Loaded) => {
                                        material = mesh_material.clone();
                                    }
                                    Some(LoadState::Failed(err)) => {
                                        debug!(
                                            "material load failed for mesh {:?}: {err}",
                                            mesh_handle
                                        );
                                    }
                                    None => {
                                        debug!(
                                            "material handle missing load state for mesh {:?}",
                                            mesh_handle
                                        );
                                    }
                                    _ => {}
                                }
                            } else {
                                debug!(
                                    "no material found for mesh {:?}, keeping random material",
                                    mesh_handle
                                );
                            }

                            if !zeroverse_meshes.normalized.contains(&mesh_handle) {
                                if let Some(mesh_asset) = meshes.get_mut(&mesh_handle) {
                                    if let Some(size) = normalize_mesh_to_unit_cube(mesh_asset) {
                                        zeroverse_meshes
                                            .original_sizes
                                            .insert(mesh_handle.clone(), size);
                                        zeroverse_meshes.normalized.insert(mesh_handle.clone());
                                    }
                                }
                            }

                            if let Some(size) = zeroverse_meshes.original_sizes.get(&mesh_handle) {
                                chosen_mesh_meta = Some((mesh_handle.clone(), *size));
                            }

                            meshes
                                .get(&mesh_handle)
                                .cloned()
                                .unwrap_or_else(|| Cuboid::default().mesh().build())
                        }
                        None => Cuboid::default().mesh().build(),
                    }
                }
            };

            if matches!(primitive_type, ZeroversePrimitives::Mesh(_)) {
                normalize_mesh_to_unit_cube(&mut mesh);
            }

            if rng.random_bool(settings.noise_probability as f64) {
                let noise_frequency = rng.random_range(
                    settings.noise_frequency_lower_bound..settings.noise_frequency_upper_bound,
                );
                let noise_scale = rng.random_range(
                    settings.noise_scale_lower_bound..settings.noise_scale_upper_bound,
                );
                displace_vertices_with_noise(&mut mesh, noise_frequency, noise_scale);
            }

            let mut position = position;
            let mut final_scale = scale;

            if let Some((min, max)) = mesh_bounds(&mesh) {
                let orig_size = if let Some((_, original_size)) = chosen_mesh_meta.as_ref() {
                    *original_size
                } else {
                    max - min
                };
                let safe_size = Vec3::new(
                    orig_size.x.max(f32::EPSILON),
                    orig_size.y.max(f32::EPSILON),
                    orig_size.z.max(f32::EPSILON),
                );
                let max_extent = safe_size.max_element();

                if settings.height_preserve_scale {
                    // Preserve the original mesh aspect: target height comes from scale.y,
                    // width/depth are derived from original aspect and modulated by scale.x/scale.z.
                    let desired_height = scale.y;
                    let desired_width = desired_height * (safe_size.x / safe_size.y) * scale.x;
                    let desired_depth = desired_height * (safe_size.z / safe_size.y) * scale.z;

                    final_scale = Vec3::new(
                        desired_width * max_extent / safe_size.x,
                        desired_height * max_extent / safe_size.y,
                        desired_depth * max_extent / safe_size.z,
                    );
                } else if settings.fit_to_sampled_box {
                    // Exact per-axis fit: hit the sampled width/height/depth.
                    final_scale = Vec3::new(
                        scale.x * max_extent / safe_size.x,
                        scale.y * max_extent / safe_size.y,
                        scale.z * max_extent / safe_size.z,
                    );
                }

                if min.y != 0.0 {
                    position.y -= min.y * final_scale.y;
                }
                if position.y < 0.0 {
                    position.y = 0.0;
                }
            }

            let transform = Transform::from_translation(position)
                .with_rotation(rotation)
                .with_scale(final_scale);

            if rng.random_bool(settings.smooth_normals_probability as f64) {
                mesh.compute_smooth_normals();
            } else {
                mesh.duplicate_vertices();
                mesh.compute_flat_normals();
            }
            // with the current texture resolution, generating tangents produces too much noise
            // mesh.generate_tangents().unwrap();

            if settings.invert_normals {
                if let Some(VertexAttributeValues::Float32x3(ref mut normals)) =
                    mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
                {
                    normals.iter_mut().for_each(|normal| {
                        normal[0] = -normal[0];
                        normal[1] = -normal[1];
                        normal[2] = -normal[2];
                    });
                }
            }

            let mut primitive = commands.spawn((
                transform,
                GlobalTransform::default(),
                InheritedVisibility::default(),
                Visibility::default(),
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                TransmittedShadowReceiver,
            ));
            if settings.track_obb {
                primitive.insert(ObbTracked);
            }

            if !settings.cast_shadows {
                primitive.insert(NotShadowCaster);
            }

            if !settings.receive_shadows {
                primitive.insert(TransmittedShadowReceiver);
            }

            if rng.random_bool(settings.wireframe_probability as f64) {
                primitive.remove::<MeshMaterial3d<StandardMaterial>>();

                // TODO: support textured wireframes
                primitive.insert((
                    Wireframe,
                    WireframeColor {
                        color: Color::srgba(
                            rng.random_range(0.0..1.0),
                            rng.random_range(0.0..1.0),
                            rng.random_range(0.0..1.0),
                            rng.random_range(0.0..1.0),
                        ),
                    },
                ));
            }
        },
    );
}

pub fn process_primitives(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
    mut zeroverse_meshes: ResMut<ZeroverseMeshes>,
    gltfs: Res<Assets<Gltf>>,
    gltf_meshes: Res<Assets<GltfMesh>>,
    asset_server: Res<AssetServer>,
    primitives: Query<(Entity, &ZeroversePrimitiveSettings), Without<ZeroversePrimitive>>,
) {
    for (entity, settings) in primitives.iter() {
        commands
            .entity(entity)
            .insert(ZeroversePrimitive)
            .with_children(|subcommands| {
                build_primitive(
                    subcommands,
                    settings,
                    &mut meshes,
                    &mut standard_materials,
                    &zeroverse_materials,
                    &mut zeroverse_meshes,
                    &gltfs,
                    &gltf_meshes,
                    &asset_server,
                );
            });
    }
}

pub fn choose_multiple_with_replacement<T: Clone>(collection: &[T], n: usize) -> Vec<T> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| collection.choose(&mut rng).unwrap().clone())
        .collect()
}

fn resolve_mesh_material(
    mesh_handle: &Handle<Mesh>,
    mesh_gltf: &Handle<Gltf>,
    gltfs: &Assets<Gltf>,
    gltf_meshes: &Assets<GltfMesh>,
) -> Option<Handle<StandardMaterial>> {
    let gltf = gltfs.get(mesh_gltf)?;
    if gltf.materials.is_empty() {
        return None;
    }

    for gltf_mesh_handle in &gltf.meshes {
        if let Some(gltf_mesh) = gltf_meshes.get(gltf_mesh_handle) {
            for primitive in &gltf_mesh.primitives {
                if primitive.mesh == *mesh_handle {
                    return primitive.material.clone();
                }
            }
        }
    }
    None
}
