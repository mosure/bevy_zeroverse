use bevy::{
    prelude::*,
    math::primitives::{
        Capsule3d,
        Cone,
        ConicalFrustum,
        Cuboid,
        Cylinder,
        Plane3d,
        Sphere,
        Tetrahedron,
        Torus,
    },
    pbr::{
        TransmittedShadowReceiver,
        wireframe::{
            Wireframe,
            WireframeColor,
        },
    },
};
use itertools::izip;
use rand::{
    Rng,
    seq::SliceRandom,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::{
    material::ZeroverseMaterials,
    manifold::ManifoldOperations,
};


pub struct ZeroversePrimitivePlugin;
impl Plugin for ZeroversePrimitivePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<PrimitiveSettings>();

        #[cfg(not(target_family = "wasm"))]  // note: web does not handle `POLYGON_MODE_LINE`, so we skip wireframes
        app.add_plugins(bevy::pbr::wireframe::WireframePlugin);

        app.add_systems(Update, process_primitives);
    }
}


#[derive(Clone, Debug, EnumIter, Reflect)]
pub enum ZeroversePrimitives {
    Capsule,
    Cone,
    ConicalFrustum,
    Cuboid,
    Cylinder,
    Plane,
    Sphere,
    Tetrahedron,
    Torus,
}

#[derive(Clone, Component, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct PrimitiveSettings {
    pub components: usize,
    pub available_types: Vec<ZeroversePrimitives>,
    pub available_operations: Vec<ManifoldOperations>,
    pub wireframe_probability: f32,
    pub rotation_lower_bound: Vec3,
    pub rotation_upper_bound: Vec3,
    pub scale_lower_bound: Vec3,
    pub scale_upper_bound: Vec3,
    pub position_bound: Vec3,
}

impl PrimitiveSettings {
    pub fn count(n: usize) -> PrimitiveSettings {
        PrimitiveSettings {
            components: n,
            ..Default::default()
        }
    }
}

impl Default for PrimitiveSettings {
    fn default() -> PrimitiveSettings {
        PrimitiveSettings {
            components: 25,
            available_types: ZeroversePrimitives::iter().collect(),
            available_operations: ManifoldOperations::iter().collect(),
            wireframe_probability: 0.1,
            rotation_lower_bound: Vec3::splat(0.0),
            rotation_upper_bound: Vec3::splat(std::f32::consts::PI),
            scale_lower_bound: Vec3::splat(0.05),
            scale_upper_bound: Vec3::splat(1.0),
            position_bound: Vec3::splat(1.5),
        }
    }
}


#[derive(Bundle, Default, Debug)]
pub struct PrimitiveBundle {
    pub settings: PrimitiveSettings,
    pub spatial: SpatialBundle,
}


#[derive(Clone, Component, Debug, Reflect)]
pub struct ZeroversePrimitive;


fn build_primitive(
    commands: &mut ChildBuilder,
    settings: &PrimitiveSettings,
    meshes: &mut ResMut<Assets<Mesh>>,
    zeroverse_materials: &Res<ZeroverseMaterials>,
) {
    let rng = &mut rand::thread_rng();

    let primitive_types = choose_multiple_with_replacement(
        &settings.available_types,
        settings.components,
    );

    let scales = (0..settings.components)
        .map(|_| Vec3::new(
            rng.gen_range(settings.scale_lower_bound.x..settings.scale_upper_bound.x),
            rng.gen_range(settings.scale_lower_bound.y..settings.scale_upper_bound.y),
            rng.gen_range(settings.scale_lower_bound.z..settings.scale_upper_bound.z),
        ))
        .collect::<Vec<_>>();

    let positions = (0..settings.components)
        .map(|_| Vec3::new(
            rng.gen_range(-settings.position_bound.x..settings.position_bound.x),
            rng.gen_range(-settings.position_bound.y..settings.position_bound.y),
            rng.gen_range(-settings.position_bound.z..settings.position_bound.z),
        ))
        .collect::<Vec<_>>();

    let rotations = (0..settings.components)
        .map(|_| Quat::from_scaled_axis(Vec3::new(
            rng.gen_range(settings.rotation_lower_bound.x..settings.rotation_upper_bound.x),
            rng.gen_range(settings.rotation_lower_bound.y..settings.rotation_upper_bound.y),
            rng.gen_range(settings.rotation_lower_bound.z..settings.rotation_upper_bound.z),
        )))
        .collect::<Vec<_>>();

    izip!(
        primitive_types,
        scales,
        positions,
        rotations,
    ).for_each(
        |(
            primitive_type,
            scale,
            position,
            rotation,
        )| {
            let mesh = match primitive_type {
                ZeroversePrimitives::Capsule => Capsule3d::default()
                    .mesh()
                    .latitudes(rng.gen_range(4..64))
                    .longitudes(rng.gen_range(4..64))
                    .rings(rng.gen_range(4..32))
                    .build(),
                ZeroversePrimitives::Cone => Cone {
                        height: rng.gen_range(0.3..1.5),
                        radius: rng.gen_range(0.3..1.5),
                    }.mesh()
                    .resolution(rng.gen_range(3..64))
                    .build(),
                ZeroversePrimitives::ConicalFrustum => ConicalFrustum {
                        radius_bottom: rng.gen_range(0.3..1.5),
                        radius_top: rng.gen_range(0.3..1.5),
                        height: rng.gen_range(0.3..1.5),
                    }.mesh()
                    .resolution(rng.gen_range(3..64))
                    .build(),
                ZeroversePrimitives::Cuboid => Cuboid::default()
                    .mesh()
                    .build(),
                ZeroversePrimitives::Cylinder => Cylinder::default()
                    .mesh()
                    .resolution(rng.gen_range(4..64))
                    .segments(rng.gen_range(3..64))
                    .build(),
                ZeroversePrimitives::Plane => Plane3d::new(Vec3::Y, Vec2::ONE)
                    .mesh()
                    .subdivisions(rng.gen_range(0..16))
                    .build(),
                ZeroversePrimitives::Sphere => Sphere::default()
                    .mesh()
                    .uv(
                        rng.gen_range(24..64),
                        rng.gen_range(12..32),
                    ),
                ZeroversePrimitives::Tetrahedron => Tetrahedron::default()
                    .mesh()
                    .build(),
                ZeroversePrimitives::Torus => {
                    let inner_radius = rng.gen_range(0.01..1.0);
                    let outer_radius = inner_radius + rng.gen_range(0.01..1.0);

                    Torus::new(inner_radius, outer_radius)
                        .mesh()
                        .major_resolution(rng.gen_range(3..64))
                        .minor_resolution(rng.gen_range(3..64))
                        .build()
                },
            };

            let transform = Transform::from_translation(position)
                .with_rotation(rotation)
                .with_scale(scale);

            let mesh = mesh.transformed_by(transform);

            // TODO: optionally spawn the base primitive (no manifold operations) for debugging

            let mut primitive = commands.spawn((
                PbrBundle {
                    mesh: meshes.add(mesh),
                    material: zeroverse_materials.materials.choose(rng).unwrap().clone(),
                    ..Default::default()
                },
                TransmittedShadowReceiver,
            ));

            if rng.gen_bool(settings.wireframe_probability as f64) {
                // TODO: support textured wireframes
                primitive.insert((
                    Wireframe,
                    WireframeColor {
                        color: Color::srgba(
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                        ),
                    },
                ));
            }
        }
    );
}


fn process_primitives(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
    primitives: Query<
        (
            Entity,
            &PrimitiveSettings
        ),
        Without<ZeroversePrimitive>,
    >,
) {
    for (entity, settings) in primitives.iter() {
        commands.entity(entity)
            .insert(ZeroversePrimitive)
            .with_children(|subcommands| {
                build_primitive(
                    subcommands,
                    settings,
                    &mut meshes,
                    &zeroverse_materials,
                );
            });
    }
}


pub fn choose_multiple_with_replacement<T: Clone>(
    collection: &[T],
    n: usize,
) -> Vec<T> {
    let rng = &mut rand::thread_rng();
    (0..n)
        .map(|_| collection.choose(rng).unwrap().clone())
        .collect()
}