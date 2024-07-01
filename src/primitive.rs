use bevy::{
    prelude::*,
    math::primitives::{
        Capsule3d,
        Cuboid,
        Cylinder,
        Plane3d,
        Sphere,
        Torus,
    },
    pbr::wireframe::{
        Wireframe,
        WireframeColor,
        WireframePlugin,
    },
    render::{
        mesh::PrimitiveTopology,
        render_asset::RenderAssetUsages,
    }
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
        app.add_plugins(WireframePlugin);

        app.add_systems(Update, process_primitives);
    }
}


#[derive(Clone, Debug, EnumIter, Reflect)]
pub enum ZeroversePrimitives {
    Capsule,
    // Cone,
    Cuboid,
    Cylinder,
    Plane,
    Sphere,
    Torus,
}

#[derive(Clone, Component, Debug, Reflect)]
pub struct PrimitiveSettings {
    pub components: usize,
    pub available_types: Vec<ZeroversePrimitives>,
    pub available_operations: Vec<ManifoldOperations>,
    pub wireframe_probability: f32,
    pub scale_bound: Vec3,
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
            components: 1,
            available_types: ZeroversePrimitives::iter().collect(),
            available_operations: ManifoldOperations::iter().collect(),
            wireframe_probability: 0.2,
            scale_bound: Vec3::splat(1.0),
            position_bound: Vec3::splat(2.0),
        }
    }
}


#[derive(Clone, Component, Debug, Reflect)]
pub struct ZeroversePrimitive {
    pub composite: Mesh,
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
    let rng = &mut rand::thread_rng();

    // TODO: break this up into smaller functions
    for (entity, settings) in primitives.iter() {
        let composite = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());

        let primitive_types = choose_multiple_with_replacement(
            &settings.available_types,
            settings.components,
        );

        let scales = (0..settings.components)
            .map(|_| Vec3::new(
                rng.gen_range(0.0..settings.scale_bound.x),
                rng.gen_range(0.0..settings.scale_bound.y),
                rng.gen_range(0.0..settings.scale_bound.z),
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
                rng.gen_range(0.0..std::f32::consts::PI),
                rng.gen_range(0.0..std::f32::consts::PI),
                rng.gen_range(0.0..std::f32::consts::PI),
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
                    ZeroversePrimitives::Capsule => Capsule3d::new(scale.x, scale.y)
                        .mesh()
                        .latitudes(rng.gen_range(4..64))
                        .longitudes(rng.gen_range(4..64))
                        .rings(rng.gen_range(4..32))
                        .build(),
                    ZeroversePrimitives::Cuboid => Cuboid::from_size(scale)
                        .mesh(),
                    ZeroversePrimitives::Cylinder => Cylinder::new(scale.x, scale.y)
                        .mesh()
                        .resolution(rng.gen_range(4..64))
                        .segments(rng.gen_range(3..64))
                        .build(),
                    ZeroversePrimitives::Plane => Plane3d::new(Vec3::Y)
                        .mesh()
                        .size(scale.x, scale.y)
                        .build(),
                    ZeroversePrimitives::Sphere => Sphere::new(scale.x)
                        .mesh()
                        .ico(rng.gen_range(3..7))
                        .unwrap(),
                    ZeroversePrimitives::Torus => Torus::new(scale.x, scale.y)
                        .mesh()
                        .major_resolution(rng.gen_range(4..64))
                        .minor_resolution(rng.gen_range(4..64))
                        .build(),
                };

                let transform = Transform::from_translation(position).with_rotation(rotation);
                let mut mesh = mesh.transformed_by(transform);
                mesh.generate_tangents().unwrap();

                // TODO: optionally spawn the base primitive (no manifold operations) for debugging

                let mut primitive = commands.spawn((
                    PbrBundle {
                        mesh: meshes.add(mesh),
                        material: zeroverse_materials.materials.choose(rng).unwrap().clone(),
                        ..Default::default()
                    },
                ));

                if rng.gen_bool(settings.wireframe_probability as f64) {
                    // TODO: support textured wireframes
                    primitive.insert((
                        Wireframe,
                        WireframeColor {
                            color: Color::rgba(
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

        commands.entity(entity).insert(ZeroversePrimitive {
            composite,
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
