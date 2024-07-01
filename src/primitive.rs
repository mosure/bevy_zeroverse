use bevy::{
    prelude::*,
    math::primitives::{
        Capsule3d,
        Cone,
        Cuboid,
        Cylinder,
        Sphere,
        Torus,
    },
    render::{
        mesh::{
            PrimitiveTopology,
            SphereKind,
        },
        render_asset::RenderAssetUsages,
    }
};
use itertools::izip;
use rand::{
    Rng,
    seq::SliceRandom,
};

use crate::{
    material::ZeroverseMaterials,
    manifold::ManifoldOperations,
};


pub struct ZeroversePrimitivePlugin;
impl Plugin for ZeroversePrimitivePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, process_primitives);
    }
}


#[derive(Clone, Debug, Reflect)]
pub enum ZeroversePrimitives {
    Capsule,
    // Cone,
    Cuboid,
    Cylinder,
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


#[derive(Clone, Component, Debug, Reflect)]
pub struct ZeroversePrimitive {
    pub composite: Mesh,
}


fn process_primitives(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<ZeroverseMaterials>,
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
        // let mut composite = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());

        let primitive_types = settings.available_types
            .choose_multiple(rng, settings.components)
            .collect::<Vec<_>>();

        let scales = (0..settings.components)
            .map(|_| Vec3::new(
                rng.gen_range(0.0..settings.scale_bound.x),
                rng.gen_range(0.0..settings.scale_bound.y),
                rng.gen_range(0.0..settings.scale_bound.z),
            ))
            .collect::<Vec<_>>();

        let positions = (0..settings.components)
            .map(|_| Vec3::new(
                rng.gen_range(0.0..settings.position_bound.x),
                rng.gen_range(0.0..settings.position_bound.y),
                rng.gen_range(0.0..settings.position_bound.z),
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
                        .latitudes(rng.gen_range(4..16))
                        .longitudes(rng.gen_range(4..16))
                        .rings(rng.gen_range(4..16))
                        .build(),
                    ZeroversePrimitives::Cuboid => Cuboid::from_size(scale)
                        .mesh(),
                    ZeroversePrimitives::Cylinder => Cylinder::new(scale.x, scale.y)
                        .mesh()
                        .resolution(rng.gen_range(4..16))
                        .segments(rng.gen_range(3..16))
                        .build(),
                    ZeroversePrimitives::Sphere => Sphere::new(scale.x)
                        .mesh()
                        .ico(rng.gen_range(3..8))
                        .unwrap(),
                    ZeroversePrimitives::Torus => Torus::new(scale.x, scale.y)
                        .mesh()
                        .major_resolution(rng.gen_range(4..16))
                        .minor_resolution(rng.gen_range(4..16))
                        .build(),
                    // _ => panic!("unsupported primitive type: {:?}", primitive_type),
                };

                let transform = Transform::from_translation(position).with_rotation(rotation);
                let mesh = mesh.transformed_by(transform);

                // TODO: optionally spawn the base primitive (no manifold operations) for debugging

                
            }
        );
    }
}
