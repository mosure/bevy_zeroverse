//! Room-local reflection volumes using Bevy's box-projected cubemaps. Bounded
//! by the largest three zones, independent of camera count and material count.
use super::*;

pub struct ReflectionVolume {
    pub transform: Transform,
    pub map: EnvironmentMapLight,
}

pub(super) fn plans(
    scene: &IndoorManifest,
    transport: &super::super::gi::BakeScene,
) -> Vec<Transform> {
    // WebGPU lacks Bevy's texture binding arrays for localized probes. Retain
    // the already-baked view environment without paying for unusable maps.
    if cfg!(target_arch = "wasm32") {
        return Vec::new();
    }
    let Some(program) = &scene.program else {
        return Vec::new();
    };
    let mut zones: Vec<_> = program.zones.iter().collect();
    zones.sort_by(|a, b| {
        (b.max - b.min)
            .element_product()
            .total_cmp(&(a.max - a.min).element_product())
    });
    let mut volumes = Vec::new();
    for zone in zones.into_iter().take(3) {
        let center = (zone.min + zone.max) * 0.5;
        let mut extent = zone.max - zone.min;
        if let Some(envelope) = &scene.envelope {
            for _ in 0..8 {
                if super::super::envelope::polygon::box_inside(
                    &envelope.footprint,
                    center - extent * 0.5,
                    center + extent * 0.5,
                    0.04,
                ) {
                    break;
                }
                extent *= 0.86;
            }
            if !super::super::envelope::polygon::box_inside(
                &envelope.footprint,
                center - extent * 0.5,
                center + extent * 0.5,
                0.04,
            ) {
                continue;
            }
        }
        let floor = scene.floor_height(center);
        let ceiling = [
            Vec2::new(-1., -1.),
            Vec2::new(-1., 1.),
            Vec2::new(1., -1.),
            Vec2::ONE,
        ]
        .map(|corner| scene.ceiling_height(center + corner * extent * 0.5))
        .into_iter()
        .fold(f32::INFINITY, f32::min);
        let height = ceiling - floor;
        let origin = Vec3::new(center.x, (floor + ceiling) * 0.5, center.y);
        if extent.min_element() < 1. || height < 1. || transport.inside_solid(origin) {
            continue;
        }
        volumes.push(
            Transform::from_translation(origin).with_scale(Vec3::new(extent.x, height, extent.y)),
        );
    }
    volumes
}

pub(super) fn bind(
    transform: Transform,
    [diffuse, specular]: [Handle<Image>; 2],
) -> ReflectionVolume {
    ReflectionVolume {
        transform,
        map: EnvironmentMapLight {
            diffuse_map: diffuse,
            specular_map: specular,
            intensity: 1.,
            // Parent already rotates the volume and sampled radiance.
            rotation: Quat::IDENTITY,
            ..default()
        },
    }
}

impl IndoorMaterials {
    pub(crate) fn spawn_reflections(&self, root: Entity, commands: &mut Commands) {
        for (i, volume) in self.reflections.iter().enumerate() {
            commands.spawn((
                Name::new(format!("indoor_reflection_{i}")),
                bevy::light::LightProbe {
                    falloff: Vec3::splat(0.08),
                },
                bevy::light::ParallaxCorrection::Auto,
                volume.map.clone(),
                volume.transform,
                ChildOf(root),
            ));
        }
    }
}
