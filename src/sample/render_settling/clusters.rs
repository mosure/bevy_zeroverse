//! Seed Bevy's adaptive cluster depth with this pose instead of the previous
//! pose. Only the far-depth reduction is repeated; Bevy still assigns, orders
//! and packs every light/probe, with its original dimensions and resolution.
use bevy::{
    camera::{
        primitives::{Frustum, Sphere},
        visibility::{RenderLayers, VisibilitySystems},
    },
    light::{
        cluster::{ClusterConfig, ClusterFarZMode, Clusters, GlobalClusterSettings},
        ClusteredDecal, LightProbe, RectLight, SimulationLightSystems,
    },
    math::Vec3A,
    prelude::*,
};

pub(super) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        prime
            .after(VisibilitySystems::CheckVisibility)
            .after(bevy::transform::TransformSystems::Propagate)
            .after(bevy::camera::CameraUpdateSystems)
            .before(SimulationLightSystems::AssignLightsToClusters),
    );
}

fn automatic(config: Option<&ClusterConfig>) -> bool {
    match config.copied().unwrap_or_default() {
        ClusterConfig::None => false,
        ClusterConfig::Single => true,
        ClusterConfig::XYZ { z_config, .. } | ClusterConfig::FixedZ { z_config, .. } => matches!(
            z_config.far_z_mode,
            ClusterFarZMode::MaxClusterableObjectRange
        ),
    }
}

pub(super) fn coherent(clusters: Option<&Clusters>, config: Option<&ClusterConfig>) -> bool {
    !automatic(config)
        || clusters.is_some_and(|c| {
            c.last_frame_farthest_z
                .is_some_and(|far| c.far == far.max(c.near))
        })
}

#[derive(Clone)]
struct Object {
    sphere: Sphere,
    layers: RenderLayers,
}

#[allow(clippy::type_complexity)]
fn prime(
    state: Res<super::SamplerState>,
    settings: Option<Res<GlobalClusterSettings>>,
    mut views: Query<
        (
            &Camera,
            &GlobalTransform,
            &Frustum,
            Option<&ClusterConfig>,
            Option<&RenderLayers>,
            &mut Clusters,
        ),
        With<super::ZeroverseCamera>,
    >,
    objects: Query<
        (
            &GlobalTransform,
            &ViewVisibility,
            Option<&RenderLayers>,
            Option<&PointLight>,
            Option<&SpotLight>,
            Option<&RectLight>,
            Has<LightProbe>,
            Has<ClusteredDecal>,
        ),
        Or<(
            With<PointLight>,
            With<SpotLight>,
            With<RectLight>,
            With<LightProbe>,
            With<ClusteredDecal>,
        )>,
    >,
    mut gathered: Local<Vec<Object>>,
) {
    let Some(settings) =
        settings.filter(|s| s.gpu_clustering.is_none() && s.supports_storage_buffers)
    else {
        return;
    };
    if !state.enabled {
        return;
    }
    gathered.clear();
    for (transform, visibility, layers, point, spot, rect, probe, decal) in &objects {
        if !visibility.get() {
            continue;
        }
        let mut push = |range, layers| {
            gathered.push(Object {
                sphere: Sphere {
                    center: transform.translation_vec3a(),
                    radius: range,
                },
                layers,
            })
        };
        for range in [
            point.map(|l| l.range),
            spot.map(|l| l.range),
            rect.map(|l| l.range),
        ]
        .into_iter()
        .flatten()
        {
            push(range, layers.cloned().unwrap_or_default());
        }
        if probe {
            push(transform.radius_vec3a(Vec3A::ONE), RenderLayers::default());
        }
        if decal && settings.clustered_decals_are_usable {
            push(transform.scale().length(), RenderLayers::default());
        }
    }
    for (camera, transform, frustum, config, layers, mut clusters) in &mut views {
        if !camera.is_active || !automatic(config) {
            continue;
        }
        let layers = layers.cloned().unwrap_or_default();
        let view = Mat4::from(transform.affine().inverse()).row(2);
        let scale = transform.compute_transform().scale.recip().z;
        let far = gathered
            .iter()
            .filter(|o| layers.intersects(&o.layers) && frustum.intersects_sphere(&o.sphere, true))
            .fold(0.0_f32, |far, o| {
                far.max(-view.dot(o.sphere.center.extend(1.0)) + o.sphere.radius * scale)
            });
        clusters.last_frame_farthest_z = Some(far);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_adaptive_depth_cannot_qualify_a_first_capture() {
        let mut clusters = Clusters {
            near: 5.,
            far: 20.,
            last_frame_farthest_z: Some(30.),
            ..default()
        };
        assert!(!coherent(Some(&clusters), None));
        clusters.far = 30.;
        assert!(coherent(Some(&clusters), None));
        clusters.last_frame_farthest_z = Some(0.);
        assert!(!coherent(Some(&clusters), None));
        clusters.far = 5.;
        assert!(coherent(Some(&clusters), None));
        clusters.last_frame_farthest_z = None;
        assert!(!coherent(Some(&clusters), None));
        let fixed = ClusterConfig::XYZ {
            dimensions: UVec3::new(8, 8, 16),
            z_config: bevy::light::cluster::ClusterZConfig {
                far_z_mode: ClusterFarZMode::Constant(50.),
                ..default()
            },
            dynamic_resizing: false,
        };
        assert!(coherent(Some(&clusters), Some(&fixed)));
    }
}
