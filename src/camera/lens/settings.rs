//! A fisheye is one photometric camera, even when rasterized as several views.
//! Keep optional view settings (including removals) identical to the owner;
//! copying exposure alone leaves custom ambient/probe/shadow settings and the
//! native draw policy different on either side of an internal face boundary.
use super::{LensCamera, LensFace};
use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::{visibility::RenderLayers, Exposure, Hdr},
    core_pipeline::{
        prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
        tonemapping::{DebandDither, Tonemapping},
    },
    light::{AmbientLight, EnvironmentMapLight, ShadowFilteringMethod},
    pbr::{DistanceFog, ScreenSpaceAmbientOcclusion, ScreenSpaceTransmission},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::{ColorGrading, Msaa},
};

pub(super) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (
            (
                inherit::<Exposure>,
                inherit::<AmbientLight>,
                inherit::<EnvironmentMapLight>,
                inherit::<ShadowFilteringMethod>,
                inherit::<ScreenSpaceTransmission>,
                inherit::<ColorGrading>,
                inherit::<Tonemapping>,
                inherit::<DebandDither>,
                inherit::<Msaa>,
                inherit::<Bloom>,
            ),
            (
                inherit::<Fxaa>,
                inherit::<ScreenSpaceAmbientOcclusion>,
                inherit::<DistanceFog>,
                inherit::<RenderLayers>,
                inherit::<Hdr>,
                inherit::<DepthPrepass>,
                inherit::<NormalPrepass>,
                inherit::<MotionVectorPrepass>,
            ),
        )
            .before(bevy::camera::CameraUpdateSystems),
    );
}

#[allow(clippy::type_complexity)] // Disjoint owner/face ECS queries.
fn inherit<T: Component + Clone>(
    mut commands: Commands,
    owners: Query<Option<Ref<T>>, (With<LensCamera>, Without<LensFace>)>,
    faces: Query<(Entity, Ref<LensFace>, Has<T>)>,
) {
    for (entity, face, present) in &faces {
        match owners.get(face.owner) {
            Ok(Some(value)) if value.is_changed() || face.is_added() || !present => {
                commands.entity(entity).insert((*value).clone());
            }
            Ok(None) if present => {
                commands.entity(entity).remove::<T>();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn faces_inherit_settings_changes_and_removals_without_rewriting_each_frame() {
        let mut app = App::new();
        app.add_systems(
            Update,
            (
                inherit::<Exposure>,
                inherit::<AmbientLight>,
                inherit::<RenderLayers>,
            ),
        );
        let owner = app
            .world_mut()
            .spawn((
                LensCamera::test_camera(),
                Exposure { ev100: 7. },
                AmbientLight {
                    brightness: 37.,
                    ..default()
                },
                RenderLayers::layer(3),
            ))
            .id();
        let face = app.world_mut().spawn(LensFace { owner }).id();
        app.update();
        assert_eq!(app.world().get::<Exposure>(face).unwrap().ev100, 7.);
        assert_eq!(
            app.world().get::<AmbientLight>(face).unwrap().brightness,
            37.
        );
        assert_eq!(
            app.world().get::<RenderLayers>(face).unwrap(),
            &RenderLayers::layer(3)
        );
        let tick = app
            .world()
            .entity(face)
            .get_ref::<Exposure>()
            .unwrap()
            .last_changed();
        app.update();
        assert_eq!(
            app.world()
                .entity(face)
                .get_ref::<Exposure>()
                .unwrap()
                .last_changed(),
            tick
        );
        app.world_mut()
            .entity_mut(owner)
            .insert(Exposure { ev100: 9. });
        app.world_mut()
            .entity_mut(owner)
            .remove::<(AmbientLight, RenderLayers)>();
        app.update();
        assert_eq!(app.world().get::<Exposure>(face).unwrap().ev100, 9.);
        assert!(app.world().get::<AmbientLight>(face).is_none());
        assert!(app.world().get::<RenderLayers>(face).is_none());
    }
}
