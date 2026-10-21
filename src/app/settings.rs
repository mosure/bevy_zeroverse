//! Keep the public runtime resources and the viewer/CLI configuration in sync.
//! Compare values per field: an unrelated UI edit must not reset a runtime edit.
use super::*;

/// Only these scalar settings participate in bidirectional runtime sync. Keep
/// large motion prompts, waypoint arrays and policies out of the per-frame path.
#[derive(Clone, PartialEq)]
pub(super) struct SyncedSettings {
    num_cameras: usize,
    scene_type: ZeroverseSceneType,
    rotation_augmentation: bool,
    max_camera_radius: f32,
    cuboid_only: bool,
    playback_mode: PlaybackMode,
    playback_speed: f32,
    render_mode: RenderMode,
}
impl From<&BevyZeroverseConfig> for SyncedSettings {
    fn from(c: &BevyZeroverseConfig) -> Self {
        Self {
            num_cameras: c.num_cameras,
            scene_type: c.scene_type.clone(),
            rotation_augmentation: c.rotation_augmentation,
            max_camera_radius: c.max_camera_radius,
            cuboid_only: c.cuboid_only,
            playback_mode: c.playback_mode,
            playback_speed: c.playback_speed,
            render_mode: c.render_mode.clone(),
        }
    }
}

fn reconcile<T: Clone + PartialEq>(config: &mut T, runtime: &mut T, previous: Option<&T>) {
    if previous != Some(config) {
        if runtime != config {
            runtime.clone_from(config);
        }
    } else if config != runtime {
        config.clone_from(runtime);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn synchronize(
    mut args: ResMut<BevyZeroverseConfig>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    mut playback: ResMut<Playback>,
    mut render: ResMut<RenderMode>,
    mut scene: ResMut<ZeroverseSceneSettings>,
    mut semantic: ResMut<ZeroverseSemanticRoomSettings>,
    mut previous: Local<Option<SyncedSettings>>,
) {
    // Bypass Bevy's change flag until a value actually changes. Otherwise every
    // frame would rebuild camera grids and annotation materials.
    let before = SyncedSettings::from(&*args);
    let config = args.bypass_change_detection();
    macro_rules! sync {
        ($field:ident, $resource:ident, $runtime:ident) => {
            reconcile(
                &mut config.$field,
                &mut $resource.bypass_change_detection().$runtime,
                previous.as_ref().map(|p| &p.$field),
            );
            if before.$field != config.$field
                || previous.as_ref().is_none_or(|p| p.$field != config.$field)
            {
                $resource.set_changed();
            }
        };
    }
    sync!(num_cameras, scene, num_cameras);
    sync!(scene_type, scene, scene_type);
    sync!(rotation_augmentation, scene, rotation_augmentation);
    sync!(max_camera_radius, scene, max_camera_radius);
    sync!(cuboid_only, semantic, cuboid_only);
    if !sampler.is_some_and(|s| s.enabled) {
        sync!(playback_mode, playback, mode);
        sync!(playback_speed, playback, speed);
    }
    let old_mode = render.clone();
    reconcile(
        &mut config.render_mode,
        render.bypass_change_detection(),
        previous.as_ref().map(|p| &p.render_mode),
    );
    if *render != old_mode {
        render.set_changed();
    }
    let changed = before.scene_type != config.scene_type
        || before.num_cameras != config.num_cameras
        || before.rotation_augmentation != config.rotation_augmentation
        || before.max_camera_radius != config.max_camera_radius
        || before.cuboid_only != config.cuboid_only
        || before.playback_mode != config.playback_mode
        || before.playback_speed != config.playback_speed
        || before.render_mode != config.render_mode;
    *previous = Some(SyncedSettings::from(&*config));
    if changed {
        args.set_changed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idle_settings_sync_does_not_mark_render_resources_changed() {
        let mut app = App::new();
        app.init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<RenderMode>()
            .init_resource::<ZeroverseSceneSettings>()
            .init_resource::<ZeroverseSemanticRoomSettings>()
            .add_systems(First, synchronize);
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .human_motion = Some("x".repeat(131072));
        app.update();
        app.world_mut().clear_trackers();
        app.update();
        assert!(!app
            .world()
            .resource_ref::<BevyZeroverseConfig>()
            .is_changed());
        assert!(!app.world().resource_ref::<RenderMode>().is_changed());
        assert!(!app
            .world()
            .resource_ref::<ZeroverseSceneSettings>()
            .is_changed());
    }
    #[test]
    fn settings_edits_sync_without_regenerating() {
        let mut app = App::new();
        app.init_resource::<BevyZeroverseConfig>()
            .init_resource::<Playback>()
            .init_resource::<RenderMode>()
            .init_resource::<ZeroverseSceneSettings>()
            .init_resource::<ZeroverseSemanticRoomSettings>()
            .add_message::<RegenerateSceneEvent>()
            .add_systems(First, synchronize);
        app.update();
        app.world_mut()
            .resource_mut::<ZeroverseSceneSettings>()
            .scene_type = ZeroverseSceneType::ProceduralIndoor;
        // Editing an unrelated config field cannot undo the scene selection.
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_density = 0.8;
        app.update();
        assert_eq!(
            app.world().resource::<BevyZeroverseConfig>().scene_type,
            ZeroverseSceneType::ProceduralIndoor
        );
        assert!(app
            .world()
            .resource::<Messages<RegenerateSceneEvent>>()
            .is_empty());
        for fraction in [0.2, 0.4, 0.8, 1.0] {
            app.world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .human_motion = Some(format!(r#"{{"fraction":{fraction}}}"#));
            app.update();
            assert!(app
                .world()
                .resource::<Messages<RegenerateSceneEvent>>()
                .is_empty());
        }
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .scene_type = ZeroverseSceneType::Room;
        app.update();
        assert_eq!(
            app.world().resource::<ZeroverseSceneSettings>().scene_type,
            ZeroverseSceneType::Room
        );
        app.world_mut()
            .resource_mut::<RenderMode>()
            .clone_from(&RenderMode::Semantic);
        app.update();
        assert_eq!(
            app.world().resource::<BevyZeroverseConfig>().render_mode,
            RenderMode::Semantic
        );
    }
}
