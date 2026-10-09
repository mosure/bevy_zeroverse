//! Input and pose adaptation for Bevy's native pan-orbit controller.
use super::*;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};

pub(crate) fn reset(camera: &mut PanOrbitCamera, radius: f32) {
    camera.current_motion = Default::default();
    camera.last_anchor_depth = -(radius.max(0.01) as f64);
}

pub(super) fn transform(focus: Vec3, yaw: f32, pitch: f32, radius: f32) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, yaw, -pitch, 0.);
    Transform {
        translation: focus + rotation * Vec3::Z * radius,
        rotation,
        ..default()
    }
}

pub(super) fn smoothing(camera: &mut PanOrbitCamera, args: &BevyZeroverseConfig) {
    // Preserve the shared 0..1 controls as explicit, bounded smoothing windows.
    let duration = |v: f32| std::time::Duration::from_secs_f32(v.clamp(0., 1.) * 0.25);
    camera.smoothing.orbit = duration(args.orbit_smoothness);
    camera.smoothing.pan = duration(args.pan_smoothness);
    camera.smoothing.zoom = duration(args.zoom_smoothness);
}

pub(super) fn controller(radius: f32, args: &BevyZeroverseConfig) -> PanOrbitCamera {
    let mut camera = PanOrbitCamera::default().with_initial_anchor_depth(radius as f64);
    smoothing(&mut camera, args);
    camera
}

/// Retain left-drag orbit, right/middle-drag pan and wheel zoom on native/WebGPU.
/// Input is always consumed, including while a control owns the pointer.
#[allow(clippy::too_many_arguments)]
pub(super) fn input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut moves: MessageReader<MouseMotion>,
    mut wheels: MessageReader<MouseWheel>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut cameras: Query<(&mut PanOrbitCamera, &Camera), With<EditorCameraMarker>>,
    mut active: Local<Option<MouseButton>>,
    editor: Option<Res<editor::model::EditorState>>,
    capture: Option<Res<editor::EditorInputCapture>>,
) {
    let delta: Vec2 = moves.read().map(|m| m.delta).sum();
    let zoom: f32 = wheels
        .read()
        .map(|m| {
            m.y * if m.unit == MouseScrollUnit::Line {
                150.
            } else {
                1.
            }
        })
        .sum();
    for (mut camera, lens) in &mut cameras {
        let enabled = lens.is_active
            && camera.enabled_motion.orbit
            && windows.single().is_ok_and(|w| {
                w.focused
                    && editor
                        .as_ref()
                        .zip(capture.as_ref())
                        .is_none_or(|(state, capture)| {
                            editor::shell::orbit_enabled(state, w, capture)
                        })
            });
        if !enabled {
            *active = None;
            camera.end_move();
            camera.current_motion = Default::default();
            continue;
        }
        if active.is_some_and(|b| buttons.just_released(b)) {
            camera.end_move();
            *active = None;
        }
        let continuing = active.is_some();
        if buttons.just_pressed(MouseButton::Left) {
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                camera.start_pan(None);
            } else {
                camera.start_orbit(None);
            }
            *active = Some(MouseButton::Left);
        } else if let Some(button) = [MouseButton::Right, MouseButton::Middle]
            .into_iter()
            .find(|b| buttons.just_pressed(*b))
        {
            camera.start_pan(None);
            *active = Some(button);
        } else if zoom != 0. && !camera.is_actively_controlled() {
            camera.start_zoom(None);
        }
        if continuing && active.is_some() {
            camera.send_screenspace_input(delta);
        }
        if zoom != 0. {
            camera.send_zoom_input(zoom);
        }
        if active.is_none() && zoom == 0. && camera.current_motion.is_zooming_only() {
            camera.end_move();
        }
    }
}
