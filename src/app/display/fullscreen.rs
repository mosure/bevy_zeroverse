use super::BevyZeroverseConfig;
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::window::PrimaryWindow;

pub(super) fn install(app: &mut App) {
    app.add_systems(PreUpdate, synchronize);
}

#[cfg(not(target_arch = "wasm32"))]
fn native_mode(enabled: bool) -> bevy::window::WindowMode {
    use bevy::window::{MonitorSelection, WindowMode};
    if enabled {
        WindowMode::BorderlessFullscreen(MonitorSelection::Current)
    } else {
        WindowMode::Windowed
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn synchronize(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<BevyZeroverseConfig>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut applied: Local<Option<bool>>,
) {
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    let current = !matches!(window.mode, bevy::window::WindowMode::Windowed);
    if config.keybinds && keys.just_pressed(KeyCode::F11) {
        config.fullscreen = !current;
    } else if *applied == Some(config.fullscreen) && current != config.fullscreen {
        config.fullscreen = current;
    }
    if config.fullscreen != current {
        window.mode = native_mode(config.fullscreen);
    }
    *applied = Some(config.fullscreen);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(module = "/src/app/display/fullscreen.js")]
extern "C" {
    fn sync_fullscreen(request: Option<bool>, shortcuts: bool) -> bool;
}

#[cfg(target_arch = "wasm32")]
fn synchronize(mut config: ResMut<BevyZeroverseConfig>, mut applied: Local<Option<bool>>) {
    let request = (*applied != Some(config.fullscreen)).then_some(config.fullscreen);
    let desired = sync_fullscreen(request, config.keybinds);
    if config.fullscreen != desired {
        config.fullscreen = desired;
    }
    *applied = Some(desired);
}
