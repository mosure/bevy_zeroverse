//! Recovery for the standalone native kiosk. A heartbeat follows a rendered window frame through
//! surface acquisition and the return from presentation. An independent thread can still act if Winit or present stops.
use super::BevyZeroverseConfig;
use crate::scene::{
    procedural_indoor::{layout::IndoorManifest, IndoorGenerationStatus},
    ZeroverseSceneType,
};
use bevy::{
    prelude::*,
    render::{
        renderer::render_system, view::window::ExtractedWindow, Render, RenderApp, RenderSystems,
    },
    window::{AppLifecycle, PrimaryWindow, WindowOccluded},
};
use std::{
    process::Command,
    sync::{mpsc, Arc, Mutex},
    time::{Duration, Instant},
};

const DEADLINE: Duration = Duration::from_secs(60);
const MAX_RESTARTS: u8 = 3;
const RECOVERY_ENV: &str = "BEVY_ZEROVERSE_KIOSK_RECOVERY";

pub(super) fn automated(config: &BevyZeroverseConfig) -> bool {
    config.kiosk
        && !config.headless
        && config.regenerate_ms != 0
        && config.scene_type == ZeroverseSceneType::ProceduralIndoor
}

pub(super) fn launch_config(cli: BevyZeroverseConfig) -> (BevyZeroverseConfig, u8) {
    std::env::var(RECOVERY_ENV)
        .ok()
        .and_then(|value| serde_json::from_str::<(BevyZeroverseConfig, u8)>(&value).ok())
        .filter(|(c, n)| automated(c) && (1..=MAX_RESTARTS).contains(n))
        .unwrap_or((cli, 0))
}

pub(super) fn open_window(config: BevyZeroverseConfig, failures: u8) -> App {
    if failures == 0 {
        return crate::app::viewer_app(None, Some(config));
    }
    // The compositor may still be resetting the connection which just stalled.
    // Reconstruct only in a fresh process: Winit event loops are process-global.
    std::thread::sleep(Duration::from_secs(1 << failures));
    match std::panic::catch_unwind(|| crate::app::viewer_app(None, Some(config.clone()))) {
        Ok(app) => app,
        Err(panic) => {
            if failures < MAX_RESTARTS {
                eprintln!("Kiosk: window initialization failed during recovery {failures}, retrying seed {:?}", config.indoor_seed);
                if let Err(error) = relaunch(&config, failures + 1) {
                    eprintln!("Kiosk: could not reopen the viewer: {error}");
                }
            }
            std::panic::resume_unwind(panic);
        }
    }
}

struct State {
    config: BevyZeroverseConfig,
    attempted: Option<u64>,
    displayed: Option<u64>,
    rooms: u32,
    since_launch: Instant,
    last_completion: Instant,
    armed: bool,
    failures: u8,
    room_timeouts: u8,
}
impl State {
    fn new(config: BevyZeroverseConfig, failures: u8, now: Instant) -> Self {
        let armed = automated(&config);
        Self {
            config,
            attempted: None,
            displayed: None,
            rooms: 0,
            since_launch: now,
            last_completion: now,
            armed,
            failures,
            room_timeouts: 0,
        }
    }
    fn arm(&mut self, enabled: bool, now: Instant) {
        if self.armed != enabled {
            self.armed = enabled;
            self.last_completion = now;
        }
    }
    fn expired(&self, now: Instant) -> bool {
        self.armed && now.duration_since(self.last_completion) >= DEADLINE
    }
    fn recovery(&self, now: Instant) -> Option<(BevyZeroverseConfig, u8)> {
        // A permanently unavailable display must not cause an unlimited relaunch loop.
        let healthy = self.rooms >= 2 && now.duration_since(self.since_launch) >= DEADLINE * 2;
        let failures = if healthy { 0 } else { self.failures };
        if failures >= MAX_RESTARTS {
            return None;
        }
        let mut config = self.config.clone();
        config.indoor_seed = self
            .attempted
            .or(self.displayed)
            .or(config.indoor_seed)
            .map(|seed| seed.wrapping_add(1));
        Some((config, failures + 1))
    }
}

#[derive(Resource)]
pub(super) struct Watch {
    state: Arc<Mutex<State>>,
    // Disconnect on normal app teardown, including an intentional window close.
    _lifetime: mpsc::Sender<()>,
}
impl Watch {
    pub(super) fn start(config: BevyZeroverseConfig, failures: u8) -> Self {
        let state = Arc::new(Mutex::new(State::new(config, failures, Instant::now())));
        let (lifetime, shutdown) = mpsc::channel();
        let monitor = state.clone();
        std::thread::Builder::new()
            .name("kiosk-watchdog".into())
            .spawn(move || monitor_render(monitor, shutdown))
            .expect("start native kiosk watchdog");
        Self {
            state,
            _lifetime: lifetime,
        }
    }

    pub(super) fn install(self, app: &mut App) {
        let render_state = self.state.clone();
        let Some(render) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // A particular driver's queue callbacks may be delayed indefinitely. Observe
        // the same SDR/HDR surface Bevy actually acquired, then the return from present.
        #[derive(Resource, Default)]
        struct Acquired(bool);
        render.init_resource::<Acquired>().add_systems(
            Render,
            (
                (|window: Query<&ExtractedWindow, With<PrimaryWindow>>,
                  mut acquired: ResMut<Acquired>| {
                    acquired.0 = window
                        .single()
                        .is_ok_and(|w| w.swap_chain_texture.is_some());
                })
                .in_set(RenderSystems::Render)
                .before(render_system),
                (move |acquired: Res<Acquired>, mut last: Local<Option<Instant>>| {
                    if !acquired.0 {
                        return;
                    }
                    let now = Instant::now();
                    if last.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(1)) {
                        *last = Some(now);
                        render_state.lock().unwrap().last_completion = now;
                    }
                })
                .in_set(RenderSystems::Cleanup),
            ),
        );
        app.insert_resource(self).add_systems(Last, observe);
    }
}

fn monitor_render(monitor: Arc<Mutex<State>>, shutdown: mpsc::Receiver<()>) {
    let mut previous = Instant::now();
    while let Err(mpsc::RecvTimeoutError::Timeout) = shutdown.recv_timeout(Duration::from_secs(1)) {
        let now = Instant::now();
        let mut state = monitor.lock().unwrap();
        // Sleep/resume can advance a platform clock while every thread is stopped.
        if now.duration_since(previous) > Duration::from_secs(10) {
            state.last_completion = now;
        }
        previous = now;
        // Dropping a future cannot preempt a synchronous CPU call. If an entire
        // preparation pool is stuck, frames may continue to present the old room.
        let workers_stalled = state.armed && state.room_timeouts >= 3;
        if !workers_stalled && !state.expired(now) {
            continue;
        }
        let recovery = state.recovery(now);
        let (displayed, attempted) = (state.displayed, state.attempted);
        drop(state);
        // The logging subscriber or graphics thread may itself be wedged.
        let reason = if workers_stalled {
            "three room generation deadlines expired without a successful transition"
        } else {
            "no completed render frame for 60 s"
        };
        eprintln!("Kiosk: {reason} (displayed {displayed:?}, attempted {attempted:?})");
        let Some((config, failures)) = recovery else {
            eprintln!("Kiosk: viewer did not recover after {MAX_RESTARTS} restarts");
            std::process::exit(1);
        };
        eprintln!(
            "Kiosk: reopening the viewer at seed {:?} (recovery {failures})",
            config.indoor_seed
        );
        if let Err(error) = relaunch(&config, failures) {
            eprintln!("Kiosk: could not reopen the viewer: {error}");
        }
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn observe(
    watch: Res<Watch>,
    args: Res<BevyZeroverseConfig>,
    generation: Option<Res<IndoorGenerationStatus>>,
    manifest: Option<Res<IndoorManifest>>,
    windows: Query<(Entity, &Window), With<PrimaryWindow>>,
    mut occlusion: MessageReader<WindowOccluded>,
    mut lifecycle: MessageReader<AppLifecycle>,
    exit: MessageReader<AppExit>,
    mut hidden: Local<bool>,
    mut suspended: Local<bool>,
) {
    let primary = windows.single().ok();
    for event in occlusion.read() {
        if primary.is_some_and(|(entity, _)| entity == event.window) {
            *hidden = event.occluded;
        }
    }
    if let Some(event) = lifecycle.read().last() {
        *suspended = matches!(event, AppLifecycle::WillSuspend | AppLifecycle::Suspended);
    }
    let mut state = watch.state.lock().unwrap();
    if args.is_changed() {
        state.config = args.clone();
    }
    state.arm(
        automated(&args)
            && !*hidden
            && !*suspended
            && exit.is_empty()
            && primary.is_some_and(|(_, window)| {
                window.visible && window.physical_width() > 0 && window.physical_height() > 0
            }),
        Instant::now(),
    );
    if let Some(status) = generation {
        state.attempted = if status.consecutive_timeouts >= 3 {
            status.skipped_seed.or(status.attempted_seed)
        } else {
            status.attempted_seed
        };
        state.room_timeouts = status.consecutive_timeouts;
    }
    if let Some(scene) = manifest {
        if state.displayed != Some(scene.seed) {
            state.displayed = Some(scene.seed);
            state.rooms = state.rooms.saturating_add(1);
        }
    }
}

fn relaunch(config: &BevyZeroverseConfig, failures: u8) -> std::io::Result<()> {
    // The installed binary can be replaced while a kiosk is running. On Linux
    // execute the kernel's reference to this image even after its path is unlinked.
    #[cfg(target_os = "linux")]
    let mut command = Command::new("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args(std::env::args_os().skip(1))
        .env(RECOVERY_ENV, serde_json::to_string(&(config, failures))?);
    #[cfg(unix)]
    {
        // Retain the PID, terminal and ordinary close/signal behavior.
        Err(std::os::unix::process::CommandExt::exec(&mut command))
    }
    #[cfg(not(unix))]
    {
        command.spawn()?;
        std::process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suspended_display_and_clean_exit_disarm_recovery_and_resume_gets_a_full_deadline() {
        let now = Instant::now();
        let config = super::super::parse_cli(["viewer", "--kiosk"]).unwrap();
        let mut state = State::new(config, 0, now);
        assert!(!state.expired(now + DEADLINE - Duration::from_millis(1)));
        assert!(state.expired(now + DEADLINE));
        state.arm(false, now);
        let next_day = now + Duration::from_secs(86_400);
        assert!(!state.expired(next_day));
        state.arm(true, next_day);
        assert!(!state.expired(next_day));
        assert!(state.expired(next_day + DEADLINE));
    }

    #[test]
    fn recovery_skips_the_attempted_seed_and_caps_consecutive_restarts() {
        let now = Instant::now();
        let config = super::super::parse_cli(["viewer", "--kiosk", "--indoor-seed", "70"]).unwrap();
        let mut state = State::new(config, 1, now);
        state.attempted = Some(75);
        state.displayed = Some(72);
        let (next, failures) = state.recovery(now + DEADLINE).unwrap();
        assert_eq!((next.indoor_seed, failures), (Some(76), 2));
        assert_eq!(next.indoor_camera, state.config.indoor_camera);
        assert_eq!(next.regenerate_ms, state.config.regenerate_ms);
        assert!(next.hdr && next.fullscreen && next.kiosk);
        state.failures = MAX_RESTARTS;
        assert!(state.recovery(now + DEADLINE * 3).is_none());
        state.rooms = 2;
        assert_eq!(state.recovery(now + DEADLINE * 3).unwrap().1, 1);
    }
}
