//! Shared normalized scene timeline and easing, used by cameras and motion.
use bevy::prelude::*;
use bevy_args::{Deserialize, Parser, Serialize, ValueEnum};
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
    Deserialize,
    Parser,
    Serialize,
    ValueEnum,
)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[type_path = "bevy_zeroverse::camera"]
pub enum PlaybackMode {
    Loop,
    Once,
    PingPong,
    Sin,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    #[default]
    Still,
}

impl PlaybackMode {
    pub fn map_progress(&self, progress: f32) -> f32 {
        let t = progress.clamp(0.0, 1.0);

        match *self {
            PlaybackMode::EaseIn => t * t,
            PlaybackMode::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv
            }
            PlaybackMode::EaseInOut => 0.5 - 0.5 * (std::f32::consts::PI * t).cos(),
            PlaybackMode::EaseInCubic => t * t * t,
            PlaybackMode::EaseOutCubic => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            PlaybackMode::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let f = -2.0 * t + 2.0;
                    1.0 - (f * f * f) / 2.0
                }
            }
            _ => t,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect, Deserialize, Parser, Serialize)]
#[cfg_attr(feature = "python", pyclass)]
#[reflect(Resource)]
#[type_path = "bevy_zeroverse::camera"]
pub struct Playback {
    pub mode: PlaybackMode,
    pub progress: f32,
    pub direction: f32,
    pub speed: f32,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            mode: PlaybackMode::default(),
            progress: 0.0,
            direction: 1.0,
            speed: 1.0,
        }
    }
}

impl Playback {
    pub fn step(&mut self, time: &Time) {
        if self.speed == 0.0 {
            return;
        }

        // bail condition
        match self.mode {
            PlaybackMode::Loop
            | PlaybackMode::EaseIn
            | PlaybackMode::EaseOut
            | PlaybackMode::EaseInOut
            | PlaybackMode::EaseInCubic
            | PlaybackMode::EaseOutCubic
            | PlaybackMode::EaseInOutCubic => {}
            PlaybackMode::Once => {
                if self.progress >= 1.0 {
                    return;
                }
            }
            PlaybackMode::PingPong => {}
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {
                return;
            }
        }

        // forward condition
        match self.mode {
            PlaybackMode::Loop
            | PlaybackMode::Once
            | PlaybackMode::PingPong
            | PlaybackMode::EaseIn
            | PlaybackMode::EaseOut
            | PlaybackMode::EaseInOut
            | PlaybackMode::EaseInCubic
            | PlaybackMode::EaseOutCubic
            | PlaybackMode::EaseInOutCubic => {
                self.progress += time.delta_secs() * self.direction * self.speed;
            }
            PlaybackMode::Sin => {
                // Recover phase from the shared progress/direction, rather than
                // elapsed application time. Pausing, changing speed and scene
                // regeneration therefore cannot jump cameras or motion actors.
                let phase =
                    (1.0 - 2.0 * self.progress.clamp(0.0, 1.0)).acos() / std::f32::consts::TAU;
                let phase = if self.direction >= 0.0 {
                    phase
                } else {
                    1.0 - phase
                };
                let phase = (phase + time.delta_secs() * self.speed).rem_euclid(1.0);
                self.progress = (1.0 - (phase * std::f32::consts::TAU).cos()) * 0.5;
                self.direction = if phase < 0.5 { 1.0 } else { -1.0 };
            }
            PlaybackMode::Still => {}
        }

        // reset condition
        match self.mode {
            PlaybackMode::Loop
            | PlaybackMode::EaseIn
            | PlaybackMode::EaseOut
            | PlaybackMode::EaseInOut
            | PlaybackMode::EaseInCubic
            | PlaybackMode::EaseOutCubic
            | PlaybackMode::EaseInOutCubic => {
                if self.progress > 1.0 {
                    self.progress = 0.0;
                }
            }
            PlaybackMode::Once => {}
            PlaybackMode::PingPong => {
                if self.progress > 1.0 {
                    self.progress = 1.0;
                    self.direction = -self.direction;
                } else if self.progress < 0.0 {
                    self.progress = 0.0;
                    self.direction = -self.direction;
                }
            }
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {}
        }
    }
}
