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
    /// Quintic easing with zero speed and acceleration at both turnarounds.
    Smootherstep,
    /// Cubic Hermite easing with zero speed at both turnarounds.
    Smoothstep,
}

impl PlaybackMode {
    /// Position along the authored trajectory for the stored clock progress.
    pub fn map_progress(&self, progress: f32) -> f32 {
        let t = f64::from(progress.clamp(0.0, 1.0));
        let position = match self {
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => sine_leg(t),
            Self::Smoothstep => t * t * (3.0 - 2.0 * t),
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => t * (3.0 - 3.0 * t + t * t),
            Self::EaseInOutCubic | Self::Smootherstep => {
                // Evaluate the nearer half to keep the curve monotone near its endpoints.
                let x = t.min(1.0 - t);
                let y = if *self == Self::Smootherstep {
                    x * x * x * (10.0 + x * (6.0 * x - 15.0))
                } else {
                    4.0 * x * x * x
                };
                if t <= 0.5 {
                    y
                } else {
                    1.0 - y
                }
            }
            Self::Loop | Self::Once | Self::PingPong | Self::Sin | Self::Still => t,
        };
        position as f32
    }

    // Used only when selecting a different curve. All authored easing functions
    // are monotone on a leg, so they can share one bounded inverse.
    fn progress_at_position(self, position: f32) -> f32 {
        if matches!(
            self,
            Self::Loop | Self::Once | Self::PingPong | Self::Sin | Self::Still
        ) || position == 0.0
            || position == 1.0
        {
            return position;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..25 {
            let mid = 0.5 * (lo + hi);
            let mapped = self.map_progress(mid);
            if mapped == position {
                return mid;
            }
            if mapped < position {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect, Deserialize, Parser, Serialize)]
#[cfg_attr(feature = "python", pyclass)]
#[reflect(Resource)]
#[type_path = "bevy_zeroverse::camera"]
pub struct Playback {
    pub mode: PlaybackMode,
    /// Leg phase for easing modes. Sin stores its sampled position for replay compatibility.
    pub progress: f32,
    /// Direction along the path at positive speed, either +1 or -1.
    pub direction: f32,
    /// Legs per second. Sin uses complete round trips per second.
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

fn sine_leg(phase: f64) -> f64 {
    (phase * std::f64::consts::FRAC_PI_2).sin().powi(2)
}

impl Playback {
    /// Normalized position shared by cameras, actors, and trajectory overlays.
    pub fn trajectory_progress(&self) -> f32 {
        self.mode.map_progress(self.progress)
    }

    /// Return to the start, ready to traverse the path forward.
    pub fn rewind(&mut self) {
        self.progress = 0.0;
        self.direction = 1.0;
    }

    /// Switch curves or pause at the same point on the trajectory.
    pub fn set_mode(&mut self, mode: PlaybackMode) {
        if self.mode != mode {
            self.progress = mode.progress_at_position(self.trajectory_progress());
            if matches!(mode, PlaybackMode::Once | PlaybackMode::Loop) {
                self.direction = 1.0;
            }
            self.mode = mode;
        }
    }

    pub fn step(&mut self, time: &Time) {
        use PlaybackMode::*;
        let delta = time.delta_secs_f64() * f64::from(self.speed);
        if self.mode == Still || delta == 0.0 || !delta.is_finite() {
            return;
        }

        let position = f64::from(self.progress.clamp(0.0, 1.0));
        match self.mode {
            Once => {
                self.progress =
                    (position + delta * f64::from(self.direction)).clamp(0.0, 1.0) as f32;
            }
            Loop => {
                self.progress =
                    (position + delta * f64::from(self.direction)).rem_euclid(1.0) as f32;
            }
            PingPong | Sin | EaseIn | EaseOut | EaseInOut | EaseInCubic | EaseOutCubic
            | EaseInOutCubic | Smootherstep | Smoothstep => {
                // Sin historically measures speed in round trips per second and
                // stores the eased position. All other modes measure legs per second.
                let (leg, travel) = if self.mode == Sin {
                    (
                        position.sqrt().asin() / std::f64::consts::FRAC_PI_2,
                        2.0 * delta,
                    )
                } else {
                    (position, delta)
                };
                let unfolded = if self.direction >= 0.0 {
                    leg
                } else {
                    2.0 - leg
                };
                let phase = (unfolded + travel).rem_euclid(2.0);
                let leg = if phase < 1.0 { phase } else { 2.0 - phase };
                self.direction = if phase < 1.0 { 1.0 } else { -1.0 };
                self.progress = if self.mode == Sin { sine_leg(leg) } else { leg } as f32;
            }
            Still => {}
        }
    }
}

#[cfg(test)]
mod tests;
