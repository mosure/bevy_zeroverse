//! Continuous support-local proposals shared by layout and built-in contents.
//! These are proposals, never a replacement for the owning surface/scene gate.
use bevy::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Footprint {
    pub center: Vec2,
    pub half: Vec2,
    pub yaw: f32,
}
impl Footprint {
    pub fn inside(self, lo: Vec2, hi: Vec2, gap: f32) -> bool {
        (self.center - self.half).cmpge(lo + Vec2::splat(gap)).all()
            && (self.center + self.half).cmple(hi - Vec2::splat(gap)).all()
    }
    /// Conservative rotated-box separation also reserves cup handles and rims.
    pub fn overlaps(self, other: Self, gap: f32) -> bool {
        (self.center - other.center)
            .abs()
            .cmplt(self.half + other.half + Vec2::splat(gap))
            .all()
    }
}

pub(crate) fn rotated_half(size: Vec2, yaw: f32) -> Vec2 {
    let (s, c) = yaw.sin_cos();
    (size * c.abs() + size.yx() * s.abs()) * 0.5
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SurfacePattern {
    heading: f32,
    alignment: f32,
    angular_spread: f32,
    clustering: f32,
    focus: Vec2,
    spread: Vec2,
}
impl SurfacePattern {
    pub fn sample(r: &mut ChaCha8Rng) -> Self {
        Self {
            heading: r.random_range(-PI..PI),
            alignment: r.random_range(0.0..0.85),
            angular_spread: r.random_range(0.15..1.4),
            clustering: r.random_range(0.0..0.75),
            focus: Vec2::new(r.random_range(0.18..0.82), r.random_range(0.18..0.82)),
            spread: Vec2::new(r.random_range(0.16..0.45), r.random_range(0.16..0.45)),
        }
    }
    pub fn yaw(self, r: &mut ChaCha8Rng) -> f32 {
        if r.random_bool(self.alignment as f64) {
            self.heading + r.random_range(-self.angular_spread..self.angular_spread)
        } else {
            r.random_range(-PI..PI)
        }
    }
    /// Sample within the legal centre interval for this rotated footprint.
    /// Rejection (not clamping) avoids probability spikes along support edges.
    pub fn propose(
        self,
        bounds: (Vec2, Vec2),
        size: Vec2,
        yaw: f32,
        gap: f32,
        r: &mut ChaCha8Rng,
    ) -> Option<Footprint> {
        let half = rotated_half(size, yaw);
        self.propose_half(bounds, half, yaw, gap, r)
    }

    /// Round vessels reserve their radius, so rotation does not artificially
    /// enlarge their footprint and select only axis-aligned survivors.
    pub fn propose_half(
        self,
        bounds: (Vec2, Vec2),
        half: Vec2,
        yaw: f32,
        gap: f32,
        r: &mut ChaCha8Rng,
    ) -> Option<Footprint> {
        let lo = bounds.0 + half + Vec2::splat(gap);
        let hi = bounds.1 - half - Vec2::splat(gap);
        if !lo.is_finite() || !hi.is_finite() || lo.cmpge(hi).any() {
            return None;
        }
        let mut uv = Vec2::new(r.random(), r.random());
        if r.random_bool(self.clustering as f64) {
            let candidate = self.focus
                + Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0)) * self.spread;
            if candidate.cmpge(Vec2::ZERO).all() && candidate.cmple(Vec2::ONE).all() {
                uv = candidate;
            }
        }
        let footprint = Footprint {
            center: lo + (hi - lo) * uv,
            half,
            yaw,
        };
        debug_assert!(footprint.inside(bounds.0, bounds.1, gap - 1e-6));
        Some(footprint)
    }
}

#[cfg(test)]
mod tests;
