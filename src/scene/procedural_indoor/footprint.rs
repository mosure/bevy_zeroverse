//! Metric, oriented footprints shared by placement and validation.
use super::layout::IndoorObject;
use bevy::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Footprint {
    center: Vec2,
    half: Vec2,
    axes: [Vec2; 2],
}

impl Footprint {
    pub fn object(object: &IndoorObject) -> Self {
        let (s, c) = object.yaw.sin_cos();
        Self {
            center: object.position.xz(),
            half: object.size.xz() * 0.5,
            axes: [Vec2::new(c, -s), Vec2::new(s, c)],
        }
    }

    pub fn bounds(lo: Vec3, hi: Vec3) -> Self {
        Self {
            center: (lo + hi).xz() * 0.5,
            half: (hi - lo).xz() * 0.5,
            axes: [Vec2::X, Vec2::Y],
        }
    }

    /// Separating-axis test. Margin reserves free space without expanding a
    /// rotated rectangle into its much larger axis-aligned bounding box.
    pub fn overlaps(self, other: Self, margin: f32) -> bool {
        let delta = other.center - self.center;
        self.axes.into_iter().chain(other.axes).all(|axis| {
            let radius = |p: Self| {
                p.half.x * p.axes[0].dot(axis).abs() + p.half.y * p.axes[1].dot(axis).abs()
            };
            delta.dot(axis).abs() < radius(self) + radius(other) + margin
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_separated_furniture_does_not_create_a_collision() {
        for i in 0..360 {
            let yaw = (i as f32).to_radians();
            let (s, c) = yaw.sin_cos();
            let a = Footprint {
                center: Vec2::ZERO,
                half: Vec2::new(1.5, 0.25),
                axes: [Vec2::new(c, -s), Vec2::new(s, c)],
            };
            let mut b = a;
            b.center += a.axes[1] * 0.8;
            assert!(!a.overlaps(b, 0.06));
            assert!(!b.overlaps(a, 0.06));
            b.center = a.axes[1] * 0.49;
            assert!(a.overlaps(b, 0.0));
            b.center = a.axes[1] * 0.55;
            assert!(a.overlaps(b, 0.06));
        }
    }
}
