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
        Self::oriented(object.position.xz(), object.size.xz(), object.yaw)
    }

    pub(crate) fn oriented(center: Vec2, size: Vec2, yaw: f32) -> Self {
        let (s, c) = yaw.sin_cos();
        Self {
            center,
            half: size * 0.5,
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
            delta.dot(axis).abs() < self.radius_on(axis) + other.radius_on(axis) + margin
        })
    }

    fn radius_on(self, axis: Vec2) -> f32 {
        self.half.x * self.axes[0].dot(axis).abs() + self.half.y * self.axes[1].dot(axis).abs()
    }

    /// In plan view the sightlines to an extended object fill the convex hull
    /// of its footprint and the observer. Test both rectangles' normals and
    /// the possible tangents from the observer to the target corners.
    pub fn obstructs_view(self, observer: Vec2, target: Self) -> bool {
        let dx = target.axes[0] * target.half.x;
        let dy = target.axes[1] * target.half.y;
        let to_center = target.center - self.center;
        let to_eye = observer - self.center;
        let tangents = [dx + dy, dx - dy, -dx + dy, -dx - dy]
            .map(|corner| (to_center + corner - to_eye).perp());
        self.axes
            .into_iter()
            .chain(target.axes)
            .chain(tangents)
            .all(|axis| {
                let panel = self.radius_on(axis);
                let center = to_center.dot(axis);
                let reach = target.radius_on(axis);
                let eye = to_eye.dot(axis);
                (center - reach).min(eye) <= panel && (center + reach).max(eye) >= -panel
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
