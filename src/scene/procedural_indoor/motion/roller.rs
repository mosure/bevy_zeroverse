//! A roller feeds inextensible fabric past a fixed rail. Store the unrevealed
//! mesh within the covered head box, maintaining nondegenerate triangles and
//! stable material-point identities for both raster and angular optical flow.
use bevy::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct RollerFeed {
    up: Vec3,
    plane: f32,
}

// A five-metre drop parks within 12.5 mm behind the fascia.
const PACKING: f32 = 0.0025;

impl RollerFeed {
    pub fn new(frame: Transform, rail_y: f32) -> Self {
        let up = frame.rotation * Vec3::Y;
        Self {
            up,
            plane: frame.transform_point(Vec3::Y * rail_y).dot(up),
        }
    }

    pub fn pack(distance_above_rail: f32) -> f32 {
        distance_above_rail.min(0.) + distance_above_rail.max(0.) * PACKING
    }

    pub fn interpolate(self, start: Vec3, end: Vec3, t: f32) -> Vec3 {
        let a = self.up.dot(start) - self.plane;
        let b = self.up.dot(end) - self.plane;
        let point = start.lerp(end, t);
        if (a > 0.) == (b > 0.) {
            return point;
        }
        let unpack = |y: f32| if y > 0. { y / PACKING } else { y };
        let physical = unpack(a).lerp(unpack(b), t);
        point + self.up * (Self::pack(physical) - a.lerp(b, t))
    }
}
