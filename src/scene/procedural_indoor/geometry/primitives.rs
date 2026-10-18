//! Fixed, immutable construction inputs. No scene, seed, size or material keys
//! accumulate here; transformed vertices always belong to the caller's geometry.
use super::*;
use std::{borrow::Cow, sync::OnceLock};

pub(super) fn unit_sphere() -> &'static Mesh {
    static SPHERE: OnceLock<Mesh> = OnceLock::new();
    SPHERE.get_or_init(|| Sphere::new(1.0).mesh().uv(20, 12))
}

#[derive(Clone, Copy)]
pub(super) struct LatheAngle {
    pub radians: f32,
    pub sin: f32,
    pub cos: f32,
}

impl LatheAngle {
    fn new(index: u32, segments: u32) -> Self {
        // Preserve the original division/multiplication order and separate trig
        // operations, including the duplicated seam and its UV coordinate.
        let radians = index as f32 / segments as f32 * TAU;
        Self {
            radians,
            sin: radians.sin(),
            cos: radians.cos(),
        }
    }
}

pub(super) fn lathe_angles(segments: u32) -> Cow<'static, [LatheAngle]> {
    // The common cylinder uses a fixed 21-entry table. Other tessellations own
    // one short-lived table per call, reused across all rings in that profile.
    static CYLINDER: OnceLock<[LatheAngle; 21]> = OnceLock::new();
    if segments == 20 {
        Cow::Borrowed(
            CYLINDER.get_or_init(|| std::array::from_fn(|i| LatheAngle::new(i as u32, 20))),
        )
    } else {
        Cow::Owned(
            (0..=segments)
                .map(|i| LatheAngle::new(i, segments))
                .collect(),
        )
    }
}
