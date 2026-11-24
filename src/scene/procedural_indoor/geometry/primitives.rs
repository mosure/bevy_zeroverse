//! Fixed, immutable construction inputs. No scene, seed, size or material keys
//! accumulate here; transformed vertices always belong to the caller's geometry.
use super::*;
use std::{borrow::Cow, sync::OnceLock};

pub(super) fn unit_sphere() -> &'static Mesh {
    static SPHERE: OnceLock<Mesh> = OnceLock::new();
    SPHERE.get_or_init(|| Sphere::new(1.0).mesh().uv(20, 12))
}

pub(super) fn small_sphere() -> &'static Mesh {
    static SPHERE: OnceLock<Mesh> = OnceLock::new();
    SPHERE.get_or_init(|| Sphere::new(1.0).mesh().uv(12, 8))
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
        Self::from_radians(index as f32 / segments as f32 * TAU)
    }

    fn from_radians(radians: f32) -> Self {
        Self {
            radians,
            sin: radians.sin(),
            cos: radians.cos(),
        }
    }
}

/// Bent tubes use a different established floating-point evaluation order from
/// lathes. Share the exact same ring across their path and across objects. The
/// cache is bounded by tessellation alone, independent of seed and room count.
pub(super) fn tube_angles(sides: u32) -> Cow<'static, [LatheAngle]> {
    static RINGS: [OnceLock<Box<[LatheAngle]>>; 33] = [const { OnceLock::new() }; 33];
    let ring = || {
        (0..=sides)
            .map(|side| LatheAngle::from_radians(TAU * side as f32 / sides as f32))
            .collect::<Vec<_>>()
    };
    match RINGS.get(sides as usize) {
        Some(entry) => Cow::Borrowed(entry.get_or_init(|| ring().into_boxed_slice())),
        None => Cow::Owned(ring()),
    }
}

pub(super) fn lathe_angles(segments: u32) -> Cow<'static, [LatheAngle]> {
    // Glides, vessels, knobs and standard cylinders use a small family of
    // ring counts. Cache only these bounded tables; custom high resolutions
    // remain transient. Lathes preserve their own angular evaluation order.
    static RINGS: [OnceLock<Box<[LatheAngle]>>; 33] = [const { OnceLock::new() }; 33];
    let ring = || {
        (0..=segments)
            .map(|i| LatheAngle::new(i, segments))
            .collect::<Vec<_>>()
    };
    match RINGS.get(segments as usize) {
        Some(entry) => Cow::Borrowed(entry.get_or_init(|| ring().into_boxed_slice())),
        None => Cow::Owned(ring()),
    }
}
