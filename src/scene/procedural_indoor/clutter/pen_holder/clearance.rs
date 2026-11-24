//! Separate swept bounds for a barrel and its short, offset spring clip.
use super::*;
use crate::scene::procedural_indoor::objects::equipment::wires::segment_distance_squared;

#[derive(Debug, Clone, Copy)]
pub(super) struct Capsule {
    pub start: Vec3,
    pub end: Vec3,
    pub radius: f32,
}

impl Capsule {
    fn separate(self, other: Self) -> bool {
        segment_distance_squared(self.start, self.end, other.start, other.end)
            > f64::from(self.radius + other.radius + 0.0003).powi(2)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ToolBounds {
    pub barrel: Capsule,
    pub clip: Option<Capsule>,
}

impl ToolBounds {
    pub fn new(tool: &writing::WritingProgram, tf: Transform) -> Self {
        let barrel = Capsule {
            start: tf.translation,
            end: tf.transform_point(Vec3::Y * tool.length),
            radius: tool.radius,
        };
        let clip = tool.clip.then(|| {
            let rings = tool.clip_rings();
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for pt in rings.into_iter().flatten() {
                lo = lo.min(pt);
                hi = hi.max(pt);
            }
            let center = (lo + hi) * 0.5;
            Capsule {
                start: tf.transform_point(center.with_y(lo.y)),
                end: tf.transform_point(center.with_y(hi.y)),
                radius: (hi - lo).xz().length() * 0.5,
            }
        });
        Self { barrel, clip }
    }

    pub fn separate(self, other: Self) -> bool {
        self.barrel.separate(other.barrel)
            && self.clip.is_none_or(|c| c.separate(other.barrel))
            && other.clip.is_none_or(|c| c.separate(self.barrel))
            && self.clip.zip(other.clip).is_none_or(|(a, b)| a.separate(b))
    }
}
