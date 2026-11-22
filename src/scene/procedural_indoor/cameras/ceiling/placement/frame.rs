//! Separate a moving mount from a sloped beam by its actual underside plane.
//! A whole-span world AABB can extend several decimetres below the high end.
//! The plane is a conservative lower bound for the full section, including flutes.
use super::*;
use crate::scene::procedural_indoor::architecture::ceiling::framing::{FrameMember, FrameSection};

pub(super) struct SlopedFrame {
    bounds: (Vec3, Vec3),
    slope: Vec2,
    intercept: f32,
}
impl SlopedFrame {
    pub(super) fn new(m: &FrameMember) -> Option<Self> {
        // Open webs include tubes protruding from the stock. Keep their full box.
        if m.section == FrameSection::OpenWeb {
            return None;
        }
        let n = (m.top[1] - m.top[0]).cross(m.top[3] - m.top[0]);
        if n.y.abs() < 1e-8 {
            return None;
        }
        let slope = -n.xz() / n.y;
        if slope.length_squared() < 1e-12 {
            return None;
        }
        // Using the lowest of all four corners also bounds a slightly warped cap.
        let intercept = m
            .top
            .iter()
            .map(|v| v.y - slope.dot(v.xz()))
            .fold(f32::INFINITY, f32::min)
            - m.depth
            - 0.0001;
        Some(Self {
            bounds: m.bounds(),
            slope,
            intercept,
        })
    }
    pub(super) fn hits(&self, (lo, hi): (Vec3, Vec3), delta: Vec3) -> bool {
        let (a, b) = self.bounds;
        if hi.max(hi + delta).cmplt(a).any() || lo.min(lo + delta).cmpgt(b).any() {
            return false;
        }
        // Maximise y - slope·xz over the entire query box and its translation.
        let xz = Vec2::select(self.slope.cmpge(Vec2::ZERO), lo.xz(), hi.xz());
        let upper = hi.y - self.slope.dot(xz) + (delta.y - self.slope.dot(delta.xz())).max(0.);
        upper >= self.intercept && segment_hits_box(Vec3::ZERO, delta, a - hi, b - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        architecture::ceiling::framing::FrameRole, materials::Surface,
    };
    #[test]
    fn sloped_stock_clears_a_parallel_rail_but_blocks_penetration_at_either_end() {
        let beam = FrameMember {
            role: FrameRole::Rafter,
            section: FrameSection::Solid,
            top: [
                Vec3::new(-4., 4., -0.2),
                Vec3::new(4., 4.8, -0.2),
                Vec3::new(4., 4.8, 0.2),
                Vec3::new(-4., 4., 0.2),
            ],
            depth: 0.5,
            material: Surface::Wood,
            finish: 0,
            web_pitch: 0.6,
        };
        let frame = SlopedFrame::new(&beam).unwrap();
        let p = Vec3::new(0., 3.83, 0.);
        let r = Vec3::splat(0.04);
        let delta = Vec3::new(3., 0.3, 0.);
        assert!(!frame.hits((p - r, p + r), delta));
        assert!(!frame.hits((p - r + delta, p + r + delta), -delta));
        for dy in [0.1, 0.4] {
            assert!(frame.hits((p - r, p + r), delta + Vec3::Y * dy));
            assert!(frame.hits(
                (p - r + delta + Vec3::Y * dy, p + r + delta + Vec3::Y * dy),
                -delta - Vec3::Y * dy
            ));
        }
    }
}
