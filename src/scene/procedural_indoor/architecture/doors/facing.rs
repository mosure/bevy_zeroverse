//! Veneer and protective sheet stock on solid, machined door faces.
use super::*;
use crate::scene::procedural_indoor::geometry::stock::TimberCuts;
use bevy::math::Affine2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VeneerMatch {
    Independent,
    Continuous,
    Book,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DoorVeneer {
    pub cut_seed: u64,
    pub matching: VeneerMatch,
    /// Grain direction relative to the vertical axis of the leaf.
    pub grain_angle_radians: f32,
}

impl DoorVeneer {
    pub(super) fn sample(seed: u64, p: &DoorProgram) -> Option<Self> {
        if !p.finish.is_cut_timber() || !matches!(p.panel, DoorPanel::Flush | DoorPanel::VisionLite)
        {
            return None;
        }
        let mut r = stream(seed, u64::from_le_bytes(*b"DoorSkin"));
        let matching = if p.mechanism != DoorMechanism::Paired {
            VeneerMatch::Independent
        } else {
            match r.random_range(0..10) {
                0..=3 => VeneerMatch::Independent,
                4..=6 => VeneerMatch::Continuous,
                _ => VeneerMatch::Book,
            }
        };
        let across = if r.random_bool(0.13) { FRAC_PI_2 } else { 0. };
        Some(Self {
            cut_seed: r.random(),
            matching,
            grain_angle_radians: across + r.random_range(-0.037..0.037),
        })
    }
    pub(super) fn valid(self, p: &DoorProgram) -> bool {
        p.finish.is_cut_timber()
            && matches!(p.panel, DoorPanel::Flush | DoorPanel::VisionLite)
            && (self.matching == VeneerMatch::Independent || p.mechanism == DoorMechanism::Paired)
            && (-0.05..=FRAC_PI_2 + 0.05).contains(&self.grain_angle_radians)
    }

    pub(super) fn charts(self, p: &DoorProgram, leaf: Leaf, index: usize) -> [Affine2; 2] {
        let paired =
            p.mechanism == DoorMechanism::Paired && self.matching != VeneerMatch::Independent;
        // Opposite faces come from separate sheets. Matched leaves meet at
        // the latch edge and hold the same cut as they swing independently.
        let mut cuts = TimberCuts::new(self.cut_seed, if paired { 0 } else { index as u64 });
        [-1., 1.].map(|side| {
            let origin = cuts.origin();
            let cut = Affine2::from_angle_translation(self.grain_angle_radians, origin);
            if !paired {
                return cut;
            }
            let right = index != 0;
            let mirror = right && self.matching == VeneerMatch::Book;
            let shift = if right && !mirror { 0.5 } else { -0.5 };
            cut * Affine2::from_scale_angle_translation(
                Vec2::new(if mirror { -1. } else { 1. }, 1.),
                0.,
                Vec2::X * (side * shift * leaf.size.x),
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectedFaces {
    Push,
    Pull,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KickPlateStock {
    pub faces: ProtectedFaces,
    pub finish: Surface,
    pub height_m: f32,
    pub edge_inset_m: f32,
    pub bottom_inset_m: f32,
    pub gauge_m: f32,
    pub corner_clip_m: f32,
}

impl KickPlateStock {
    pub(super) fn eligible(p: &DoorProgram) -> bool {
        matches!(p.mechanism, DoorMechanism::Hinged | DoorMechanism::Paired)
            && matches!(p.panel, DoorPanel::Flush | DoorPanel::VisionLite)
    }

    pub(super) fn sample(seed: u64, p: &DoorProgram) -> Option<Self> {
        if !Self::eligible(p) {
            return None;
        }
        let mut r = stream(seed, u64::from_le_bytes(*b"KickPlat"));
        if !r.random_bool(if p.handle == DoorHandle::PanicBar {
            0.79
        } else {
            0.46
        }) {
            return None;
        }
        let faces = match r.random_range(0..12) {
            0 => ProtectedFaces::Pull,
            1..=4 => ProtectedFaces::Both,
            _ => ProtectedFaces::Push,
        };
        let height_m = match r.random_range(0..10) {
            0..=1 => r.random_range(0.085..0.155),
            2..=7 => r.random_range(0.16..0.36),
            _ => r.random_range(0.38..0.62),
        };
        let finish = match r.random_range(0..10) {
            0..=5 => Surface::Metal,
            6..=7 => Surface::Chrome,
            _ => Surface::Plastic,
        };
        Some(Self {
            faces,
            finish,
            height_m,
            edge_inset_m: r.random_range(0.012..0.042),
            bottom_inset_m: r.random_range(0.004..0.026),
            gauge_m: if finish == Surface::Plastic {
                r.random_range(0.0018..0.0032)
            } else {
                r.random_range(0.0009..0.0017)
            },
            corner_clip_m: r.random_range(0.0015..0.014),
        })
    }

    pub(super) fn valid(self, p: &DoorProgram) -> bool {
        Self::eligible(p)
            && matches!(
                self.finish,
                Surface::Metal | Surface::Chrome | Surface::Plastic
            )
            && (0.065..=0.64).contains(&self.height_m)
            && (0.009..=0.045).contains(&self.edge_inset_m)
            && (0.003..=0.03).contains(&self.bottom_inset_m)
            && (0.0008..=0.0035).contains(&self.gauge_m)
            && (0.001..=0.016).contains(&self.corner_clip_m)
    }
}

/// Installed protection in leaf-local metres at the door's current rest pose.
#[derive(Clone, Copy, Debug)]
pub struct FittedKickPlate {
    pub leaf_index: usize,
    pub leaf_transform: Transform,
    pub side: f32,
    pub bounds: Rect,
    pub base_z: f32,
    pub gauge_m: f32,
    pub corner_clip_m: f32,
    pub finish: Surface,
}
