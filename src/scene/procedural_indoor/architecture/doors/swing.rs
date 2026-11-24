//! A swing door hangs inside its lining. The pin lies just past the opening
//! face of the jamb, so the heel clears the jamb throughout a quarter turn.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwingHanging {
    /// Lining centre relative to the opening datum along its local +Z.
    /// Partitions are centred. The entry datum is the room-side wall face.
    pub lining_offset_m: f32,
    /// Setback of the swing-side leaf face from the front of the jamb.
    pub reveal_m: f32,
    pub barrel_radius_m: f32,
    pub hinge_length_m: f32,
    pub knuckles: u8,
    /// Nominal leaf area per hinge, at a thickness of 42 mm.
    pub area_per_hinge_m2: f32,
    pub bottom_inset_m: f32,
    pub top_inset_m: f32,
    pub spacing_bias: f32,
    pub stop_projection_m: f32,
}

impl SwingHanging {
    pub(super) fn sample(seed: u64, mechanism: DoorMechanism) -> Option<Self> {
        matches!(mechanism, DoorMechanism::Hinged | DoorMechanism::Paired).then(|| {
            let mut r = stream(seed, u64::from_le_bytes(*b"ButtHang"));
            Self {
                lining_offset_m: 0.,
                reveal_m: r.random_range(0.0015..0.010),
                barrel_radius_m: r.random_range(0.0045..0.008),
                hinge_length_m: r.random_range(0.078..0.124),
                knuckles: if r.random_bool(0.6) { 5 } else { 3 },
                area_per_hinge_m2: r.random_range(0.68..1.22),
                bottom_inset_m: r.random_range(0.19..0.31),
                top_inset_m: r.random_range(0.14..0.24),
                spacing_bias: r.random_range(-0.20..0.20),
                stop_projection_m: r.random_range(0.012..0.021),
            }
        })
    }

    pub(super) fn valid(self) -> bool {
        (0.0..=0.15).contains(&self.lining_offset_m)
            && (0.0..=0.012).contains(&self.reveal_m)
            && (0.004..=0.009).contains(&self.barrel_radius_m)
            && (0.075..=0.13).contains(&self.hinge_length_m)
            && matches!(self.knuckles, 3 | 5)
            && (0.65..=1.25).contains(&self.area_per_hinge_m2)
            && (0.18..=0.32).contains(&self.bottom_inset_m)
            && (0.13..=0.25).contains(&self.top_inset_m)
            && (-0.25..=0.25).contains(&self.spacing_bias)
            && (0.012..=0.025).contains(&self.stop_projection_m)
    }

    pub(super) fn leaf_count(self, p: &DoorProgram) -> usize {
        if p.mechanism == DoorMechanism::Paired {
            2
        } else {
            1
        }
    }

    pub(super) fn gauge(self) -> f32 {
        self.barrel_radius_m * 0.30
    }

    pub(super) fn half_gap(self) -> f32 {
        self.gauge() + 0.0006
    }

    pub(super) fn swing_face(self, d: &DoorInstallation) -> f32 {
        // Frame stock projects 10 mm beyond each wall face.
        (d.wall_depth + 0.020) * 0.5 - self.reveal_m
    }

    pub(super) fn fit(self, d: &DoorInstallation, index: usize) -> HangingFit {
        let p = &d.program;
        let paired = p.mechanism == DoorMechanism::Paired;
        let side = if !paired {
            p.handedness
        } else if index == 0 {
            -1.
        } else {
            1.
        };
        let clear_half = (d.width - FRAME_WIDTH_M) * 0.5;
        let gap = self.half_gap();
        let x = clear_half - gap;
        let face = self.swing_face(d);
        let lead = self.reveal_m + self.barrel_radius_m + 0.0008;
        // At a small opening angle it is the back corner, not the latch edge
        // at rest, that reaches furthest toward the other jamb (or other leaf).
        let reach = if paired {
            x - 0.0015
        } else {
            x + clear_half - 0.002
        };
        let back = lead + p.thickness;
        let far_edge = ((reach - back) * (reach + back)).sqrt();
        let width = far_edge - gap;
        let height = d.height - FRAME_WIDTH_M * 0.5 - FLOOR_GAP - 1.5 * gap;
        HangingFit {
            side,
            pin: Vec3::new(side * x, FLOOR_GAP, p.swing_side * (face + lead)),
            offset: Vec3::new(
                -side * (width * 0.5 + gap),
                height * 0.5,
                -p.swing_side * (lead + p.thickness * 0.5),
            ),
            size: Vec3::new(width, height, p.thickness),
        }
    }

    pub fn hinges_per_leaf(self, d: &DoorInstallation) -> usize {
        let size = self.fit(d, 0).size;
        self.hinge_count(size)
    }

    fn hinge_count(self, size: Vec3) -> usize {
        ((size.x * size.y * (size.z / 0.042) / self.area_per_hinge_m2).ceil() as usize).clamp(2, 4)
    }

    pub(super) fn heights(self, size: Vec3) -> impl Iterator<Item = f32> {
        let count = self.hinge_count(size);
        let span = size.y - self.bottom_inset_m - self.top_inset_m;
        (0..count).map(move |i| {
            let t = i as f32 / (count - 1) as f32;
            self.bottom_inset_m + span * (t + self.spacing_bias * t * (1. - t))
        })
    }
}

pub(super) struct HangingFit {
    pub(super) side: f32,
    /// In installation coordinates, at the height of the bottom of the leaf.
    pub(super) pin: Vec3,
    /// Leaf centre relative to the pin with the door shut.
    pub(super) offset: Vec3,
    pub(super) size: Vec3,
}

impl HangingFit {
    pub(super) fn leaf(&self, d: &DoorInstallation) -> Leaf {
        let rotation =
            Quat::from_rotation_y(self.side * d.program.swing_side * d.program.opening * FRAC_PI_2);
        Leaf {
            transform: d.transform().mul_transform(
                Transform::from_translation(self.pin + rotation * self.offset)
                    .with_rotation(rotation),
            ),
            size: self.size,
        }
    }
}
