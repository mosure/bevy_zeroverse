//! Door-local papers; positions, formats, headings and mounts are independent.
//! They stay on the real skin/viewport and share their door's motion ownership.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Clone, Debug, serde::Serialize)]
pub struct DoorNote {
    pub center: Vec2,
    pub size: Vec2,
    pub yaw: f32,
    pub seed: u64,
    pub photo: bool,
}

pub fn placements(
    seed: u64,
    width: f32,
    height: f32,
    side: f32,
    p: &KitchenProgram,
) -> Vec<DoorNote> {
    let mut r = stream(seed, 0x4652494447454e4f);
    // Keep paper on the raised field (or inside the viewport), not across trim.
    let half = Vec2::new(width * 0.34, height * 0.36);
    let pattern = SurfacePattern::sample(&mut r);
    let handle = Footprint {
        center: Vec2::new(-side * (width * 0.5 - 0.045), 0.),
        half: Vec2::new(0.025, 0.075),
        yaw: 0.,
    };
    let mut footprints = Vec::new();
    let mut notes = Vec::new();
    for _ in 0..64 {
        if notes.len() >= p.door_notes as usize {
            break;
        }
        let w = (width * r.random_range(0.13..0.38)).min(0.22);
        let h = (w * r.random_range(0.6..1.65)).min(height * 0.24);
        let yaw = r.random_range(-0.28..0.28);
        let Some(q) = pattern.propose((-half, half), Vec2::new(w, h), yaw, 0.003, &mut r) else {
            continue;
        };
        if (p.handles != 4 && q.overlaps(handle, 0.006))
            || footprints.iter().any(|&old| q.overlaps(old, 0.008))
        {
            continue;
        }
        footprints.push(q);
        notes.push(DoorNote {
            center: q.center,
            size: Vec2::new(w, h),
            yaw,
            seed: r.random(),
            photo: r.random_bool(0.48),
        });
    }
    notes
}

pub(super) fn build(a: &mut Assembly, notes: &[DoorNote], door: Transform, p: &KitchenProgram) {
    let front = if p.viewport { 0.0032 } else { 0.0162 };
    for note in notes {
        let mut r = stream(note.seed, 0x4e4f5445434f4e54);
        let tf = door.mul_transform(
            Transform::from_xyz(note.center.x, note.center.y, front)
                .with_rotation(Quat::from_rotation_z(note.yaw)),
        );
        let label = format!("paper#finish{}", r.random_range(0..12));
        box_at(
            a,
            if note.photo {
                Surface::Art
            } else {
                Surface::Paper
            },
            &label,
            Vec3::ZERO,
            note.size.extend(0.0008),
            tf,
        );
        if !note.photo {
            let count = r.random_range(2..=6);
            let left = note.size.x * r.random_range(-0.38..-0.22);
            for line in 0..count {
                let width = note.size.x * r.random_range(0.32..0.68);
                box_at(
                    a,
                    Surface::Ink,
                    "paper",
                    Vec3::new(
                        left + width * 0.5,
                        note.size.y * (0.30 - line as f32 * 0.58 / count as f32),
                        0.00065,
                    ),
                    Vec3::new(width, note.size.y * 0.014, 0.00025),
                    tf,
                );
            }
        }
        let x = note.size.x * r.random_range(-0.30..0.30);
        let y = note.size.y * 0.38;
        if p.viewport {
            // Adhesive tab on a glass door; no unsupported magnet assumption.
            box_at(
                a,
                Surface::Paper,
                "paper",
                Vec3::new(x, y, 0.0007),
                Vec3::new(note.size.x * 0.28, note.size.y * 0.14, 0.0003),
                tf,
            );
        } else {
            a.part(Surface::Plastic, &label).cylinder(
                (note.size.min_element() * r.random_range(0.055..0.10)).min(0.012),
                0.003,
                tf.mul_transform(
                    Transform::from_xyz(x, y, 0.0022)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                ),
            );
        }
    }
}
