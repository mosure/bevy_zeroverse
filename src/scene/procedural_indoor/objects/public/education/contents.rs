//! Compartment-local editions and poses on the actual shelf surface.
use super::*;
use crate::scene::procedural_indoor::{
    objects::storage::bookcase::contents::{self, Content},
    placement::rotated_half,
};

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &LockerProgram,
    bay: usize,
    row: usize,
    (lo, hi): (f32, f32),
    version: u32,
) {
    let mut rng = stream(
        o.seed,
        0x4c4f_434b_4649_4c4c ^ ((bay as u64) << 8) ^ row as u64,
    );
    if !rng.random_bool(p.contents_probability as f64) {
        return;
    }
    let b = &p.bays[bay];
    let rear = -o.size.z * 0.5 + p.panel_m + 0.008;
    let front = -o.size.z * 0.5 + p.body_depth_m - 0.012;
    let usable = b.width - 0.018;
    let lay_flat = rng.random_bool(0.34);
    let count = if lay_flat { 1 } else { rng.random_range(1..=4) };
    let mut entries = Vec::with_capacity(count);
    let mut span = 0.;
    for _ in 0..count {
        let size = if lay_flat {
            Vec3::new(
                usable * rng.random_range(0.58..0.86),
                (hi - lo) * rng.random_range(0.12..0.25),
                (front - rear) * rng.random_range(0.52..0.76),
            )
            .min(Vec3::new(0.24, 0.10, 0.30))
        } else {
            Vec3::new(
                rng.random_range(0.017..0.047),
                ((hi - lo) * rng.random_range(0.50..0.84)).min(0.33),
                ((front - rear) * rng.random_range(0.48..0.78)).min(0.27),
            )
        };
        let yaw = rng.random_range(-0.16..0.16);
        let half = rotated_half(size.xz(), yaw);
        let gap = if entries.is_empty() {
            0.
        } else {
            rng.random_range(0.001..0.008)
        };
        if span + gap + 2. * half.x > usable || 2. * half.y >= front - rear {
            break;
        }
        let z = rng.random_range(rear + half.y..front - half.y);
        let mut book = Content::upright_book(
            rng.random(),
            size,
            Vec3::new(span + gap + half.x, lo, z),
            yaw,
        );
        book.upright = !lay_flat;
        span += gap + 2. * half.x;
        entries.push(book);
    }
    let offset = b.left + 0.009 + rng.random_range(0.0..=usable - span);
    for mut c in entries {
        c.position.x += offset;
        contents::build(a, o, &c, version);
    }
}
