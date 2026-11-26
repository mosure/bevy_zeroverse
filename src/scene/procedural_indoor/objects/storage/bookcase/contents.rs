//! Small bounded shelf assemblies. Sampling and mesh tests use the same occupied volumes.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint, SurfacePattern};

#[derive(Clone, Debug)]
pub(crate) struct Content {
    pub kind: ObjectKind,
    pub upright: bool,
    pub seed: u64,
    pub size: Vec3,
    pub position: Vec3,
    pub yaw: f32,
    /// Rotation about the shelf depth axis, with the origin at the book's lower edge.
    pub lean: f32,
    pub edition: Option<u64>,
    pub bookend: Option<runs::Bookend>,
}

impl Content {
    pub(crate) fn upright_book(seed: u64, size: Vec3, position: Vec3, yaw: f32) -> Self {
        Self {
            kind: ObjectKind::Books,
            upright: true,
            seed,
            size,
            position,
            yaw,
            lean: 0.,
            edition: None,
            bookend: None,
        }
    }
}

pub(crate) fn placements(
    o: &IndoorObject,
    p: &BookcaseProgram,
    bay: usize,
    level: usize,
) -> Vec<Content> {
    let mut r = stream(o.seed, 0x424f_4f4b_434f_4e54 + (bay * 16 + level) as u64);
    let b = &p.bays[bay];
    let (rear, front) = p.deck_limits(o.size.z);
    let bounds = (Vec2::new(b.left, rear), Vec2::new(b.right, front));
    let stock_depth = if p.rack.is_some() {
        // Preserve ordinary book/box stock on deeper cases. Even the deepest
        // upright volume gets some fore-aft play on a shallow recessed deck.
        o.size.z.min((front - rear) * 1.10)
    } else {
        o.size.z
    };
    let base = b.levels[level] + p.panel_m * 0.5;
    let headroom = b.levels[level + 1] - p.panel_m * 0.5 - base - 0.006;
    let mut result = Vec::new();
    if r.random_bool((1. - p.fill) as f64 * 0.55) {
        return result;
    }
    let theme = r.random_range(0..6);
    if theme < 3 {
        let count = ((b.right - b.left) * p.fill / 0.045).round().clamp(2., 10.) as usize;
        let mut x = b.left + r.random_range(0.01..0.10) * (b.right - b.left);
        for _ in 0..count {
            let size = Vec3::new(
                r.random_range(0.022..0.055),
                (headroom * r.random_range(0.60..0.94)).min(0.37),
                (stock_depth * r.random_range(0.48..0.80)).min(0.31),
            );
            let yaw = r.random_range(-0.10..0.10);
            let half = rotated_half(size.xz(), yaw);
            if x + half.x * 2. > b.right - 0.012 {
                break;
            }
            let space = rear + half.y + 0.006..front - half.y - 0.006;
            if space.is_empty() {
                continue;
            }
            let z = r.random_range(space);
            result.push(Content::upright_book(
                r.random(),
                size,
                Vec3::new(x + half.x, base, z),
                yaw,
            ));
            x += half.x * 2. + r.random_range(0.002..0.013);
        }
        // Shelved books may be left-, right- or loosely aligned. Move the
        // already non-overlapping row within its usable interval; no clamping
        // or privileged first proposal creates an atom at the left divider.
        if !result.is_empty() {
            let left = result
                .iter()
                .map(|c| c.position.x - rotated_half(c.size.xz(), c.yaw).x)
                .fold(f32::INFINITY, f32::min);
            let right = result
                .iter()
                .map(|c| c.position.x + rotated_half(c.size.xz(), c.yaw).x)
                .fold(f32::NEG_INFINITY, f32::max);
            let lo = b.left + 0.004 - left;
            let hi = b.right - 0.004 - right;
            if hi > lo {
                let shift = r.random_range(lo..hi);
                for c in &mut result {
                    c.position.x += shift;
                }
            }
        }
        runs::arrange(o, p, bay, level, &mut result);
        return result;
    }
    let pattern = SurfacePattern::sample(&mut r);
    let mut occupied: Vec<Footprint> = Vec::new();
    for _ in 0..r.random_range(1..=3) {
        let kind = match theme {
            3 => ObjectKind::Books,
            4 => ObjectKind::StorageBox,
            _ => ObjectKind::Vase,
        };
        let seed = r.random();
        let size = match kind {
            ObjectKind::Books => Vec3::new(
                r.random_range(0.13..0.24),
                r.random_range(0.04_f32..0.14).min(headroom * 0.85),
                r.random_range(0.16..0.27),
            ),
            ObjectKind::StorageBox => Vec3::new(
                r.random_range(0.16..0.34),
                headroom * r.random_range(0.42..0.78),
                stock_depth * r.random_range(0.52..0.76),
            ),
            _ => {
                let d = r.random_range(0.075..0.16);
                Vec3::new(d, (d * r.random_range(0.8..2.)).min(headroom * 0.90), d)
            }
        };
        for _ in 0..12 {
            let yaw = pattern.yaw(&mut r);
            let Some(q) = pattern.propose(bounds, size.xz(), yaw, 0.012, &mut r) else {
                continue;
            };
            if occupied.iter().any(|a| a.overlaps(q, 0.01)) {
                continue;
            }
            occupied.push(q);
            result.push(Content {
                kind,
                upright: false,
                seed,
                size,
                position: Vec3::new(q.center.x, base, q.center.y),
                yaw,
                lean: 0.,
                edition: None,
                bookend: None,
            });
            break;
        }
    }
    result
}

pub(crate) fn build(a: &mut Assembly, parent: &IndoorObject, c: &Content, version: u32) {
    let tf = Transform::from_translation(c.position)
        .with_rotation(Quat::from_rotation_y(c.yaw) * Quat::from_rotation_z(c.lean));
    if c.kind == ObjectKind::Books && !c.upright {
        // Bound volumes share one construction in either orientation. Avoid
        // expanding a shelf stack into many rings of spiral-notebook geometry.
        let mut r = stream(c.seed, 0x424f_4f4b_5354_414b);
        let n = r.random_range(2..=4);
        let weights: Vec<f32> = (0..n).map(|_| r.random_range(0.7..1.3)).collect();
        let total: f32 = weights.iter().sum();
        let mut y = 0.;
        for weight in weights {
            let h = c.size.y * weight / total;
            let angle: f32 = r.random_range(-0.12..0.12);
            let width = c.size.x * r.random_range(0.82..1.);
            let depth = c.size.z * r.random_range(0.82..1.);
            let fit = (c.size.x / (width * angle.cos().abs() + depth * angle.sin().abs()))
                .min(c.size.z / (depth * angle.cos().abs() + width * angle.sin().abs()))
                .min(1.);
            let w = width * fit;
            let q = Quat::from_rotation_y(angle);
            let local = q * Vec3::new(-w * 0.5, y + h * 0.5, 0.);
            binding::volume(
                a,
                r.random(),
                None,
                Vec3::new(h, w, depth * fit),
                Transform::from_translation(tf.transform_point(local))
                    .with_rotation(tf.rotation * q * Quat::from_rotation_z(-FRAC_PI_2)),
                version,
            );
            y += h;
        }
        return;
    }
    if !c.upright {
        let mut child = parent.clone();
        child.kind = c.kind;
        child.seed = c.seed;
        child.finish_seed = None;
        child.size = c.size;
        super::super::super::public::append_at(a, build_object(&child), tf);
        return;
    }
    binding::volume(a, c.seed, c.edition, c.size, tf, version);
    if let Some(bookend) = &c.bookend {
        bookend.build(a, parent, c);
    }
}
