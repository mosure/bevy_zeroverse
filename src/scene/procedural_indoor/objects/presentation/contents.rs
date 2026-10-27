//! Bounded support-local composition, independent from accessory pigments.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrayItem {
    pub eraser: bool,
    pub size: Vec3,
    pub center: Vec2,
    pub yaw: f32,
    pub cap_fraction: f32,
    pub finish_slot: usize,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Magnet {
    pub center: Vec2,
    pub radius_m: f32,
    pub finish_slot: usize,
}

pub(super) fn tray_items(o: &IndoorObject, p: &BoardProgram) -> Vec<TrayItem> {
    let mut r = stream(o.seed, 0x424f415244545241);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4d41524b46494e49);
    let pattern = SurfacePattern::sample(&mut r);
    let requested = r.random_range(0..=6);
    let want_eraser = r.random_bool(0.45);
    let mut occupied: Vec<Footprint> = Vec::new();
    let mut items = Vec::new();
    for i in 0..requested + usize::from(want_eraser) {
        let eraser = i == requested;
        let radius = r.random_range(0.004..0.0065);
        let width = p.tray_hi.x - p.tray_lo.x;
        let depth = p.tray_hi.y - p.tray_lo.y;
        let size = if eraser {
            Vec3::new(
                r.random_range(0.070_f32..0.13).min(width * 0.60),
                r.random_range(0.012..0.022),
                r.random_range(0.022_f32..0.043).min(depth * 0.60),
            )
        } else {
            Vec3::new(
                r.random_range(0.095_f32..0.16).min(width * 0.80),
                radius * 2.,
                radius * 2.,
            )
        };
        let cap = r.random_range(0.12..0.24);
        for _ in 0..16 {
            // Narrow trays favor nearly horizontal markers but retain continuous
            // yaw and both directions. Broader trays admit larger angles.
            let max_yaw = ((depth - size.z - 0.012).max(0.) / size.x).min(0.7).asin();
            let yaw =
                r.random_range(-1.0..1.0) * max_yaw + if r.random_bool(0.5) { PI } else { 0. };
            let Some(q) = pattern.propose((p.tray_lo, p.tray_hi), size.xz(), yaw, 0.005, &mut r)
            else {
                continue;
            };
            if occupied.iter().any(|&a| a.overlaps(q, 0.004)) {
                continue;
            }
            occupied.push(q);
            items.push(TrayItem {
                eraser,
                size,
                center: q.center,
                yaw,
                cap_fraction: cap,
                finish_slot: f
                    .random_range(0..crate::scene::procedural_indoor::materials::variants::COUNT),
            });
            break;
        }
    }
    items
}

pub(super) fn magnets(o: &IndoorObject, p: &BoardProgram) -> Vec<Magnet> {
    let mut r = stream(o.seed, 0x424f4152444d4147);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4d41474e46494e49);
    let mut result: Vec<Magnet> = Vec::new();
    let count = if r.random_bool(0.50) {
        0
    } else {
        r.random_range(1..=6)
    };
    for _ in 0..count {
        let radius = r.random_range(0.0045..0.010);
        let lo = Vec2::new(
            -o.size.x * 0.5 + p.border_m + radius + 0.006,
            (p.tray_base_top_m + p.tray_lip_height_m + 0.018).max(p.border_m + radius),
        );
        let hi = Vec2::new(
            o.size.x * 0.5 - p.border_m - radius - 0.006,
            o.size.y - p.border_m - radius - 0.006,
        );
        if lo.cmpge(hi).any() {
            continue;
        }
        for _ in 0..12 {
            let center = Vec2::new(r.random_range(lo.x..hi.x), r.random_range(lo.y..hi.y));
            if result
                .iter()
                .any(|v| v.center.distance(center) < v.radius_m + radius + 0.006)
            {
                continue;
            }
            result.push(Magnet {
                center,
                radius_m: radius,
                finish_slot: f
                    .random_range(0..crate::scene::procedural_indoor::materials::variants::COUNT),
            });
            break;
        }
    }
    result
}

pub(super) fn build_item(a: &mut Assembly, o: &IndoorObject, p: &BoardProgram, item: &TrayItem) {
    let label = format!("{}#finish{}", o.kind.class_name(), item.finish_slot);
    let tf = Transform::from_xyz(item.center.x, p.tray_base_top_m, item.center.y)
        .with_rotation(Quat::from_rotation_y(item.yaw));
    let s = item.size;
    if item.eraser {
        a.part(Surface::Rubber, &label).cuboid(
            Vec3::new(s.x, 0.003, s.z),
            0.0005,
            tf.with_translation(tf.transform_point(Vec3::Y * 0.0015)),
        );
        a.part(Surface::Plastic, &label).cuboid(
            Vec3::new(s.x, s.y - 0.003, s.z),
            0.002,
            tf.with_translation(tf.transform_point(Vec3::Y * (s.y + 0.003) * 0.5)),
        );
    } else {
        let start = -s.x * 0.5;
        let end = s.x * 0.5;
        let cap = s.x * item.cap_fraction;
        for (surface, left, right, radius) in [
            (Surface::Plastic, start, start + cap, s.y * 0.5),
            (Surface::Paper, start + cap, end - cap * 0.30, s.y * 0.45),
            (Surface::Plastic, end - cap * 0.30, end, s.y * 0.5),
        ] {
            a.part(surface, &label).rod(
                tf.transform_point(Vec3::new(left, s.y * 0.5, 0.)),
                tf.transform_point(Vec3::new(right, s.y * 0.5, 0.)),
                radius,
            );
        }
    }
}
