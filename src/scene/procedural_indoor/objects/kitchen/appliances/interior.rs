//! Real appliance cavities and supported racks, with bounded construction cost.
use super::*;
#[derive(Clone, Debug, serde::Serialize)]
pub struct Basket {
    pub position: Vec3,
    pub size: Vec3,
    pub cutlery: bool,
    pub seed: u64,
}
pub fn dishwasher_baskets(o: &IndoorObject, p: &KitchenProgram) -> Vec<Basket> {
    let product = program::parameters(o);
    let (d, z) = body(o, p);
    let w = o.size.x - 0.018;
    let mut r = stream(o.seed, 0x444953484241534b);
    let bottom = product.toe_m + r.random_range(0.042..0.065);
    let top = o.size.y * (1. - product.fascia_fraction) - 0.035;
    let available = top - bottom;
    let tray = available > 0.55 && r.random_bool(0.45);
    let tray_h = if tray {
        r.random_range(0.055..0.075)
    } else {
        0.
    };
    let gap = r.random_range(0.025..0.042);
    let main = available - tray_h - if tray { 2. * gap } else { gap };
    let split = r.random_range(0.44..0.59);
    let mut y = bottom;
    (0..if tray { 3 } else { 2 })
        .map(|i| {
            let h = match i {
                0 => main * split,
                1 => main * (1. - split),
                _ => tray_h,
            };
            let size = Vec3::new(
                w * r.random_range(0.81..0.90),
                h,
                d * r.random_range(0.73..0.84),
            );
            let position = Vec3::new(
                w * r.random_range(-0.02..0.02),
                y,
                z + d * r.random_range(-0.025..0.025),
            );
            let size = Vec3::new(
                size.x.min(o.size.x - 0.054 - 2. * position.x.abs()),
                size.y,
                size.z.min(d - 0.046 - 2. * (position.z - z).abs()),
            );
            y += h + gap;
            Basket {
                position,
                size,
                cutlery: i == 2,
                seed: r.random(),
            }
        })
        .collect()
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct OvenRack {
    pub y_m: f32,
    pub width_m: f32,
    pub depth_m: f32,
    pub wire_m: f32,
    pub bars: usize,
    pub transverse: bool,
    pub pan: bool,
    pub pan_size: Vec3,
    pub pan_offset: Vec2,
}
pub fn oven_racks(
    o: &IndoorObject,
    p: &KitchenProgram,
    front: &program::OvenFront,
    index: usize,
) -> Vec<OvenRack> {
    let mut r = stream(o.seed, 0x4f56454e5241434b + index as u64);
    let (d, _) = body(o, p);
    let count = r.random_range(1..=3);
    let low = front.bottom_m + front.height_m * r.random_range(0.14..0.23);
    let high = front.bottom_m + front.height_m * r.random_range(0.65..0.77);
    let width = front.width_m * r.random_range(0.85..0.94);
    let depth = d * r.random_range(0.72..0.84);
    let pan_index = r.random_range(0..count);
    (0..count)
        .map(|i| {
            let y = low
                + 0.045 * i as f32
                + (high - low - 0.045 * (count - 1) as f32)
                    * (i as f32 + r.random_range(0.15..0.80))
                    / count as f32;
            let transverse = r.random_bool(0.4);
            let span = if transverse { depth } else { width };
            OvenRack {
                y_m: y,
                width_m: width,
                depth_m: depth,
                wire_m: r.random_range(0.0018..0.0032),
                bars: ((span / r.random_range(0.035..0.065)).round() as usize).clamp(5, 16),
                transverse,
                pan: p.contents && i == pan_index,
                pan_size: Vec3::new(
                    width * r.random_range(0.48..0.75),
                    r.random_range(0.008..0.020),
                    depth * r.random_range(0.48..0.75),
                ),
                pan_offset: Vec2::new(
                    r.random_range(-0.06..0.06) * width,
                    r.random_range(-0.06..0.06) * depth,
                ),
            }
        })
        .collect()
}

pub(super) fn build_oven_rack(a: &mut Assembly, r: &OvenRack, center: Vec2, label: &str) {
    let w = r.width_m * 0.5;
    let d = r.depth_m * 0.5;
    let at = |x: f32, z: f32| Vec3::new(center.x + x, r.y_m, center.y + z);
    for z in [-d, d] {
        a.part(Surface::Metal, label)
            .rod(at(-w, z), at(w, z), r.wire_m);
    }
    for x in [-w, w] {
        a.part(Surface::Metal, label)
            .rod(at(x, -d), at(x, d), r.wire_m);
    }
    for i in 0..r.bars {
        let t = (i as f32 + 1.) / (r.bars + 1) as f32 * 2. - 1.;
        let (a0, b0) = if r.transverse {
            (at(-w, d * t), at(w, d * t))
        } else {
            (at(w * t, -d), at(w * t, d))
        };
        a.part(Surface::Metal, label).rod(a0, b0, r.wire_m * 0.7);
    }
    if r.pan {
        a.box_part(
            Surface::Metal,
            "dishware",
            Vec3::new(
                center.x + r.pan_offset.x,
                r.y_m + r.wire_m * 0.7 + r.pan_size.y * 0.5,
                center.y + r.pan_offset.y,
            ),
            r.pan_size,
            0.002,
        );
    }
}
