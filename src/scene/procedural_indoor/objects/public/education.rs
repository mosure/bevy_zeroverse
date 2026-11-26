//! Metric locker compartments, manufactured fronts and individually sampled use states.
use super::*;
use rand_chacha::ChaCha8Rng;
mod contents;
mod fronts;
mod sheet;
pub use fronts::{Fittings, Handing};
pub use sheet::VentStock;
pub const FITTED_VERSION: u32 = 219;
pub(crate) const WALL_BANK: u32 = 16;
pub(crate) const FREESTANDING_BANK: u32 = 17;
pub fn installation(o: &IndoorObject) -> Option<&'static str> {
    match o.variant {
        WALL_BANK => Some("wall_backed"),
        FREESTANDING_BANK => Some("freestanding"),
        _ => None,
    }
}
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, serde::Serialize)]
pub struct LockerBay {
    pub left: f32,
    pub width: f32,
    pub levels: Vec<f32>,
    pub open: Vec<bool>,
    pub angles: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LockerProgram {
    /// Steel vented, laminate flush, framed glazed, or open cubbies.
    pub front: u8,
    pub handle: u8,
    pub feet_m: f32,
    pub panel_m: f32,
    pub body_depth_m: f32,
    pub vent_rows: usize,
    pub contents_probability: f32,
    pub bays: Vec<LockerBay>,
    pub finish: Surface,
    pub accent: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fittings: Option<Fittings>,
}

pub fn parameters(o: &IndoorObject) -> LockerProgram {
    parameters_at(
        o,
        crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
    )
}
pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> LockerProgram {
    let mut r = stream(o.seed, 0x4c4f_434b_4552_4745);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4c4f_434b_4552_4649);
    let front = r.random_range(0..4);
    let t = r.random_range(0.014..0.023);
    let feet = r.random_range(0.035..0.13);
    let n = (o.size.x / r.random_range(0.25..0.44))
        .round()
        .clamp(2., 6.) as usize;
    let fittings = (version >= FITTED_VERSION).then(|| Fittings::sample(o, front));
    // Equal modules remain common in manufactured lockers, mixed-width banks
    // and independently tiered compartments also occur. Never perturb joins.
    let mixed = r.random_bool(0.48);
    let weights: Vec<f32> = (0..n)
        .map(|_| if mixed { r.random_range(0.8..1.25) } else { 1. })
        .collect();
    let sum: f32 = weights.iter().sum();
    let mut x = -o.size.x * 0.5 + t;
    let depth = o.size.z * r.random_range(0.72..0.84);
    let rows = r.random_range(1..=4);
    let openness = r.random_range(0.0..0.48);
    let mut bays = Vec::with_capacity(n);
    for weight in weights {
        let width = (o.size.x - t * (n + 1) as f32) * weight / sum;
        let rows = if mixed { r.random_range(1..=4) } else { rows };
        let rows = rows.min(((o.size.y - feet) / 0.31).floor().max(1.) as usize);
        let levels: Vec<f32> = (0..=rows)
            .map(|i| feet + t + (o.size.y - feet - t * 2.) * i as f32 / rows as f32)
            .collect();
        let open: Vec<bool> = (0..rows)
            .map(|_| front != 3 && r.random_bool(openness))
            .collect();
        // Opening reserve belongs to the admitted envelope. A single hinged
        // leaf replaces the previous pair of shrinking sliding panels.
        let limit = ((o.size.z - depth - if fittings.is_some() { t + 0.026 } else { 0.030 })
            / width)
            .clamp(0.02, 0.65)
            .asin();
        let angles = open
            .iter()
            .map(|&open| {
                if open {
                    r.random_range(0.35..1.) * limit
                } else {
                    0.
                }
            })
            .collect();
        bays.push(LockerBay {
            left: x,
            width,
            levels,
            open,
            angles,
        });
        x += width + t;
    }
    LockerProgram {
        front,
        handle: r.random_range(0..3),
        feet_m: feet,
        panel_m: t,
        body_depth_m: depth,
        vent_rows: r.random_range(2..=6),
        contents_probability: r.random_range(0.15..0.8),
        bays,
        finish: if front == 0 {
            if fittings.is_some() {
                Surface::Metal
            } else {
                Surface::Plastic
            }
        } else {
            [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)]
        },
        accent: [Surface::Metal, Surface::Chrome, Surface::Plastic][f.random_range(0..3)],
        fittings,
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    let s = o.size;
    let label = p.fittings.as_ref().and_then(|v| v.case_label());
    let l = label.as_deref().unwrap_or(o.kind.class_name());
    let t = p.panel_m;
    let rear = -s.z * 0.5;
    let front = rear + p.body_depth_m;
    let low = p.feet_m;
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., low * 0.5, (rear + front) * 0.5),
        Vec3::new(s.x * 0.88, low, p.body_depth_m * 0.85),
        0.003,
    );
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., (low + s.y) * 0.5, rear + t * 0.5),
        Vec3::new(s.x, s.y - low, t),
        0.002,
    );
    for y in [low + t * 0.5, s.y - t * 0.5] {
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., y, (rear + t + front) * 0.5),
            Vec3::new(s.x, t, p.body_depth_m - t),
            0.002,
        );
    }
    for x in std::iter::once(-s.x * 0.5 + t * 0.5)
        .chain(p.bays.iter().map(|b| b.left + b.width + t * 0.5))
    {
        a.box_part(
            p.finish,
            l,
            Vec3::new(x, (low + s.y) * 0.5, (rear + t + front) * 0.5),
            Vec3::new(t, s.y - low - 2. * t, p.body_depth_m - t),
            0.002,
        );
    }
    let mut r = stream(o.seed, 0x4c4f_434b_434f_4e54);
    for (bay_index, b) in p.bays.iter().enumerate() {
        for (row, limits) in b.levels.windows(2).enumerate() {
            let bottom = limits[0];
            let top = limits[1];
            if row > 0 {
                a.box_part(
                    p.finish,
                    l,
                    Vec3::new(b.left + b.width * 0.5, bottom, (rear + t + front) * 0.5),
                    Vec3::new(b.width, t, p.body_depth_m - t),
                    0.002,
                );
            }
            let lo = bottom + if row > 0 { t * 0.5 } else { 0. };
            let hi = top - if row + 1 < b.open.len() { t * 0.5 } else { 0. };
            if let Some(fittings) = &p.fittings {
                if matches!(p.front, 2 | 3) || b.open[row] {
                    contents::build(a, o, &p, bay_index, row, (lo, hi), version);
                }
                if p.front != 3 {
                    fittings.door(a, o, &p, bay_index, row, (lo, hi), l);
                }
            } else {
                legacy::cell(a, o, &p, b, row, (lo, hi), &mut r, version);
            }
        }
    }
}

mod legacy;
