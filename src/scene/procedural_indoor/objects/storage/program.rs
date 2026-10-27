//! Shared cabinet compartments, finishes and operating clearances.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct CabinetBay {
    pub left: f32,
    pub right: f32,
    /// Board centre heights; drawer fronts use these same boundaries.
    pub levels: Vec<f32>,
    pub drawers: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShelfProgram {
    pub levels: usize,
    pub columns: usize,
    pub open_back: bool,
    pub metal: bool,
    pub fill: f32,
    pub panel_m: f32,
    /// Hinged, drawer, mixed, partially open, sliding, slatted fronts.
    pub front: u8,
    pub handle: u8,
    pub open_fraction: f32,
    pub opening_radians: f32,
    pub bays: Vec<CabinetBay>,
    pub bottom_m: f32,
    pub body_depth_m: f32,
    pub frame: Surface,
    pub shelves: Surface,
    pub fronts: Surface,
    pub handle_height_fraction: f32,
    pub handle_span_fraction: f32,
    pub slat_pitch_m: f32,
    pub slat_fill: f32,
}

pub fn parameters(o: &IndoorObject) -> ShelfProgram {
    let mut r = stream(o.seed, 3050);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4341_4249_4649_4e49);
    let s = o.size;
    let t = r.random_range(0.015_f32..0.032).min(s.min_element() * 0.07);
    let front = r.random_range(0..6);
    let mut columns = (s.x / r.random_range(0.36..0.72)).round().clamp(1., 4.) as usize;
    if matches!(front, 2 | 4) {
        columns = columns.max(2);
    }
    if front == 4 {
        columns = 2;
    }
    let bottom = r.random_range(0.035_f32..0.13).min(s.y * 0.15).max(t);
    let top = s.y - t * 0.5;
    let count = ((top - bottom) / r.random_range(0.23..0.43))
        .round()
        .clamp(2., 6.) as usize;
    let aligned = r.random_bool(0.30);
    let weights: Vec<f32> = (0..columns).map(|_| r.random_range(0.7..1.4)).collect();
    let sum: f32 = weights.iter().sum();
    let clear = s.x - t * (columns + 1) as f32;
    let mixed_drawer = r.random_range(0..columns);
    let mut left = -s.x * 0.5 + t;
    let mut bays: Vec<CabinetBay> = Vec::with_capacity(columns);
    for (index, weight) in weights.into_iter().enumerate() {
        let right = left + clear * weight / sum;
        let levels = if aligned && index > 0 {
            bays[0].levels.clone()
        } else {
            let spaces: Vec<f32> = (0..count).map(|_| r.random_range(0.45..1.55)).collect();
            let sum: f32 = spaces.iter().sum();
            let minimum = ((top - bottom) / count as f32 * 0.6).min(0.16);
            let free = top - bottom - minimum * count as f32;
            let mut levels = vec![bottom];
            for weight in spaces {
                levels.push(levels.last().unwrap() + minimum + free * weight / sum);
            }
            *levels.last_mut().unwrap() = top;
            levels
        };
        bays.push(CabinetBay {
            left,
            right,
            levels,
            drawers: front == 1 || (front == 2 && index == mixed_drawer),
        });
        left = right + t;
    }
    let widest = bays.iter().map(|b| b.right - b.left).fold(0., f32::max);
    let opening = if matches!(front, 0 | 2 | 5) && r.random_bool(0.55) {
        ((s.z * 0.19).min(0.10) / widest).min(0.8).asin() * r.random_range(0.25..1.)
    } else {
        0.
    };
    let substrates = [Surface::Wood, Surface::Metal, Surface::Plastic];
    let frame = substrates[f.random_range(0..substrates.len())];
    ShelfProgram {
        levels: count,
        columns,
        open_back: false,
        metal: frame == Surface::Metal,
        fill: r.random_range(0.20..0.92),
        panel_m: t,
        front,
        handle: r.random_range(0..4),
        open_fraction: r.random_range(0.22..0.92),
        opening_radians: opening,
        bays,
        bottom_m: bottom,
        body_depth_m: s.z - 2. * widest * opening.sin(),
        frame,
        shelves: if f.random_bool(0.6) {
            frame
        } else {
            substrates[f.random_range(0..substrates.len())]
        },
        fronts: if f.random_bool(0.6) {
            frame
        } else {
            substrates[f.random_range(0..substrates.len())]
        },
        handle_height_fraction: r.random_range(0.40..0.70),
        handle_span_fraction: r.random_range(0.16..0.38),
        slat_pitch_m: r.random_range(0.026..0.080),
        slat_fill: r.random_range(0.25..0.60),
    }
}

pub(super) fn open_levels(p: &ShelfProgram, bay: &CabinetBay) -> usize {
    ((bay.levels.len() - 1) as f32 * p.open_fraction)
        .round()
        .max(1.) as usize
}
