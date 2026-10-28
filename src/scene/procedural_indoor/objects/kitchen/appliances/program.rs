//! Physical appliance parameters shared by bounds, construction and diagnostics.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ApplianceProgram {
    pub shell: Surface,
    pub front: Surface,
    pub toe_m: f32,
    pub door_split: f32,
    pub control_fraction: f32,
    pub fascia_fraction: f32,
    /// Actual shelf centre heights, ordered from bottom to top.
    pub shelf_levels_m: Vec<f32>,
    pub shelf_top_m: f32,
}

pub(super) fn proportions(seed: u64) -> (f32, f32) {
    let mut r = stream(seed, 0x4150504c50524f50);
    (r.random_range(0.37..0.63), r.random_range(0.15..0.26))
}

pub(crate) fn microwave_door_width(o: &IndoorObject, p: &KitchenProgram) -> f32 {
    let w = o.size.x - 0.018;
    w * if p.controls == 2 {
        0.94
    } else {
        1. - proportions(o.seed).1
    } - 0.006
}

pub(crate) fn refrigerator_leaf_width(o: &IndoorObject, p: &KitchenProgram) -> f32 {
    let w = o.size.x - 0.036;
    if p.paired {
        let split = proportions(o.seed).0;
        w * split.max(1. - split)
    } else {
        w
    }
}

pub fn parameters(o: &IndoorObject) -> ApplianceProgram {
    let mut r = stream(o.seed, 0x4150504c5348454c);
    let (shell, front) = finishes(o);
    let toe = if o.kind == ObjectKind::Microwave {
        0.012
    } else {
        0.075
    };
    let (door_split, control_fraction) = proportions(o.seed);
    let shelf_top_m = o.size.y - r.random_range(0.045..0.075);
    let mut levels = Vec::new();
    if o.kind == ObjectKind::Refrigerator {
        let bottom = toe + r.random_range(0.03..0.075);
        let count = ((shelf_top_m - bottom) / r.random_range(0.26..0.45))
            .round()
            .clamp(2., 6.) as usize;
        let weights: Vec<f32> = (0..count).map(|_| r.random_range(0.45..1.55)).collect();
        let sum: f32 = weights.iter().sum();
        let minimum = 0.14_f32.min((shelf_top_m - bottom) / count as f32 * 0.65);
        let free = shelf_top_m - bottom - minimum * count as f32;
        let mut y = bottom;
        for weight in weights {
            levels.push(y);
            y += minimum + free * weight / sum;
        }
    }
    ApplianceProgram {
        shell,
        front,
        toe_m: toe,
        door_split,
        control_fraction,
        fascia_fraction: fascia_fraction(o.seed),
        shelf_levels_m: levels,
        shelf_top_m,
    }
}

pub(super) fn finishes(o: &IndoorObject) -> (Surface, Surface) {
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4150504c46494e49);
    let finishes: &[Surface] = if matches!(o.kind, ObjectKind::Range | ObjectKind::Oven) {
        &[Surface::Metal, Surface::Chrome]
    } else {
        &[Surface::Metal, Surface::Chrome, Surface::Plastic]
    };
    let shell = finishes[f.random_range(0..finishes.len())];
    // Timber is a fitted decorative door skin only, never a hot cavity lining.
    let front = if matches!(o.kind, ObjectKind::Refrigerator | ObjectKind::Dishwasher)
        && f.random_bool(0.24)
    {
        Surface::Wood
    } else if f.random_bool(0.55) {
        shell
    } else {
        finishes[f.random_range(0..finishes.len())]
    };
    (shell, front)
}

pub(super) fn fascia_fraction(seed: u64) -> f32 {
    stream(seed, 0x4150504c46415343).random_range(0.055..0.18)
}
pub(crate) fn dishwasher_door_height(o: &IndoorObject) -> f32 {
    (o.size.y * (1. - fascia_fraction(o.seed)) - 0.075 - 0.024 - 0.012).max(0.10)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct OvenFront {
    pub center_x: f32,
    pub bottom_m: f32,
    pub height_m: f32,
    pub width_m: f32,
}
/// Does not call `body`: door travel uses these same measured fronts before
/// expanding the object's occupied depth, including after worktop alignment.
pub fn oven_fronts(o: &IndoorObject, p: &KitchenProgram) -> Vec<OvenFront> {
    let mut r = stream(o.seed, 0x4f56454e46524f4e);
    let tall = o.kind == ObjectKind::Oven && o.size.y > 1.3;
    let n = if tall && p.stacked_ovens { 2 } else { 1 };
    let h = o.size.y;
    let w = o.size.x - 0.018;
    let bottom = if tall {
        h * r.random_range(if n == 2 { 0.20..0.28 } else { 0.31..0.44 })
    } else {
        0.087
    };
    let gap = h * r.random_range(0.035..0.065);
    let span = if tall {
        h * r.random_range(0.75..0.82) - bottom
    } else {
        h * r.random_range(0.68..0.73)
    };
    let first = if n == 2 {
        r.random_range(0.43..0.57)
    } else {
        1.
    };
    let width = w * r.random_range(if p.controls == 0 {
        0.88..0.95
    } else {
        0.68..0.77
    });
    let center_x = match p.controls {
        1 => w * 0.08,
        2 => -w * 0.08,
        _ => 0.,
    };
    let free = span - gap * (n - 1) as f32;
    let mut y = bottom;
    (0..n)
        .map(|i| {
            let height = free * if i == 0 { first } else { 1. - first };
            let f = OvenFront {
                center_x,
                bottom_m: y,
                height_m: height,
                width_m: width,
            };
            y += height + gap;
            f
        })
        .collect()
}
