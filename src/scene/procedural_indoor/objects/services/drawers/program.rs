//! Actual drawer-bank boundaries, shared by mesh, motion and diagnostics.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DrawerBank {
    pub left: f32,
    pub right: f32,
    pub first: usize,
    pub count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DrawerFront {
    pub index: usize,
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DrawerProgram {
    /// Per-drawer weights, grouped by bank; length is the total moving count.
    pub weights: Vec<f32>,
    pub banks: Vec<DrawerBank>,
    pub panel: f32,
    pub base: u8,
    pub foot: f32,
    pub foot_span: Vec2,
    pub top: f32,
    pub front: Surface,
    pub frame: Surface,
    pub profile: u8,
    pub hardware: u8,
    pub hardware_finish: Surface,
    pub handle_depth: f32,
    pub handle_width: f32,
    pub handle_height: f32,
}

/// Pedestals through wider low storage. Width is continuous; construction only
/// adds another bank when there is room for real drawer boxes and a partition.
pub fn dimensions(r: &mut impl Rng) -> Vec3 {
    Vec3::new(
        r.random_range(0.34_f32.ln()..0.96_f32.ln()).exp(),
        r.random_range(0.48..0.85),
        r.random_range(0.36..0.59),
    )
}

pub fn parameters(o: &IndoorObject) -> DrawerProgram {
    let mut r = stream(o.seed, 0x4452415745525052);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4452_4157_4649_4e49);
    let s = o.size;
    let foot = r.random_range(0.035_f32..0.095).min(s.y * 0.15);
    let top = r.random_range(0.016_f32..0.033).min(s.y * 0.075);
    let panel = 0.018_f32.min((s.y - foot - top).min(s.x * 0.98).min(s.z - 0.065) * 0.08);
    let available = s.y - foot - top - 2. * panel - 0.008;
    // Compact tool/stationery drawers can use an 85 mm pitch. A 100 mm
    // capacity cutoff unnecessarily removed valid five-drawer pedestals.
    let capacity = ((available / 0.085).floor() as usize).clamp(2, 5);
    let width = s.x * 0.98 - 2. * panel - 0.006;
    let mut boundaries = vec![-width * 0.5];
    if width > 0.53 && r.random_bool(0.72) {
        boundaries.push(width * r.random_range(-0.12..0.12));
    }
    boundaries.push(width * 0.5);
    let mut weights = Vec::new();
    let mut banks = Vec::new();
    for (index, pair) in boundaries.windows(2).enumerate() {
        let count = r.random_range(2..=capacity);
        let first = weights.len();
        weights.extend((0..count).map(|_| r.random_range(0.55..1.65)));
        banks.push(DrawerBank {
            left: pair[0] + if index > 0 { 0.004 } else { 0. },
            right: pair[1]
                - if index + 2 < boundaries.len() {
                    0.004
                } else {
                    0.
                },
            first,
            count,
        });
    }
    let finishes = [Surface::Wood, Surface::Metal, Surface::Plastic];
    let frame = finishes[f.random_range(0..3)];
    DrawerProgram {
        weights,
        banks,
        panel,
        foot,
        top,
        base: r.random_range(0..3),
        foot_span: Vec2::new(r.random_range(0.32..0.41), r.random_range(0.29..0.39)),
        front: if f.random_bool(0.60) {
            frame
        } else {
            finishes[f.random_range(0..3)]
        },
        frame,
        profile: r.random_range(0..3),
        hardware: r.random_range(0..3),
        hardware_finish: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Wood,
            Surface::Plastic,
        ][f.random_range(0..4)],
        handle_depth: r.random_range(0.013..0.027),
        handle_width: r.random_range(0.17..0.39),
        handle_height: r.random_range(-0.06..0.31),
    }
}

impl DrawerProgram {
    pub fn fronts(&self, height: f32) -> impl Iterator<Item = DrawerFront> + '_ {
        self.banks.iter().flat_map(move |bank| {
            let weights = &self.weights[bank.first..bank.first + bank.count];
            let total: f32 = weights.iter().sum();
            let available = height - self.foot - self.top - 2. * self.panel - 0.008;
            let minimum = (available / bank.count as f32 * 0.65).min(0.075);
            let free = available - minimum * bank.count as f32;
            weights.iter().enumerate().scan(
                self.foot + self.panel + 0.004,
                move |bottom, (i, w)| {
                    let pitch = minimum + free * w / total;
                    let front = DrawerFront {
                        index: bank.first + i,
                        left: bank.left,
                        right: bank.right,
                        bottom: *bottom + 0.003,
                        top: *bottom + pitch - 0.003,
                    };
                    *bottom += pitch;
                    Some(front)
                },
            )
        })
    }
}
