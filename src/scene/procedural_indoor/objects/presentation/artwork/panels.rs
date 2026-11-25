//! Wall-local stretcher sizes for a single image hung across multiple canvases.
use super::*;

pub(crate) const INTRODUCED: u32 = 237;
const SINGLE_STYLES: u32 = 6;
const STRETCHER_MIN_M: f32 = 0.22;
const INSTALLATION_MIN_M: f32 = 0.55;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelAxis {
    /// A row from left to right on the wall.
    Horizontal,
    /// A column from bottom to top.
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct CanvasPanel {
    /// Bounds in the artwork's local X/Y plane, in metres.
    pub min_m: Vec2,
    pub max_m: Vec2,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CanvasSet {
    pub axis: PanelAxis,
    pub gap_m: f32,
    pub panels: Vec<CanvasPanel>,
}

fn fits(size: Vec2) -> bool {
    size.min_element() >= STRETCHER_MIN_M && size.max_element() >= INSTALLATION_MIN_M
}

/// Only stretched canvases and timber float frames are offered as sets.
/// Keep the sampled mounting style and record the installation in the manifest.
pub(crate) fn select_installation(o: &mut IndoorObject) {
    if !matches!(o.variant, 0 | 4) || !fits(o.size.xy()) {
        return;
    }
    let size = o.size.xy();
    let scale = ((size.max_element() - INSTALLATION_MIN_M) / 1.25).clamp(0., 1.);
    let elongation = (size.max_element() / size.min_element() - 1.).min(1.);
    let probability = 0.33 + 0.34 * scale + 0.13 * elongation;
    if stream(o.seed, 0x4152_5453_4554_4651).random_bool(f64::from(probability)) {
        o.variant += SINGLE_STYLES;
    }
}

/// The measured panels used by mesh construction, object audits and dataset metrics.
/// A small resized fixture falls back to a single mount of its original style.
pub fn canvas_set(o: &IndoorObject) -> Option<CanvasSet> {
    if o.kind != ObjectKind::WallArt || !matches!(o.variant, 6 | 10) || !fits(o.size.xy()) {
        return None;
    }
    let size = o.size.xy();
    let mut rng = stream(o.seed, 0x4152_5453_4554_4c59);
    let horizontal = size.x >= INSTALLATION_MIN_M
        && (size.y < INSTALLATION_MIN_M
            || rng.random_bool(f64::from(
                (0.55 + 0.8 * (size.x / size.y).ln()).clamp(0.04, 0.96),
            )));
    let (axis, span, cross) = if horizontal {
        (PanelAxis::Horizontal, size.x, size.y)
    } else {
        (PanelAxis::Vertical, size.y, size.x)
    };
    let gap_m = rng.random_range(0.018..0.078_f32).min(span * 0.07);
    let limit = ((span + gap_m) / (STRETCHER_MIN_M + gap_m)).floor() as usize;
    let count = match rng.random_range(0..100) {
        0..=17 => 4,
        18..=58 => 3,
        _ => 2,
    }
    .min(limit);
    let variation = if rng.random_bool(0.45) {
        0.
    } else {
        rng.random_range(0.14..0.88)
    };
    let weights: [f32; 4] =
        std::array::from_fn(|_| (variation * rng.random_range(-1.0_f32..1.)).exp());
    let total: f32 = weights[..count].iter().sum();
    let extra = span - (count - 1) as f32 * gap_m - count as f32 * STRETCHER_MIN_M;
    let tallest = rng.random_range(0..count);
    let stagger = if rng.random_bool(0.53) {
        0.
    } else {
        rng.random_range(0.09..0.40)
    };
    let alignment = rng.random_range(0.12..0.88);
    let to_xy = |along, across| {
        if horizontal {
            Vec2::new(along - size.x * 0.5, across)
        } else {
            Vec2::new(across - size.x * 0.5, along)
        }
    };
    let mut panels = Vec::with_capacity(count);
    let mut start = 0.;
    for (i, weight) in weights[..count].iter().enumerate() {
        let end = if i + 1 == count {
            span
        } else {
            start + STRETCHER_MIN_M + extra * weight / total
        };
        let trim = if i == tallest {
            0.
        } else {
            (cross * stagger * rng.random_range(0.35..1.)).min(cross - STRETCHER_MIN_M)
        };
        let inset = trim * alignment;
        panels.push(CanvasPanel {
            min_m: to_xy(start, inset),
            max_m: to_xy(end, cross - (trim - inset)),
        });
        start = end + gap_m;
    }
    Some(CanvasSet {
        axis,
        gap_m,
        panels,
    })
}
