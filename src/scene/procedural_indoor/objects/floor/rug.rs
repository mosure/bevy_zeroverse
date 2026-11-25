//! Cut, binding and pile share one measured outline. Placement only needs the cut.
use super::*;
use std::sync::LazyLock;

mod legacy;
mod mesh;
mod radii;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(u32)]
pub enum Cut {
    Straight = 4,
    Rounded,
    Chamfered,
    Oval,
    Capsule,
    Asymmetric,
}

impl Cut {
    /// Keep the cut independent of the finish and the order of placement retries.
    pub(crate) fn sample(seed: u64, size: Vec3) -> Self {
        let sides = size.xz();
        let runner = ((sides.max_element() / sides.min_element() - 1.) / 3.).clamp(0., 1.);
        let draw = stream(seed, 0x5255_4743_5554).random::<f32>();
        if draw < 0.36 + 0.10 * runner {
            Self::Straight
        } else if draw < 0.57 + 0.09 * runner {
            Self::Rounded
        } else if draw < 0.67 + 0.07 * runner {
            Self::Chamfered
        } else if draw < 0.81 - 0.01 * runner {
            Self::Oval
        } else if draw < 0.88 + 0.09 * runner {
            Self::Capsule
        } else {
            Self::Asymmetric
        }
    }

    fn from_variant(variant: u32) -> Option<Self> {
        match variant {
            4 => Some(Self::Straight),
            5 => Some(Self::Rounded),
            6 => Some(Self::Chamfered),
            7 => Some(Self::Oval),
            8 => Some(Self::Capsule),
            9 => Some(Self::Asymmetric),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Binding {
    Serged,
    FoldedTape,
    WideBand,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RugProgram {
    pub cut: Cut,
    pub corner_m: f32,
    pub taper: f32,
    /// Ellipse radii at each corner, starting at (+X,+Z) and proceeding towards -X.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corner_radii_xz_m: Option<[Vec2; 4]>,
    pub binding: Binding,
    pub binding_width_m: f32,
    pub turn_under_m: f32,
}

fn cut_dimensions(o: &IndoorObject, cut: Cut) -> (f32, f32) {
    if !matches!(cut, Cut::Straight | Cut::Chamfered | Cut::Rounded) {
        return (0., 0.);
    }
    let mut r = stream(o.seed, 0x5255_4743_4f52_4e52);
    let half = o.size.x.min(o.size.z) * 0.5;
    let corner = half
        * match cut {
            Cut::Rounded => r.random_range(0.06..0.98),
            Cut::Chamfered => r.random_range(0.07..0.45),
            _ => 0.,
        };
    let taper = if matches!(cut, Cut::Straight | Cut::Chamfered) && r.random_bool(0.24) {
        r.random_range(-0.035..0.035)
    } else {
        0.
    };
    (corner, taper)
}

/// Metric product specification. Archived variants retain their original builder.
pub fn parameters(o: &IndoorObject) -> Option<RugProgram> {
    let cut = Cut::from_variant(o.variant)?;
    let (corner_m, taper) = cut_dimensions(o, cut);
    let corner_radii_xz_m = radii::sample(o, cut);
    let mut r = stream(o.seed, 0x5255_4742_494e_4447);
    let (binding, width) = match r.random_range(0..10) {
        0..=3 => (Binding::Serged, r.random_range(0.004_f32..0.014)),
        4..=8 => (Binding::FoldedTape, r.random_range(0.014..0.043)),
        _ => (Binding::WideBand, r.random_range(0.043..0.085)),
    };
    let half = o.size.xz() * 0.5;
    // Keep the parallel inner contour inside the tightest bend of the product.
    // An ellipse has its smallest radius of curvature at the end of its major axis.
    let bend = if let Some(radii) = corner_radii_xz_m {
        radii
            .into_iter()
            .map(radii::tightest_bend)
            .fold(f32::INFINITY, f32::min)
    } else if cut == Cut::Oval {
        radii::tightest_bend(half)
    } else if corner_m > 0. {
        corner_m
    } else {
        half.min_element()
    };
    let binding_width_m = width.min(bend * 0.55);
    Some(RugProgram {
        cut,
        corner_m,
        taper,
        corner_radii_xz_m,
        binding,
        binding_width_m,
        turn_under_m: (o.size.y * r.random_range(0.20..0.63)).min(binding_width_m * 0.45),
    })
}

const QUARTER_STEPS: usize = 12;
static QUARTER: LazyLock<[Vec2; QUARTER_STEPS + 1]> = LazyLock::new(|| {
    std::array::from_fn(|i| match i {
        0 => Vec2::X,
        QUARTER_STEPS => Vec2::Y,
        _ => Vec2::from_angle(FRAC_PI_2 * i as f32 / QUARTER_STEPS as f32),
    })
});

fn quadrant(p: Vec2, quarter: usize) -> Vec2 {
    match quarter {
        0 => p,
        1 => p.perp(),
        2 => -p,
        _ => -p.perp(),
    }
}

fn cut_outline(
    size: Vec2,
    cut: Cut,
    corner: f32,
    taper: f32,
    radii: Option<[Vec2; 4]>,
) -> Vec<Vec2> {
    let half = size * 0.5;
    let mut points = Vec::with_capacity(match cut {
        Cut::Straight => 4,
        Cut::Chamfered => 8,
        _ => 4 * (QUARTER_STEPS + 1),
    });
    for i in 0..4 {
        let sign = quadrant(Vec2::ONE, i);
        match cut {
            Cut::Straight => points.push(sign * half),
            Cut::Oval => points.extend(
                QUARTER[..QUARTER_STEPS]
                    .iter()
                    .map(|&p| quadrant(p, i) * half),
            ),
            Cut::Rounded | Cut::Chamfered | Cut::Capsule | Cut::Asymmetric => {
                let radius = radii.map_or(Vec2::splat(corner), |r| r[i]);
                let center = sign * (half - radius);
                let stride = if cut == Cut::Chamfered {
                    QUARTER_STEPS
                } else if radius.max_element() < 0.15 {
                    3
                } else {
                    1
                };
                points.extend(
                    QUARTER
                        .iter()
                        .step_by(stride)
                        .map(|&p| center + quadrant(p, i) * radius),
                );
            }
        }
    }
    // The adjoining quarters of a half-elliptical end share its tip.
    if cut == Cut::Capsule {
        points.dedup();
        if points.first() == points.last() {
            points.pop();
        }
    }
    if taper != 0. {
        for p in &mut points {
            p.x *= 1. - taper.abs() + taper * (p.y / half.y);
        }
    }
    points
}

pub(crate) fn outline(o: &IndoorObject) -> Vec<Vec2> {
    let Some(cut) = Cut::from_variant(o.variant) else {
        return legacy::outline(o);
    };
    let (corner, taper) = cut_dimensions(o, cut);
    cut_outline(o.size.xz(), cut, corner, taper, radii::sample(o, cut))
}

impl RugProgram {
    pub(crate) fn outline(&self, size: Vec2) -> Vec<Vec2> {
        cut_outline(
            size,
            self.cut,
            self.corner_m,
            self.taper,
            self.corner_radii_xz_m,
        )
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match parameters(o) {
        Some(p) => mesh::build(a, o, p),
        None => legacy::build(a, o),
    }
}
