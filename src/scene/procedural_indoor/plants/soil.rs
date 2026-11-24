//! A continuous fine bed and a bounded population of larger, partly buried grains.
//! Sampling takes place inside the fitted mouth, including the corners of square pots.
use super::{pots::Fill, *};
use std::borrow::Cow;

pub(super) mod legacy;
mod mesh;
#[cfg(test)]
mod tests;
pub(crate) const INTRODUCED: u32 = 222;
pub(crate) const FIRST_SLOT: usize = 784;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Medium {
    Loam,
    PerliteMix,
    BarkChips,
    ExpandedClay,
    MineralGravel,
}

#[derive(Clone, Copy)]
pub(crate) enum Layer {
    Bed,
    Grain,
    Accent,
}

impl Medium {
    pub(crate) const GRADED: [Self; 4] = [
        Self::PerliteMix,
        Self::BarkChips,
        Self::ExpandedClay,
        Self::MineralGravel,
    ];
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let mut rng = stream(o.seed, u64::from_le_bytes(*b"PotMedia"));
        match rng.random_range(0..100) {
            0..30 if matches!(o.variant % SPECIES, 7..=9) && rng.random_bool(0.45) => {
                Self::MineralGravel
            }
            0..30 => Self::Loam,
            30..57 => Self::PerliteMix,
            57..77 => Self::BarkChips,
            77..89 => Self::ExpandedClay,
            _ => Self::MineralGravel,
        }
    }
    pub(crate) fn slot(self, layer: Layer) -> Option<usize> {
        (self != Self::Loam).then(|| FIRST_SLOT + (self as usize - 1) * 3 + layer as usize)
    }
    pub(super) fn label(self, layer: Layer) -> Cow<'static, str> {
        self.slot(layer)
            .map_or(Cow::Borrowed("other_prop"), |slot| {
                Cow::Owned(format!("plant#finish{slot}"))
            })
    }
}

#[derive(Clone, Copy)]
struct Grain {
    center: Vec2,
    half: Vec2,
    height: f32,
    yaw: f32,
    shape_seed: u32,
    accent: bool,
}

pub(super) struct Dressing {
    medium: Medium,
    grains: Vec<Grain>,
    /// Representative diameter of the coarser, geometric fraction, in metres.
    diameter: f32,
    area: f32,
}

/// Export the same realized population used by meshing. Never count the fine
/// texture as separately identifiable objects.
#[derive(serde::Serialize)]
pub(crate) struct Stats {
    pub medium: Medium,
    pub coarse_grains: usize,
    pub nominal_grain_diameter_m: f32,
    pub coarse_footprint_fraction: f32,
}
pub(crate) fn stats(o: &IndoorObject, pot: &PotProfile, r: f32, h: f32) -> Stats {
    let fill = Fill::new(r, h, pot);
    let d = Dressing::sample(o.seed, Medium::sample(o), fill);
    Stats {
        medium: d.medium,
        coarse_grains: d.grains.len(),
        nominal_grain_diameter_m: d.diameter,
        // Bounding ellipses of the coarse grains; these do not overlap.
        coarse_footprint_fraction: d
            .grains
            .iter()
            .map(|g| PI * g.half.x * g.half.y)
            .sum::<f32>()
            / d.area,
    }
}

impl Dressing {
    fn sample(seed: u64, medium: Medium, fill: Fill) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"PotGrade"));
        let (grade, cover, aspect, rise) = match medium {
            Medium::Loam => (0.004_f32..0.016, 0.0..0.028, 0.56..0.93, 0.26..0.60),
            Medium::PerliteMix => (0.003..0.009, 0.015..0.085, 0.64..0.98, 0.45..0.88),
            Medium::BarkChips => (0.013..0.039, 0.12..0.29, 0.28..0.64, 0.15..0.43),
            Medium::ExpandedClay => (0.007..0.018, 0.13..0.32, 0.75..1.0, 0.55..0.98),
            Medium::MineralGravel => (0.005..0.022, 0.09..0.27, 0.48..0.91, 0.32..0.70),
        };
        let n = fill.sides as f32;
        let step = TAU / n;
        let apothem = (step * 0.5).cos();
        let edge = Vec2::from_angle(step);
        let area = 0.5 * n * fill.radius.powi(2) * edge.y;
        let diameter = rng.random_range(grade).min(fill.radius * 0.52);
        let typical_area = PI * (diameter * 0.5).powi(2) * (aspect.start + aspect.end) * 0.5;
        let expected = area * rng.random_range(cover) / typical_area;
        // Fine grains already appear in the bed texture. Let the number of
        // individually modelled grains taper off without giving every large
        // planter exactly the same count or inflating its grains to fill it.
        let limit = rng.random_range(18.0_f32..32.0);
        let target = (-limit * (-expected / limit).exp_m1()).round() as usize;
        let mut grains = Vec::<Grain>::with_capacity(target);
        for _ in 0..target * 3 {
            if grains.len() == target {
                break;
            }
            let length = diameter * 0.5 * rng.random_range(0.62..1.35);
            let width = length * rng.random_range(aspect.clone());
            // Select an equal-area sector then a uniform point in its triangle.
            // Insetting all edges by the grain's radius keeps its entire cap inside.
            let sector = rng.random_range(0.0..n);
            let offset = Vec2::X.lerp(edge, sector.fract());
            let scale = (fill.radius - length / apothem) * rng.random::<f32>().sqrt();
            let center = Vec2::from_angle(sector.floor() * step).rotate(offset) * scale;
            if grains
                .iter()
                .any(|g| g.center.distance_squared(center) < (g.half.x + length).powi(2))
            {
                continue;
            }
            grains.push(Grain {
                center,
                half: Vec2::new(length, width),
                height: (width * rng.random_range(rise.clone())).min(fill.rim_gap * 0.78),
                yaw: rng.random_range(0.0..TAU),
                shape_seed: rng.random(),
                accent: rng.random_bool(0.31),
            });
        }
        Self {
            medium,
            grains,
            diameter,
            area,
        }
    }
}

pub(super) fn build(a: &mut Assembly, seed: u64, medium: Medium, fill: Fill) {
    let mut chart = stream(seed, u64::from_le_bytes(*b"PotChart"));
    let axis = Vec2::from_angle(chart.random_range(0.0..TAU));
    let offset = Vec2::new(chart.random_range(0.0..2.), chart.random_range(0.0..2.));
    let bed = medium.label(Layer::Bed);
    for uv in &mut a.part(Surface::Soil, &bed).uvs {
        *uv = (axis.rotate(Vec2::from_array(*uv)) + offset).to_array();
    }
    let d = Dressing::sample(seed, medium, fill);
    for accent in [false, true] {
        let label = medium.label(if accent { Layer::Accent } else { Layer::Grain });
        let mut grains = d.grains.iter().filter(|g| g.accent == accent).peekable();
        if grains.peek().is_none() {
            continue;
        }
        let g = a.part(Surface::Soil, &label);
        for grain in grains {
            mesh::append(g, *grain, medium, fill.level);
        }
    }
}
