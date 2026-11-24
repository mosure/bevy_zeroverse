//! A split-leaf canopy progresses from small entire leaves to broader blades
//! with marginal clefts and enclosed fenestrations. Each leaf has its own petiole.
use super::*;
mod lamina;

#[derive(Clone, Copy)]
enum Opening {
    Entire,
    Cleft { depth: f32 },
    Window { center: f32, radius: f32 },
}

pub(in crate::scene::procedural_indoor) const VERSION: u32 = 207;
const MAX_BAYS: usize = 4;

struct Leaf {
    bays: [[Opening; 2]; MAX_BAYS],
    count: usize,
    rows: [f32; MAX_BAYS + 1],
    age: f32,
    elevation: f32,
    reach: f32,
    breadth: f32,
    pitch: f32,
    azimuth: f32,
    petiole_reach: f32,
    arch: f32,
    twist: f32,
    rake: f32,
    asymmetry: f32,
    light: bool,
    vein_span: f32,
    vein_offset: f32,
}

fn leaves(seed: u64, growth: &Growth) -> impl Iterator<Item = Leaf> {
    let mut rng = stream(seed, 9851);
    let mut finish = stream(seed, 9852);
    let count = growth.count(11);
    let propensity = (growth.maturity * 0.79 + rng.random_range(0.04..0.26)).min(1.);
    (0..count).map(move |i| {
        // The last shoot is new growth. Older outside leaves have longer, lower petioles.
        let age =
            (1. - i as f32 / (count - 1) as f32 + rng.random_range(-0.13..0.13)).clamp(0., 1.);
        let development = propensity * (0.28 + age * 0.72);
        let count = if development > 0.55 { 4 } else { 3 };
        let mut bays = [[Opening::Entire; 2]; MAX_BAYS];
        for (j, pair) in bays[..count].iter_mut().enumerate() {
            for side in pair {
                let chance: f32 = rng.random();
                // Fenestrations are more common towards the broad basal shoulder.
                let basal = 1. - j as f32 / count as f32;
                let rate = ((development - 0.20) * (1.08 + basal * 0.22)).clamp(0., 0.94);
                *side = if chance < rate * (0.20 + basal * 0.40) {
                    Opening::Window {
                        center: rng.random_range(0.47..0.60),
                        radius: rng.random_range(0.14..0.26),
                    }
                } else if chance < rate {
                    Opening::Cleft {
                        depth: rng.random_range(0.19..0.57),
                    }
                } else {
                    Opening::Entire
                };
            }
        }
        let first = rng.random_range(0.09..0.16);
        let last = rng.random_range(0.81..0.90);
        let mut rows = [last; MAX_BAYS + 1];
        rows[0] = first;
        for (j, row) in rows.iter_mut().enumerate().take(count).skip(1) {
            *row =
                first + (last - first) * (j as f32 + rng.random_range(-0.17..0.17)) / count as f32;
        }
        let vein_span = finish.random_range(0.68..0.90) + age * 0.09;
        Leaf {
            bays,
            count,
            rows,
            age,
            elevation: rng.random_range(0.52..0.75) - age * 0.12,
            reach: rng.random_range(0.48..0.66) + age * 0.10,
            breadth: rng.random_range(0.19..0.26) + age * 0.09,
            pitch: rng.random_range(0.03..0.53) - age * 0.69,
            azimuth: rng.random_range(-0.40..0.40),
            petiole_reach: rng.random_range(0.08..0.19) + age * 0.07,
            arch: rng.random_range(0.14..0.33),
            twist: rng.random_range(-0.19..0.19),
            rake: rng.random_range(0.07..0.15),
            asymmetry: rng.random_range(-0.10..0.10),
            vein_span,
            vein_offset: finish.random::<f32>() * (1. - vein_span),
            light: rng.random::<f32>() < 0.06 + (1. - age) * (1. - age) * 0.64,
        }
    })
}

#[derive(Default, serde::Serialize)]
pub(in crate::scene::procedural_indoor) struct CanopyStats {
    pub(in crate::scene::procedural_indoor) leaves: u32,
    pub(in crate::scene::procedural_indoor) intact_leaves: u32,
    pub(in crate::scene::procedural_indoor) marginal_clefts: u32,
    pub(in crate::scene::procedural_indoor) closed_windows: u32,
}

pub(in crate::scene::procedural_indoor) fn canopy_stats(seed: u64, growth: &Growth) -> CanopyStats {
    let mut stats = CanopyStats::default();
    for leaf in leaves(seed, growth) {
        stats.leaves += 1;
        let mut intact = true;
        for opening in leaf.bays[..leaf.count].iter().flatten() {
            match opening {
                Opening::Entire => continue,
                Opening::Cleft { .. } => stats.marginal_clefts += 1,
                Opening::Window { .. } => stats.closed_windows += 1,
            }
            intact = false;
        }
        stats.intact_leaves += u32::from(intact);
    }
    stats
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, growth: &Growth, base: Vec3, spread: f32) {
    let height = o.size.y - base.y;
    let count = growth.count(11);
    let mut rng = stream(o.seed, 9853);
    let lean = radial(growth.shoot_phase) * spread * rng.random_range(0.04..0.20);
    let rise = Vec3::Y * height * rng.random_range(0.19..0.35);
    let leader = [base, base + rise * 0.48 + lean * 0.25, base + rise + lean];
    let radius = spread.min(height) * rng.random_range(0.020..0.035);
    a.part(Surface::Succulent, "other_prop")
        .tapered_tube(&leader, radius, radius * 0.48, 8);
    for (i, leaf) in leaves(o.seed, growth).enumerate() {
        let angle = growth.angle(i);
        let outward = radial(angle);
        let end = base + Vec3::Y * height * leaf.elevation + outward * spread * leaf.petiole_reach;
        let node = along(&leader, 0.05 + 0.95 * growth.node(i, count - 1));
        let elbow = node.lerp(end, 0.53) - outward * (spread * 0.055);
        let petiole_radius = radius * (0.37 + 0.18 * leaf.age);
        a.part(Surface::Succulent, "other_prop").tapered_tube(
            &[node, elbow, end],
            petiole_radius,
            petiole_radius * 0.58,
            8,
        );
        let delta = incline(
            radial(angle + leaf.azimuth),
            spread * leaf.reach,
            leaf.pitch,
            end.y,
            base.y,
            o.size.y,
        );
        let width = spread * leaf.breadth * growth.leaf_aspect;
        let surface = if leaf.light {
            Surface::LeafLight
        } else {
            Surface::Leaf
        };
        lamina::append(
            a.part(surface, "other_prop"),
            end,
            delta,
            width,
            growth,
            &leaf,
        );
    }
}

#[cfg(test)]
mod tests;
