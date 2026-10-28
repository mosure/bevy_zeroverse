//! Metric, installation-coherent joinery. Profiles are actual solids, clipped
//! by the same apertures as the wall; no texture-only or depth-biased trim.
use super::super::{
    envelope::{construction::triangle, polygon},
    layout::{stream, IndoorManifest},
    materials::{variants, Surface},
    objects::Assembly,
};
use super::facade::{solid_rectangles, WindowOpening};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrimProfile {
    Flat,
    Chamfer,
    Stepped,
    Round,
    Cove,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimSection {
    pub height_m: f32,
    pub depth_m: f32,
    pub shoulder: f32,
    pub tip_fraction: f32,
    pub profile: TrimProfile,
    pub surface: Surface,
    pub finish_slot: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimProgram {
    pub base: Option<TrimSection>,
    pub crown: Option<TrimSection>,
    /// Real shadow/reveal joint at the floor/roof, rather than a coincident face.
    pub joint_m: f32,
    pub corner_reveal_m: f32,
}

impl TrimProgram {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 3160);
        let mut section = |crown: bool| TrimSection {
            // Log-uniform scale spans slim commercial skirting and tall domestic
            // mouldings without concentrating every room near the same mean.
            // Tall fitted furnishings reserve 30 cm below the roof. Keep crown
            // relief within that budget, including its reveal and roof slope.
            height_m: r
                .random_range(
                    if crown {
                        0.035_f32.ln()
                    } else {
                        0.025_f32.ln()
                    }..if crown { 0.25_f32.ln() } else { 0.42_f32.ln() },
                )
                .exp(),
            depth_m: r
                .random_range(0.006_f32.ln()..if crown { 0.065_f32.ln() } else { 0.07_f32.ln() })
                .exp(),
            shoulder: r.random_range(0.36..0.85),
            tip_fraction: r.random_range(0.18..0.72),
            profile: [
                TrimProfile::Flat,
                TrimProfile::Chamfer,
                TrimProfile::Stepped,
                TrimProfile::Round,
                TrimProfile::Cove,
            ][r.random_range(0..5)],
            surface: if crown {
                [
                    Surface::Paint,
                    Surface::Paint,
                    Surface::WoodEdge,
                    Surface::Metal,
                ][r.random_range(0..4)]
            } else {
                [
                    Surface::Paint,
                    Surface::WoodEdge,
                    Surface::Metal,
                    Surface::Plastic,
                    Surface::Ceramic,
                ][r.random_range(0..5)]
            },
            finish_slot: r.random_range(0..variants::COUNT),
        };
        let base = section(false);
        let crown = section(true);
        let base_present = r.random_bool(0.92);
        let crown_present = r.random_bool(0.70);
        let joint_m = r.random_range(0.003..0.013);
        // The endpoint reveal must clear the deepest adjoining section. A tall
        // crown can project farther than a skirting; independent sampling of
        // the old corner gap would allow the two strips to intersect.
        let corner_reveal_m = r
            .random_range(0.045_f32..0.085)
            .max(base.depth_m.max(crown.depth_m) + 0.008);
        Self {
            base: base_present.then_some(base),
            crown: crown_present.then_some(crown),
            joint_m,
            corner_reveal_m,
        }
    }

    pub fn for_scene(scene: &IndoorManifest) -> Option<&Self> {
        scene.program.as_ref()?.finishes.as_ref()?.trim.as_ref()
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.002..=0.015).contains(&self.joint_m)
            || !(0.04..=0.09).contains(&self.corner_reveal_m)
            || self.base.iter().chain(&self.crown).any(|s| {
                !(0.025..=0.42).contains(&s.height_m)
                    || !(0.005..=0.07).contains(&s.depth_m)
                    || !(0.3..=0.9).contains(&s.shoulder)
                    || !(0.1..=0.8).contains(&s.tip_fraction)
                    || s.finish_slot >= variants::COUNT
                    || self.corner_reveal_m < s.depth_m + 0.004
                    || !matches!(
                        s.surface,
                        Surface::Paint
                            | Surface::WoodEdge
                            | Surface::Metal
                            | Surface::Plastic
                            | Surface::Ceramic
                    )
            })
            || self
                .crown
                .as_ref()
                .is_some_and(|s| s.height_m + self.joint_m > 0.28)
        {
            return Err("invalid architectural trim profile".into());
        }
        Ok(())
    }
}

impl TrimSection {
    /// A small bounded section, with continuous dimensions and profile curvature.
    /// Each consecutive pair bounds disjoint surface area, including at steps.
    pub(crate) fn knots(&self) -> Vec<Vec2> {
        let s = self.shoulder;
        let t = self.tip_fraction;
        let mut result = vec![Vec2::new(0., 1.)];
        match self.profile {
            TrimProfile::Flat => result.push(Vec2::new(1., 1.)),
            TrimProfile::Chamfer => result.extend([Vec2::new(s, 1.), Vec2::new(1., t)]),
            TrimProfile::Stepped => result.extend([
                Vec2::new(s, 1.),
                Vec2::new(s + (1. - s) * 0.12, t),
                Vec2::new(1., t),
            ]),
            TrimProfile::Round | TrimProfile::Cove => {
                result.push(Vec2::new(s, 1.));
                for i in 1..=5 {
                    let u = i as f32 / 5.;
                    let depth = if self.profile == TrimProfile::Round {
                        (1. - u * u).sqrt()
                    } else {
                        1. - (2. * u - u * u).sqrt()
                    };
                    result.push(Vec2::new(s + (1. - s) * u, t + (1. - t) * depth));
                }
            }
        }
        result
    }
}

/// `level` is the height at each end of the wall-local span. The same builder
/// follows a sloped roof and a flat floor. It does not subdivide by room size.
pub(crate) fn build(
    a: &mut Assembly,
    tf: Transform,
    span: Vec2,
    level: Vec2,
    section: &TrimSection,
    holes: &[WindowOpening],
    crown: bool,
) {
    if span.y - span.x < 0.01 {
        return;
    }
    let slope = (level.y - level.x) / (span.y - span.x);
    let intercept = level.x - slope * span.x;
    let label = format!(
        "{}#finish{}",
        if crown { "ceiling" } else { "wall" },
        section.finish_slot
    );
    let g = a.part(section.surface, &label);
    let mut knots = section.knots();
    if crown {
        knots.reverse();
        for p in &mut knots {
            p.x = 1. - p.x;
        }
    }
    for pair in knots.windows(2) {
        let low = intercept + pair[0].x * section.height_m;
        let high = intercept + pair[1].x * section.height_m;
        for (lo, hi) in solid_rectangles(
            Vec2::new(span.x, level.min_element() + pair[0].x * section.height_m),
            Vec2::new(span.y, level.max_element() + pair[1].x * section.height_m),
            holes,
        ) {
            let rect = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
            let p = polygon::clip(&rect, Vec2::new(-slope, 1.), low);
            let p = polygon::clip(&p, Vec2::new(slope, -1.), -high);
            if p.len() < 3 {
                continue;
            }
            let front: Vec<_> = p
                .iter()
                .map(|p| {
                    let u = ((p.y - slope * p.x - low) / (high - low)).clamp(0., 1.);
                    tf.transform_point(Vec3::new(
                        p.x,
                        p.y,
                        section.depth_m * pair[0].y.lerp(pair[1].y, u),
                    ))
                })
                .collect();
            let back: Vec<_> = p
                .iter()
                .map(|p| tf.transform_point(Vec3::new(p.x, p.y, -0.002)))
                .collect();
            let normal = tf.rotation * Vec3::Z;
            for i in 1..p.len() - 1 {
                triangle(g, [front[0], front[i], front[i + 1]], normal);
                triangle(g, [back[0], back[i], back[i + 1]], -normal);
            }
            for i in 0..p.len() {
                let j = (i + 1) % p.len();
                let n = (back[j] - back[i]).cross(normal);
                triangle(g, [back[i], back[j], front[j]], n);
                triangle(g, [back[i], front[j], front[i]], n);
            }
        }
    }
}

#[cfg(test)]
mod tests;

/// Fitted cabinets terminate the skirting instead of intersecting a deeper
/// plinth. Navigation envelopes need not move when this architectural join changes.
pub(crate) fn fitted_openings(
    scene: &IndoorManifest,
    tf: Transform,
    section: &TrimSection,
    holes: &[WindowOpening],
) -> Vec<WindowOpening> {
    let mut cutouts = holes.to_vec();
    for object in scene
        .objects
        .iter()
        .filter(|o| o.solid && !o.neighbor && o.support.is_none())
    {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        let object_tf = object.transform();
        for x in [-0.5, 0.5] {
            for y in [0., 1.] {
                for z in [-0.5, 0.5] {
                    let p = tf.rotation.inverse()
                        * (object_tf.transform_point(object.size * Vec3::new(x, y, z))
                            - tf.translation);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        if lo.z < section.depth_m + 0.005
            && hi.z > 0.
            && lo.y < section.height_m + 0.02
            && hi.y > 0.
        {
            cutouts.push(WindowOpening {
                min: lo.truncate() - Vec2::splat(0.006),
                max: hi.truncate() + Vec2::splat(0.006),
                columns: 1,
                transom: 0.,
                transoms: vec![],
                glazing: None,
                operable: false,
            });
        }
    }
    cutouts
}
