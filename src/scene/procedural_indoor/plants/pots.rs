//! Continuously shaped, hollow planters with fitted soil and separate saucers.
use super::super::geometry::{ProfileJoints, RevolvedOutline};
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PotProfile {
    /// Desired height/diameter before reserving room for the living canopy.
    pub aspect: f32,
    pub saucer: bool,
    pub belly: f32,
    pub neck: f32,
    pub wall_fraction: f32,
    pub flute_depth: f32,
    pub flute_count: u32,
    pub lip_height_fraction: f32,
    /// Circular, softly faceted or square/octagonal nursery and decorative pots.
    pub sides: u32,
    pub shoulder_height_fraction: f32,
    pub pedestal_fraction: f32,
    pub saucer_radius: f32,
    pub base_rounding: f32,
}
impl PotProfile {
    pub(super) fn outer_radius_factor(&self) -> f32 {
        (self.belly.max(self.neck + 0.025) * (1. + self.flute_depth)).max(if self.saucer {
            self.saucer_radius
        } else {
            0.
        })
    }

    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 6220);
        let sides = [4, 6, 8, 12, 24, 32, 32, 40, 40][rng.random_range(0..9)];
        Self {
            aspect: rng.random_range(0.38_f32.ln()..1.65_f32.ln()).exp(),
            saucer: rng.random_bool(0.66),
            belly: rng.random_range(0.76..1.10),
            neck: rng.random_range(0.67..1.06),
            wall_fraction: rng.random_range(0.04..0.095),
            flute_depth: rng.random_range(0.0..0.065),
            // At least four vertices per flute on smooth pots. Undersampling
            // the radial wave produces aliasing instead of physical ribbing.
            flute_count: rng.random_range(4..=(sides / 4).max(4)),
            lip_height_fraction: rng.random_range(0.018..0.085),
            sides,
            shoulder_height_fraction: rng.random_range(0.28..0.78),
            pedestal_fraction: rng.random_range(0.025..0.12),
            saucer_radius: rng.random_range(1.06..1.16),
            base_rounding: rng.random_range(0.04..0.22),
        }
    }
}

/// Root volume responds to both stature and canopy width. Sample proportions
/// independently of plant height, with room left for the canopy and the real
/// outer rim/saucer. This gives shallow bowls and tall cachepots on the same
/// growth habit instead of a fixed-height fraction with an almost fixed radius.
pub(crate) fn dimensions(o: &IndoorObject, growth: &Growth) -> (f32, f32) {
    let form = match o.variant % SPECIES {
        7 => 1.25,
        8 | 9 => 1.40,
        _ => 1.,
    };
    let p = PotProfile::sample(o.seed);
    let mut rng = stream(o.seed, 6230);
    let width = o.size.x.min(o.size.z);
    let radius = (width * 0.40 * growth.pot_radius * form)
        .max(o.size.y * rng.random_range(0.055..0.085))
        .min(width * 0.45 / p.outer_radius_factor());
    let height = (2. * radius * p.aspect).min(o.size.y * growth.pot_fraction);
    (radius, height)
}

pub(super) fn build(
    a: &mut Assembly,
    material: Surface,
    r: f32,
    h: f32,
    taper: f32,
    p: &PotProfile,
) -> (f32, f32) {
    let foot = if p.saucer {
        h * p.pedestal_fraction
    } else {
        0.
    };
    let wall = p.wall_fraction;
    let lip = p.lip_height_fraction;
    let profile = [
        (0., foot),
        (r * (taper - p.base_rounding), foot),
        (r * taper, h * 0.15),
        (
            r * (taper + p.belly) * 0.5,
            h * p.shoulder_height_fraction * 0.65,
        ),
        (r * p.belly, h * p.shoulder_height_fraction),
        (r * p.neck, h * (1. - lip)),
        (r * (p.neck + 0.025), h * (1. - lip * 0.7)),
        (r * (p.neck + 0.025), h),
        (r * (p.neck - wall), h),
        (r * (p.neck - wall), h * (1. - lip)),
        (r * (p.belly - wall), h * p.shoulder_height_fraction),
        // Floor stays above the outer base rounding, including tall saucers.
        (r * (taper - wall), h * 0.16),
        (0., h * 0.16),
    ];
    let outline = if p.sides < 16 {
        RevolvedOutline::polygon(p.sides)
    } else {
        RevolvedOutline::round(p.sides, p.flute_depth, p.flute_count)
    };
    outline.sweep(
        a.part(material, "other_prop"),
        &profile,
        ProfileJoints::Creased,
        Transform::IDENTITY,
    );
    if p.saucer {
        a.part(material, "other_prop").lathe(
            &[
                (0., 0.),
                (r * p.saucer_radius, 0.),
                (r * p.saucer_radius, foot),
                (r * (p.saucer_radius - 0.06), foot * 1.6),
                (r * (p.saucer_radius - 0.10), foot),
                (0., foot),
            ],
            32,
            Transform::IDENTITY,
        );
    }
    let soil = h * (1. - lip - 0.065);
    // Intersect the inner shoulder at each fill height. A small embed inside
    // the solid wall closes the seam even under floating-point roundoff.
    let inner_radius = |y: f32| {
        let t = ((y / h - p.shoulder_height_fraction) / (1. - lip - p.shoulder_height_fraction))
            .clamp(0., 1.);
        r * ((p.belly - wall) * (1. - t) + (p.neck - wall) * t)
    };
    let bottom = soil - h * 0.025;
    let embed = r * wall * 0.12;
    outline.sweep(
        a.part(Surface::Soil, "other_prop"),
        &[
            (0., bottom),
            (inner_radius(bottom) + embed, bottom),
            (inner_radius(soil) + embed, soil),
            (0., soil),
        ],
        ProfileJoints::Creased,
        Transform::IDENTITY,
    );
    // Plant roots use the original conservative clear radius in every direction.
    let roots = inner_radius(soil) * (1. - p.flute_depth) * (PI / p.sides as f32).cos() - r * 0.008;
    (soil, roots)
}
