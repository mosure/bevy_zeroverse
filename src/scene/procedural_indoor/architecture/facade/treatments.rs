//! Seeded, metric window coverings. Meshes stay inside the aperture's lateral
//! extent and the existing 20 cm wall-clearance allowance. Static by default.
mod cloth;
mod mesh;
#[cfg(test)]
mod tests;

use super::*;
use crate::scene::procedural_indoor::materials::variants;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stack {
    Pair,
    Left,
    Right,
}

/// Coverage/tilt remain on `Facade`; this independent seed stream preserves
/// aperture dimensions and supplies installation geometry and shared PBR slots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadeProgram {
    pub identity: u64,
    pub slat_pitch: f32,
    pub fold_pitch: f32,
    pub fold_depth: f32,
    pub fullness: f32,
    pub hem_sag: f32,
    pub split: f32,
    pub tieback: f32,
    pub stack: Stack,
    pub slat_material: Surface,
    pub fabric: Surface,
    pub finish_slot: usize,
}
impl ShadeProgram {
    pub fn sample(seed: u64) -> (Shade, Self) {
        let mut r = stream(seed, 0x5348414445);
        let shade = match r.random_range(0..20) {
            0..6 => Shade::None,
            6..9 => Shade::Venetian,
            9..12 => Shade::Roller,
            12..14 => Shade::Vertical,
            14..16 => Shade::Roman,
            16..18 => Shade::Pleated,
            _ => Shade::Curtain,
        };
        (
            shade,
            Self {
                identity: r.random(),
                slat_pitch: r.random_range(0.028..0.082),
                fold_pitch: r.random_range(0.10..0.25),
                fold_depth: r.random_range(0.012..0.034),
                fullness: r.random_range(1.4..2.4),
                hem_sag: r.random_range(0.001..0.018),
                split: r.random_range(0.38..0.62),
                tieback: if r.random_bool(0.32) {
                    r.random_range(0.35..0.85)
                } else {
                    0.
                },
                stack: [Stack::Pair, Stack::Pair, Stack::Left, Stack::Right][r.random_range(0..4)],
                slat_material: [Surface::WoodEdge, Surface::Metal, Surface::Plastic]
                    [r.random_range(0..3)],
                fabric: if r.random_bool(0.78) {
                    Surface::Fabric
                } else {
                    Surface::FabricAlt
                },
                finish_slot: r.random_range(0..variants::COUNT),
            },
        )
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.02..=0.10).contains(&self.slat_pitch)
            || !(0.08..=0.30).contains(&self.fold_pitch)
            || !(0.006..=0.04).contains(&self.fold_depth)
            || !(1.0..=2.6).contains(&self.fullness)
            || !(0.0..=0.025).contains(&self.hem_sag)
            || !(0.25..=0.75).contains(&self.split)
            || !(0.0..=1.0).contains(&self.tieback)
            || !matches!(
                self.slat_material,
                Surface::WoodEdge | Surface::Metal | Surface::Plastic
            )
            || !matches!(self.fabric, Surface::Fabric | Surface::FabricAlt)
            || self.finish_slot >= variants::COUNT
        {
            return Err("invalid window treatment program".into());
        }
        Ok(())
    }
}

/// Append the production installation without adding an occluding proxy plane.
/// Covers are opaque PBR geometry, including on the annotation render paths.
pub fn build(a: &mut Assembly, tf: Transform, f: &Facade, o: &WindowOpening) {
    if f.shade == Shade::None {
        return;
    }
    let fallback;
    let p = if let Some(p) = &f.treatment {
        p
    } else {
        fallback = ShadeProgram::sample(0).1;
        &fallback
    };
    use super::super::super::motion::{construction as motion, MotionKind};
    let seed = p.identity ^ o.min.x.to_bits() as u64 ^ ((o.min.y.to_bits() as u64) << 32);
    if a.motion_selected(MotionKind::Covering, seed) {
        let mut first = Assembly::default();
        let mut last = Assembly::default();
        mesh::build(&mut first, tf, f, o, p);
        let mut end = f.clone();
        end.shade_coverage = motion::endpoint(seed, f.shade_coverage, 1.);
        // Translate rigid slats without shearing or shrinking them. Tilting and
        // lifting are distinct degrees of freedom; this program opens the stack.
        let tilt = f.shade_tilt.clamp(-1.2, 1.2);
        let angle = match f.shade {
            Shade::Venetian => Some(tilt * f.shade_coverage.sqrt()),
            Shade::Vertical => Some(
                if tilt < 0. { -1. } else { 1. }
                    * (tilt.abs().max(0.12) * f.shade_coverage + 1.50 * (1. - f.shade_coverage)),
            ),
            _ => None,
        };
        if f.shade == Shade::Venetian {
            let height = o.max.y - o.min.y - 0.028 - 0.095;
            let count = (height / p.slat_pitch).floor().clamp(2., 80.);
            let pitch = height / count;
            let depth = (pitch * 1.07).min(0.092);
            let min_coverage = ((depth * angle.unwrap().sin().abs() + 0.002 - 0.004)
                / (pitch - 0.004))
                .clamp(0., 1.);
            end.shade_coverage = end.shade_coverage.max(min_coverage);
        }
        mesh::build_with_angle(&mut last, tf, &end, o, p, angle);
        motion::morph(a, first, last, seed, tf);
    } else {
        mesh::build(a, tf, f, o, p);
    }
}
