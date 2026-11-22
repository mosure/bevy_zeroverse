//! Seeded, metric window coverings. Meshes stay inside the aperture's lateral
//! extent and the existing 20 cm wall-clearance allowance. Static by default.
mod cloth;
pub(crate) mod drapery;
pub use drapery::CurtainHeading;
mod mesh;
pub(crate) mod minification;
pub(crate) mod slats;
pub use slats::SlatStock;
mod track;
pub use track::PanelTrack;
#[cfg(test)]
mod tests;

use super::*;
use crate::scene::procedural_indoor::materials::variants;
use crate::scene::procedural_indoor::materials::window_textile::WindowTextile;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slats: Option<SlatStock>,
    pub fold_pitch: f32,
    /// Fraction of a folded face occupied by the rounded pressed crease.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pleat_crease: Option<f32>,
    pub fold_depth: f32,
    pub fullness: f32,
    pub hem_sag: f32,
    pub split: f32,
    pub tieback: f32,
    pub stack: Stack,
    pub slat_material: Surface,
    pub fabric: Surface,
    pub finish_slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textile: Option<WindowTextile>,
    /// Absent in archived installations using the original curtain mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curtain_heading: Option<CurtainHeading>,
    /// Maximum per-opening shift in log odds of deployment. A shared draw
    /// or an archived installation uses the bank's coverage directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment_spread: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<PanelTrack>,
}
impl ShadeProgram {
    pub fn sample(seed: u64) -> (Shade, Self) {
        Self::sample_installation(seed, true)
    }

    fn sample_installation(seed: u64, metric_stock: bool) -> (Shade, Self) {
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
        // Separate stock/installation draws leave the original aperture, tilt,
        // coverage, hardware, and all unrelated room/camera streams intact.
        let mut stock = stream(seed, 0x534841444553544b);
        let track = (matches!(shade, Shade::Roller | Shade::Vertical) && stock.random_bool(0.28))
            .then(|| PanelTrack::sample(&mut stock));
        let shade = if track.is_some() {
            Shade::PanelTrack
        } else {
            shade
        };
        let textile = (!matches!(shade, Shade::None | Shade::Venetian))
            .then(|| WindowTextile::sample(&mut stock, shade == Shade::Curtain));
        let mut program = Self {
            identity: r.random(),
            slat_pitch: r.random_range(0.028..0.082),
            slats: None,
            fold_pitch: r.random_range(0.10..0.25),
            pleat_crease: None,
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
            textile,
            curtain_heading: None,
            deployment_spread: None,
            track,
        };
        if metric_stock && shade == Shade::Curtain {
            program.curtain_heading = Some(match stock.random_range(0..3) {
                0 => CurtainHeading::Wave,
                1 => CurtainHeading::PinchPleat,
                _ => CurtainHeading::Gathered,
            });
        }
        if metric_stock && shade == Shade::Venetian {
            let slats = SlatStock::sample(&mut stock, program.slat_material);
            program.slat_pitch = slats.depth_m * stock.random_range(0.86..0.96);
            program.slats = Some(slats);
        }
        if metric_stock && shade == Shade::Pleated {
            program.pleat_crease = Some(stock.random_range(0.12..0.26));
        }
        if let Some(track) = &mut program.track {
            track.reed_variation = Some(stock.random_range(0.10..0.24));
        }
        if metric_stock && shade != Shade::None {
            let mut controls = stream(seed, 0x44524157434f4e54);
            if controls.random_bool(0.60) {
                program.deployment_spread = Some(controls.random_range(0.7..2.6));
            }
        }
        (shade, program)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.02..=0.10).contains(&self.slat_pitch)
            || self
                .slats
                .is_some_and(|slats| !slats.valid(self.slat_pitch))
            || self
                .pleat_crease
                .is_some_and(|r| !(0.10..=0.28).contains(&r))
            || !(0.08..=0.30).contains(&self.fold_pitch)
            || !(0.006..=0.04).contains(&self.fold_depth)
            || !(1.0..=2.6).contains(&self.fullness)
            || !(0.0..=0.025).contains(&self.hem_sag)
            || !(0.25..=0.75).contains(&self.split)
            || !(0.0..=1.0).contains(&self.tieback)
            || self
                .deployment_spread
                .is_some_and(|v| !(0.0..=3.).contains(&v))
            || !matches!(
                self.slat_material,
                Surface::WoodEdge | Surface::Metal | Surface::Plastic
            )
            || !matches!(self.fabric, Surface::Fabric | Surface::FabricAlt)
            || self.finish_slot >= variants::COUNT
            || self.textile.is_some_and(|t| t.colorway >= 4)
            || self.track.is_some_and(|t| !t.valid())
        {
            return Err("invalid window treatment program".into());
        }
        Ok(())
    }
}

fn installation_seed(p: &ShadeProgram, o: &WindowOpening) -> u64 {
    p.identity ^ o.min.x.to_bits() as u64 ^ ((o.min.y.to_bits() as u64) << 32)
}

impl Facade {
    /// Realized draw in this opening. The two endpoints stay fully parked and
    /// fully deployed, and every opening advances monotonically with the bank.
    pub fn shade_coverage_at(&self, opening: &WindowOpening) -> f32 {
        let c = self.shade_coverage.clamp(0., 1.);
        let Some(p) = self.treatment.as_ref() else {
            return c;
        };
        let Some(spread) = p.deployment_spread else {
            return c;
        };
        if self.openings.len() < 2
            || self.shade == Shade::None
            || c == 0.
            || c == 1.
            || spread == 0.
        {
            return c;
        }
        let mut rng = stream(installation_seed(p, opening), 0x42415944524157);
        let odds = (rng.random_range(-1.0_f32..1.) * spread).exp();
        c * odds / ((1. - c) + c * odds)
    }
}

fn head_contact(kind: Shade, p: &ShadeProgram, o: &WindowOpening, tf: Transform) -> Option<String> {
    let tracked = matches!(kind, Shade::Roller | Shade::PanelTrack) && p.textile.is_some()
        || kind == Shade::Curtain && p.curtain_heading.is_some();
    tracked.then(|| {
        // Separate even repeated aperture templates on different wall segments.
        let center = tf.transform_point(((o.min + o.max) * 0.5).extend(0.11));
        format!(
            "contact_shade{:016x}_{:08x}_{:08x}_{:08x}",
            installation_seed(p, o),
            center.x.to_bits(),
            center.y.to_bits(),
            center.z.to_bits()
        )
    })
}

/// Append the production installation without adding an occluding proxy plane.
/// Covers are opaque PBR geometry, including on the annotation render paths.
pub fn build(a: &mut Assembly, tf: Transform, f: &Facade, o: &WindowOpening) {
    if f.shade == Shade::None {
        return;
    }
    let mut fallback;
    let p = if let Some(p) = &f.treatment {
        p
    } else {
        fallback = ShadeProgram::sample_installation(0, false).1;
        fallback.textile = None;
        fallback.track = None;
        &fallback
    };
    use super::super::super::motion::{construction as motion, MotionKind};
    let seed = installation_seed(p, o);
    let coverage = f.shade_coverage_at(o);
    if a.motion_selected(MotionKind::Covering, seed)
        && !(f.shade == Shade::Curtain && drapery::tied(p, o, coverage))
    {
        let mut first = Assembly::default();
        let mut last = Assembly::default();
        mesh::build(
            &mut first,
            tf,
            f,
            o,
            p,
            mesh::Draw {
                coverage,
                fixed_angle: None,
                moving: true,
            },
        );
        let mut end = motion::endpoint(seed, coverage, 1.);
        // Translate rigid slats without shearing or shrinking them. Tilting and
        // lifting are distinct degrees of freedom; this program opens the stack.
        let tilt = f.shade_tilt.clamp(-1.2, 1.2);
        let angle = match f.shade {
            Shade::Venetian => Some(tilt * coverage.sqrt()),
            Shade::Vertical => Some(
                if tilt < 0. { -1. } else { 1. }
                    * (tilt.abs().max(0.12) * coverage + 1.50 * (1. - coverage)),
            ),
            _ => None,
        };
        if f.shade == Shade::Venetian && p.slats.is_none() {
            let height = o.max.y - o.min.y - 0.028 - 0.095;
            let count = (height / p.slat_pitch).floor().clamp(2., 80.);
            let pitch = height / count;
            let depth = (pitch * 1.07).min(0.092);
            let min_coverage = ((depth * angle.unwrap().sin().abs() + 0.002 - 0.004)
                / (pitch - 0.004))
                .clamp(0., 1.);
            end = end.max(min_coverage);
        }
        mesh::build(
            &mut last,
            tf,
            f,
            o,
            p,
            mesh::Draw {
                coverage: end,
                fixed_angle: angle,
                moving: true,
            },
        );
        let feed =
            (f.shade == Shade::Roller && p.textile.is_some()).then(|| mesh::feed_plane(tf, o));
        motion::morph(
            a,
            first,
            last,
            seed,
            tf,
            feed,
            head_contact(f.shade, p, o, tf),
        );
    } else {
        mesh::build(
            a,
            tf,
            f,
            o,
            p,
            mesh::Draw {
                coverage,
                fixed_angle: None,
                moving: false,
            },
        );
    }
}
