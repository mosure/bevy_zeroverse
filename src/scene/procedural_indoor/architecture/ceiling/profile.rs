//! Recorded continuous shape controls and repeated installation rhythms.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrussWeb {
    Warren,
    Pratt,
    Space,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeilingDetail {
    /// Reuses the existing bounded PBR finish pool; no installation textures.
    pub finish: u8,
    pub shape: CeilingShape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum CeilingShape {
    Coffer {
        aspect: f32,
        beam_fraction: f32,
        recess_fraction: f32,
        timber: bool,
    },
    Diffuser {
        round: bool,
        blades: u8,
        angle: f32,
    },
    Panel {
        camber: f32,
        taper: f32,
        phase: f32,
    },
    Grid {
        aspect: f32,
        recess_m: f32,
    },
    Duct {
        section: f32,
        end_scale: f32,
        bend: f32,
        rise: f32,
        branches: u8,
        seam_pitch_m: f32,
    },
    Sprinkler {
        radius_m: f32,
        branch_reach: f32,
        head_pitch_m: f32,
        upright: bool,
    },
    Conduit {
        count: u8,
        radius_m: f32,
        bend: f32,
        rise: f32,
        junction_pitch_m: f32,
    },
    Tray {
        cables: u8,
        rung_pitch_m: f32,
        rise: f32,
    },
    Truss {
        web: TrussWeb,
        chord_radius_m: f32,
        cell_pitch_m: f32,
        taper: f32,
    },
}

impl CeilingDetail {
    pub(super) fn sample(system: CeilingSystem, r: &mut impl Rng, finish: u8) -> Self {
        use CeilingShape::*;
        let shape = match system {
            CeilingSystem::Coffers => Coffer {
                aspect: r.random_range(0.7..1.7),
                beam_fraction: r.random_range(0.06..0.16),
                recess_fraction: r.random_range(0.42..0.86),
                timber: r.random_bool(0.42),
            },
            CeilingSystem::Ventilation => Diffuser {
                round: r.random_bool(0.45),
                blades: r.random_range(3..=8),
                angle: r.random_range(0.15..0.65),
            },
            CeilingSystem::AcousticCloud | CeilingSystem::Baffles | CeilingSystem::TimberSlats => {
                Panel {
                    camber: r.random_range(0.05..0.90),
                    taper: r.random_range(0.65..1.),
                    phase: r.random_range(0.0..std::f32::consts::TAU),
                }
            }
            CeilingSystem::SuspendedTiles | CeilingSystem::OpenCell => Grid {
                aspect: r.random_range(0.7..2.0),
                recess_m: r.random_range(0.003..0.028),
            },
            CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct => Duct {
                section: r.random_range(0.60..0.95),
                end_scale: r.random_range(0.62..1.),
                bend: r.random_range(-0.65..0.65),
                rise: r.random_range(-0.45..0.45),
                branches: r.random_range(0..=3),
                seam_pitch_m: r.random_range(0.45..1.2),
            },
            CeilingSystem::SprinklerMain => Sprinkler {
                radius_m: r.random_range(0.016..0.029),
                branch_reach: r.random_range(0.55..0.92),
                head_pitch_m: r.random_range(1.1..2.8),
                upright: r.random_bool(0.3),
            },
            CeilingSystem::Conduit => Conduit {
                count: r.random_range(2..=5),
                radius_m: r.random_range(0.008..0.016),
                bend: r.random_range(-0.7..0.7),
                rise: r.random_range(-0.55..0.55),
                junction_pitch_m: r.random_range(1.5..3.),
            },
            CeilingSystem::CableTray => Tray {
                cables: r.random_range(3..=8),
                rung_pitch_m: r.random_range(0.16..0.35),
                rise: r.random_range(-0.4..0.4),
            },
            CeilingSystem::Truss => Truss {
                web: [TrussWeb::Warren, TrussWeb::Pratt, TrussWeb::Space][r.random_range(0..3)],
                chord_radius_m: r.random_range(0.014..0.029),
                cell_pitch_m: r.random_range(0.40..0.95),
                taper: r.random_range(0.50..1.),
            },
        };
        Self { finish, shape }
    }

    pub(super) fn valid(&self, system: CeilingSystem) -> bool {
        let in_range = |v: f32, a, b| (a..=b).contains(&v);
        if self.finish >= 12 {
            return false;
        }
        use CeilingShape::*;
        match self.shape {
            Coffer {
                aspect,
                beam_fraction,
                recess_fraction,
                ..
            } => {
                system == CeilingSystem::Coffers
                    && in_range(aspect, 0.5, 2.)
                    && in_range(beam_fraction, 0.04, 0.20)
                    && in_range(recess_fraction, 0.3, 0.9)
            }
            Diffuser { blades, angle, .. } => {
                system == CeilingSystem::Ventilation
                    && (3..=8).contains(&blades)
                    && in_range(angle, 0., 0.75)
            }
            Panel {
                camber,
                taper,
                phase,
            } => {
                matches!(
                    system,
                    CeilingSystem::AcousticCloud
                        | CeilingSystem::Baffles
                        | CeilingSystem::TimberSlats
                ) && in_range(camber, 0., 1.)
                    && in_range(taper, 0.5, 1.)
                    && in_range(phase, 0., std::f32::consts::TAU)
            }
            Grid { aspect, recess_m } => {
                matches!(
                    system,
                    CeilingSystem::SuspendedTiles | CeilingSystem::OpenCell
                ) && in_range(aspect, 0.5, 2.5)
                    && in_range(recess_m, 0.002, 0.03)
            }
            Duct {
                section,
                end_scale,
                bend,
                rise,
                branches,
                seam_pitch_m,
            } => {
                matches!(
                    system,
                    CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct
                ) && in_range(section, 0.5, 1.)
                    && in_range(end_scale, 0.5, 1.)
                    && in_range(bend, -0.8, 0.8)
                    && in_range(rise, -0.6, 0.6)
                    && branches <= 4
                    && in_range(seam_pitch_m, 0.3, 1.5)
            }
            Sprinkler {
                radius_m,
                branch_reach,
                head_pitch_m,
                ..
            } => {
                system == CeilingSystem::SprinklerMain
                    && in_range(radius_m, 0.012, 0.035)
                    && in_range(branch_reach, 0.4, 0.95)
                    && in_range(head_pitch_m, 0.8, 3.5)
            }
            Conduit {
                count,
                radius_m,
                bend,
                rise,
                junction_pitch_m,
            } => {
                system == CeilingSystem::Conduit
                    && (1..=6).contains(&count)
                    && in_range(radius_m, 0.006, 0.018)
                    && in_range(bend, -0.8, 0.8)
                    && in_range(rise, -0.6, 0.6)
                    && in_range(junction_pitch_m, 1., 4.)
            }
            Tray {
                cables,
                rung_pitch_m,
                rise,
            } => {
                system == CeilingSystem::CableTray
                    && (1..=10).contains(&cables)
                    && in_range(rung_pitch_m, 0.14, 0.4)
                    && in_range(rise, -0.5, 0.5)
            }
            Truss {
                chord_radius_m,
                cell_pitch_m,
                taper,
                ..
            } => {
                system == CeilingSystem::Truss
                    && in_range(chord_radius_m, 0.012, 0.03)
                    && in_range(cell_pitch_m, 0.3, 1.2)
                    && in_range(taper, 0.4, 1.)
            }
        }
    }
}
