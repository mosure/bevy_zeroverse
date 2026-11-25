//! The same opening determines visible source area, reflected luminance and
//! direct-light cutoff. A spot approximates the azimuthal average of a louver.
use super::*;
use crate::scene::procedural_indoor::envelope::construction::triangle;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub enum OpticalInsert {
    Plain,
    Basket {
        width: f32,
        recess: f32,
        reflector: Surface,
    },
    Cup {
        throat: f32,
        recess: f32,
        reflector: Surface,
    },
}

impl OpticalInsert {
    pub(super) fn basket_width(self) -> f32 {
        if let Self::Basket { width, .. } = self {
            width
        } else {
            1.
        }
    }
    pub(super) fn throat(self) -> f32 {
        if let Self::Cup { throat, .. } = self {
            throat
        } else {
            1.
        }
    }
}

pub fn optical_insert(scene: &IndoorManifest, p: &FixtureProgram) -> OpticalInsert {
    if scene.generator_version < 128
        || scene.program.is_none()
        || !matches!(p.shape, FixtureShape::Panel | FixtureShape::Downlight)
    {
        return OpticalInsert::Plain;
    }
    let mut r = stream(scene.seed, u64::from_le_bytes(*b"OpticSku"));
    match p.shape {
        FixtureShape::Panel if r.random_bool(0.62) => OpticalInsert::Basket {
            width: r.random_range(0.32..0.70),
            recess: r.random_range(0.27..0.77),
            reflector: if r.random_bool(0.72) {
                Surface::Ceiling
            } else {
                Surface::Chrome
            },
        },
        FixtureShape::Downlight => OpticalInsert::Cup {
            throat: r.random_range(0.48..0.88),
            recess: r.random_range(0.40..0.86),
            reflector: [Surface::Chrome, Surface::Metal, Surface::Ceiling][r.random_range(0..3)],
        },
        _ => OpticalInsert::Plain,
    }
}

/// The recess has to fit both the available housing depth and the diameter of
/// its reflector. A small trim in a deep housing need not become a pinhole.
pub(super) fn cup_depth(height: f32, rim: f32, recess: f32) -> f32 {
    (height * recess).min(rim * (0.65 + 1.9 * recess))
}

pub(crate) fn beam_angles(scene: &IndoorManifest, index: usize, position: Vec3) -> (f32, f32) {
    if scene.generator_version < 128 || scene.program.is_none() {
        let mut r = stream(
            super::super::super::photometry::lighting_seed(scene),
            192 + index as u64,
        );
        return (r.random_range(0.48..0.90), r.random_range(1.02..1.40));
    }
    let mut r = stream(scene.seed, 192 + index as u64);
    let p = parameters_at(scene, position);
    let insert = optical_insert(scene, &p);
    let outer = match insert {
        OpticalInsert::Cup { throat, recess, .. } => {
            let s = size(scene, position);
            let rim = cells(s).2 * p.opening_fraction;
            // At this angle the near rim hides the far edge of the emitter.
            let cutoff = ((1. + throat) * rim).atan2(cup_depth(s.y, rim, recess));
            cutoff.min(r.random_range(0.94..1.29))
        }
        OpticalInsert::Basket { width, recess, .. } => {
            let s = size(scene, position);
            let half = s.x.min(s.z) * (0.5 - p.border_fraction);
            // Along the long axis the open well has a broader distribution.
            let across = ((1. + width) * half).atan2(s.y * recess);
            0.5 * (across + r.random_range(1.29..1.49))
        }
        _ => match p.shape {
            FixtureShape::Louver => {
                let s = size(scene, position);
                let length = s.x.max(s.z);
                let cells = (length / p.louver_pitch_m).floor().clamp(2., 20.);
                let spacing = length * (1. - 2. * p.border_fraction) / cells;
                let cutoff = spacing.atan2(0.60 * s.y);
                0.57 * cutoff + 0.43 * r.random_range(1.29..1.49)
            }
            FixtureShape::Tube => {
                let broad = r.random_range(1.28..1.49);
                fitted_linear_at(scene, position, p.shape).map_or(broad, |l| l.beam(broad))
            }
            FixtureShape::Ring | FixtureShape::Drum => r.random_range(1.19..1.46),
            _ => r.random_range(1.28..1.49),
        },
    };
    // A frosted diffuser softens the shoulder. The direct and indirect lights
    // both normalize Bevy's squared cone falloff to the same delivered lumens.
    let shoulder = if matches!(insert, OpticalInsert::Cup { .. }) {
        r.random_range(0.47..0.74)
    } else {
        r.random_range(0.33..0.58)
    };
    (outer * shoulder, outer)
}

/// The diffuser and two splayed return faces exactly tile the aperture.
/// The surrounding four rails close the end cheeks up to the back pan.
pub(super) struct Basket {
    center: Vec3,
    along: Vec3,
    across: Vec3,
    end: f32,
    section: [(f32, f32); 4],
}
impl Basket {
    pub(super) fn new(center: Vec3, s: Vec3, width: f32, recess: f32) -> Self {
        let half = s.x.min(s.z) * 0.5;
        let across = if s.x >= s.z { Vec3::Z } else { Vec3::X };
        let rim = -s.y * 0.5 + 0.002;
        let glass = -s.y * 0.5 + recess * s.y;
        Self {
            center,
            across,
            along: Vec3::new(across.z, 0., across.x),
            end: s.x.max(s.z) * 0.5,
            section: [
                (-half, rim),
                (-half * width, glass),
                (half * width, glass),
                (half, rim),
            ],
        }
    }
    pub(super) fn strip(&self, g: &mut Geometry, i: usize) {
        let xy = |i: usize, z| {
            self.center
                + self.across * self.section[i].0
                + Vec3::Y * self.section[i].1
                + self.along * z
        };
        let [p, q, r, s] = [
            xy(i, -self.end),
            xy(i + 1, -self.end),
            xy(i + 1, self.end),
            xy(i, self.end),
        ];
        triangle(g, [p, q, r], Vec3::NEG_Y);
        triangle(g, [p, r, s], Vec3::NEG_Y);
    }
}
