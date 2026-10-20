//! A bounded, deterministic search within requested travel limits. Fractions
//! select eligible mechanisms; large amplitudes are ceilings, not minimums.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Adjustment {
    pub translation: f32,
    pub rotation: f32,
    pub heading: f32,
    pub morph: f32,
}
impl Default for Adjustment {
    fn default() -> Self {
        Self {
            translation: 1.,
            rotation: 1.,
            heading: 0.,
            morph: 1.,
        }
    }
}

pub(super) fn candidates(joints: &[Joint]) -> Vec<(Adjustment, &'static str)> {
    let mut out = vec![(Adjustment::default(), "requested")];
    let kind = joints[0].kind;
    // Spinning rotors need their entire disk at every nonzero speed. Slowing
    // one cannot make an obstructed installation safe.
    if kind == MotionKind::Fan {
        return out;
    }
    for scale in [0.5, 0.25, 0.125, 0.0625] {
        out.push((
            Adjustment {
                translation: scale,
                rotation: scale,
                morph: scale,
                ..default()
            },
            "shortened",
        ));
    }
    let travel = joints
        .iter()
        .map(|j| j.translation.length())
        .fold(0., f32::max);
    let angle = joints.iter().map(|j| j.angle.abs()).fold(0., f32::max);
    // End the search at metric/angular floors, rather than a fraction of a
    // potentially huge requested path. Raising the maximum still permits a
    // useful centimetre of travel or one degree of swivel in tight clearance.
    let minimum = Adjustment {
        translation: (0.01 / travel).min(1.),
        rotation: (1_f32.to_radians() / angle).min(1.),
        morph: 0.015,
        ..default()
    };
    out.push((minimum, "shortened"));
    if matches!(
        kind,
        MotionKind::Chair | MotionKind::Prop | MotionKind::Cart
    ) {
        // Search nearby headings as well as the opposite direction; do not
        // invent translation or swivel when that component was disabled.
        if joints.iter().any(|j| j.translation.length_squared() > 0.) {
            for heading in [
                std::f32::consts::PI,
                std::f32::consts::FRAC_PI_2,
                -std::f32::consts::FRAC_PI_2,
            ] {
                for scale in [0.5, 0.125, minimum.translation] {
                    out.push((
                        Adjustment {
                            heading,
                            translation: scale,
                            rotation: 0.,
                            ..default()
                        },
                        "rerouted",
                    ));
                }
            }
            out.push((
                Adjustment {
                    rotation: 0.,
                    ..default()
                },
                "translation_only",
            ));
        }
        if joints.iter().any(|j| j.angle != 0.) {
            for scale in [
                1.,
                0.5,
                0.25,
                0.125,
                -0.5,
                -0.125,
                minimum.rotation,
                -minimum.rotation,
            ] {
                out.push((
                    Adjustment {
                        translation: 0.,
                        rotation: scale,
                        ..default()
                    },
                    "swivel_only",
                ));
            }
        }
    }
    out
}
