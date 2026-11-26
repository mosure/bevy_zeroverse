//! A bounded, deterministic search within requested travel limits. Fractions
//! select eligible mechanisms; large amplitudes are ceilings, not minimums.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Adjustment {
    pub translation: f32,
    pub rotation: f32,
    pub heading: f32,
    pub morph: f32,
    pub lift: f32,
}
impl Default for Adjustment {
    fn default() -> Self {
        Self {
            translation: 1.,
            rotation: 1.,
            heading: 0.,
            morph: 1.,
            lift: 1.,
        }
    }
}

pub(super) fn candidates(joints: &[Joint], version: u32) -> Vec<(Adjustment, &'static str)> {
    let primary = &joints[0];
    let modern_prop = version >= TEMPORAL_PROGRAM_VERSION && primary.kind == MotionKind::Prop;
    if modern_prop && primary.contact == Contact::Surface {
        return on_surface(primary);
    }
    // An articulated laptop lid can reduce its requested hinge excursion. Reversing
    // a freely swivelling prop is valid; reversing a hinge can violate its stops.
    let restrict_hinge = modern_prop && primary.contact == Contact::Free;
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
        // A screen hem and its fabric share one transmission, including the shortest fallback.
        translation: if matches!(
            primary.cloth_motion,
            Some(ClothMotion::ProjectionScreen { .. })
        ) {
            0.015
        } else {
            (0.01 / travel).min(1.)
        },
        rotation: (1_f32.to_radians() / angle).min(1.),
        morph: 0.015,
        ..default()
    };
    out.push((minimum, "shortened"));
    if !restrict_hinge
        && matches!(
            kind,
            MotionKind::Chair | MotionKind::Prop | MotionKind::Cart
        )
    {
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

/// Try useful alternative headings on a worktop before reducing a gesture to
/// a nudge. Every candidate must still pass the continuous clearance tests.
fn on_surface(j: &Joint) -> Vec<(Adjustment, &'static str)> {
    let mut choices = Vec::with_capacity(24);
    let travel = j.translation != Vec3::ZERO;
    let turn = j.angle != 0.;
    for scale in [1., 0.5, 0.125] {
        let fit = Adjustment {
            translation: scale,
            rotation: scale,
            lift: scale,
            ..default()
        };
        choices.push((
            fit,
            if scale == 1. {
                "requested"
            } else {
                "shortened"
            },
        ));
        if j.lift_m > 0. && (travel || turn) {
            choices.push((Adjustment { lift: 0., ..fit }, "surface_only"));
        }
        if travel {
            for heading in [
                std::f32::consts::PI,
                std::f32::consts::FRAC_PI_2,
                -std::f32::consts::FRAC_PI_2,
            ] {
                choices.push((
                    Adjustment {
                        heading,
                        rotation: 0.,
                        lift: 0.,
                        ..fit
                    },
                    "rerouted",
                ));
            }
        }
        if j.lift_m > 0. && (travel || turn) {
            choices.push((
                Adjustment {
                    rotation: 0.,
                    translation: 0.,
                    ..fit
                },
                "pickup_only",
            ));
        }
        if turn && (travel || j.lift_m > 0.) {
            choices.push((
                Adjustment {
                    translation: 0.,
                    lift: 0.,
                    ..fit
                },
                "swivel_only",
            ));
        }
    }
    if turn {
        for scale in [-0.5, -0.125] {
            choices.push((
                Adjustment {
                    translation: 0.,
                    rotation: scale,
                    lift: 0.,
                    ..default()
                },
                "swivel_only",
            ));
        }
    }
    choices.push((
        Adjustment {
            translation: (0.01 / j.translation.length()).min(1.),
            rotation: (1_f32.to_radians() / j.angle.abs()).min(1.),
            lift: (0.01 / j.lift_m).min(1.),
            ..default()
        },
        "shortened",
    ));
    choices
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn laptop_hinge_stays_within_its_physical_stops_while_loose_items_can_reverse() {
        let mut j = construction::rigid(MotionKind::Prop, 0, Vec3::ZERO, Vec3::X, 0.85, Vec3::ZERO);
        let hinge = candidates(std::slice::from_ref(&j), TEMPORAL_PROGRAM_VERSION);
        assert_eq!(hinge.len(), 6);
        assert!(hinge.iter().all(|(a, _)| (0.0..=1.).contains(&a.rotation)));
        j.contact = Contact::Surface;
        let prop = candidates(&[j], TEMPORAL_PROGRAM_VERSION);
        assert!(prop.iter().any(|(a, _)| a.rotation < 0.));
        assert!(prop.len() <= 24);
    }
}
