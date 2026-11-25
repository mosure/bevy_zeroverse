//! Adapt ceiling stock dimensions to the shared indexed rotor mesh.
use super::*;
use crate::scene::procedural_indoor::geometry::rotor::RotorBlades;

pub(super) fn build(g: &mut Geometry, o: &IndoorObject, p: &FanProgram, section: &BladeProfile) {
    RotorBlades {
        blades: p.blades,
        phase: p.phase,
        hub_radius: p.hub_radius,
        radius_m: p.radius_m,
        rotor_y: p.rotor_y,
        blade_width_m: p.blade_width_m,
        root_chord_ratio: section.root_chord_ratio,
        shoulder_span: section.shoulder_span,
        tip_chord_ratio: section.tip_chord_ratio,
        pitch_radians: p.pitch_radians,
        tip_pitch_radians: section.tip_pitch_radians,
        sweep_m: p.sweep_m,
        thickness_m: p.thickness_m,
        camber_m: section.camber_m,
        tip_rise_m: section.tip_rise_m,
    }
    .append(g, o.seed, |v| v);
}
