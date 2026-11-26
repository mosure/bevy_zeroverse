//! Feasible camera proposals on the approach, beside the flight or upstairs.
//! The normal trajectory, subject, rig spread and overlap gates still apply.
use crate::scene::procedural_indoor::envelope::Mezzanine;
use bevy::prelude::*;
use rand::Rng;

pub(super) fn propose(m: &Mezzanine, rng: &mut impl Rng) -> (Vec2, f32, Vec3) {
    let axis = m.stair_axis;
    let cross = 1 - axis;
    let sign = if m.ascending_positive() { 1. } else { -1. };
    let u = rng.random_range(2..m.steps - 2);
    let (lo, hi) = m.treads().nth(u as usize).unwrap();
    let at = lo.xz()
        + (hi.xz() - lo.xz())
            * Vec2::new(rng.random_range(0.12..0.88), rng.random_range(0.12..0.88));
    let target = Vec3::new(at.x, hi.y + rng.random_range(0.04..0.74), at.y);
    let mut eye = at;
    let support = if rng.random_bool(0.34) {
        let low = m.deck.min + Vec2::splat(0.46);
        let high = m.deck.max - Vec2::splat(0.46);
        eye = low + (high - low) * Vec2::new(rng.random(), rng.random());
        m.deck.height
    } else {
        if rng.random_bool(0.64) {
            let foot = if sign > 0. {
                m.stair_min[axis]
            } else {
                m.stair_max[axis]
            };
            eye[axis] = foot - sign * rng.random_range(1.0..3.45);
            eye[cross] += rng.random_range(-1.65..1.65);
        } else {
            let high = rng.random_bool(0.5);
            let edge = if high {
                m.stair_max[cross]
            } else {
                m.stair_min[cross]
            };
            eye[cross] = edge + if high { 1. } else { -1. } * rng.random_range(1.30..3.25);
            eye[axis] -= sign * rng.random_range(0.75..3.40);
        }
        0.
    };
    (eye, support, target)
}

#[cfg(test)]
mod tests {
    use super::super::frame_person;
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::CameraSettings,
        layout::{IndoorLayout, IndoorManifest},
    };

    #[test]
    fn stair_hall_person_composition_preserves_visibility_for_narrow_lenses() {
        let person = Vec3::new(0., 1.5, -4.);
        let eye = Vec3::Y * 1.5;
        let stair = Vec3::new(3., 3., -4.);
        for scale in [0.12, 0.5, 1.] {
            let at = frame_person(eye, person, stair, scale);
            let angle = (at - eye).angle_between(person - eye);
            assert!(angle > 0. && angle.to_degrees() <= 12.0 * scale + 0.001);
            assert!(
                (at - eye).angle_between(stair - eye) < (person - eye).angle_between(stair - eye)
            );
        }
        assert_eq!(frame_person(eye, person, person, 1.), person);
    }

    #[test]
    fn stair_hall_independent_room_cameras_use_both_floor_levels() {
        let settings = CameraSettings {
            lenses: None,
            ceiling_mount: None,
            multiview: None,
            ..Default::default()
        };
        let mut upper = 0;
        let mut lower = 0;
        for seed in 0..32 {
            let mut s =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::StairHall, 0.65, 0, 0.)
                    .unwrap();
            s.resample_cameras(1, settings.clone(), 1.).unwrap();
            let m = s.envelope.as_ref().unwrap().mezzanine.as_ref().unwrap();
            let camera = &s.cameras[0];
            let eye = camera.transform_at(0.).translation;
            upper += usize::from(m.deck.contains(eye.xz()) && eye.y > m.deck.height + 0.50);
            lower += usize::from(eye.y < m.deck.height);
            assert!(s.camera_curve_clear(camera));
        }
        assert!(upper >= 2 && lower >= 10, "upper={upper} lower={lower}");
    }
}
