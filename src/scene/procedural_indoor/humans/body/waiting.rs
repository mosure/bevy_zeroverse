//! Resolve the seat contact against this person's skinned posterior. The
//! anatomical pelvis-to-seat distance varies with stature and body shape.
use super::*;

pub(super) fn fit(
    h: &IndoorHuman,
    seat: super::super::poses::WaitingSeat,
    joints: &mut [Vec3],
    probes: impl Iterator<Item = (bool, Vec3)>,
) -> bool {
    let pelvis = joints[0];
    let scale = h.stature / 1.75;
    let mut floor = f32::INFINITY;
    let mut underside = f32::INFINITY;
    for (foot, p) in probes {
        if foot {
            floor = floor.min(p.y);
        } else {
            let d = (p - pelvis) / scale;
            if d.x.abs() < 0.145 && (-0.15..0.12).contains(&d.z) && (-0.235..0.).contains(&d.y) {
                underside = underside.min(p.y);
            }
        }
    }
    let underside = underside - floor;
    if !underside.is_finite() {
        return false;
    }
    let ease = h
        .appearance
        .as_ref()
        .map_or(0.008, |a| a.garment.trouser_ease);
    let travel = super::super::poses::WaitingSeat::FIT_TRAVEL_M;
    let offset = (seat.seat_height_m + ease + 0.002 - underside).clamp(-travel, travel);
    if offset.abs() < 0.001 {
        return false;
    }
    for joint in &mut joints[..13] {
        joint.y += offset;
    }
    for (hip, side) in [(13, -1.), (17, 1.)] {
        let root = joints[hip];
        let knee = joints[hip + 1];
        let ankle = joints[hip + 2];
        joints[hip].y += offset;
        (joints[hip + 1], _) = super::super::poses::two_bone(
            joints[hip],
            ankle,
            Vec3::new(side * 0.12, 0., -1.),
            knee.distance(root),
            knee.distance(ankle),
        );
    }
    true
}
