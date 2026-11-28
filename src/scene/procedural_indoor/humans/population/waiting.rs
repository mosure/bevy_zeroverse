//! Existing passengers may sit on an installed waiting seat. The bank faces
//! local +Z, while a person faces local -Z. Table bays never support a person.
use super::*;
#[cfg(test)]
mod tests;
use crate::scene::procedural_indoor::{
    layout::{
        functional::{ProgramAssembly, ReservationRole},
        IndoorLayout, IndoorObject,
    },
    objects::public::{banquette, beam_seating::BeamSeating},
};

pub(in super::super) const VERSION: u32 = 316;

fn transit(zone: &ProgramAssembly) -> bool {
    matches!(
        zone.activity,
        IndoorLayout::ElevatorLanding | IndoorLayout::Hallway
    )
}

#[derive(Clone, Copy)]
struct Place {
    seat: poses::WaitingSeat,
    x: f32,
    width: f32,
    back: f32,
    front: f32,
    arms: [Option<f32>; 2],
}

fn places(bank: &IndoorObject) -> Vec<Place> {
    match bank.kind {
        ObjectKind::BeamSeating => {
            let product = BeamSeating::for_object(bank);
            (0..product.bays)
                .filter(|&bay| Some(bay) != product.table_bay)
                .map(|bay| {
                    let x = product.center_x(bay);
                    let arms = [bay, bay + 1].map(|i| product.arm_x(bank, i));
                    Place {
                        seat: poses::WaitingSeat {
                            bay,
                            seat_height_m: product.seat_height_m,
                        },
                        x,
                        width: product.seat_width_m,
                        back: -bank.size.z * 0.215,
                        front: bank.size.z * 0.44,
                        arms,
                    }
                })
                .collect()
        }
        ObjectKind::Banquette => {
            let product = banquette::parameters(bank);
            let (back, front) = product.seat_depth_range(bank);
            let full_width = banquette::cushion_width(bank);
            let total = product.seat_weights.iter().sum::<f32>();
            let mut places = Vec::new();
            let mut left = -full_width * 0.5;
            for &weight in &product.seat_weights {
                let pad = full_width * weight / total;
                // A single long cushion accommodates several passengers. On
                // divided cushions keep each pelvis away from the seam.
                let count = (pad / 0.57).floor().clamp(1., 6.) as usize;
                let width = pad / count as f32;
                for i in 0..count {
                    places.push(Place {
                        seat: poses::WaitingSeat {
                            bay: places.len(),
                            seat_height_m: product.seat_height_m,
                        },
                        x: left + (i as f32 + 0.5) * width,
                        width,
                        back,
                        front,
                        arms: [None; 2],
                    });
                }
                left += pad;
            }
            places
        }
        _ => Vec::new(),
    }
}

impl Place {
    fn fits(&self, bank: &IndoorObject, zone: &ProgramAssembly, h: &IndoorHuman) -> bool {
        let local = bank.transform().compute_affine().inverse() * h.transform().compute_affine();
        let pt = |joint: usize| local.transform_point3(h.joints[joint]);
        let forward = local.transform_vector3(Vec3::NEG_Z);
        let hip = pt(0);
        let scale = h.stature / 1.75;
        let seat_y = self.seat.seat_height_m;
        let travel = poses::WaitingSeat::FIT_TRAVEL_M;
        if forward.z < 0.975
            || local.translation.y.abs() > 0.001
            || (hip.y - self.seat.pelvis_height(h.stature)).abs() > 0.001
            || (hip.x - self.x).abs() + 0.155 * h.build * scale > self.width * 0.5
            || hip.z < self.back + 0.14 * scale
            || hip.z > self.front - 0.155 * scale
        {
            return false;
        }
        // Feet bear on the floor ahead of the plinth or metal frame. Protect
        // the through aisle and the door approaches, including the toe boxes.
        for (heel, toe, radius) in footwear::seated_soles(h) {
            if local.transform_point3(heel).z < self.front + 0.024 {
                return false;
            }
            let tf = h.transform();
            let a = tf.transform_point(heel);
            let b = tf.transform_point(toe);
            if zone.reservations.iter().any(|r| {
                matches!(
                    r.role,
                    ReservationRole::Circulation | ReservationRole::DoorApproach
                ) && r.overlaps(a.min(b) - radius, a.max(b) + radius)
            }) {
                return false;
            }
        }
        // Test the resting and gesturing arms against the actual bank's arm
        // positions. Its own seat pan is an intentional contact surface.
        let back = Vec3::new(
            self.x - self.width * 0.5,
            seat_y + 0.04 - travel,
            -bank.size.z * 0.5,
        );
        let face = Vec3::new(self.x + self.width * 0.5, bank.size.y + travel, self.back);
        if box_hit(pt(0), pt(2), 0.13 * scale, back, face) {
            return false;
        }
        for arm in [5, 9] {
            let hand = pt(arm + 3);
            let tip = hand + (hand - pt(arm + 2)).normalize_or_zero() * (0.085 * scale);
            for (a, b, radius) in [
                (pt(arm), pt(arm + 1), 0.078 * h.build),
                (pt(arm + 1), pt(arm + 2), 0.059 * h.build),
                (pt(arm + 2), tip, 0.050 * scale),
            ] {
                if box_hit(a, b, radius, back, face)
                    || a.y.min(b.y) - radius < seat_y + travel + 0.006
                    || self.arms.into_iter().flatten().any(|x| {
                        box_hit(
                            a,
                            b,
                            radius,
                            Vec3::new(x - 0.019, seat_y - 0.10 - travel, -0.121),
                            Vec3::new(
                                x + 0.019,
                                seat_y + 0.21 + travel,
                                bank.size.z * 0.24 + 0.016,
                            ),
                        )
                    })
                {
                    return false;
                }
            }
        }
        true
    }
}

pub(in super::super) fn valid(
    scene: &IndoorManifest,
    bank: &IndoorObject,
    h: &IndoorHuman,
    seat: poses::WaitingSeat,
) -> bool {
    scene.generator_version >= VERSION
        && h.pose.seated()
        && bank.neighbor == h.neighbor
        && scene
            .functional_programs()
            .iter()
            .filter(|p| transit(p))
            .any(|p| {
                p.members.contains(&bank.id)
                    && places(bank).into_iter().any(|place| {
                        place.seat.bay == seat.bay
                            && (place.seat.seat_height_m - seat.seat_height_m).abs() < 0.0001
                            && place.fits(bank, p, h)
                    })
            })
}

pub(super) fn stage(scene: &mut IndoorManifest, first: usize) {
    if !scene.functional_programs().iter().any(transit) {
        return;
    }
    let mut rng = stream(scene.seed, u64::from_le_bytes(*b"WaitPax1"));
    let objects = &scene.objects;
    let mut candidates: Vec<_> = scene
        .functional_programs()
        .iter()
        .enumerate()
        .filter(|(_, p)| transit(p))
        .flat_map(|(zone, p)| {
            p.members.iter().flat_map(move |&id| {
                places(&objects[id])
                    .into_iter()
                    .map(move |place| (zone, id, place))
            })
        })
        .collect();
    candidates.shuffle(&mut rng);
    let probability = rng.random_range(0.42..0.83);
    for i in first..scene.humans.len() {
        let original = &scene.humans[i];
        if candidates.is_empty() {
            break;
        }
        if original.neighbor || original.workstation.is_some() || !rng.random_bool(probability) {
            continue;
        }
        // Preserve the admitted person's seed, appearance and fallback pose.
        // Work stays bounded and a failed fit never removes an occupant.
        let mut h = original.clone();
        h.pose = seated_pose(false, false, 0.5, &mut rng);
        let base = poses::PoseProgram::sample(h.seed, h.pose);
        for attempt in 0..8 {
            let which = attempt % candidates.len();
            let (zone, id, place) = candidates[which];
            let p = &scene.functional_programs()[zone];
            if scene.layout == IndoorLayout::PublicSuite
                && !(original.position.xz().cmpge(p.min).all()
                    && original.position.xz().cmple(p.max).all())
            {
                continue;
            }
            let bank = &scene.objects[id];
            h.chair = Some(id);
            h.yaw = bank.yaw + PI + rng.random_range(-0.17..0.17);
            h.position = bank.transform().transform_point(Vec3::new(
                place.x + rng.random_range(-0.028..0.028),
                0.,
                place.front - rng.random_range(0.187..0.260) * h.stature / 1.75,
            ));
            let mut pose = base.clone();
            pose.waiting_seat = Some(place.seat);
            pose.stride *= 0.40;
            let neutral = pose.solve(h.stature, h.build, h.shoulder_width, true);
            let gesture = (h.pose == HumanPoseKind::SeatedTalking).then(|| rng.random_range(0..2));
            for (arm, shoulder) in [5, 9].into_iter().enumerate() {
                if Some(arm) == gesture {
                    continue;
                }
                let side = if arm == 0 { -1. } else { 1. };
                let wrist = Vec3::new(
                    side * rng.random_range(0.06..0.20),
                    neutral[0].y + rng.random_range(0.055..0.155),
                    -rng.random_range(0.15..0.32),
                );
                pose.aim_wrist(
                    arm,
                    pose.torso_rotation().inverse() * (wrist - neutral[shoulder]),
                    h.stature,
                );
            }
            h.joints = pose.solve(h.stature, h.build, h.shoulder_width, true);
            h.pose_program = Some(pose);
            if !place.fits(bank, p, &h) {
                continue;
            }
            update_bounds(&mut h);
            if placement_clear(scene, &h) {
                scene.humans[i] = h;
                candidates.swap_remove(which);
                break;
            }
        }
    }
}
