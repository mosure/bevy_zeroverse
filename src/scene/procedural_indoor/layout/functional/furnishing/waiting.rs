//! Waiting banks face the through aisle. Keep knees out of circulation and
//! reserve a standing apron plus a walkable connection back to the aisle.
use super::*;

const KNEES_M: f32 = 0.38;
const APRON_M: f32 = 0.64;
const LINK_WIDTH_M: f32 = 0.68;

pub(super) fn is_waiting_seat(activity: IndoorLayout, kind: ObjectKind) -> bool {
    matches!(
        activity,
        IndoorLayout::ElevatorLanding | IndoorLayout::Hallway
    ) && matches!(kind, ObjectKind::BeamSeating | ObjectKind::Banquette)
}

fn strip(axis: usize, from: f32, to: f32, a: f32, b: f32) -> Reservation {
    let mut min = Vec2::ZERO;
    let mut max = min;
    min[axis] = from.min(to);
    max[axis] = from.max(to);
    min[1 - axis] = a;
    max[1 - axis] = b;
    Reservation {
        min,
        max,
        role: ReservationRole::SeatingAccess,
    }
}

pub(super) fn routes(p: &ProgramAssembly, o: &IndoorObject) -> Option<[Option<Reservation>; 2]> {
    let spine = p
        .reservations
        .first()
        .filter(|s| s.role == ReservationRole::Circulation)?;
    let span = p.max - p.min;
    let axis = usize::from(p.activity == IndoorLayout::Hallway && span.x > span.y);
    let along = 1 - axis;
    let left = o.position.xz()[axis] < spine.min[axis];
    let sign = if left { 1. } else { -1. };
    let (lo, hi) = o.bounds();
    let (lo, hi) = (lo.xz(), hi.xz());
    let (front, aisle) = if left {
        (hi[axis], spine.min[axis])
    } else {
        (lo[axis], spine.max[axis])
    };
    let distance = (aisle - front) * sign;
    let forward = Vec2::new(o.yaw.sin(), o.yaw.cos());
    let mid = o.position.xz()[along];
    if forward[axis] * sign < 0.97
        || distance < KNEES_M
        || mid - LINK_WIDTH_M * 0.5 < spine.min[along]
        || mid + LINK_WIDTH_M * 0.5 > spine.max[along]
    {
        return None;
    }
    let knees = strip(axis, front, front + sign * KNEES_M, lo[along], hi[along]);
    if p.reservations
        .iter()
        .filter(|r| r.role == ReservationRole::DoorApproach)
        .any(|r| r.min.cmplt(knees.max).all() && r.max.cmpgt(knees.min).all())
    {
        return None;
    }
    let end = front + sign * APRON_M;
    let apron = strip(axis, front, end, lo[along], hi[along]);
    let link = (distance > APRON_M).then(|| {
        strip(
            axis,
            end,
            aisle + sign * 0.035,
            mid - LINK_WIDTH_M * 0.5,
            mid + LINK_WIDTH_M * 0.5,
        )
    });
    Some([Some(apron), link])
}

pub(in super::super) fn validate(
    scene: &IndoorManifest,
    p: &ProgramAssembly,
) -> Result<(), &'static str> {
    let mut installed = p
        .reservations
        .iter()
        .filter(|r| r.role == ReservationRole::SeatingAccess);
    for o in p
        .members
        .iter()
        .filter_map(|&id| scene.objects.get(id))
        .filter(|o| is_waiting_seat(p.activity, o.kind))
    {
        let required =
            routes(p, o).ok_or("waiting seats face away from the aisle or block a doorway")?;
        for route in required.iter().flatten() {
            if installed.next() != Some(route) {
                return Err("missing waiting-seat apron or connection to the aisle");
            }
        }
    }
    if installed.next().is_some() {
        return Err("unassigned waiting-seat approach");
    }
    Ok(())
}

impl IndoorManifest {
    pub(super) fn waiting_seats(&mut self, index: usize, axis: usize, stations: usize) {
        if stations == 0 {
            return;
        }
        let p = &self.functional_programs()[index];
        let (lo, hi) = (p.min, p.max);
        let mut r = stream(self.seed, u64::from_le_bytes(*b"WaitBays") + index as u64);
        let linked_share = r.random_range(0.36..0.89);
        let structure = PlacementStructure::new();
        let initial_side = r.random_bool(0.5);
        for station in 0..stations {
            let kind = if r.random_bool(linked_share) {
                ObjectKind::BeamSeating
            } else {
                ObjectKind::Banquette
            };
            let size = geometry::dimensions(kind, r.random());
            let along = 1 - axis;
            let margin = size.x * 0.5 + 0.23;
            let span = hi[along] - lo[along] - 2. * margin;
            if span <= 0. {
                continue;
            }
            let phase = r.random_range(0..6);
            let mut low = if station % 2 == 0 {
                initial_side
            } else {
                !initial_side
            };
            for attempt in 0..6 {
                if attempt == 3 {
                    low = !low;
                }
                let mut point = Vec2::ZERO;
                let stand_off = size.z * 0.5 + r.random_range(0.11..0.38);
                point[axis] = if low {
                    lo[axis] + stand_off
                } else {
                    hi[axis] - stand_off
                };
                // A new stratum on every attempt. Do not reflect the opposite
                // bank's accepted position across the centreline.
                let bin = (phase + 5 * attempt) % 6;
                point[along] =
                    lo[along] + margin + (bin as f32 + r.random_range(0.05..0.95)) * span / 6.;
                let yaw = match (axis, low) {
                    (0, true) => FRAC_PI_2,
                    (0, false) => -FRAC_PI_2,
                    (_, true) => 0.,
                    (_, false) => PI,
                } + r.random_range(-0.10..0.10);
                let site = FloorProposal {
                    kind,
                    size,
                    yaw,
                    position: Vec3::new(point.x, 0., point.y),
                };
                // The six candidates already cover both sides. An additional
                // ring of recovery offsets can push a waiting bank off its wall.
                if self
                    .public_fit_at_offsets(index, site, &mut r, &structure, &[Vec2::ZERO], |_| true)
                    .is_some()
                {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;

    #[test]
    fn transit_seats_face_an_accessible_through_aisle_and_reject_a_reversed_bank() {
        let mut kinds = [false; 2];
        let mut directions = [false; 4];
        let mut without = [false; 2];
        let mut tested_corruption = false;
        for (v, activity) in [IndoorLayout::Hallway, IndoorLayout::ElevatorLanding]
            .into_iter()
            .enumerate()
        {
            for seed in 70510..70550 {
                let scene =
                    IndoorManifest::generate_with_humans(seed, activity, 0.65, 0, 0.).unwrap();
                validate_layout(&scene).unwrap();
                let p = &scene.functional_programs()[0];
                let mut count = 0;
                for &id in &p.members {
                    let o = &scene.objects[id];
                    if activity == IndoorLayout::ElevatorLanding
                        && o.kind == ObjectKind::InformationStand
                    {
                        let spine = &p.reservations[0];
                        let pos = o.position.xz();
                        let approach = (pos.clamp(spine.min, spine.max) - pos).normalize_or_zero();
                        assert!(approach.dot(Vec2::new(o.yaw.sin(), o.yaw.cos())) > 0.98);
                    }
                    if !is_waiting_seat(activity, o.kind) {
                        continue;
                    }
                    count += 1;
                    kinds[usize::from(o.kind == ObjectKind::BeamSeating)] = true;
                    let f = Vec2::new(o.yaw.sin(), o.yaw.cos());
                    let axis = usize::from(f.y.abs() > f.x.abs());
                    directions[axis * 2 + usize::from(f[axis] > 0.)] = true;
                    let (lo, hi) = o.bounds();
                    let spine = &p.reservations[0];
                    let edge = if f[axis] > 0. {
                        spine.min[axis] - hi.xz()[axis]
                    } else {
                        lo.xz()[axis] - spine.max[axis]
                    };
                    assert!(edge >= KNEES_M);
                    let routes = routes(p, o).unwrap();
                    let walk: Vec<_> = routes.iter().flatten().collect();
                    assert!(walk.iter().all(|r| p.reservations.contains(r)));
                    let last = walk.last().unwrap();
                    assert!(last.min.cmple(spine.max).all() && last.max.cmpge(spine.min).all());
                    if !tested_corruption {
                        let mut bad = scene.clone();
                        bad.objects[id].yaw += PI;
                        assert!(bad.validate_public_programs().is_err());
                        bad = scene.clone();
                        bad.assembly_diagnostics.programs[0]
                            .reservations
                            .retain(|r| r.role != ReservationRole::SeatingAccess);
                        assert!(bad.validate_public_programs().is_err());
                        tested_corruption = true;
                    }
                }
                without[v] |= count == 0;
            }
        }
        assert!(
            tested_corruption
                && kinds.into_iter().all(|x| x)
                && directions.into_iter().all(|x| x)
                && without.into_iter().all(|x| x)
        );
    }
}
