//! Perimeter cubicles face the trading floor. Checkout and minimum stock have first claim
//! on the floor; additional merchandise fills around the cubicle and its approach.
use super::*;
use rand::seq::SliceRandom;

pub(in super::super) const APPROACH_DEPTH: f32 = 0.98;

/// Cover the whole frontage, including both doors on a two-bay installation.
pub(in super::super) fn serves(route: &Reservation, o: &IndoorObject) -> bool {
    let transform = o.transform();
    [-0.5, 0.5].into_iter().all(|side| {
        [0.002, APPROACH_DEPTH - 0.002].into_iter().all(|depth| {
            let corner = transform
                .transform_point(Vec3::new(
                    side * (o.size.x - 0.004),
                    0.,
                    o.size.z * 0.5 + depth,
                ))
                .xz();
            corner.cmpge(route.min).all() && corner.cmple(route.max).all()
        })
    })
}

impl IndoorManifest {
    pub(super) fn fit_changing_rooms(&mut self, index: usize, structure: &PlacementStructure) {
        let mut rng = stream(self.seed, u64::from_le_bytes(*b"FitRoom ") ^ index as u64);
        if !rng.random_bool(0.87) {
            return;
        }
        let program = &self.functional_programs()[index];
        let (lo, hi) = (program.min, program.max);
        let mut size = geometry::dimensions(ObjectKind::FittingRoom, rng.random());
        let mut sides = [(0, 1.), (0, -1.), (1, 1.), (1, -1.)];
        sides.shuffle(&mut rng);
        loop {
            for &(axis, sign) in &sides {
                let along = 1 - axis;
                let free = hi[along] - lo[along] - size.x - 0.12;
                if free < 0. || hi[axis] - lo[axis] < size.z + 1.10 {
                    continue;
                }
                let yaw = match (axis, sign > 0.) {
                    (0, true) => FRAC_PI_2,
                    (0, false) => -FRAC_PI_2,
                    (_, true) => 0.,
                    _ => PI,
                };
                let wall = if sign > 0. { lo[axis] } else { hi[axis] };
                let mut sections = [0, 1, 2];
                sections.shuffle(&mut rng);
                for section in sections {
                    let mut p = Vec2::ZERO;
                    p[axis] = wall + sign * (size.z * 0.5 + rng.random_range(0.05..0.25));
                    p[along] = lo[along]
                        + size.x * 0.5
                        + 0.06
                        + free * (section as f32 + rng.random_range(0.08..0.92)) / 3.;
                    let proposal = FloorProposal {
                        kind: ObjectKind::FittingRoom,
                        position: Vec3::new(p.x, 0., p.y),
                        size,
                        yaw,
                    };
                    if self
                        .public_fit_with_structure(index, proposal, &mut rng, structure, |o| {
                            let gap = (o.position.xz()[axis] - wall) * sign - size.z * 0.5;
                            (0.025..0.39).contains(&gap)
                        })
                        .is_some()
                    {
                        return;
                    }
                }
            }
            if size.x < 2.4 {
                return;
            }
            // The same kit can be installed as a single cubicle in a smaller shop.
            size.x *= 0.5;
        }
    }
}
