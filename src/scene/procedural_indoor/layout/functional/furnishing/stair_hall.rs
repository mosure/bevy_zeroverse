//! Keep a continuous clear path from the shared entrance, around the stair
//! structure when necessary, up the flight and into the upper seating bay.
use super::*;

pub(in crate::scene::procedural_indoor::layout::functional) fn routes(
    scene: &IndoorManifest,
) -> Vec<Reservation> {
    let m = scene.envelope.as_ref().unwrap().mezzanine.as_ref().unwrap();
    let half = scene.room_size.xz() * 0.5;
    let a = m.stair_axis;
    let w = (m.stair_width() * 0.38).min(0.65);
    let dir = if m.ascending_positive() { 1. } else { -1. };
    let mut toe = (m.stair_min + m.stair_max) * 0.5;
    toe[a] = if dir > 0. {
        m.stair_min[a]
    } else {
        m.stair_max[a]
    };
    let mut approach = toe;
    approach[a] -= dir * (w + 0.65);
    let mut upper = toe;
    // A turning circle fits on the gallery beyond the last riser.
    upper[a] = (m.deck.min[a] + m.deck.max[a]) * 0.5;
    let front = half.y - 1.37;
    let mut point = Vec2::new(scene.door_x, half.y - 0.16 - w);
    let mut path = Vec::with_capacity(6);
    let mut walk_to = |next: Vec2| {
        if next != point {
            path.push(Reservation {
                role: ReservationRole::Circulation,
                min: point.min(next) - Vec2::splat(w),
                max: point.max(next) + Vec2::splat(w),
            });
        }
        point = next;
    };
    walk_to(Vec2::new(scene.door_x, front));
    if a == 1 && dir > 0. {
        // The deck lies between entrance and stair foot. Take a side aisle
        // outside the columns/guard, then turn across the lower landing.
        let sign = if m.deck.min.x.abs() > m.deck.max.x.abs() {
            1.
        } else {
            -1.
        };
        let side = sign * (half.x - 1.40);
        walk_to(Vec2::new(side, front));
        walk_to(Vec2::new(side, approach.y));
    } else {
        walk_to(Vec2::new(approach.x, front));
    }
    walk_to(approach);
    walk_to(upper);
    path
}

impl IndoorManifest {
    pub(super) fn furnish_stair_hall(&mut self) {
        let m = self.envelope.as_ref().unwrap().mezzanine.as_ref().unwrap();
        let deck = m.deck.clone();
        let a = m.stair_axis;
        let c = 1 - a;
        let left = m.stair_min[c] - deck.min[c] > deck.max[c] - m.stair_max[c];
        let bay_lo = if left { deck.min[c] } else { m.stair_max[c] };
        let bay_hi = if left { m.stair_min[c] } else { deck.max[c] };
        let half = self.room_size.xz() * 0.5;
        let center = (deck.min + deck.max) * 0.5;
        let mut r = stream(self.seed, u64::from_le_bytes(*b"StairFit"));
        let seated = r.random_bool(0.52 + 0.22 * self.density as f64);
        let paired =
            seated && bay_hi - bay_lo > 5.0 && r.random_bool(0.16 + 0.31 * self.density as f64);
        let stations = if paired { 2 } else { 1 };
        self.assembly_diagnostics.programs.push(ProgramAssembly {
            zone: 0,
            activity: IndoorLayout::StairHall,
            role: SpaceRole::Transit,
            capacity_unit: CapacityUnit::SeatingStations,
            parameters: ProgramParameters::sample(self.seed, 0, self.density),
            min: -half,
            max: half,
            members: vec![],
            required: vec![],
            requested_capacity: if seated { stations } else { 0 },
            accepted_capacity: 0,
            rejected_candidates: 0,
            reservations: routes(self),
            services: vec![],
            focus: Vec3::new(center.x, deck.height + 0.62, center.y),
        });
        // A gallery can overlook the stair, hold artwork or offer a place to
        // sit. Leave a turning area at the head in every arrangement.
        let inward = match (c, left) {
            (0, true) => FRAC_PI_2,
            (0, false) => -FRAC_PI_2,
            (_, true) => 0.,
            (_, false) => PI,
        };
        let display = !seated && r.random_bool(0.70 + 0.23 * self.density as f64);
        if seated || display {
            for station in 0..stations {
                let kind = if seated {
                    if r.random_bool(0.60) {
                        ObjectKind::Bench
                    } else {
                        ObjectKind::Banquette
                    }
                } else if r.random_bool(0.60) {
                    ObjectKind::ExhibitionPanel
                } else {
                    ObjectKind::InformationStand
                };
                let size = if seated {
                    Vec3::new(
                        r.random_range(1.16..(deck.max[a] - deck.min[a] - 0.64).min(2.45)),
                        r.random_range(0.72..1.08),
                        r.random_range(0.52..0.82),
                    )
                } else {
                    let mut size = geometry::dimensions(kind, r.random());
                    size.x = size.x.min(deck.max[a] - deck.min[a] - 0.65);
                    size
                };
                let distance = match (paired, station) {
                    (true, 0) => r.random_range(0.18..0.31),
                    (true, _) => r.random_range(0.65..0.79),
                    _ => r.random_range(0.19..0.68),
                };
                let mut p = center;
                p[c] = bay_lo + (bay_hi - bay_lo) * if left { distance } else { 1. - distance };
                let span = (deck.max[a] - deck.min[a] - size.x - 0.48).max(0.) * 0.5;
                p[a] += r.random_range(-1.0..1.0) * span;
                let yaw = inward + if station > 0 { PI } else { 0. } + r.random_range(-0.21..0.21);
                self.public_fit_with(
                    0,
                    FloorProposal {
                        kind,
                        position: Vec3::new(p.x, deck.height, p.y),
                        size,
                        yaw,
                    },
                    &mut r,
                    |o| {
                        if o.kind == ObjectKind::Bench {
                            crate::scene::procedural_indoor::objects::services::bench::fit_height(
                                o,
                            );
                        }
                        true
                    },
                );
            }
        }

        // A directory at ground level faces into the room. Vary the chosen wall
        // and setback; the shared gate rejects doorway and route obstructions.
        if r.random_bool(0.55 + 0.32 * self.density as f64) {
            let size = geometry::dimensions(ObjectKind::InformationStand, r.random());
            for _ in 0..6 {
                let side = r.random_range(0..4);
                let mut p =
                    half * Vec2::new(r.random_range(-0.74..0.74), r.random_range(-0.74..0.74));
                let i = side / 2;
                p[i] =
                    (half[i] - r.random_range(0.94..1.60)) * if side % 2 == 0 { 1. } else { -1. };
                let yaw = (-p.x).atan2(-p.y) + r.random_range(-0.35..0.35);
                if self
                    .public_fit(
                        0,
                        ObjectKind::InformationStand,
                        Vec3::new(p.x, 0., p.y),
                        size,
                        yaw,
                        &mut r,
                    )
                    .is_some()
                {
                    break;
                }
            }
        }
        let receipt = &mut self.assembly_diagnostics.programs[0];
        receipt.members = (0..self.objects.len()).collect();
        receipt.accepted_capacity = receipt.realized_capacity(&self.objects);
    }
}
