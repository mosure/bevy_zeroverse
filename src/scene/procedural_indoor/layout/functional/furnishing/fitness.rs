use super::*;
use crate::scene::procedural_indoor::objects::public::strength;
use rand::seq::SliceRandom;

const DUMBBELL_ACCESS: f32 = 0.83;

/// The operational footprint includes its rotated object box plus the working
/// margins in equipment coordinates. Its outer strips remain free during furnishing.
struct AccessEnvelope {
    a: Vec2,
    b: Vec2,
    min: Vec2,
    max: Vec2,
}

impl AccessEnvelope {
    fn for_object(o: &IndoorObject, lo: Vec3, hi: Vec3) -> Option<Self> {
        let (a, b) = (lo.xz(), hi.xz());
        let (sin, cos) = o.yaw.sin_cos();
        // Margins are measured in the equipment frame. The moving belt exits at
        // local -Z even when installed on the opposite side of a room.
        let (side, rear, front) = match o.kind {
            ObjectKind::Treadmill => (0.50, 2.00, 0.14),
            ObjectKind::StationaryBike => (0.35, 0.19, 0.19),
            ObjectKind::PowerRack => strength::ACCESS,
            ObjectKind::DumbbellRack => (0., 0., DUMBBELL_ACCESS),
            ObjectKind::WeightBench => (0.49, 0.30, 0.30),
            ObjectKind::ExerciseMatRack => (0., 0., 0.69),
            ObjectKind::ExerciseMat => (0.29, 0.21, 0.21),
            _ => return None,
        };
        let side_pad = Vec2::new(cos.abs(), sin.abs()) * side;
        let forward = Vec2::new(sin, cos);
        let min = a - side_pad + (-rear * forward).min(front * forward);
        let max = b + side_pad + (-rear * forward).max(front * forward);
        Some(Self { a, b, min, max })
    }
    fn strips(&self) -> impl Iterator<Item = Reservation> {
        let Self { a, b, min, max } = *self;
        [
            (min, Vec2::new(a.x, max.y)),
            (Vec2::new(b.x, min.y), max),
            (Vec2::new(a.x, min.y), Vec2::new(b.x, a.y)),
            (Vec2::new(a.x, b.y), Vec2::new(b.x, max.y)),
        ]
        .into_iter()
        .filter(|(l, h)| (h - l).min_element() > 0.001)
        .map(|(min, max)| Reservation {
            role: ReservationRole::EquipmentAccess,
            min,
            max,
        })
    }
}

pub(super) fn equipment_access(
    scene: &IndoorManifest,
    index: usize,
    o: &IndoorObject,
    structure: &PlacementStructure,
) -> Option<Vec<Reservation>> {
    let (lo, hi) = o.bounds();
    let area = AccessEnvelope::for_object(o, lo, hi)?;
    let (min, max) = (area.min, area.max);
    let program = &scene.functional_programs()[index];
    if min.cmplt(program.min).any() || max.cmpgt(program.max).any() {
        return None;
    }
    let mut clear = o.clone();
    let mid = (min + max) * 0.5;
    clear.position = Vec3::new(mid.x, lo.y, mid.y);
    let headroom = match o.kind {
        ObjectKind::Treadmill => 2.35,
        ObjectKind::PowerRack => o.size.y + 0.22,
        _ => 1.90,
    };
    clear.size = Vec3::new(max.x - min.x, headroom, max.y - min.y);
    clear.yaw = 0.;
    if !scene.placement_clear_with_structure(&clear, 0.035, false, structure) {
        return None;
    }
    Some(area.strips().collect())
}

/// Recompute the entire footprint, including rotated corners. The solid object's
/// own envelope already excludes furniture; only the outside strips are archived.
pub(in super::super) fn rack_access_present(
    scene: &IndoorManifest,
    p: &ProgramAssembly,
    o: &IndoorObject,
) -> bool {
    let (lo, hi) = o.bounds();
    let Some(area) = AccessEnvelope::for_object(o, lo, hi) else {
        return false;
    };
    if scene.minimum_ceiling_height(area.min, area.max) < hi.y + 0.22 - 0.001 {
        return false;
    }
    area.strips().all(|strip| {
        p.reservations.iter().any(|r| {
            r.role == strip.role
                && r.min.abs_diff_eq(strip.min, 0.001)
                && r.max.abs_diff_eq(strip.max, 0.001)
        })
    })
}

impl IndoorManifest {
    /// A wide dumbbell shelf may only fit lengthwise in a bank. Search for the
    /// whole shelf and standing area together, then offset back to the shelf.
    pub(super) fn recover_dumbbell_anchor(&mut self, index: usize, size: Vec3, r: &mut ChaCha8Rng) {
        let mut headings = [0., PI, FRAC_PI_2, -FRAC_PI_2];
        headings.shuffle(r);
        let structure = PlacementStructure::new();
        let compact = Vec3::new(1.05, size.y, 0.48);
        for stock in [size, compact] {
            let footprint = Vec3::new(stock.x, 1.90, stock.z + DUMBBELL_ACCESS);
            for yaw in headings {
                let forward = Vec2::new(yaw.sin(), yaw.cos());
                for center in recovery::anchor_points(self, index, footprint, yaw, r) {
                    let p = center - forward * (DUMBBELL_ACCESS * 0.5);
                    let proposal = FloorProposal {
                        kind: ObjectKind::DumbbellRack,
                        position: Vec3::new(p.x, 0., p.y),
                        size: stock,
                        yaw,
                    };
                    if self
                        .public_fit_with_structure(index, proposal, r, &structure, |_| true)
                        .is_some()
                    {
                        return;
                    }
                }
            }
        }
    }

    /// Try the side and both end walls of each bank. A bare sleeve has the
    /// same loading envelope as a heavy bar. Existing required fittings win.
    fn fit_strength_bay(
        &mut self,
        index: usize,
        banks: [(f32, f32); 2],
        start: f32,
        end: f32,
        cardio_share: f64,
    ) {
        if self.generator_version < strength::VERSION {
            return;
        }
        let mut r = stream(self.seed, u64::from_le_bytes(*b"LiftSite") ^ index as u64);
        if !r.random_bool(0.88 - 0.65 * cardio_share) {
            return;
        }
        self.assembly_diagnostics.programs[index].requested_capacity += 1;
        let size = geometry::dimensions(ObjectKind::PowerRack, r.random());
        let margin = size.x * 0.5 + strength::ACCESS.0 + 0.12;
        let mut order = [0usize, 1];
        order.shuffle(&mut r);
        let structure = PlacementStructure::new();
        for bank in order {
            let (left, right) = banks[bank];
            let low = Vec2::new(left, start);
            let high = Vec2::new(right, end);
            let inward = if bank == 0 { 1. } else { -1. };
            // Two disjoint stretches of the outer side and one candidate on
            // each end wall. Both the position and bar axis can change banks.
            let mut sites = [
                (0usize, inward, 0., 0.5),
                (0, inward, 0.5, 1.),
                (1, 1., 0., 1.),
                (1, -1., 0., 1.),
            ];
            sites.shuffle(&mut r);
            for (axis, sign, from, to) in sites {
                let along = 1 - axis;
                let gap =
                    (high[axis] - low[axis] - size.z - strength::ACCESS.1 - strength::ACCESS.2)
                        .min(0.84);
                let reach = high[along] - low[along] - 2. * margin;
                if gap <= 0.07 || reach <= 0. {
                    continue;
                }
                let wall = if sign > 0. { low[axis] } else { high[axis] };
                let mut pos = Vec2::ZERO;
                pos[axis] =
                    wall + sign * (size.z * 0.5 + strength::ACCESS.1 + r.random_range(0.06..gap));
                pos[along] = low[along] + margin + reach * r.random_range(from..to);
                let yaw = if axis == 0 {
                    sign * FRAC_PI_2
                } else if sign > 0. {
                    0.
                } else {
                    PI
                };
                let proposal = FloorProposal {
                    kind: ObjectKind::PowerRack,
                    position: Vec3::new(pos.x, 0., pos.y),
                    size,
                    yaw: yaw + r.random_range(-0.17..0.17),
                };
                if self
                    .public_fit_with_structure(index, proposal, &mut r, &structure, |_| true)
                    .is_some()
                {
                    return;
                }
            }
        }
    }

    pub(super) fn fitness_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = &self.functional_programs()[index];
        let (lo, hi) = (p.min, p.max);
        let aisle = &p.reservations[0];
        let banks = [(lo.x, aisle.min.x), (aisle.max.x, hi.x)];
        let pitch = r.random_range(2.55..3.75) + (1. - self.density) * 0.75;
        let rows = (((hi.y - lo.y) / pitch).floor() as usize).clamp(1, 6);
        let mut sites = Vec::with_capacity(2 * rows);
        for (left, right) in banks {
            let heading =
                r.random_range(-0.32..0.32) + if r.random_bool(0.52) { FRAC_PI_2 } else { 0. };
            for row in 0..rows {
                let x = left + (right - left) * r.random_range(0.36..0.64);
                let z =
                    lo.y + (hi.y - lo.y) * (row as f32 + r.random_range(0.33..0.67)) / rows as f32;
                sites.push((Vec3::new(x, 0., z), heading + r.random_range(-0.18..0.18)));
            }
        }
        sites.shuffle(r);
        let bench_share = r.random_range(0.22..0.86);
        let mut cardio = stream(self.seed, u64::from_le_bytes(*b"GymMixFt") + index as u64);
        let cardio_share = if cardio.random_bool(0.19) {
            0.
        } else {
            cardio.random_range(0.25..0.92)
        };
        let treadmill_share = cardio.random_range(0.20..0.84);
        let mut has_bench = false;
        for (pos, yaw) in sites {
            let kind = if has_bench && cardio.random_bool(cardio_share) {
                if cardio.random_bool(treadmill_share) {
                    ObjectKind::Treadmill
                } else {
                    ObjectKind::StationaryBike
                }
            } else if !has_bench || r.random_bool(bench_share) {
                ObjectKind::WeightBench
            } else {
                ObjectKind::ExerciseMat
            };
            let size = geometry::dimensions(kind, r.random());
            self.assembly_diagnostics.programs[index].requested_capacity += 1;
            // Put the front console toward the room end, with the belt runout
            // opening into the bank. Other stations retain their own heading.
            let heading = if kind == ObjectKind::Treadmill {
                (if pos.z > (lo.y + hi.y) * 0.5 { 0. } else { PI })
                    + cardio.random_range(-0.085..0.085)
            } else {
                yaw
            };
            let accepted = self
                .public_fit(index, kind, pos, size, heading, r)
                .is_some();
            if !has_bench && accepted && kind == ObjectKind::WeightBench {
                has_bench = true;
                self.fit_strength_bay(index, banks, lo.y, hi.y, cardio_share);
            }
            if !accepted && kind == ObjectKind::Treadmill {
                let size = geometry::dimensions(ObjectKind::StationaryBike, cardio.random());
                self.public_fit(index, ObjectKind::StationaryBike, pos, size, yaw, r);
            }
        }
        if !has_bench {
            // Narrow/chamfered rooms can lose a whole bank. Try opposite banks
            // at compact bench size before declaring the required station absent.
            let size = Vec3::new(0.48, 0.44, 1.22);
            for uv in variation::recovery_points(4, r) {
                let q = lo + (hi - lo) * uv;
                self.assembly_diagnostics.programs[index].requested_capacity += 1;
                if self
                    .public_fit(
                        index,
                        ObjectKind::WeightBench,
                        Vec3::new(q.x, 0., q.y),
                        size,
                        0.,
                        r,
                    )
                    .is_some()
                {
                    break;
                }
            }
        }
        self.assembly_diagnostics.programs[index]
            .required
            .push(ObjectKind::WeightBench);
    }
}
