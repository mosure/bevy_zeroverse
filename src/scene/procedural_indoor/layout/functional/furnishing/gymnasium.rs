//! A clear playing volume takes precedence over sideline furniture.
use super::*;
use crate::scene::procedural_indoor::objects::public::bleachers;
use crate::scene::procedural_indoor::program::court::VolleyballCourt;
mod goals;
pub(in super::super) use goals::{goal_access, valid_goal};

const APPROACH: f32 = 0.70;

impl IndoorManifest {
    pub(super) fn furnish_gymnasium(&mut self) {
        let Some(c) = self.program.as_ref().and_then(|p| p.volleyball) else {
            return;
        };
        let half = self.room_size.xz() * 0.5;
        let (lo, hi) = c.protected_bounds();
        let mut r = stream(self.seed, u64::from_le_bytes(*b"VBSeats "));
        let first = self.objects.len();
        let o = self.floor_candidate(
            ObjectKind::VolleyballNet,
            c.center.extend(0.).xzy(),
            c.net_size(),
            c.net_yaw(),
            &mut r,
        );
        // The equipment occupies its own centreline. Every subsequent solid
        // fixture must clear the entire court and its surrounding run-off.
        if self.placement_clear(&o, 0.035) {
            self.objects.push(o);
        }
        let target = c.world(Vec2::new(
            r.random_range(-0.33..0.33) * c.width,
            r.random_range(-0.18..0.18) * c.length,
        ));
        self.assembly_diagnostics.programs.push(ProgramAssembly {
            zone: 0,
            activity: IndoorLayout::Gymnasium,
            role: SpaceRole::Exercise,
            capacity_unit: CapacityUnit::PlayingCourts,
            parameters: ProgramParameters::sample(self.seed, 0, self.density),
            min: -half,
            max: half,
            members: vec![],
            required: vec![ObjectKind::VolleyballNet],
            requested_capacity: 1,
            accepted_capacity: 0,
            rejected_candidates: 0,
            reservations: vec![Reservation {
                min: lo,
                max: hi,
                role: ReservationRole::PlayingArea,
            }],
            services: vec![],
            focus: Vec3::new(target.x, r.random_range(1.35..2.15), target.y),
        });
        let banks = if self.generator_version >= bleachers::VERSION {
            self.spectator_banks(c)
        } else {
            [false; 2]
        };
        let primary = if r.random_bool(0.5) { 1. } else { -1. };
        for side in [primary, -primary] {
            if !r.random_bool(if side == primary { 0.93 } else { 0.49 })
                || banks[usize::from(side > 0.)]
            {
                continue;
            }
            let stations = (r.random_range(1.0..2.55) + self.density * 1.3).round() as usize;
            for i in 0..stations {
                let q = (i as f32 + r.random_range(0.24..0.76)) / stations as f32;
                let at = c.world(Vec2::new(
                    side * (c.width * 0.5 + c.free_zone + r.random_range(0.42..0.88)),
                    (q - 0.5) * c.length,
                ));
                let size = Vec3::new(
                    r.random_range(1.45..2.85),
                    r.random_range(0.76..0.98),
                    r.random_range(0.46..0.64),
                );
                let yaw = facing_court(c, side);
                self.public_fit(
                    0,
                    ObjectKind::Bench,
                    Vec3::new(at.x, 0., at.y),
                    size,
                    yaw,
                    &mut r,
                );
            }
        }
        self.install_practice_goals(c);
        let p = &mut self.assembly_diagnostics.programs[0];
        p.members = (first..self.objects.len()).collect();
        p.accepted_capacity = p.realized_capacity(&self.objects);
    }

    fn spectator_banks(&mut self, c: VolleyballCourt) -> [bool; 2] {
        let mut installed = [false; 2];
        let mut r = stream(self.seed, u64::from_le_bytes(*b"GymCrowd"));
        if !r.random_bool(if c.full_size { 0.74 } else { 0.53 }) {
            return installed;
        }
        let axis = if c.long_x { 1 } else { 0 };
        let wall = self.room_size.xz()[axis] * 0.5;
        let edge = c.width * 0.5 + c.free_zone;
        let wider = if c.center[axis] > 0. { -1. } else { 1. };
        let first = if r.random_bool(0.76) { wider } else { -wider };
        let both = r.random_bool(if c.full_size { 0.25 } else { 0.12 });
        let structure = PlacementStructure::new();
        for side in [first, -first] {
            if !both && installed.contains(&true) {
                break;
            }
            let passage = r.random_range(APPROACH..0.99);
            let depth = wall - side * c.center[axis] - edge - passage - 0.36;
            let pitch = r.random_range(0.605..0.700);
            let max_rows = (depth / pitch).floor().clamp(0., 5.) as usize;
            if max_rows < 2 {
                continue;
            }
            let rows = r.random_range(2..=max_rows);
            let width = r.random_range((c.length * 0.29).max(3.55)..(c.length * 0.59).min(8.30));
            let size = bleachers::dimensions(width, rows, pitch, r.random_range(0.235..0.300));
            let cross = side * (edge + passage + size.z * 0.5);
            // Try separate sections of this sideline if a door or column occupies
            // the preferred one. Each candidate retains the sampled width and tiers.
            let reach = (c.length - width) * 0.5;
            let favored = r.random_range(-0.90..0.90);
            for offset in [favored, if favored > 0. { -0.86 } else { 0.86 }, 0.] {
                let p = c.world(Vec2::new(cross, reach * offset));
                let mut o = self.floor_candidate(
                    ObjectKind::BleacherBank,
                    Vec3::new(p.x, 0., p.y),
                    size,
                    facing_court(c, side),
                    &mut r,
                );
                o.finish_seed = Some(self.seed);
                let access = bank_access(&o);
                if self.placement_clear_with_structure(&o, 0.035, false, &structure)
                    && !structure
                        .get_or_init(|| self.columns())
                        .iter()
                        .any(|&(lo, hi)| lo.y < 2.1 && hi.y > 0. && access.overlaps(lo, hi))
                    && self
                        .objects
                        .iter()
                        .filter(|item| item.solid && !item.neighbor)
                        .all(|item| {
                            let (lo, hi) = item.bounds();
                            !access.overlaps(lo, hi)
                        })
                {
                    self.objects.push(o);
                    self.assembly_diagnostics.programs[0]
                        .reservations
                        .push(access);
                    installed[usize::from(side > 0.)] = true;
                    break;
                }
                self.assembly_diagnostics.programs[0].rejected_candidates += 1;
            }
        }
        installed
    }
}

pub(in super::super) fn bank_access(o: &IndoorObject) -> Reservation {
    let forward = (o.transform().rotation * Vec3::Z).xz().round();
    let center = o.position.xz() + forward * ((o.size.z + APPROACH) * 0.5);
    let half = (forward.perp().abs() * o.size.x + forward.abs() * APPROACH) * 0.5;
    Reservation {
        min: center - half,
        max: center + half,
        role: ReservationRole::Circulation,
    }
}

fn facing_court(c: VolleyballCourt, side: f32) -> f32 {
    if c.long_x {
        if side > 0. {
            PI
        } else {
            0.
        }
    } else {
        -side * FRAC_PI_2
    }
}

pub(in super::super) fn valid_bank(o: &IndoorObject, c: VolleyballCourt) -> bool {
    let d = o.position.xz() - c.center;
    let local = if c.long_x { Vec2::new(d.y, d.x) } else { d };
    let dir = (o.transform().rotation * Vec3::Z).xz();
    o.solid
        && !o.neighbor
        && o.position.y.abs() < 0.005
        && dir[if c.long_x { 1 } else { 0 }] * -local.x.signum() > 0.999
        && local.x.abs() - o.size.z * 0.5 >= c.width * 0.5 + c.free_zone + APPROACH - 0.001
        && local.y.abs() + o.size.x * 0.5 <= c.length * 0.5 + 0.002
        && bleachers::fits_stock(o.size)
}

pub(in super::super) fn matches_net(o: &IndoorObject, c: VolleyballCourt) -> bool {
    o.kind == ObjectKind::VolleyballNet
        && o.solid
        && !o.neighbor
        && o.position
            .distance_squared(Vec3::new(c.center.x, 0., c.center.y))
            < 1e-8
        && o.size.distance_squared(c.net_size()) < 1e-8
        && (o.yaw - c.net_yaw()).abs() < 1e-6
}
