//! Fit dining groups between the service frontage, public aisle and zone edge.
use super::*;

pub(super) struct TableSite {
    pub position: Vec3,
    pub width: f32,
    pub yaw: f32,
}

#[derive(Default)]
pub(super) struct DiningPlan {
    pub tables: Vec<TableSite>,
    pub perimeter_bench: Option<FloorProposal>,
}

/// Use the actual front of service fixtures and queue hardware for each side.
/// A recovered fixture elsewhere in the room remains a local collision obstacle.
pub(super) fn dining_bays(p: &ProgramAssembly, objects: &[IndoorObject]) -> Option<[Rect; 2]> {
    let aisle = p.dining_aisle()?;
    let bounds = [(p.min.x, aisle.min.x), (aisle.max.x, p.max.x)];
    let mut bays = bounds.map(|(left, right)| Rect {
        min: Vec2::new(left, p.min.y),
        max: Vec2::new(right, p.max.y),
    });
    let front_limit = p.min.y + (p.max.y - p.min.y) * 0.46;
    for bay in &mut bays {
        for r in &p.reservations {
            if r.max.x <= bay.min.x || r.min.x >= bay.max.x {
                continue;
            }
            if r.role == ReservationRole::Circulation && r.max.y >= p.max.y {
                bay.max.y = bay.max.y.min(r.min.y);
            } else if (r.role == ReservationRole::Circulation && r.min.y <= p.min.y)
                || (matches!(
                    r.role,
                    ReservationRole::Queue | ReservationRole::StaffAccess
                ) && r.min.y < front_limit)
            {
                bay.min.y = bay.min.y.max(r.max.y);
            }
        }
        for o in objects.iter().filter(|o| {
            !o.neighbor
                && matches!(
                    o.kind,
                    ObjectKind::ServiceCounter
                        | ObjectKind::DisplayCase
                        | ObjectKind::TrayReturn
                        | ObjectKind::QueueBarrier
                )
                && o.position.xz().cmpge(p.min).all()
                && o.position.xz().cmple(p.max).all()
        }) {
            let (lo, hi) = o.bounds();
            if lo.z < front_limit && hi.x > bay.min.x && lo.x < bay.max.x {
                bay.min.y = bay.min.y.max(hi.z);
            }
        }
    }
    Some(bays)
}

impl DiningPlan {
    pub(super) fn sample(
        p: &ProgramAssembly,
        objects: &[IndoorObject],
        seed: u64,
        communal: bool,
    ) -> Self {
        let mut plan = Self::default();
        let Some(bays) = dining_bays(p, objects) else {
            return plan;
        };
        let mut venue = stream(seed, u64::from_le_bytes(*b"CafeBays") ^ p.zone as u64);
        let looseness = venue.random_range(0.06..0.97);
        let along_aisle = venue.random_bool(0.45);
        // Denser venues use shorter gaps. Even their small tables need space for seats.
        let restaurant = p.activity == IndoorLayout::Restaurant;
        let pitch = venue.random_range(if restaurant { 2.58..3.27 } else { 2.95..3.65 })
            - (p.parameters.occupancy - 0.6) * 1.30
            + if communal { 0.18 } else { 0. };
        for bay in bays {
            let mut bank = ChaCha8Rng::seed_from_u64(venue.random());
            let heading = p.parameters.group_rotation_radians
                + if along_aisle ^ bank.random_bool(0.22) {
                    FRAC_PI_2
                } else {
                    0.
                }
                + bank.random_range(-0.53..0.53) * looseness;
            sample_bank(
                &mut plan.tables,
                bay,
                pitch,
                heading,
                looseness,
                p.activity,
                &mut bank,
            );
        }
        // Random admission order gives neither wall nor the service end first
        // choice when a seat footprint can fit at only one of two stations.
        plan.tables.shuffle(&mut venue);
        // A venue need not install a table at every usable station. Keep quieter
        // and fully furnished rooms possible at the same requested density.
        let fill = (venue.random_range(if restaurant { 0.72..1.05 } else { 0.37..1.02 })
            + 0.78 * (p.parameters.occupancy - 0.80))
            .clamp(0.30, 1.0);
        let count = (plan.tables.len() as f32 * fill + venue.random::<f32>()) as usize;
        plan.tables.truncate(count.max(1));
        if p.activity == IndoorLayout::Cafe {
            plan.perimeter_bench = perimeter_bench(bays, &mut venue);
        }
        plan
    }
}

fn sample_bank(
    sites: &mut Vec<TableSite>,
    bay: Rect,
    pitch: f32,
    heading: f32,
    looseness: f32,
    activity: IndoorLayout,
    bank: &mut ChaCha8Rng,
) {
    let usable = bay.size();
    if usable.x < 1.25 || usable.y < 1.6 {
        return;
    }
    let lanes = if usable.x
        >= bank.random_range(if activity == IndoorLayout::Restaurant {
            4.7..5.8
        } else {
            5.85..7.20
        }) {
        2
    } else {
        1
    };
    let lane_width = usable.x / lanes as f32;
    let bay_pitch = pitch * bank.random_range(0.90..1.12);
    for lane in 0..lanes {
        let left = bay.min.x + lane as f32 * lane_width;
        let start = bay.min.y + bank.random_range(0.86..1.18);
        let end = bay.max.y - bank.random_range(0.78..1.12);
        if end < start {
            continue;
        }
        let n = (1 + ((end - start) / bay_pitch) as usize).min(8);
        let slack = end - start - (n - 1) as f32 * bay_pitch;
        let first = start + slack * bank.random_range(0.07..0.93);
        let line = bank.random_range(0.05..0.95);
        for row in 0..n {
            let width = match activity {
                IndoorLayout::Cafeteria => bank.random_range(1.35..1.95),
                IndoorLayout::Restaurant => match bank.random_range(0..10) {
                    0..=3 => bank.random_range(0.74..1.0),
                    4..=7 => bank.random_range(1.06..1.55),
                    _ => bank.random_range(1.58..2.22),
                },
                _ => bank.random_range(0.78..1.20),
            };
            let yaw = heading + bank.random_range(-0.36..0.36) * looseness;
            let (sin, cos) = yaw.sin_cos();
            // Allow room for a chair on the aisle side. Final admission uses
            // the actual collection dimensions, seat rings and circulation.
            let margin =
                ((cos.abs() * width + sin.abs() * 0.85) * 0.5 + 0.36).min(lane_width * 0.5);
            let across = line + (bank.random::<f32>() - line) * looseness;
            let x = left + margin + (lane_width - 2. * margin) * across;
            let z = (first + row as f32 * bay_pitch + bank.random_range(-0.34..0.34) * looseness)
                .clamp(start, end);
            sites.push(TableSite {
                position: Vec3::new(x, 0., z),
                width,
                yaw,
            });
        }
    }
}

fn perimeter_bench(bays: [Rect; 2], venue: &mut ChaCha8Rng) -> Option<FloorProposal> {
    let first = usize::from(venue.random_bool(0.5));
    for side in [first, 1 - first] {
        let bay = bays[side];
        let mut size = geometry::dimensions(ObjectKind::Banquette, venue.random());
        size.x = size.x.min(bay.height() - 0.45);
        if size.x < 1.05 || bay.width() < size.z + 0.4 {
            continue;
        }
        let inset = size.z * 0.5 + venue.random_range(0.04..0.20);
        let x = if side == 0 {
            bay.min.x + inset
        } else {
            bay.max.x - inset
        };
        let z = bay.min.y + size.x * 0.5 + (bay.height() - size.x) * venue.random_range(0.10..0.90);
        return Some(FloorProposal {
            kind: ObjectKind::Banquette,
            position: Vec3::new(x, 0., z),
            size,
            yaw: if side == 0 { FRAC_PI_2 } else { -FRAC_PI_2 },
        });
    }
    None
}
