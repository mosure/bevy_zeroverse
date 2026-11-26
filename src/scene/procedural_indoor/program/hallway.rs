//! A circulation floor has a continuous public passage and enclosed rooms along
//! its sides. The ordinary partitions supply openings, glazing, doors and labels.
use super::*;
use rand_chacha::ChaCha8Rng;

pub(crate) struct Section {
    /// Normal to the corridor frontages (X or Z).
    across: usize,
    passage: f32,
    wings: [f32; 2],
    length: f32,
    height: f32,
}

impl Section {
    pub(crate) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"HallPlan"));
        let double = r.random_bool(0.49);
        // A transverse single-loaded hall meets the shared +Z wall directly.
        // Longitudinal halls meet that same entrance at their end.
        let across = usize::from(!double && r.random_bool(0.53));
        let mut wings = [r.random_range(4.1..6.2), r.random_range(4.1..6.2)];
        if !double {
            let empty = if across == 1 { 1 } else { r.random_range(0..2) };
            wings[empty] = 0.;
        }
        Self {
            across,
            passage: r.random_range(3.15..5.25),
            wings,
            length: r.random_range(12.5..19.6),
            height: r.random_range(3.35..4.45),
        }
    }

    pub(crate) fn size(&self) -> Vec3 {
        let cross = self.wings[0] + self.passage + self.wings[1];
        let mut floor = Vec2::splat(self.length);
        floor[self.across] = cross;
        Vec3::new(floor.x, self.height, floor.y)
    }

    pub(super) fn subdivide(&self, seed: u64, size: Vec2, p: &mut IndoorProgram) {
        let mut r = stream(seed, u64::from_le_bytes(*b"HallBays"));
        let a = self.across;
        let along = 1 - a;
        let half = size * 0.5;
        let scale = size[a] / self.size().xz()[a];
        // Callers may supply their own building dimensions. Leave an infeasible
        // plan for topology validation instead of sampling an empty door span.
        if !size.is_finite()
            || size[along] < 8.2
            || self.passage * scale < 3.
            || self.wings.iter().any(|&w| w > 0. && w * scale < 3.15)
        {
            return;
        }
        let hall = &mut p.zones[0];
        hall.min[a] = -half[a] + self.wings[0] * scale;
        hall.max[a] = half[a] - self.wings[1] * scale;
        let hall = hall.clone();
        p.primary_zone = Some(0);
        let thickness = r.random_range(0.105..0.20);
        let glass = r.random_range(0.16..0.85);
        let sill = r.random_range(0.16..1.22);
        for (side, wing) in self.wings.iter().enumerate() {
            if *wing == 0. {
                continue;
            }
            let boundary = if side == 0 { hall.min[a] } else { hall.max[a] };
            let outer = if side == 0 { -half[a] } else { half[a] };
            let bays = (size[along] / r.random_range(4.8..7.1))
                .round()
                .clamp(2., 4.) as usize;
            let mut front = -half[along];
            for i in 0..bays {
                let end = if i + 1 == bays {
                    half[along]
                } else {
                    -half[along]
                        + size[along] * (i as f32 + r.random_range(0.87..1.13)) / bays as f32
                };
                let index = p.zones.len();
                let mut zone = hall.clone();
                zone.activity = if r.random_bool(0.36) {
                    IndoorLayout::Lounge
                } else {
                    IndoorLayout::OpenOffice
                };
                zone.composition = Some(activity::ActivityMix::sample(seed, index, zone.activity));
                zone.furnishing = Some(furnishing::FurnishingField::sample(
                    seed,
                    index,
                    zone.activity,
                ));
                zone.min[a] = outer.min(boundary);
                zone.max[a] = outer.max(boundary);
                zone.min[along] = front;
                zone.max[along] = end;
                zone.orientation = r.random_range(0.0..std::f32::consts::TAU);
                p.zones.push(zone);
                let glazing = if r.random_bool(glass) {
                    r.random_range(0.35..0.94)
                } else {
                    0.
                };
                p.partitions.push(partition(
                    &mut r,
                    a,
                    boundary,
                    (front, end),
                    thickness,
                    sill,
                    glazing,
                ));
                if i > 0 {
                    // A suite divider butts into the room-side face of the
                    // continuous frontage. Its communicating door joins peers.
                    let face = boundary + (outer - boundary).signum() * (thickness * 0.5 + 0.002);
                    p.partitions.push(partition(
                        &mut r,
                        along,
                        front,
                        (outer.min(face), outer.max(face)),
                        thickness,
                        sill,
                        0.,
                    ));
                }
                front = end;
            }
        }
    }
}

fn partition(
    r: &mut ChaCha8Rng,
    axis: usize,
    coordinate: f32,
    span: (f32, f32),
    thickness: f32,
    sill: f32,
    glazing_fraction: f32,
) -> Partition {
    let (start, end) = span;
    let width = r.random_range(0.93..1.48);
    let margin = width * 0.5 + 0.51;
    Partition {
        axis,
        coordinate,
        start,
        end,
        door_center: r.random_range(start + margin..end - margin),
        door_width: width,
        door_height: r.random_range(2.05..2.34),
        arch_rise: 0.,
        door: None,
        thickness,
        sill,
        glazing_fraction,
        mullion_pitch: r.random_range(0.79..1.94),
        transom_fraction: r.random_bool(0.36).then(|| r.random_range(0.38..0.79)),
    }
}

pub(crate) fn across(hall: &Zone) -> usize {
    usize::from(hall.max.x - hall.min.x > hall.max.y - hall.min.y)
}

/// Signed coordinate-axis normal pointing from this frontage into the hall.
pub(crate) fn front_normal(hall: &Zone, wall: &Partition) -> Option<f32> {
    let a = across(hall);
    if wall.axis != a {
        return None;
    }
    if (wall.coordinate - hall.min[a]).abs() < 0.001 {
        Some(1.)
    } else if (wall.coordinate - hall.max[a]).abs() < 0.001 {
        Some(-1.)
    } else {
        None
    }
}

/// Corner chamfers belong to the side rooms. Keep the entrance wall continuous
/// and protect the full clear width of the corridor to its far end.
pub(crate) fn outline(scene: &IndoorManifest, size: Vec3, r: &mut impl Rng) -> Vec<Vec2> {
    let hall = &scene.program.as_ref().unwrap().zones[0];
    let h = size.xz() * 0.5;
    let mut points = Vec::with_capacity(6);
    for x in [-h.x, h.x] {
        let corner = Vec2::new(x, -h.y);
        let dist = (corner - corner.clamp(hall.min, hall.max)).length();
        if dist > 2. && r.random_bool(0.64) {
            let cut = r.random_range(0.48..1.53);
            let bottom = Vec2::new(x - x.signum() * cut, -h.y);
            let side = Vec2::new(x, -h.y + cut * r.random_range(0.70..1.25));
            points.extend(if x < 0. {
                [side, bottom]
            } else {
                [bottom, side]
            });
        } else {
            points.push(corner);
        }
    }
    points.extend([h, Vec2::new(-h.x, h.y)]);
    points
}

/// This floor is a graph: side rooms open onto the hall and adjacent suites
/// also communicate. Check each real boundary instead of assuming a BSP tree.
pub(super) fn connected(p: &IndoorProgram) -> bool {
    let Some((hall, rooms)) = p.zones.split_first() else {
        return false;
    };
    if p.primary_zone != Some(0) || hall.activity != IndoorLayout::Hallway || rooms.len() < 2 {
        return false;
    }
    let a = across(hall);
    let along = 1 - a;
    let mut assigned = vec![false; p.partitions.len()];
    for (i, room) in rooms.iter().enumerate() {
        let mut frontage = 0;
        for (j, w) in p.partitions.iter().enumerate() {
            if w.axis > 1 {
                return false;
            }
            if let Some(n) = front_normal(hall, w) {
                let face = if n > 0. { room.max[a] } else { room.min[a] };
                if (face - w.coordinate).abs() < 0.001
                    && w.door_center - w.door_width * 0.5 >= room.min[along]
                    && w.door_center + w.door_width * 0.5 <= room.max[along]
                {
                    if assigned[j] {
                        return false;
                    }
                    assigned[j] = true;
                    frontage += 1;
                }
            }
        }
        if frontage != 1 {
            return false;
        }
        for peer in &rooms[..i] {
            if room.min[a] != peer.min[a] || room.max[a] != peer.max[a] {
                continue;
            }
            let boundary = if room.min[along] == peer.max[along] {
                room.min[along]
            } else if room.max[along] == peer.min[along] {
                room.max[along]
            } else {
                continue;
            };
            let mut count = 0;
            for (j, w) in p.partitions.iter().enumerate() {
                if w.axis == along
                    && (w.coordinate - boundary).abs() < 0.001
                    && w.door_center - w.door_width * 0.5 > room.min[a]
                    && w.door_center + w.door_width * 0.5 < room.max[a]
                {
                    if assigned[j] {
                        return false;
                    }
                    assigned[j] = true;
                    count += 1;
                }
            }
            if count != 1 {
                return false;
            }
        }
    }
    assigned.into_iter().all(|v| v)
}

/// A continuous rail stock for the building, cut at jambs and glazing. The
/// outside face lies inside the wall's existing 22.5 mm navigation envelope.
pub(crate) struct WallGuard {
    height: f32,
    width: f32,
    depth: f32,
    surface: super::super::materials::Surface,
}

impl WallGuard {
    pub(crate) fn sample(seed: u64) -> Option<Self> {
        use super::super::materials::Surface;
        let mut r = stream(seed, u64::from_le_bytes(*b"HallRail"));
        r.random_bool(0.64).then(|| Self {
            height: r.random_range(0.77..1.05),
            width: r.random_range(0.095..0.20),
            depth: r.random_range(0.012..0.021),
            surface: [Surface::WoodEdge, Surface::Plastic, Surface::Metal][r.random_range(0..3)],
        })
    }

    pub(crate) fn build(
        &self,
        wall: &Partition,
        inward: f32,
        a: &mut super::super::objects::Assembly,
    ) {
        use super::super::architecture::doors::DoorMechanism;
        if (wall.glazing_fraction > 0. && self.height + self.width * 0.5 > wall.sill - 0.08)
            || wall.door.as_ref().is_some_and(|d| {
                matches!(
                    d.mechanism,
                    DoorMechanism::Sliding | DoorMechanism::Telescoping
                )
            })
        {
            return;
        }
        let normal = if wall.axis == 0 { Vec3::X } else { Vec3::Z };
        for (lo, hi) in wall.spans() {
            let len = hi - lo - 0.18;
            if len < 0.28 {
                continue;
            }
            let center = wall.position((lo + hi) * 0.5, self.height)
                + normal * inward * (wall.thickness * 0.5 + self.depth * 0.5 - 0.001);
            a.box_part(
                self.surface,
                "wall",
                center,
                wall.size(len, self.width, self.depth),
                self.depth * 0.25,
            );
        }
    }
}
