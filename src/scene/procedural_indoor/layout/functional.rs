//! Capacity-aware spatial programs. These are not aliases for the office activity
//! mixture. Geometry, reserved routes, support levels and receipts share one plan.
mod architecture;
mod furnishing;
mod services;
mod validation;
pub use services::MountedUnit;
#[cfg(test)]
mod tests;
use super::super::{envelope::FloorPatch, program::Zone};
use super::*;
pub use architecture::{prepare_architecture, size_room, support_regions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpaceRole {
    Service,
    Dining,
    Teaching,
    Audience,
    Transit,
    Waiting,
    Exhibition,
    Demonstration,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationRole {
    Circulation,
    Queue,
    StaffAccess,
    Viewing,
    DoorApproach,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reservation {
    pub role: ReservationRole,
    pub min: Vec2,
    pub max: Vec2,
}
impl Reservation {
    pub fn overlaps(&self, lo: Vec3, hi: Vec3) -> bool {
        lo.x < self.max.x - 0.001
            && hi.x > self.min.x + 0.001
            && lo.z < self.max.y - 0.001
            && hi.z > self.min.y + 0.001
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceKind {
    Power,
    Water,
    Drain,
    Ventilation,
    Display,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceConnection {
    pub kind: ServiceKind,
    pub source: usize,
    pub target: usize,
    pub source_point: Vec3,
    pub target_point: Vec3,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramParameters {
    pub aisle_metres: f32,
    pub row_pitch_metres: f32,
    pub seat_pitch_metres: f32,
    pub row_curvature_metres: f32,
    pub group_rotation_radians: f32,
    pub tier_rise_metres: f32,
    pub collaborative_fraction: f32,
    pub display_spacing_metres: f32,
    pub service_length_metres: f32,
    pub occupancy: f32,
}
impl ProgramParameters {
    pub fn sample(seed: u64, zone: usize, density: f32) -> Self {
        let mut r = stream(seed, 8230 + zone as u64);
        Self {
            aisle_metres: r.random_range(1.10..1.50),
            row_pitch_metres: r.random_range(1.32..1.65),
            seat_pitch_metres: r.random_range(0.84..1.07),
            row_curvature_metres: r.random_range(0.0..0.20),
            group_rotation_radians: r.random_range(-0.22..0.22),
            tier_rise_metres: r.random_range(0.24..0.30),
            collaborative_fraction: r.random(),
            display_spacing_metres: r.random_range(2.0..3.1),
            service_length_metres: r.random_range(1.7..2.7),
            occupancy: 0.60 + 0.40 * density,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramAssembly {
    pub zone: usize,
    pub activity: IndoorLayout,
    pub role: SpaceRole,
    pub capacity_unit: CapacityUnit,
    pub parameters: ProgramParameters,
    pub min: Vec2,
    pub max: Vec2,
    pub members: Vec<usize>,
    pub required: Vec<ObjectKind>,
    pub requested_capacity: usize,
    pub accepted_capacity: usize,
    pub rejected_candidates: usize,
    pub reservations: Vec<Reservation>,
    pub services: Vec<ServiceConnection>,
    pub focus: Vec3,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacityUnit {
    ChairStations,
    DisplayStations,
    ElevatorCars,
}
impl ProgramAssembly {
    pub fn realized_capacity(&self, objects: &[IndoorObject]) -> usize {
        self.members
            .iter()
            .filter_map(|&id| objects.get(id))
            .filter(|o| match self.capacity_unit {
                CapacityUnit::ChairStations => o.kind == ObjectKind::Chair,
                CapacityUnit::DisplayStations => matches!(
                    o.kind,
                    ObjectKind::ExhibitPlinth
                        | ObjectKind::ExhibitionPanel
                        | ObjectKind::DisplayCase
                ),
                CapacityUnit::ElevatorCars => o.kind == ObjectKind::Elevator,
            })
            .count()
    }
}
/// A named architectural support surface can overlap another one in X/Z (e.g. a mezzanine).
/// Elevation is authoritative; this is an export view of the construction model.
/// Furniture overlays retain their separate `IndoorManifest::rug_supports` contours.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportRegion {
    pub name: String,
    pub min: Vec2,
    pub max: Vec2,
    pub elevation_metres: f32,
    /// Actual support footprint. Bounds alone may include holes or other levels.
    pub triangles: Vec<[Vec2; 3]>,
    pub overhead: bool,
}
fn usable_region(scene: &IndoorManifest, zone: &Zone) -> (Vec2, Vec2) {
    let lo = zone.min + Vec2::splat(0.55);
    let hi = zone.max - Vec2::splat(0.55);
    let Some(e) = &scene.envelope else {
        return (lo, hi);
    };
    // Bounded maximum-area search. A single chamfer must not shrink all four
    // sides and erase otherwise usable teaching or dining space.
    let step = (hi - lo) * 0.075;
    let mut best = None;
    for a in 0..5 {
        for b in 0..5 {
            for c in 0..5 {
                for d in 0..5 {
                    let min = lo + step * Vec2::new(a as f32, b as f32);
                    let max = hi - step * Vec2::new(c as f32, d as f32);
                    let area = (max - min).x * (max - min).y;
                    if best.as_ref().is_some_and(|(score, _, _)| *score >= area) {
                        continue;
                    }
                    if super::super::envelope::polygon::box_inside(&e.footprint, min, max, 0.35) {
                        best = Some((area, min, max));
                    }
                }
            }
        }
    }
    best.map_or((lo, hi), |(_, a, b)| (a, b))
}

impl IndoorManifest {
    pub(crate) fn program_space_clear(&self, lo: Vec3, hi: Vec3) -> bool {
        self.assembly_diagnostics
            .programs
            .iter()
            .all(|p| p.reservations.iter().all(|r| !r.overlaps(lo, hi)))
    }
    pub fn functional_programs(&self) -> &[ProgramAssembly] {
        &self.assembly_diagnostics.programs
    }
}
