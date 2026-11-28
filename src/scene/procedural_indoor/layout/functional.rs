//! Capacity-aware spatial programs. These are not aliases for the office activity
//! mixture. Geometry, reserved routes, support levels and receipts share one plan.
mod architecture;
mod furnishing;
mod seating;
mod services;
mod table_rows;
mod ventilation;
pub use seating::LectureSeating;
pub use table_rows::TableRowSeating;
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
    Exercise,
    Retail,
    Work,
    Changing,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationRole {
    Circulation,
    Queue,
    StaffAccess,
    Viewing,
    DoorApproach,
    EquipmentAccess,
    Browsing,
    FittingAccess,
    PlayingArea,
    SeatingAccess,
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
    Data,
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
    #[serde(default)]
    pub lecture_seating: LectureSeating,
    #[serde(default)]
    pub table_rows: TableRowSeating,
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
            lecture_seating: LectureSeating::sample(seed, zone),
            table_rows: TableRowSeating::sample(seed, zone),
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

    pub(super) fn lecture_rows(&self, start: f32, end: f32) -> Vec<[f32; 2]> {
        if !self.table_rows.lecture_tables {
            return self.lecture_seating.rows(start, end, self.row_pitch_metres);
        }
        // A writing top, a full chair envelope, and a passage behind each row.
        // The surplus varies continuously; even the shortest tier fits the group.
        let minimum = self.table_rows.lecture_depth_metres + 1.02 + self.table_rows.aisle_metres;
        self.lecture_seating.rows_with_clearance(
            start,
            end,
            minimum + self.row_pitch_metres - 1.15,
            minimum,
        )
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
    /// Individual chairs only. Retained for teaching programs and old receipts.
    ChairStations,
    DisplayStations,
    ElevatorCars,
    /// One station per chair or banquette, conservatively counting a shared
    /// bench once instead of estimating how many people fit along its length.
    SeatingStations,
    /// Each bench, mat, bike or treadmill is one independently usable station.
    ExerciseStations,
    /// One stocked fixture accessible from the sales floor.
    MerchandisingFixtures,
    PlayingCourts,
    /// One independently installed cabinet. Stacked dryers have two drum pockets.
    LaundryMachines,
    /// Installed, usable banks. Each bank can contain multiple individual compartments.
    LockerBanks,
}
/// Both stations must face into the same class even after a local placement offset.
fn class_stations_face(
    activity: IndoorLayout,
    instructor: &IndoorObject,
    participant: &IndoorObject,
) -> bool {
    let heading = |yaw: f32| Vec2::new(yaw.sin(), yaw.cos());
    let toward = (instructor.position - participant.position)
        .xz()
        .normalize_or_zero();
    let alignment = if activity == IndoorLayout::YogaStudio {
        0.22
    } else {
        0.55
    };
    toward.dot(heading(participant.yaw)) >= alignment
        && (-toward).dot(heading(instructor.yaw)) >= 0.18
}

impl ProgramAssembly {
    /// The accepted instructor bike at the focus of an instructor-led class.
    /// Its local +Z faces the participants. Capacity includes this station.
    pub fn cycling_instructor<'a>(&self, objects: &'a [IndoorObject]) -> Option<&'a IndoorObject> {
        if self.activity != IndoorLayout::CyclingStudio {
            return None;
        }
        self.class_instructor(objects)
    }

    /// Instructor station at the class focus, facing the participants along local +Z.
    pub fn class_instructor<'a>(&self, objects: &'a [IndoorObject]) -> Option<&'a IndoorObject> {
        let kind = match self.activity {
            IndoorLayout::CyclingStudio => ObjectKind::StationaryBike,
            IndoorLayout::YogaStudio => ObjectKind::ExerciseMat,
            _ => return None,
        };
        self.members
            .iter()
            .filter_map(|&id| objects.get(id))
            .find(|o| o.kind == kind && o.position.xz().distance_squared(self.focus.xz()) < 1e-6)
    }

    /// The longitudinal public aisle separates the two dining areas. A suite
    /// can also have a transverse route spanning the full width of the zone.
    pub(crate) fn dining_aisle(&self) -> Option<&Reservation> {
        if !matches!(
            self.activity,
            IndoorLayout::Cafe | IndoorLayout::Cafeteria | IndoorLayout::Restaurant
        ) {
            return None;
        }
        self.reservations.iter().find(|r| {
            r.role == ReservationRole::Circulation && r.min.x > self.min.x && r.max.x < self.max.x
        })
    }

    fn counts_toward_capacity(&self, object: &IndoorObject) -> bool {
        match self.capacity_unit {
            CapacityUnit::LockerBanks => object.kind == ObjectKind::Locker,
            CapacityUnit::LaundryMachines => matches!(
                object.kind,
                ObjectKind::WashingMachine | ObjectKind::TumbleDryer
            ),
            CapacityUnit::PlayingCourts => object.kind == ObjectKind::VolleyballNet,
            CapacityUnit::ChairStations => object.kind == ObjectKind::Chair,
            CapacityUnit::SeatingStations => {
                matches!(
                    object.kind,
                    ObjectKind::Chair | ObjectKind::Banquette | ObjectKind::BeamSeating
                ) || (matches!(
                    self.activity,
                    IndoorLayout::StairHall | IndoorLayout::Courtyard
                ) && object.kind == ObjectKind::Bench)
            }
            CapacityUnit::DisplayStations => matches!(
                object.kind,
                ObjectKind::ExhibitPlinth | ObjectKind::ExhibitionPanel | ObjectKind::DisplayCase
            ),
            CapacityUnit::MerchandisingFixtures => matches!(
                object.kind,
                ObjectKind::GroceryDisplay | ObjectKind::ApparelDisplay
            ),
            CapacityUnit::ElevatorCars => object.kind == ObjectKind::Elevator,
            CapacityUnit::ExerciseStations => matches!(
                object.kind,
                ObjectKind::WeightBench
                    | ObjectKind::PowerRack
                    | ObjectKind::ExerciseMat
                    | ObjectKind::StationaryBike
                    | ObjectKind::Treadmill
            ),
        }
    }

    /// Fitted wall equipment can occupy the perimeter of its architectural zone.
    /// The interior rectangle for loose furnishings may be inset from that wall.
    pub(crate) fn member_region(
        &self,
        scene: &IndoorManifest,
        object: &IndoorObject,
    ) -> (Vec2, Vec2) {
        use super::super::objects::public::education;
        if object.kind == ObjectKind::DrinkingFountain
            || (object.kind == ObjectKind::Locker
                && education::wall_bank(object)
                && scene.generator_version >= education::FITTED_VERSION)
        {
            if let Some(z) = scene.program.as_ref().and_then(|p| p.zones.get(self.zone)) {
                return (z.min, z.max);
            }
        }
        (self.min, self.max)
    }

    pub fn realized_capacity(&self, objects: &[IndoorObject]) -> usize {
        self.members
            .iter()
            .filter_map(|&id| objects.get(id))
            .filter(|o| self.counts_toward_capacity(o))
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
            let min = lo + step * Vec2::new(a as f32, b as f32);
            for c in 0..5 {
                for d in 0..5 {
                    let max = hi - step * Vec2::new(c as f32, d as f32);
                    let area = (max - min).x * (max - min).y;
                    // Increasing d only shortens this rectangle at fixed width.
                    if best.as_ref().is_some_and(|(score, _, _)| *score >= area) {
                        break;
                    }
                    if super::super::envelope::polygon::box_inside(&e.footprint, min, max, 0.35) {
                        if min == lo && max == hi {
                            return (min, max);
                        }
                        best = Some((area, min, max));
                        break;
                    }
                }
            }
        }
    }
    best.map_or((lo, hi), |(_, a, b)| (a, b))
}

impl ProgramAssembly {
    /// The installed net occupies the centreline of its own reserved court.
    /// Every other object, and every other kind of reservation, uses ordinary clearance.
    pub(super) fn space_clear_for(
        &self,
        scene: &IndoorManifest,
        o: &IndoorObject,
        lo: Vec3,
        hi: Vec3,
    ) -> bool {
        self.reservations.iter().all(|r| {
            !r.overlaps(lo, hi)
                || (r.role == ReservationRole::PlayingArea
                    && self.activity == IndoorLayout::Gymnasium
                    && self.members.contains(&o.id)
                    && scene
                        .program
                        .as_ref()
                        .and_then(|p| p.volleyball)
                        .is_some_and(|c| furnishing::gymnasium::matches_net(o, c)))
        })
    }
}

impl IndoorManifest {
    pub fn functional_programs(&self) -> &[ProgramAssembly] {
        &self.assembly_diagnostics.programs
    }
}
