//! Seeded, renderer-independent layout grammar. All lengths are metres.
mod adjoining;
pub use adjoining::{
    AdjoiningElevation, AdjoiningFloor, AdjoiningReturn, AdjoiningSpace, AdjoiningWindows,
};
mod amenities;
pub mod assemblies;
pub(crate) mod botany;
mod clutter;
mod decor;
pub(crate) mod dining_places;
pub use decor::{electrical_parameters, ElectricalProgram, TelevisionMount};
pub(crate) mod equipment;
pub(crate) mod floor;
pub mod functional;
mod furnishing;
pub mod kitchen;
mod lounge_storage;
pub use lounge_storage::SupportPlane;
mod printers;
mod programs;
pub(crate) mod refreshments;
mod sampling;
mod seating;
pub(crate) mod surfaces;
pub(crate) mod wall_composition;
mod waste;
pub(crate) use seating::furniture_overlap;
#[cfg(test)]
mod tests;
use bevy::prelude::*;
use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub use bevy_zeroverse_capture::GENERATOR_VERSION;
pub const CAMERA_CLEARANCE: f32 = 0.28;
enum PropSurface<'a> {
    Top(Option<&'a [Vec2]>),
    Shelf(super::objects::tables::LoungeShelf),
}
/// Depth used by manifests predating metric adjoining floor plans.
pub const NEIGHBOR_DEPTH: f32 = 3.2;

/// Architecture is immutable while admitting a group of fixtures. Furniture
/// and walkway occupancy still come from the current manifest on every trial.
type PlacementStructure = std::cell::OnceCell<Vec<(Vec3, Vec3)>>;

#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect, ValueEnum,
)]
pub enum IndoorLayout {
    /// Uniformly sample every concrete room program, including public suites.
    #[default]
    #[value(alias = "all")]
    #[serde(alias = "All", alias = "all", alias = "mixed")]
    Mixed,
    /// Executive boardrooms, smaller meeting rooms and partitioned conference suites.
    Conference,
    OpenOffice,
    Lounge,
    Training,
    Coworking,
    Breakroom,
    Reception,
    /// Reading rooms with browsable two-sided stacks, perimeter shelving and study seats.
    Library,
    Workshop,
    Studio,
    Kitchenette,
    Kitchen,
    ElevatorLanding,
    Cafe,
    Cafeteria,
    Classroom,
    LectureHall,
    Showroom,
    Gallery,
    PublicSuite,
    /// Table-service dining with a waiter station, paired seating and clear service aisles.
    Restaurant,
    /// Free weights, floor exercise and independently sampled cardio equipment.
    #[value(alias = "workout-room", alias = "gym")]
    FitnessRoom,
    /// Staffed checkout, packaged groceries and open produce islands.
    GroceryStore,
    /// Apparel sales floor with hanging collections and folded stock.
    ClothingStore,
    /// Public passage with one or two banks of connected, furnished rooms.
    #[value(alias = "corridor")]
    #[serde(alias = "Corridor")]
    Hallway,
    /// A double-height circulation hall with a guarded flight and upper gallery.
    #[value(alias = "stairwell", alias = "stairway")]
    #[serde(alias = "Stairwell", alias = "Stairway")]
    StairHall,
    /// An open-air planted court surrounded by covered walks.
    Courtyard,
    /// Instructor-led indoor cycling with front-facing riders and a through aisle.
    #[value(alias = "spin-studio")]
    #[serde(alias = "SpinStudio")]
    CyclingStudio,
    /// Clear-span volleyball hall with optional spectator seating and basketball practice goals.
    #[value(alias = "sports-hall", alias = "volleyball-hall")]
    Gymnasium,
    /// Mat-based movement classes with an instructor station and accessible equipment storage.
    YogaStudio,
    /// Self-service laundry with washer and dryer banks and accessible loading aisles.
    Laundromat,
    /// Chamber ensemble with player-facing music desks and an optional accompanist.
    RehearsalRoom,
    /// Dry changing room with fitted locker banks, paired benches and privacy glazing.
    #[value(alias = "changing-room")]
    #[serde(alias = "ChangingRoom")]
    LockerRoom,
}

impl IndoorLayout {
    const INITIAL_MIXED_COUNT: usize = 20;
    /// Every concrete whole-room program eligible for `Mixed` (`all`).
    pub const PROFILES: [Self; 33] = [
        Self::Conference,
        Self::OpenOffice,
        Self::Lounge,
        Self::Training,
        Self::Coworking,
        Self::Breakroom,
        Self::Reception,
        Self::Library,
        Self::Workshop,
        Self::Studio,
        Self::Kitchenette,
        Self::Kitchen,
        Self::ElevatorLanding,
        Self::Cafe,
        Self::Cafeteria,
        Self::Classroom,
        Self::LectureHall,
        Self::Showroom,
        Self::Gallery,
        Self::PublicSuite,
        Self::Restaurant,
        Self::FitnessRoom,
        Self::GroceryStore,
        Self::ClothingStore,
        Self::Hallway,
        Self::StairHall,
        Self::Courtyard,
        Self::CyclingStudio,
        Self::Gymnasium,
        Self::YogaStudio,
        Self::Laundromat,
        Self::RehearsalRoom,
        Self::LockerRoom,
    ];
    /// Compatible secondary activities inside an office's partitioned zones.
    /// Service/teaching/exhibition programs need their whole-room reservations.
    pub const OFFICE_PROFILES: [Self; 10] = [
        Self::Conference,
        Self::OpenOffice,
        Self::Lounge,
        Self::Training,
        Self::Coworking,
        Self::Breakroom,
        Self::Reception,
        Self::Library,
        Self::Workshop,
        Self::Studio,
    ];
    /// Individual service, teaching, exhibition and other public room programs.
    pub const PUBLIC_PROFILES: [Self; 22] = [
        Self::Kitchenette,
        Self::Kitchen,
        Self::Cafe,
        Self::Cafeteria,
        Self::Classroom,
        Self::LectureHall,
        Self::ElevatorLanding,
        Self::Showroom,
        Self::Gallery,
        Self::Restaurant,
        Self::FitnessRoom,
        Self::GroceryStore,
        Self::ClothingStore,
        Self::Hallway,
        Self::StairHall,
        Self::Courtyard,
        Self::CyclingStudio,
        Self::Gymnasium,
        Self::YogaStudio,
        Self::Laundromat,
        Self::RehearsalRoom,
        Self::LockerRoom,
    ];
    fn sample_profile(seed: u64, profiles: &[Self], rng: &mut ChaCha8Rng) -> Self {
        // Keep the original draw (and downstream RNG position) for established
        // rooms. Each appended program replaces 1/n of the existing assignments,
        // which leaves every program equally probable after each extension.
        let original = profiles.len().min(Self::INITIAL_MIXED_COUNT);
        let mut selected = profiles[rng.random_range(0..original)];
        for (i, &profile) in profiles.iter().enumerate().skip(original) {
            if stream(seed, 0x5645_4e55_4500 + i as u64).random_range(0..=i) == 0 {
                selected = profile;
            }
        }
        selected
    }
    pub fn is_public(self) -> bool {
        matches!(
            self,
            Self::ElevatorLanding
                | Self::Cafe
                | Self::Cafeteria
                | Self::Classroom
                | Self::LectureHall
                | Self::Showroom
                | Self::Gallery
                | Self::PublicSuite
                | Self::Restaurant
                | Self::FitnessRoom
                | Self::GroceryStore
                | Self::ClothingStore
                | Self::Hallway
                | Self::StairHall
                | Self::Courtyard
                | Self::CyclingStudio
                | Self::Gymnasium
                | Self::YogaStudio
                | Self::Laundromat
                | Self::RehearsalRoom
                | Self::LockerRoom
        )
    }
    pub fn is_exercise(self) -> bool {
        matches!(
            self,
            Self::FitnessRoom | Self::CyclingStudio | Self::YogaStudio
        )
    }
    pub fn is_retail(self) -> bool {
        matches!(self, Self::GroceryStore | Self::ClothingStore)
    }
    pub fn is_kitchen(self) -> bool {
        matches!(self, Self::Kitchen | Self::Kitchenette)
    }
    pub fn prefers_soft_seating(self) -> bool {
        matches!(self, Self::Lounge | Self::Reception | Self::Breakroom)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureStyle {
    #[default]
    Contemporary,
    Timber,
    Industrial,
    Classic,
}

// Compatibility path retained for existing scene consumers.
pub use super::objects::ObjectKind;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndoorObject {
    /// Shared manufacturing/content identity, independent of geometric detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_seed: Option<u64>,
    pub id: usize,
    pub kind: ObjectKind,
    /// Floor or support contact, not the centre of the object.
    pub position: Vec3,
    pub size: Vec3,
    pub yaw: f32,
    pub variant: u32,
    pub seed: u64,
    /// Furniture blocks placement and the complete camera path.
    pub solid: bool,
    /// Stable instance ID of the furnishing that physically supports this item.
    pub support: Option<usize>,
    pub neighbor: bool,
    /// Chair or dining banquette -> table/work surface. Served dish -> its seat.
    /// Display/laptop or aimed desk fan -> work chair. Music desk or piano -> player chair.
    /// Recycling bin -> its general-waste partner. Changing bench -> its locker bank.
    /// Each side of a back-to-back locker island identifies the opposing bank.
    #[serde(default)]
    pub interaction_target: Option<usize>,
    /// Seeded local centreline of a routed wire. Non-cables never carry a path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cable_path: Option<Vec<Vec3>>,
    /// Cable destination instance; `interaction_target` identifies its source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cable_target: Option<usize>,
    /// Zero-based mains receptacle on a wall plate, floor box or power strip.
    /// Older routes use its first receptacle. Dedicated data and USB ports omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cable_target_socket: Option<u8>,
}

impl IndoorObject {
    /// The backless changing bench supports loose belongings on its flat seat.
    /// General work surfaces also participate in the room's clutter sampler.
    pub(crate) fn supports_item(&self, kind: ObjectKind) -> bool {
        self.kind.is_surface()
            || (self.kind == ObjectKind::ExerciseMat
                && matches!(kind, ObjectKind::YogaBlock | ObjectKind::YogaBolster))
            || (self.kind == ObjectKind::Bench
                && self.variant == super::objects::services::bench::CHANGING
                && matches!(kind, ObjectKind::Bag | ObjectKind::KitchenTowel))
    }

    /// Physical cable segments leave the empty space inside a hanging route
    /// navigable. Ordinary objects retain their allocation-free boxes; decorative
    /// strands reserve bounded cable, hanger and bulb segments.
    pub(crate) fn obstacle_bounds(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        let tf = self.transform();
        let radius = if self.kind == ObjectKind::Cable {
            super::objects::equipment::wires::bound_radius(self)
        } else {
            0.
        };
        (!matches!(
            self.kind,
            ObjectKind::Cable | ObjectKind::CeilingFan | ObjectKind::StringLights
        ))
        .then(|| self.bounds())
        .into_iter()
        .chain(
            (self.kind == ObjectKind::CeilingFan)
                .then(|| super::objects::fans::local_reservations(self))
                .into_iter()
                .flatten()
                .map(|(a, b)| (a + self.position, b + self.position)),
        )
        .chain(
            (self.kind == ObjectKind::StringLights)
                .then(|| {
                    super::objects::equipment::signals::string_program(self)
                        .local_reservations(self)
                })
                .into_iter()
                .flatten()
                .map(move |(a, b)| {
                    let c = tf.transform_point((a + b) * 0.5);
                    let half = Mat3::from_quat(tf.rotation).abs() * ((b - a) * 0.5);
                    (c - half, c + half)
                }),
        )
        .chain(super::objects::equipment::wires::MainsPlug::world(self).map(|plug| plug.bounds()))
        .chain(self.cable_path.iter().flat_map(move |path| {
            path.windows(2).map(move |pair| {
                let a = tf.transform_point(pair[0]);
                let b = tf.transform_point(pair[1]);
                (
                    a.min(b) - Vec3::splat(radius),
                    a.max(b) + Vec3::splat(radius),
                )
            })
        }))
    }

    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.position).with_rotation(Quat::from_rotation_y(self.yaw))
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let c = self.yaw.cos().abs();
        let s = self.yaw.sin().abs();
        let half = Vec3::new(
            c * self.size.x + s * self.size.z,
            self.size.y,
            s * self.size.x + c * self.size.z,
        ) * 0.5;
        let centre = self.position + Vec3::Y * self.size.y * 0.5;
        // Preserve the exact support plane: subtracting half-height after
        // adding it loses low-pile rug offsets to cancellation.
        (
            (centre - half).with_y(self.position.y),
            (centre + half).with_y(self.position.y + self.size.y),
        )
    }

    /// Fitted kitchen carcasses may approach the wall past the general furniture
    /// setback, while remaining clear of the at-most 40 mm projecting skirting.
    pub(crate) fn overhead_clearance(&self) -> f32 {
        if super::objects::public::elevator::is_core(self) {
            0.008
        } else {
            0.30
        }
    }

    pub(crate) fn wall_clearance(&self, generator: u32) -> f32 {
        if self.kind == ObjectKind::DrinkingFountain {
            0.0
        } else if self.kind == ObjectKind::BaseboardHeater {
            0.006
        } else if self.kind == ObjectKind::Locker
            && super::objects::public::education::wall_bank(self)
            && generator >= super::objects::public::education::FITTED_VERSION
        {
            0.045
        } else if super::objects::kitchen::is_island(self) {
            0.90
        } else if self.kind.is_kitchen() && self.solid {
            0.045
        } else {
            0.30
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndoorCamera {
    /// Persistent room-camera bank around local +Z. Explicit motion quaternions
    /// already include this offset. Defaults to zero for archived cameras.
    #[serde(default)]
    pub roll_radians: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ceiling_mount: Option<super::cameras::ceiling::CeilingCameraMount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fisheye: Option<bevy_zeroverse_capture::calibration::FisheyeSensor>,
    /// Optical center displacement from (0.5, 0.5), in image width/height units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal_point_offset: Option<Vec2>,
    pub start: Vec3,
    pub end: Vec3,
    pub target: Vec3,
    pub fov_degrees: f32,
    #[serde(default)]
    pub motion: Option<super::cameras::CameraMotion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingMood {
    Daylight,
    Overcast,
    Evening,
}

#[derive(Debug, Clone, Resource, Serialize, Deserialize, PartialEq)]
pub struct IndoorManifest {
    /// Per-table loose belongings policy, recorded for exact replay.
    #[serde(default)]
    pub clutter: super::items::distribution::ClutterSettings,
    /// Bounded exact support contours for furniture-bearing area rugs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rug_supports: Vec<floor::area::RugSupport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indoor_motion: Option<super::motion::IndoorMotionReport>,
    #[serde(default)]
    pub camera_diagnostics: super::cameras::diagnostics::CameraSamplingDiagnostics,
    #[serde(default)]
    pub assembly_diagnostics: assemblies::AssemblyDiagnostics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<super::appearance::AppearanceSettings>,
    pub generator_version: u32,
    #[serde(default)]
    pub program: Option<super::program::IndoorProgram>,
    pub seed: u64,
    pub layout: IndoorLayout,
    pub world_yaw: f32,
    pub room_size: Vec3,
    #[serde(default)]
    pub envelope: Option<super::envelope::EnvelopeProgram>,
    /// Connected circulation space beyond the primary reconstruction region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjoining: Option<AdjoiningSpace>,
    /// Construction of the exterior walls beyond the shared interior partition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjoining_elevation: Option<AdjoiningElevation>,
    #[serde(default)]
    pub ceiling_installation: Option<super::architecture::ceiling::CeilingInstallation>,
    pub palette: u32,
    pub furniture_style: u32,
    #[serde(default)]
    pub floor_plan: super::floorplan::FloorPlan,
    #[serde(default)]
    pub furnishing_quarter_turn: u8,
    pub floor_style: u32,
    pub ceiling_style: u32,
    #[serde(default)]
    pub architecture_style: ArchitectureStyle,
    /// Per-wall exterior geometry. Missing on legacy manifests.
    #[serde(default)]
    pub exterior: Option<super::architecture::facade::ExteriorProgram>,
    /// Legacy west-wall settings, used only when `exterior` is absent.
    pub window_bays: u32,
    pub window_sill: f32,
    pub glazing_height: f32,
    pub blinds: bool,
    pub door_x: f32,
    #[serde(default)]
    pub entry_door: Option<super::architecture::doors::DoorInstallation>,
    pub column_width: f32,
    pub lighting: LightingMood,
    pub sun_elevation: f32,
    pub sun_azimuth: f32,
    pub light_kelvin: f32,
    #[serde(default)]
    pub lighting_design: u8,
    #[serde(default = "default_target_lux")]
    pub target_lux: f32,
    #[serde(default = "default_daylight_lux")]
    pub daylight_lux: f32,
    pub density: f32,
    pub objects: Vec<IndoorObject>,
    #[serde(default)]
    pub human_density: f32,
    #[serde(default)]
    pub humans: Vec<super::humans::IndoorHuman>,
    #[serde(default)]
    pub rejected_human_placements: usize,
    #[serde(
        default = "super::cameras::CameraSettings::archived_independent",
        deserialize_with = "super::cameras::deserialize_archived_settings"
    )]
    pub camera_settings: super::cameras::CameraSettings,
    /// Width / height used by the visibility sampler, including portrait captures.
    #[serde(default = "default_camera_aspect")]
    pub camera_aspect_ratio: f32,
    pub cameras: Vec<IndoorCamera>,
    pub rejected_placements: usize,
}

/// Separate streams keep camera count and material changes from perturbing the layout.
pub fn stream(seed: u64, domain: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(domain);
    rng
}

fn default_target_lux() -> f32 {
    450.0
}
fn default_camera_aspect() -> f32 {
    1.0
}
fn default_daylight_lux() -> f32 {
    18000.0
}

impl IndoorManifest {
    pub fn generate(
        seed: u64,
        layout: IndoorLayout,
        density: f32,
        cameras: usize,
    ) -> Result<Self, String> {
        Self::generate_with_humans(seed, layout, density, cameras, 0.25)
    }

    pub fn generate_with_humans(
        seed: u64,
        layout: IndoorLayout,
        density: f32,
        cameras: usize,
        human_density: f32,
    ) -> Result<Self, String> {
        Self::generate_with_clutter(
            seed,
            layout,
            density,
            cameras,
            human_density,
            Default::default(),
        )
    }

    pub fn generate_with_clutter(
        seed: u64,
        layout: IndoorLayout,
        density: f32,
        cameras: usize,
        human_density: f32,
        clutter: super::items::distribution::ClutterSettings,
    ) -> Result<Self, String> {
        clutter.validate()?;
        Self::generate_with_profiles(
            seed,
            layout,
            density,
            cameras,
            human_density,
            &IndoorLayout::PROFILES,
            clutter,
        )
    }

    // Preserve reported office-only seed fixtures when expanding the production
    // Mixed distribution. This is not a second public sampler or viewer setting.
    #[cfg(test)]
    pub(super) fn office_regression_fixture(
        seed: u64,
        density: f32,
        cameras: usize,
        human_density: f32,
    ) -> Result<Self, String> {
        Self::generate_with_profiles(
            seed,
            IndoorLayout::Mixed,
            density,
            cameras,
            human_density,
            &IndoorLayout::OFFICE_PROFILES,
            Default::default(),
        )
    }

    /// Reproduce fixtures recorded when Mixed first included public venues.
    #[cfg(test)]
    pub(super) fn initial_mixed_fixture(
        seed: u64,
        density: f32,
        cameras: usize,
        humans: f32,
    ) -> Result<Self, String> {
        Self::generate_with_profiles(
            seed,
            IndoorLayout::Mixed,
            density,
            cameras,
            humans,
            &IndoorLayout::PROFILES[..IndoorLayout::INITIAL_MIXED_COUNT],
            Default::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_with_profiles(
        seed: u64,
        layout: IndoorLayout,
        density: f32,
        cameras: usize,
        human_density: f32,
        profiles: &[IndoorLayout],
        clutter: super::items::distribution::ClutterSettings,
    ) -> Result<Self, String> {
        if !human_density.is_finite() || !(0.0..=1.0).contains(&human_density) {
            return Err("indoor human density must be finite and between 0 and 1".into());
        }
        if !density.is_finite() || !(0.0..=1.0).contains(&density) {
            return Err("indoor density must be finite and between 0 and 1".into());
        }
        if cameras > super::cameras::MAX_INDOOR_CAMERAS {
            return Err("at most 256 indoor cameras are supported per scene".into());
        }
        let mut rng = stream(seed, 0);
        let layout = if layout == IndoorLayout::Mixed {
            IndoorLayout::sample_profile(seed, profiles, &mut rng)
        } else {
            layout
        };
        let mut room_size = super::domain::room_size(seed);
        if layout.is_kitchen() {
            let mut kitchen_rng = stream(seed, 8050);
            room_size.x = kitchen_rng.random_range(6.2..11.0);
            room_size.z = kitchen_rng.random_range(6.2..10.0);
        }
        let boardroom = super::program::boardroom::BoardroomPlan::sample(seed, layout);
        if let Some(plan) = boardroom {
            let span = plan.span();
            room_size.x = span.x;
            room_size.z = span.y;
        } else {
            room_size.y = super::envelope::EnvelopeProgram::room_height(seed, room_size);
        }
        functional::size_room(seed, layout, &mut room_size);
        let mut scene = Self {
            indoor_motion: None,
            camera_diagnostics: default(),
            assembly_diagnostics: default(),
            appearance: None,
            clutter,
            generator_version: GENERATOR_VERSION,
            program: Some(super::program::IndoorProgram::sample_for_floor(
                seed, room_size, layout, density, boardroom,
            )),
            seed,
            layout,
            world_yaw: 0.0,
            room_size,
            envelope: None,
            adjoining: Some(AdjoiningSpace::sample(seed, room_size.x)),
            adjoining_elevation: None,
            ceiling_installation: None,
            rug_supports: Vec::new(),
            palette: rng.random_range(0..6),
            furniture_style: rng.random_range(0..3),
            floor_plan: [
                super::floorplan::FloorPlan::OpenHall,
                super::floorplan::FloorPlan::CornerCore,
                super::floorplan::FloorPlan::WindowGallery,
                super::floorplan::FloorPlan::DividedSuite,
            ][stream(seed, 21).random_range(0..4)],
            furnishing_quarter_turn: stream(seed, 22).random_range(0..4),
            floor_style: rng.random_range(0..3),
            ceiling_style: rng.random_range(0..4),
            architecture_style: [
                ArchitectureStyle::Contemporary,
                ArchitectureStyle::Timber,
                ArchitectureStyle::Industrial,
                ArchitectureStyle::Classic,
            ][stream(seed, 20).random_range(0..4)],
            exterior: None,
            window_bays: (room_size.z / rng.random_range(1.15..3.5))
                .round()
                .clamp(2.0, 14.0) as u32,
            window_sill: rng.random_range(0.12..1.4),
            glazing_height: room_size.y - rng.random_range(0.22..0.85),
            blinds: rng.random_bool(0.4),
            door_x: room_size.x * 0.5 - 1.05,
            entry_door: None,
            column_width: rng.random_range(0.18..0.55),
            lighting: [
                LightingMood::Daylight,
                LightingMood::Overcast,
                LightingMood::Evening,
            ][rng.random_range(0..3)],
            sun_elevation: rng.random_range(0.08..1.3),
            sun_azimuth: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            light_kelvin: stream(seed, 23).random_range(2900.0..5400.0),
            lighting_design: stream(seed, 24).random_range(0..3),
            target_lux: stream(seed, 25).random_range(240.0..580.0),
            daylight_lux: stream(seed, 26).random_range(8000.0..32000.0),
            density,
            objects: Vec::new(),
            human_density,
            humans: Vec::new(),
            rejected_human_placements: 0,
            camera_settings: default(),
            camera_aspect_ratio: default_camera_aspect(),
            cameras: Vec::new(),
            rejected_placements: 0,
        };
        // Choose the installed floor before sampling its finish and metric scale.
        if layout == IndoorLayout::Courtyard {
            scene.floor_style = 2;
        } else if layout == IndoorLayout::Gymnasium {
            scene.floor_style = 0;
        } else if layout.is_retail() {
            // Washable hard stock in food retail. Apparel shops also use carpet.
            let roll = stream(seed, u64::from_le_bytes(*b"ShopFlrs")).random::<f32>();
            scene.floor_style = match layout {
                IndoorLayout::GroceryStore if roll < 0.18 => 0,
                IndoorLayout::ClothingStore if roll < 0.42 => 0,
                IndoorLayout::ClothingStore if roll < 0.60 => 1,
                _ => 2,
            };
        } else if layout == IndoorLayout::RehearsalRoom {
            // Finished timber, low-pile carpet and resilient institutional floors.
            let roll = stream(seed, u64::from_le_bytes(*b"MuscFlrs")).random::<f32>();
            scene.floor_style = if roll < 0.54 {
                0
            } else if roll < 0.81 {
                1
            } else {
                2
            };
        } else if layout == IndoorLayout::LockerRoom {
            scene.floor_style = 2;
            scene.window_sill =
                stream(seed, u64::from_le_bytes(*b"LockSill")).random_range(1.94..2.16);
        } else if layout == IndoorLayout::Laundromat {
            scene.floor_style = 2; // Washable tile, terrazzo, concrete or resilient stock.
        } else if layout.is_exercise() {
            let timber_share = if layout == IndoorLayout::YogaStudio {
                0.81
            } else {
                0.35
            };
            scene.floor_style =
                if stream(seed, u64::from_le_bytes(*b"GymCover")).random_bool(timber_share) {
                    0
                } else {
                    2
                };
        } else if layout.is_kitchen() || layout == IndoorLayout::Restaurant {
            scene.floor_style =
                if stream(seed, 8061).random_bool(if layout == IndoorLayout::Restaurant {
                    0.61
                } else {
                    0.38
                }) {
                    0
                } else {
                    2
                };
        }
        super::materials::program::room_floor_finish(
            &mut scene.program.as_mut().unwrap().materials,
            scene.floor_style,
            layout,
        );
        let domain = scene.domain().unwrap().clone();
        scene.target_lux = domain.target_lux;
        scene.daylight_lux = domain.photometry.sun_lux;
        scene.light_kelvin = domain.fixture_kelvin;
        scene.lighting = domain.photometry.mood();
        scene.exterior = Some(super::architecture::facade::ExteriorProgram::sample(
            seed,
            room_size,
            scene.column_width,
        ));
        if layout == IndoorLayout::LockerRoom {
            for facade in &mut scene.exterior.as_mut().unwrap().facades {
                facade.fit_clerestory(scene.window_sill, room_size.y);
            }
        }
        super::architecture::doors::sample_entry(&mut scene)?;
        scene.envelope = Some(super::envelope::EnvelopeProgram::sample(&mut scene));
        functional::prepare_architecture(&mut scene);
        scene.adjoining_elevation = Some(AdjoiningElevation::sample(&scene));
        super::architecture::doors::sample_partitions(&mut scene);
        super::architecture::doors::fit_floor_states(&mut scene);
        scene.ceiling_installation = Some(
            super::architecture::ceiling::CeilingInstallation::sample(&scene),
        );
        scene.furnish(&mut rng);
        scene.furnish_neighbor();
        let open_floor = matches!(layout, IndoorLayout::Courtyard | IndoorLayout::Gymnasium);
        let printer = if open_floor {
            scene.select_primary_room();
            None
        } else {
            scene.furnish_interior_details()
        };
        super::humans::populate(&mut scene, human_density);
        scene.settle_seating();
        if !open_floor {
            scene.finish_interior_details(printer);
        }
        scene.record_program_services();
        scene.assign_assembly_finishes();
        if scene.layout == IndoorLayout::Gymnasium {
            super::program::court::FreeThrowMark::install(&mut scene);
        }
        scene.settle_folded_table_fronts();
        if scene.layout == IndoorLayout::Studio
            && scene.generator_version >= super::objects::public::easel::VERSION
        {
            scene.settle_studio_sightlines();
        }
        // Routing depends on settled people, but may append new object instances.
        // Assign the final disjoint human ID range before camera/annotation export.
        for (index, human) in scene.humans.iter_mut().enumerate() {
            human.id = scene.objects.len() + index;
        }
        scene.meter_indoor_lighting();
        scene.sample_cameras(cameras)?;
        Ok(scene)
    }

    fn furnish_interior_details(&mut self) -> Option<IndoorObject> {
        self.assign_work_surfaces();
        if !self.layout.is_public() {
            self.furnish_amenities();
        }
        self.furnish_botany();
        self.furnish_equipment();
        self.furnish_waste_stations();
        // Install fixed desk fittings first. Computers face their own operator,
        // and later poses, loose belongings and cable routes clear the installation.
        self.furnish_desk_screens();
        self.decorate_surfaces();
        if self.layout.is_public() {
            self.dress_public_programs();
        }
        let printer = self.furnish_worktop_printer();
        self.furnish_supported_botany();
        self.scatter_clutter();
        self.finish_program_equipment();
        // Ceiling fixtures belong to the static furnishing stream. Place them
        // before occupancy-dependent seating, floor details and cable routes so
        // toggling people cannot renumber or resample an unrelated fixture.
        // Pin the reconstruction room before its fixtures and occupants use it.
        self.add_baseboard_heaters();
        self.select_primary_room();
        self.furnish_entry_coat_rack();
        self.furnish_ceiling_fans();
        self.furnish_wall_installations();
        self.furnish_area_rugs();
        printer
    }

    fn finish_interior_details(&mut self, printer: Option<IndoorObject>) {
        self.furnish_floor_details();
        if let Some(printer) = printer {
            self.furnish_print_stand(printer);
        }
        self.orient_desk_fans();
        self.connect_equipment();
        self.finish_floor_outlets();
        self.finish_counter_flatware();
        self.furnish_thermostat();
        super::objects::services::cart::stock::dress_carts(
            &mut self.objects,
            self.layout,
            self.program.as_ref(),
        );
        self.furnish_lounge_shelves();
    }

    fn candidate(
        &self,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> IndoorObject {
        let mut object = IndoorObject {
            finish_seed: None,
            id: self.objects.len(),
            kind,
            position: pos,
            size,
            yaw,
            variant: if kind == ObjectKind::Plant {
                super::plants::sample_species(size, rng)
            } else {
                rng.random_range(
                    0..if kind == ObjectKind::Chair {
                        super::objects::chairs::FAMILIES
                    } else if kind == ObjectKind::Laptop {
                        super::objects::computers::LAPTOP_FAMILIES
                    } else if kind == ObjectKind::WallArt {
                        6
                    } else {
                        3
                    },
                )
            },
            seed: rng.random(),
            solid: true,
            support: None,
            neighbor: false,
            interaction_target: None,
            cable_path: None,
            cable_target: None,
            cable_target_socket: None,
        };
        if kind == ObjectKind::Clock && self.generator_version >= 201 {
            super::objects::clocks::install(&mut object, self.seed, self.lighting);
        }
        if kind == ObjectKind::PcTower
            && self.generator_version >= super::objects::desktop::tower::INTRODUCED
        {
            object.variant |= super::objects::desktop::tower::FITTED_IO;
        }
        if kind == ObjectKind::Laptop && self.generator_version >= 205 {
            object.variant |= super::objects::computers::io::FITTED_IO;
        }
        if kind == ObjectKind::Notebook && self.generator_version >= 199 {
            object.variant |= super::objects::literature::SPREAD_BINDING;
        }
        if kind == ObjectKind::TissueBox {
            super::objects::containers::tissue::fit_proposal(&mut object);
        }
        if kind == ObjectKind::Rug {
            object.variant = super::objects::floor::rug::Cut::sample(object.seed, size) as u32;
        }
        object
    }

    /// Resolve base-floor proposals without modifying explicit upper-floor heights.
    /// Supported props and wall attachments use `candidate` with exact positions:
    /// a tabletop over a sunken floor can legitimately lie near world height zero.
    fn floor_candidate(
        &self,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> IndoorObject {
        let pos = if pos.y.abs() < 0.0001 && pos.z < self.room_size.z * 0.5 {
            pos.with_y(self.floor_height(pos.xz()))
        } else {
            pos
        };
        self.candidate(kind, pos, size, yaw, rng)
    }

    fn add(
        &mut self,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> Option<usize> {
        let mut obj = self.floor_candidate(kind, pos, size, yaw, rng);
        if matches!(kind, ObjectKind::Desk | ObjectKind::Table) {
            obj.variant = self.furniture_style;
        }
        if kind == ObjectKind::Bench {
            super::objects::services::bench::fit_height(&mut obj);
        }
        if !self.placement_clear(&obj, 0.06) {
            self.rejected_placements += 1;
            return None;
        }
        let id = obj.id;
        self.objects.push(obj);
        Some(id)
    }

    fn fixture(&mut self, kind: ObjectKind, pos: Vec3, size: Vec3, yaw: f32, rng: &mut ChaCha8Rng) {
        let mut obj = self.floor_candidate(kind, pos, size, yaw, rng);
        obj.solid = false;
        self.objects.push(obj);
    }

    pub(crate) fn installation_supported(&self, object: &IndoorObject, lo: Vec3, hi: Vec3) -> bool {
        if object.kind == ObjectKind::DrinkingFountain {
            let floor = self
                .envelope
                .as_ref()
                .map_or(0., |e| e.floor_height(object.position.xz()));
            let mount = super::objects::public::fountain::mount_height_m(object.seed);
            return (lo.y - floor - mount).abs() < 0.002 && self.wall_attachment_clear(object);
        }
        self.floor_support_clear(lo, hi)
    }

    pub fn placement_clear(&self, object: &IndoorObject, margin: f32) -> bool {
        self.placement_clear_internal(object, margin, false)
    }

    fn placement_clear_internal(
        &self,
        object: &IndoorObject,
        margin: f32,
        allow_furniture: bool,
    ) -> bool {
        self.placement_clear_with_structure(
            object,
            margin,
            allow_furniture,
            &PlacementStructure::new(),
        )
    }

    fn placement_clear_with_structure(
        &self,
        object: &IndoorObject,
        margin: f32,
        allow_furniture: bool,
        structure: &PlacementStructure,
    ) -> bool {
        use super::footprint::Footprint;
        let access = programs::access::envelope(object);
        let structural_extent = access.as_ref().unwrap_or(object);
        let footprint = Footprint::object(structural_extent);
        let (lo, hi) = structural_extent.bounds();
        let half = self.room_size * 0.5;
        let wall_clearance = object.wall_clearance(self.generator_version);
        if self.envelope.as_ref().is_some_and(|e| {
            !super::envelope::polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), wall_clearance)
                || if access.is_some() {
                    // A low-pile rug can cross a browsing aisle. The carcass needs
                    // full support and the underlying slab must span the entire aisle.
                    let (base, top) = object.bounds();
                    !self.installation_supported(object, base, top)
                        || !e.support_clear(lo.with_y(e.support_height(object.position)), hi)
                } else {
                    !self.floor_support_clear(lo, hi)
                }
        }) || hi.y > self.minimum_ceiling_height(lo.xz(), hi.xz()) - object.overhead_clearance()
        {
            return false;
        }
        if lo.x < -half.x + wall_clearance
            || hi.x > half.x - wall_clearance
            || lo.z < -half.z + wall_clearance
            || hi.z > half.z - wall_clearance
        {
            return false;
        }
        if object.kind.is_kitchen()
            && object.solid
            && super::architecture::facade::overlaps_opening(self, lo, hi, 0.03)
        {
            return false;
        }
        if self
            .program
            .as_ref()
            .is_some_and(|p| !p.portal_clear(lo, hi))
        {
            return false;
        }
        if !self
            .functional_programs()
            .iter()
            .all(|p| p.space_clear_for(self, object, lo, hi))
        {
            return false;
        }
        if self.functional_programs().iter().any(|p| {
            if !p.members.contains(&object.id) {
                return false;
            }
            let (min, max) = p.member_region(self, object);
            lo.x < min.x || hi.x > max.x || lo.z < min.y || hi.z > max.y
        }) {
            return false;
        }
        // Keep the floor-level landing usable, including the portion on the platform.
        if object.solid
            && !object.neighbor
            && lo.y < 1.0
            && self
                .envelope
                .as_ref()
                .and_then(|e| e.step_installation.as_ref())
                .and_then(|s| s.passage)
                .is_some_and(|passage| {
                    footprint.overlaps(
                        Footprint::oriented(passage.center(), passage.size(), 0.),
                        0.,
                    )
                })
        {
            return false;
        }
        // A clear 1.3 m door approach is reserved across every grammar.
        if self.entry_approach_overlaps(lo, hi) {
            return false;
        }
        if structure
            .get_or_init(|| self.columns())
            .iter()
            .any(|(a, b)| {
                lo.y < b.y - 0.001
                    && hi.y > a.y + 0.001
                    && footprint.overlaps(Footprint::bounds(*a, *b), margin)
            })
        {
            return false;
        }
        !self
            .objects
            .iter()
            // Early service mats and floor sockets have real thickness even
            // though they are not navigation obstacles. Later furnishings must
            // respect the clear floor area under which they were admitted.
            .filter(|o| {
                ((o.solid && !allow_furniture)
                    || matches!(o.kind, ObjectKind::Rug | ObjectKind::FloorOutlet))
                    && !o.neighbor
                    && o.id != object.id
            })
            .any(|other| {
                programs::access::blocks(object, other)
                    || seating::furniture_overlap(
                        object,
                        other,
                        seating::furniture_margin(object, other, margin),
                    )
            })
    }

    fn assign_work_surfaces(&mut self) {
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::Desk | ObjectKind::Table))
            .filter(|o| !super::objects::tables::is_raised_desk(o))
            .cloned()
            .collect();
        let walls = self.columns();
        let mut loose_stools = Vec::new();
        for chair in self
            .objects
            .iter_mut()
            .filter(|o| o.kind == ObjectKind::Chair)
        {
            // An accepted functional group owns this relationship. Nearest
            // surface assignment is only for independently placed seating.
            if chair.interaction_target.is_some()
                || (self.layout == IndoorLayout::RehearsalRoom && !chair.neighbor)
            {
                continue;
            }
            let forward = Quat::from_rotation_y(chair.yaw) * Vec3::NEG_Z;
            let target = surfaces
                .iter()
                .filter_map(|surface| {
                    if surface.neighbor != chair.neighbor
                        || (matches!(chair.variant, 6 | 7)
                            && !super::objects::chairs::stool::same_floor(chair, surface))
                        || (surface.position.y - chair.position.y).abs() > 0.05
                    {
                        return None;
                    }
                    let local = surface
                        .transform()
                        .compute_affine()
                        .inverse()
                        .transform_point3(chair.position);
                    let nearest = local
                        .clamp(-surface.size * 0.5, surface.size * 0.5)
                        .with_y(0.0);
                    let edge = surface
                        .transform()
                        .transform_point(nearest)
                        .with_y(chair.position.y);
                    let delta = edge - chair.position;
                    let distance = delta.length();
                    if distance > 1.25
                        || delta.normalize_or_zero().dot(forward) < 0.5
                        || (!chair.neighbor
                            && walls.iter().any(|(a, b)| {
                                segment_hits_box(chair.position + Vec3::Y, edge + Vec3::Y, *a, *b)
                            }))
                    {
                        return None;
                    }
                    Some((surface.id, distance))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id);
            if matches!(chair.variant, 6 | 7) {
                if let Some(id) = target {
                    loose_stools.push((chair.id, id));
                }
            } else {
                chair.interaction_target = target;
            }
        }
        // A loose stool can be paired only if it still fits at working height.
        // Prearranged sets already used their worktop during construction.
        for (id, target) in loose_stools {
            let mut chair = self.objects[id].clone();
            chair.interaction_target = Some(target);
            chair.size.y =
                super::objects::chairs::stool::fitted_envelope(&chair, Some(&self.objects[target]));
            if self.placement_clear(&chair, 0.0) {
                self.objects[id] = chair;
            }
        }
    }

    fn decorate_surfaces(&mut self) {
        if self.layout.is_public() {
            return;
        }
        let office = surfaces::OfficeFitout::sample(self);
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface() && !o.kind.is_kitchen())
            .cloned()
            .collect();
        for surface in surfaces {
            if surface.kind == ObjectKind::Workbench {
                self.dress_workbench(&surface);
                continue;
            }
            if surface.kind == ObjectKind::ReceptionDesk {
                self.dress_reception(&surface);
                continue;
            }
            let mut local = sampling::Domain::SurfaceDressing.rng(surface.seed);
            let rng = &mut local;
            // Refreshment carts, personal drawers and lounge side tables use
            // their own task program rather than inheriting a laptop workstation.
            if matches!(
                surface.kind,
                ObjectKind::MobileCart
                    | ObjectKind::DrawerUnit
                    | ObjectKind::SideTable
                    | ObjectKind::WallShelf
                    | ObjectKind::Bookcase
            ) {
                if surface.kind == ObjectKind::MobileCart {
                    self.appliance_group(&surface);
                }
                continue;
            }
            if surface.kind == ObjectKind::Cabinet {
                if super::items::parameters(self, &surface).dominant_task() == "refreshment" {
                    self.appliance_group(&surface);
                }
                self.place_prop(
                    &surface,
                    ObjectKind::Books,
                    Vec3::new(-0.40, 0.0, 0.0),
                    Vec3::new(0.35, 0.12, 0.22),
                    0.05,
                    rng,
                );
                // Greenery is selected and fitted by the shared botanical pass.
                continue;
            }
            if surface.kind == ObjectKind::Desk {
                let mut light_rng = stream(surface.seed, 5540);
                if light_rng.random_bool(0.26) {
                    let handed = if light_rng.random_bool(0.5) { -1. } else { 1. };
                    let chair = self.objects.iter().find(|o| {
                        o.kind == ObjectKind::Chair && o.interaction_target == Some(surface.id)
                    });
                    let (offset, size) = surfaces::desk_lamp_placement(
                        &surface,
                        chair.map(|o| o.position),
                        handed,
                        light_rng.random_range(0.29..0.54),
                    );
                    self.prop(
                        &surface,
                        ObjectKind::DeskLamp,
                        offset,
                        size,
                        light_rng.random_range(-0.12..0.12),
                        &mut light_rng,
                    );
                }
            }
            if surface.kind == ObjectKind::Table && self.layout.is_kitchen() && !surface.neighbor {
                self.dress_kitchen_dining(surface.id, rng);
                continue;
            }
            if surface.kind == ObjectKind::Table {
                if !self.is_dining_table(&surface) {
                    self.conference_props(&surface, rng);
                }
                continue;
            }
            self.decorate_workspace(&surface, office);
        }
    }

    fn prop(
        &mut self,
        support: &IndoorObject,
        kind: ObjectKind,
        offset: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) {
        if let Some(domain) = self.domain() {
            if !rng.random_bool((0.12 + 0.88 * domain.clutter) as f64) {
                return;
            }
        }
        self.supported_prop(support, kind, offset, size, yaw, rng);
    }

    /// Functional equipment has already been selected by its activity program.
    /// Apply the same physical gate without independently thinning its members
    /// through the optional loose-clutter probability.
    fn supported_prop(
        &mut self,
        support: &IndoorObject,
        kind: ObjectKind,
        offset: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) {
        let pos = support
            .transform()
            .transform_point(offset + Vec3::Y * support.size.y);
        let obj = self.candidate(kind, pos, size, support.yaw + yaw, rng);
        self.admit_supported_prop(support, obj);
    }

    fn admit_supported_prop(&mut self, support: &IndoorObject, obj: IndoorObject) -> bool {
        self.admit_supported_prop_with_outline(support, obj, None)
    }

    /// Dining service already has the outline used to locate the diner. The
    /// same immutable polygon can qualify each proposal on that tabletop.
    fn admit_supported_prop_with_outline(
        &mut self,
        support: &IndoorObject,
        mut obj: IndoorObject,
        outline: Option<&[Vec2]>,
    ) -> bool {
        let kind = obj.kind;
        let pos = obj.position;
        // Consume the candidate's small fixed draw even when the table is
        // clear, keeping later proposals on this stream reproducible.
        if !self.permits_table_item(support, kind) {
            return false;
        }
        obj.solid = false;
        obj.support = Some(support.id);
        obj.neighbor = support.neighbor;
        if matches!(kind, ObjectKind::Laptop | ObjectKind::Monitor) {
            if let Some(chair) = self
                .objects
                .iter()
                .filter(|o| {
                    o.kind == ObjectKind::Chair
                        && o.interaction_target == Some(support.id)
                        && obj.interaction_target.is_none_or(|id| o.id == id)
                })
                .min_by(|a, b| {
                    a.position
                        .distance_squared(pos)
                        .total_cmp(&b.position.distance_squared(pos))
                })
            {
                obj.yaw = surfaces::screen_facing_yaw(&obj, chair.position);
                obj.interaction_target = Some(chair.id);
            }
        }
        if !self.orient_and_check_prop(&mut obj, support, outline) {
            self.rejected_placements += 1;
            return false;
        }
        if kind == ObjectKind::Mouse
            && self.generator_version >= super::objects::desktop::pointer::INTRODUCED
        {
            super::objects::desktop::pointer::fit_operator_side(&mut obj, &self.objects);
        }
        self.objects.push(obj);
        true
    }

    /// Check actual rotated prop footprints in the supporting surface's frame.
    pub fn prop_clear(&self, object: &IndoorObject, support: &IndoorObject, margin: f32) -> bool {
        self.prop_clear_cached(object, support, margin, None)
    }

    /// A validation pass sees the same immutable tabletop once per child. Reuse
    /// its sampled convex outline for the duration of that pass only.
    pub(super) fn prop_clear_cached(
        &self,
        object: &IndoorObject,
        support: &IndoorObject,
        margin: f32,
        tops: Option<&mut std::collections::HashMap<usize, Vec<Vec2>>>,
    ) -> bool {
        if object.kind == ObjectKind::Cable {
            return object.support == Some(support.id)
                && equipment::cable_route_clear(self, object);
        }
        if self.has_lower_contact_candidate(object, support) {
            let Some(deck) = self.lower_shelf_contact(object, support) else {
                return false;
            };
            return self.prop_clear_on(object, support, margin, PropSurface::Shelf(deck));
        }
        let outline = tops
            .filter(|_| support.kind.has_table_outline())
            .map(|cache| {
                cache
                    .entry(support.id)
                    .or_insert_with(|| super::objects::tables::outline(support))
                    .as_slice()
            });
        self.prop_clear_on(object, support, margin, PropSurface::Top(outline))
    }

    fn prop_clear_on(
        &self,
        object: &IndoorObject,
        support: &IndoorObject,
        margin: f32,
        surface: PropSurface<'_>,
    ) -> bool {
        let inverse = support.transform().compute_affine().inverse();
        let local_bounds = |o: &IndoorObject| {
            let centre = inverse.transform_point3(o.position);
            let half = super::placement::rotated_half(o.size.xz(), o.yaw - support.yaw);
            (centre.xz() - half, centre.xz() + half)
        };
        let (lo, hi) = local_bounds(object);
        let half = Vec2::new(support.size.x, support.size.z) * 0.5;
        if lo.cmplt(-half + Vec2::splat(margin)).any() || hi.cmpgt(half - Vec2::splat(margin)).any()
        {
            return false;
        }
        if support.kind == ObjectKind::Workbench {
            let p = super::objects::public::workbench::Bench::sample(support);
            if (object.position.y - support.position.y - support.size.y).abs() <= 0.002 {
                if !p.supports_top(support, lo, hi, margin) {
                    return false;
                }
            } else if !p.grips(support, object) {
                return false;
            }
        }
        if support.kind == ObjectKind::ExerciseMat
            && (!support.supports_item(object.kind)
                || !super::objects::public::mats::PracticeMat::for_object(support).supports_prop(
                    support.size,
                    lo,
                    hi,
                    margin,
                ))
        {
            return false;
        }
        if support.kind == ObjectKind::ReceptionDesk
            && !super::objects::public::reception::Counter::sample(support).supports(
                support,
                object.position.y - support.position.y,
                lo,
                hi,
                margin,
            )
        {
            return false;
        }
        if support.kind == ObjectKind::KitchenCounter {
            let (top_lo, top_hi) = super::objects::kitchen::counter_top_bounds(support);
            if lo.cmplt(top_lo + Vec2::splat(margin)).any()
                || hi.cmpgt(top_hi - Vec2::splat(margin)).any()
            {
                return false;
            }
        }
        if support.kind == ObjectKind::WallShelf
            && self.generator_version >= super::objects::equipment::wall_shelf::DECK_VERSION
        {
            let deck = super::objects::equipment::wall_shelf::ShelfDeck::sample(support);
            if !deck.supports(
                support,
                (lo + hi) * 0.5,
                object.size.xz(),
                object.yaw - support.yaw,
                margin,
            ) {
                return false;
            }
        }
        if support.kind == ObjectKind::CuttingBoard
            && !super::objects::kitchen::supplies::board_supports(support, lo, hi, margin)
        {
            return false;
        }
        match surface {
            PropSurface::Shelf(deck) => {
                let (min, max) = deck.storage_bounds();
                if !deck.carries(object.kind, object.size, margin)
                    || lo.cmplt(min + Vec2::splat(margin)).any()
                    || hi.cmpgt(max - Vec2::splat(margin)).any()
                    || !self.shelf_access_clear(object, support, margin)
                {
                    return false;
                }
            }
            PropSurface::Top(outline) if support.kind.has_table_outline() => {
                use super::objects::tables;
                if !outline.map_or_else(
                    || tables::supports(support, lo, hi, margin),
                    |poly| tables::supports_outline(poly, lo, hi, margin),
                ) {
                    return false;
                }
            }
            _ => {}
        }
        // Most rejected proposals already overhang the worktop or intersect a
        // neighbour. Only the survivors need room-wide fixture and cable checks.
        if self
            .objects
            .iter()
            .filter(|o| {
                o.support == Some(support.id) && o.id != object.id && o.kind != ObjectKind::Cable
            })
            .any(|o| {
                use super::objects::services::divider::desktop;
                // Belongings on two different levels may share a plan footprint.
                if (o.position.y + o.size.y + margin <= object.position.y
                    || object.position.y + object.size.y + margin <= o.position.y)
                    && (matches!(surface, PropSurface::Shelf(_))
                        || self.has_lower_contact_candidate(o, support))
                {
                    return false;
                }
                let (a, b) = if o.kind == ObjectKind::HotFoodWell {
                    local_bounds(&refreshments::serving_envelope(o))
                } else {
                    local_bounds(o)
                };
                let overlap = lo.x < b.x + margin
                    && hi.x + margin > a.x
                    && lo.y < b.y + margin
                    && hi.y + margin > a.y;
                if desktop::is_desktop(object) {
                    desktop::obscures_display(self, object, o)
                        || (overlap && desktop::overlaps_peer(object, o, margin))
                } else if desktop::is_desktop(o) {
                    desktop::obscures_display(self, o, object)
                        || (overlap && desktop::overlaps_peer(o, object, margin))
                } else {
                    overlap
                }
            })
        {
            return false;
        }
        // Surface containment alone does not protect a tall prop from sloping
        // roofs or narrow soffits. Check its complete transformed footprint.
        let (world_lo, world_hi) = object.bounds();
        // Tables can fit below mounted artwork while taller belongings cannot.
        // Check installed attachments as well as peers on this support; use
        // obstacle bounds so suspended cable/string-light voids stay usable.
        if self.objects.iter().any(|other| {
            other.id != object.id
                && other.id != support.id
                && other.neighbor == object.neighbor
                && other.support.is_none()
                && super::objects::wall::mount_offset(other).is_some()
                && other.obstacle_bounds().into_iter().any(|(lo, hi)| {
                    world_lo.cmplt(hi + Vec3::splat(margin)).all()
                        && world_hi.cmpgt(lo - Vec3::splat(margin)).all()
                })
        }) {
            return false;
        }
        if world_hi.y > self.minimum_ceiling_height(world_lo.xz(), world_hi.xz()) - 0.08 {
            return false;
        }
        if !object.neighbor
            && self.ceiling_installation.as_ref().is_some_and(|c| {
                c.obstacles().any(|(lo, hi)| {
                    world_lo.cmplt(hi + Vec3::splat(0.025)).all()
                        && world_hi.cmpgt(lo - Vec3::splat(0.025)).all()
                })
            })
        {
            return false;
        }
        equipment::cables_clear_proposed_prop(self, object)
    }

    pub fn camera_clear(&self, p: Vec3) -> bool {
        super::cameras::clearance::Clearance::default().point_clear(self, p)
    }

    pub fn camera_path_clear(&self, start: Vec3, end: Vec3) -> bool {
        super::cameras::clearance::Clearance::default().path_clear(self, start, end)
    }

    /// Reject a foreground obstruction along the optical axis, including props.
    pub fn camera_view_clear(&self, position: Vec3, target: Vec3) -> bool {
        super::cameras::clearance::Clearance::default().view_clear(self, position, target)
    }

    /// Lens clearance is continuous; view suitability is checked over the path.
    pub fn camera_trajectory_clear(&self, start: Vec3, end: Vec3, target: Vec3) -> bool {
        let camera = IndoorCamera {
            roll_radians: 0.,
            ceiling_mount: None,
            fisheye: None,
            principal_point_offset: None,
            start,
            end,
            target,
            fov_degrees: 60.0,
            motion: None,
        };
        let clearance = super::cameras::clearance::Clearance::default();
        clearance.path_clear(self, start, end)
            && (0..=32).all(|step| {
                let pose = camera.transform_at(step as f32 / 32.0);
                pose.translation.distance(target) >= 1.5
                    && clearance.view_clear(
                        self,
                        pose.translation,
                        pose.translation + pose.rotation * Vec3::NEG_Z * 2.0,
                    )
            })
    }

    pub(crate) fn camera_obstacles(&self) -> Vec<(Vec3, Vec3)> {
        let mut obstacles = self.structural_obstacles();
        // Fixture sizes, aim and suspension vary independently. The nominal
        // circuit drop alone cannot reserve their actual bodies or wires.
        obstacles.extend(super::architecture::fixture_obstacles(self));
        obstacles
    }

    pub(crate) fn structural_obstacles(&self) -> Vec<(Vec3, Vec3)> {
        self.structural_obstacles_with_ceiling(true)
    }

    /// Visibility may replace ceiling-installation reserves with their actual
    /// triangles. Collision queries always include the full reserves.
    pub(crate) fn structural_obstacles_with_ceiling(&self, ceiling: bool) -> Vec<(Vec3, Vec3)> {
        let mut obstacles = self.column_obstacles(ceiling, true);
        if let Some(program) = &self.program {
            for p in &program.partitions {
                obstacles.extend(p.portal_obstacles(self.room_size.y));
            }
        }
        obstacles
    }

    pub(super) fn columns(&self) -> Vec<(Vec3, Vec3)> {
        self.column_obstacles(true, true)
    }

    pub(super) fn append_installed_obstacles(
        &self,
        boxes: &mut Vec<(Vec3, Vec3)>,
        ceiling: bool,
        partitions: bool,
    ) {
        boxes.extend(
            super::architecture::under_mezzanine_fixtures(self)
                .into_iter()
                .map(|p| {
                    let half = super::architecture::fixture_half_extents(self, p);
                    (p - half - Vec3::Y * 0.008, p + half + Vec3::Y * 0.075)
                }),
        );
        if partitions {
            boxes.extend(super::floorplan::obstacles(self));
        }
        if let Some(installation) = self.ceiling_installation.as_ref().filter(|_| ceiling) {
            boxes.extend(installation.obstacles());
            if let Some(grid) = &installation.room_grid {
                boxes.extend(grid.air_reservations(self.room_size.y));
            }
        }
    }

    pub(super) fn column_obstacles(&self, ceiling: bool, partitions: bool) -> Vec<(Vec3, Vec3)> {
        if let Some(envelope) = &self.envelope {
            let mut boxes = envelope.structural_boxes(self.room_size);
            self.append_installed_obstacles(&mut boxes, ceiling, partitions);
            return boxes;
        }
        let half = self.room_size * 0.5;
        let mut boxes = Vec::with_capacity(4);
        for sx in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let c = Vec3::new(
                    sx * (half.x - self.column_width * 0.5),
                    half.y,
                    sz * (half.z - self.column_width * 0.5),
                );
                let r = Vec3::new(self.column_width, self.room_size.y, self.column_width) * 0.5;
                boxes.push((c - r, c + r));
            }
        }
        if partitions {
            boxes.extend(super::floorplan::obstacles(self));
        }
        boxes
    }
}

/// Continuous slab intersection, including parallel and zero-length segments.
pub fn segment_hits_box(start: Vec3, end: Vec3, lo: Vec3, hi: Vec3) -> bool {
    let d = end - start;
    let mut tmin: f32 = 0.0;
    let mut tmax: f32 = 1.0;
    for axis in 0..3 {
        if d[axis].abs() < 1e-7 {
            if start[axis] < lo[axis] || start[axis] > hi[axis] {
                return false;
            }
        } else {
            let a = (lo[axis] - start[axis]) / d[axis];
            let b = (hi[axis] - start[axis]) / d[axis];
            tmin = tmin.max(a.min(b));
            tmax = tmax.min(a.max(b));
            if tmax < tmin {
                return false;
            }
        }
    }
    true
}
