//! Seeded, renderer-independent layout grammar. All lengths are metres.
mod amenities;
pub mod assemblies;
pub(crate) mod botany;
mod clutter;
mod decor;
pub(crate) mod equipment;
pub(crate) mod floor;
pub mod functional;
mod furnishing;
pub mod kitchen;
mod programs;
pub(crate) mod refreshments;
mod seating;
pub(crate) mod surfaces;
pub(crate) mod wall_composition;
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
pub const NEIGHBOR_DEPTH: f32 = 3.2;

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
    Conference,
    OpenOffice,
    Lounge,
    Training,
    Coworking,
    Breakroom,
    Reception,
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
}

impl IndoorLayout {
    /// Every concrete whole-room program eligible for `Mixed` (`all`).
    pub const PROFILES: [Self; 20] = [
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
    /// Single functional programs available to public-suite composition.
    pub const PUBLIC_PROFILES: [Self; 9] = [
        Self::Kitchenette,
        Self::Kitchen,
        Self::Cafe,
        Self::Cafeteria,
        Self::Classroom,
        Self::LectureHall,
        Self::ElevatorLanding,
        Self::Showroom,
        Self::Gallery,
    ];
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
        )
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
    /// Stable instance id of a supporting table/desk/cabinet.
    pub support: Option<usize>,
    pub neighbor: bool,
    /// Chair -> work surface; display/laptop -> intended seated user (chair id).
    #[serde(default)]
    pub interaction_target: Option<usize>,
    /// Seeded local centreline of a routed wire. Non-cables never carry a path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cable_path: Option<Vec<Vec3>>,
    /// Cable destination instance; `interaction_target` identifies its source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cable_target: Option<usize>,
}

impl IndoorObject {
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
                        .into_iter()
                        .map(|(a, b)| {
                            let c = tf.transform_point((a + b) * 0.5);
                            let half = Mat3::from_quat(tf.rotation).abs() * ((b - a) * 0.5);
                            (c - half, c + half)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten(),
        )
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

    pub(crate) fn wall_clearance(&self) -> f32 {
        if super::objects::kitchen::is_island(self) {
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
        if cameras > 256 {
            return Err("at most 256 indoor cameras are supported per scene".into());
        }
        let mut rng = stream(seed, 0);
        let layout = if layout == IndoorLayout::Mixed {
            profiles[rng.random_range(0..profiles.len())]
        } else {
            layout
        };
        let mut room_size = super::domain::room_size(seed);
        if layout.is_kitchen() {
            let mut kitchen_rng = stream(seed, 8050);
            room_size.x = kitchen_rng.random_range(6.2..11.0);
            room_size.z = kitchen_rng.random_range(6.2..10.0);
        }
        room_size.y = super::envelope::EnvelopeProgram::room_height(seed, room_size);
        functional::size_room(seed, layout, &mut room_size);
        let mut scene = Self {
            indoor_motion: None,
            camera_diagnostics: default(),
            assembly_diagnostics: default(),
            appearance: None,
            clutter,
            generator_version: GENERATOR_VERSION,
            program: Some(super::program::IndoorProgram::sample(
                seed, room_size, layout, density,
            )),
            seed,
            layout,
            world_yaw: 0.0,
            room_size,
            envelope: None,
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
        // Wet/preparation areas use hard flooring. Textile variation belongs to
        // removable kitchen mats, not wall-to-wall office carpet.
        if layout.is_kitchen() {
            scene.floor_style = if stream(seed, 8061).random_bool(0.38) {
                0
            } else {
                2
            };
        }
        super::materials::program::floor_finish(
            &mut scene.program.as_mut().unwrap().materials,
            scene.floor_style,
        );
        let domain = scene.domain().unwrap().clone();
        scene.target_lux = domain.target_lux;
        scene.daylight_lux = domain.photometry.sun_lux;
        scene.light_kelvin = domain.fixture_kelvin;
        scene.lighting = if scene.daylight_lux > 8000.0 {
            LightingMood::Daylight
        } else if scene.daylight_lux > 100.0 {
            LightingMood::Overcast
        } else {
            LightingMood::Evening
        };
        scene.exterior = Some(super::architecture::facade::ExteriorProgram::sample(
            seed,
            room_size,
            scene.column_width,
        ));
        super::architecture::doors::sample_entry(&mut scene)?;
        scene.envelope = Some(super::envelope::EnvelopeProgram::sample(&mut scene));
        functional::prepare_architecture(&mut scene);
        super::architecture::doors::sample_partitions(&mut scene);
        super::architecture::doors::fit_floor_states(&mut scene);
        scene.ceiling_installation = Some(
            super::architecture::ceiling::CeilingInstallation::sample(&scene),
        );
        scene.furnish(&mut rng);
        scene.assign_work_surfaces();
        if !scene.layout.is_public() {
            scene.furnish_amenities();
        }
        scene.furnish_botany();
        scene.furnish_equipment();
        scene.compose_wall_groups();
        scene.decorate(&mut rng);
        if scene.layout.is_public() {
            scene.dress_public_programs();
        }
        scene.scatter_clutter();
        scene.finish_program_equipment();
        // Ceiling fixtures belong to the static furnishing stream. Place them
        // before occupancy-dependent seating, floor details and cable routes so
        // toggling people cannot renumber or resample an unrelated fixture.
        // Pin the reconstruction room before its fixtures and occupants use it.
        scene.select_primary_room();
        scene.furnish_ceiling_fans();
        scene.furnish_room_services();
        scene.furnish_area_rugs();
        super::humans::populate(&mut scene, human_density);
        scene.settle_seating();
        scene.furnish_floor_details();
        scene.connect_equipment();
        scene.finish_floor_outlets();
        scene.record_program_services();
        scene.assign_assembly_finishes();
        // Routing depends on settled people, but may append new object instances.
        // Assign the final disjoint human ID range before camera/annotation export.
        for (index, human) in scene.humans.iter_mut().enumerate() {
            human.id = scene.objects.len() + index;
        }
        scene.sample_cameras(cameras)?;
        Ok(scene)
    }

    fn candidate(
        &self,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> IndoorObject {
        IndoorObject {
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
        }
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
        if kind == ObjectKind::Bench && !super::objects::services::parameters(&obj).backed {
            obj.size.y = obj.size.y.min(0.48);
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

    pub fn placement_clear(&self, object: &IndoorObject, margin: f32) -> bool {
        self.placement_clear_internal(object, margin, false)
    }

    fn placement_clear_internal(
        &self,
        object: &IndoorObject,
        margin: f32,
        allow_furniture: bool,
    ) -> bool {
        use super::footprint::Footprint;
        let footprint = Footprint::object(object);
        let (lo, hi) = object.bounds();
        let half = self.room_size * 0.5;
        let wall_clearance = object.wall_clearance();
        if self.envelope.as_ref().is_some_and(|e| {
            !super::envelope::polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), wall_clearance)
                || !self.floor_support_clear(lo, hi)
                || [
                    lo.xz(),
                    hi.xz(),
                    Vec2::new(lo.x, hi.z),
                    Vec2::new(hi.x, lo.z),
                ]
                .into_iter()
                .any(|p| hi.y > e.ceiling_height(self.room_size, p) - object.overhead_clearance())
        }) {
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
        if !self.program_space_clear(lo, hi) {
            return false;
        }
        if self.functional_programs().iter().any(|p| {
            p.members.contains(&object.id)
                && (lo.x < p.min.x || hi.x > p.max.x || lo.z < p.min.y || hi.z > p.max.y)
        }) {
            return false;
        }
        // A clear 1.3 m door approach is reserved across every grammar.
        if self.entry_approach_overlaps(lo, hi) {
            return false;
        }
        if self.columns().iter().any(|(a, b)| {
            lo.y < b.y - 0.001
                && hi.y > a.y + 0.001
                && footprint.overlaps(Footprint::bounds(*a, *b), margin)
        }) {
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
                seating::furniture_overlap(
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
            .cloned()
            .collect();
        let walls = self.columns();
        for chair in self
            .objects
            .iter_mut()
            .filter(|o| o.kind == ObjectKind::Chair)
        {
            // An accepted functional group owns this relationship. Nearest
            // surface assignment is only for independently placed seating.
            if chair.interaction_target.is_some() {
                continue;
            }
            let forward = Quat::from_rotation_y(chair.yaw) * Vec3::NEG_Z;
            chair.interaction_target = surfaces
                .iter()
                .filter_map(|surface| {
                    if surface.neighbor != chair.neighbor
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
        }
    }

    fn decorate(&mut self, rng: &mut ChaCha8Rng) {
        self.wall_decorations(rng);
        self.wall_hardware(rng);
        if self.layout.is_public() {
            return;
        }
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface() && !o.kind.is_kitchen())
            .cloned()
            .collect();
        for surface in surfaces {
            // Refreshment carts, personal drawers and lounge side tables use
            // their own task program rather than inheriting a laptop workstation.
            if matches!(
                surface.kind,
                ObjectKind::MobileCart
                    | ObjectKind::DrawerUnit
                    | ObjectKind::SideTable
                    | ObjectKind::WallShelf
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
                if rng.random_bool(0.72) {
                    self.place_prop(
                        &surface,
                        ObjectKind::Plant,
                        Vec3::new(surface.size.x * 0.28, 0.0, 0.0),
                        Vec3::new(0.32, 0.46, 0.32),
                        rng.random_range(0.0..6.0),
                        rng,
                    );
                }
                continue;
            }
            if surface.kind == ObjectKind::Desk {
                let mut light_rng = stream(surface.seed, 5540);
                if light_rng.random_bool(0.26) {
                    let handed = if light_rng.random_bool(0.5) { -1. } else { 1. };
                    self.prop(
                        &surface,
                        ObjectKind::DeskLamp,
                        Vec3::new(handed * surface.size.x * 0.34, 0., -surface.size.z * 0.27),
                        Vec3::new(0.20, light_rng.random_range(0.31..0.48), 0.22),
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
                self.conference_props(&surface, rng);
                continue;
            }
            self.decorate_workspace(&surface);
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
        let mut obj = self.candidate(kind, pos, size, support.yaw + yaw, rng);
        // Consume the candidate's small fixed draw even when the table is
        // clear, keeping later proposals on this stream reproducible.
        if !self.permits_table_item(support, kind) {
            return;
        }
        obj.solid = false;
        obj.support = Some(support.id);
        obj.neighbor = support.neighbor;
        if matches!(kind, ObjectKind::Laptop | ObjectKind::Monitor) {
            if let Some(chair) = self
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(support.id))
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
        if !self.prop_clear(&obj, support, 0.012) {
            self.rejected_placements += 1;
            return;
        }
        self.objects.push(obj);
    }

    /// Check actual rotated prop footprints in the supporting surface's frame.
    pub fn prop_clear(&self, object: &IndoorObject, support: &IndoorObject, margin: f32) -> bool {
        if object.kind == ObjectKind::Cable {
            return object.support == Some(support.id)
                && equipment::cable_route_clear(self, object);
        }
        // Surface containment alone does not protect a tall prop from a sloped
        // ceiling. Check the complete transformed box, including its top.
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
        if [
            world_lo.xz(),
            world_hi.xz(),
            Vec2::new(world_lo.x, world_hi.z),
            Vec2::new(world_hi.x, world_lo.z),
        ]
        .into_iter()
        .any(|p| {
            world_hi.y
                > if object.neighbor {
                    self.room_size.y
                } else {
                    self.ceiling_height(p)
                } - 0.08
        }) {
            return false;
        }
        if !object.neighbor
            && self.ceiling_installation.as_ref().is_some_and(|c| {
                c.modules.iter().any(|m| {
                    let (lo, hi) = m.bounds();
                    world_lo.cmplt(hi + Vec3::splat(0.025)).all()
                        && world_hi.cmpgt(lo - Vec3::splat(0.025)).all()
                })
            })
        {
            return false;
        }
        let inverse = support.transform().compute_affine().inverse();
        let local_bounds = |o: &IndoorObject| {
            let centre = inverse.transform_point3(o.position);
            let yaw = o.yaw - support.yaw;
            let half = Vec2::new(
                yaw.cos().abs() * o.size.x + yaw.sin().abs() * o.size.z,
                yaw.sin().abs() * o.size.x + yaw.cos().abs() * o.size.z,
            ) * 0.5;
            let centre = Vec2::new(centre.x, centre.z);
            (centre - half, centre + half)
        };
        let (lo, hi) = local_bounds(object);
        let half = Vec2::new(support.size.x, support.size.z) * 0.5;
        if lo.cmplt(-half + Vec2::splat(margin)).any() || hi.cmpgt(half - Vec2::splat(margin)).any()
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
        if support.kind.has_table_outline()
            && !super::objects::tables::supports(support, lo, hi, margin)
        {
            return false;
        }
        equipment::cables_clear_proposed_prop(self, object)
            && !self
                .objects
                .iter()
                // Cables reserve their actual segments above, rather than their
                // enclosing rectangle, while ordinary props retain footprint gaps.
                .filter(|o| {
                    o.support == Some(support.id)
                        && o.id != object.id
                        && o.kind != ObjectKind::Cable
                })
                .any(|o| {
                    let (a, b) = local_bounds(o);
                    lo.x < b.x + margin
                        && hi.x + margin > a.x
                        && lo.y < b.y + margin
                        && hi.y + margin > a.y
                })
    }

    pub fn camera_clear(&self, p: Vec3) -> bool {
        let half = self.room_size * 0.5;
        if !p.is_finite()
            || (self.camera_settings.primary_room && !self.in_primary_room(p, CAMERA_CLEARANCE))
            || p.x.abs() > half.x - 0.50
            || p.z.abs() > half.z - 0.50
            || p.y < self.floor_height(p.xz()) + 0.70
            || self.envelope.as_ref().is_some_and(|e| !e.volume_clear(self.room_size,p,CAMERA_CLEARANCE))
            // Include the deepest sampled fixture and continuous lens clearance.
            || p.y > self.ceiling_height(p.xz()) - self.program.as_ref().map_or([0.10, 0.42, 0.06][self.lighting_design as usize % 3], |p| p.light_drop) - 0.04 - CAMERA_CLEARANCE
        {
            return false;
        }
        if self.camera_obstacles().iter().any(|(a, b)| {
            p.cmpge(*a - Vec3::splat(CAMERA_CLEARANCE)).all()
                && p.cmple(*b + Vec3::splat(CAMERA_CLEARANCE)).all()
        }) {
            return false;
        }
        !self
            .objects
            .iter()
            .flat_map(IndoorObject::obstacle_bounds)
            .chain(self.humans.iter().map(super::humans::IndoorHuman::bounds))
            .any(|(a, b)| {
                p.cmpge(a - Vec3::splat(CAMERA_CLEARANCE)).all()
                    && p.cmple(b + Vec3::splat(CAMERA_CLEARANCE)).all()
            })
    }

    pub fn camera_path_clear(&self, start: Vec3, end: Vec3) -> bool {
        self.camera_clear(start)
            && self.camera_clear(end)
            && self
                .envelope
                .as_ref()
                .is_none_or(|e| e.segment_clear(self.room_size, start, end, CAMERA_CLEARANCE))
            && !self.camera_obstacles().iter().any(|(a, b)| {
                segment_hits_box(
                    start,
                    end,
                    *a - Vec3::splat(CAMERA_CLEARANCE),
                    *b + Vec3::splat(CAMERA_CLEARANCE),
                )
            })
            && !self
                .objects
                .iter()
                .flat_map(IndoorObject::obstacle_bounds)
                .chain(self.humans.iter().map(super::humans::IndoorHuman::bounds))
                .any(|(a, b)| {
                    segment_hits_box(
                        start,
                        end,
                        a - Vec3::splat(CAMERA_CLEARANCE),
                        b + Vec3::splat(CAMERA_CLEARANCE),
                    )
                })
    }

    /// Reject a foreground obstruction along the optical axis, including props.
    pub fn camera_view_clear(&self, position: Vec3, target: Vec3) -> bool {
        let distance = position.distance(target);
        if !target.is_finite() || distance < 1.5 {
            return false;
        }
        let near_target = position.lerp(target, 1.0 / distance);
        !self
            .objects
            .iter()
            .flat_map(IndoorObject::obstacle_bounds)
            .chain(self.humans.iter().map(super::humans::IndoorHuman::bounds))
            .chain(self.camera_obstacles())
            .any(|(lo, hi)| {
                segment_hits_box(
                    position,
                    near_target,
                    lo - Vec3::splat(0.01),
                    hi + Vec3::splat(0.01),
                )
            })
    }

    /// Lens clearance is continuous; view suitability is checked over the path.
    pub fn camera_trajectory_clear(&self, start: Vec3, end: Vec3, target: Vec3) -> bool {
        let camera = IndoorCamera {
            ceiling_mount: None,
            fisheye: None,
            principal_point_offset: None,
            start,
            end,
            target,
            fov_degrees: 60.0,
            motion: None,
        };
        self.camera_path_clear(start, end)
            && (0..=32).all(|step| {
                let pose = camera.transform_at(step as f32 / 32.0);
                pose.translation.distance(target) >= 1.5
                    && self.camera_view_clear(
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
        let mut obstacles = self.column_obstacles(ceiling);
        if let Some(program) = &self.program {
            for p in &program.partitions {
                obstacles.extend(p.portal_obstacles(self.room_size.y));
            }
        }
        obstacles
    }

    pub(super) fn columns(&self) -> Vec<(Vec3, Vec3)> {
        self.column_obstacles(true)
    }

    fn column_obstacles(&self, ceiling: bool) -> Vec<(Vec3, Vec3)> {
        if let Some(envelope) = &self.envelope {
            let mut boxes = envelope.structural_boxes(self.room_size);
            boxes.extend(
                super::architecture::under_mezzanine_fixtures(self)
                    .into_iter()
                    .map(|p| {
                        let half = super::architecture::fixture_half_extents(self, p);
                        (p - half - Vec3::Y * 0.008, p + half + Vec3::Y * 0.075)
                    }),
            );
            boxes.extend(super::floorplan::obstacles(self));
            if let Some(installation) = self.ceiling_installation.as_ref().filter(|_| ceiling) {
                boxes.extend(installation.modules.iter().map(|m| m.bounds()));
            }
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
        boxes.extend(super::floorplan::obstacles(self));
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
