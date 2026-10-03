//! Seeded, renderer-independent layout grammar. All lengths are metres.
mod decor;
mod furnishing;
mod programs;
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
    #[default]
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
}

impl IndoorLayout {
    pub const PROFILES: [Self; 10] = [
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectKind {
    Table,
    Desk,
    Chair,
    Sofa,
    CoffeeTable,
    Cabinet,
    Bookcase,
    Plant,
    TrashCan,
    Whiteboard,
    Display,
    Laptop,
    Monitor,
    Notebook,
    Mug,
    Books,
    Rug,
    WallArt,
    Clock,
    FloorLamp,
    Keyboard,
    Mouse,
    WaterBottle,
    PenHolder,
    Printer,
    StorageBox,
    CoatRack,
    Bag,
    CoffeeCup,
    SodaCan,
    Notepad,
    Pencil,
    Microphone,
    Phone,
    WallOutlet,
    LightSwitch,
}

impl ObjectKind {
    pub fn class_name(self) -> &'static str {
        match self {
            Self::Table | Self::CoffeeTable => "table",
            Self::Desk => "desk",
            Self::Chair => "chair",
            Self::Sofa => "sofa",
            Self::Cabinet => "cabinet",
            Self::Bookcase => "bookshelf",
            Self::Whiteboard => "whiteboard",
            Self::Display | Self::Monitor => "television",
            Self::Notebook | Self::Notepad => "paper",
            Self::Books => "books",
            Self::Rug => "floormat",
            Self::WallArt => "picture",
            Self::FloorLamp => "lamp",
            _ => "other_prop",
        }
    }

    pub fn is_surface(self) -> bool {
        matches!(
            self,
            Self::Table | Self::Desk | Self::CoffeeTable | Self::Cabinet
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndoorObject {
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
}

impl IndoorObject {
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
        (centre - half, centre + half)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndoorCamera {
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
        default = "super::cameras::CameraSettings::independent",
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
            IndoorLayout::PROFILES[rng.random_range(0..IndoorLayout::PROFILES.len())]
        } else {
            layout
        };
        let mut room_size = super::domain::room_size(seed);
        room_size.y = super::envelope::EnvelopeProgram::room_height(seed, room_size);
        let mut scene = Self {
            appearance: None,
            generator_version: GENERATOR_VERSION,
            program: Some(super::program::IndoorProgram::sample(
                seed, room_size, layout, density,
            )),
            seed,
            layout,
            world_yaw: 0.0,
            room_size,
            envelope: None,
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
        scene.envelope = Some(super::envelope::EnvelopeProgram::sample(&mut scene));
        scene.furnish(&mut rng);
        scene.assign_work_surfaces();
        scene.decorate(&mut rng);
        scene.scatter_clutter(&mut rng);
        super::humans::populate(&mut scene, human_density);
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
            id: self.objects.len(),
            kind,
            position: if pos.y.abs() < 0.0001 && pos.z < self.room_size.z * 0.5 {
                pos.with_y(self.floor_height(pos.xz()))
            } else {
                pos
            },
            size,
            yaw,
            variant: rng.random_range(
                0..if kind == ObjectKind::Chair {
                    super::objects::chairs::FAMILIES
                } else if kind == ObjectKind::Laptop {
                    super::objects::computers::LAPTOP_FAMILIES
                } else if kind == ObjectKind::Plant {
                    super::plants::SPECIES
                } else {
                    3
                },
            ),
            seed: rng.random(),
            solid: true,
            support: None,
            neighbor: false,
            interaction_target: None,
        }
    }

    fn add(
        &mut self,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> Option<usize> {
        let mut obj = self.candidate(kind, pos, size, yaw, rng);
        if matches!(kind, ObjectKind::Desk | ObjectKind::Table) {
            obj.variant = self.furniture_style;
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
        let mut obj = self.candidate(kind, pos, size, yaw, rng);
        obj.solid = false;
        self.objects.push(obj);
    }

    pub fn placement_clear(&self, object: &IndoorObject, margin: f32) -> bool {
        use super::footprint::Footprint;
        let footprint = Footprint::object(object);
        let (lo, hi) = object.bounds();
        let half = self.room_size * 0.5;
        if self.envelope.as_ref().is_some_and(|e| {
            !super::envelope::polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.30)
                || !e.support_clear(lo, hi)
                || [
                    lo.xz(),
                    hi.xz(),
                    Vec2::new(lo.x, hi.z),
                    Vec2::new(hi.x, lo.z),
                ]
                .into_iter()
                .any(|p| hi.y > e.ceiling_height(self.room_size, p) - 0.30)
        }) {
            return false;
        }
        if lo.x < -half.x + 0.30
            || hi.x > half.x - 0.30
            || lo.z < -half.z + 0.30
            || hi.z > half.z - 0.30
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
        // A clear 1.3 m door approach is reserved across every grammar.
        if hi.x > self.door_x - 0.70 && lo.x < self.door_x + 0.70 && hi.z > half.z - 1.45 {
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
            .filter(|o| o.solid && !o.neighbor && o.id != object.id)
            .any(|other| {
                let (a, b) = other.bounds();
                lo.y < b.y && hi.y > a.y && footprint.overlaps(Footprint::object(other), margin)
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
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface())
            .cloned()
            .collect();
        for surface in surfaces {
            if surface.kind == ObjectKind::Cabinet {
                self.prop(
                    &surface,
                    ObjectKind::Books,
                    Vec3::new(-0.40, 0.0, 0.0),
                    Vec3::new(0.35, 0.12, 0.22),
                    0.05,
                    rng,
                );
                if rng.random_bool(0.72) {
                    self.prop(
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
            if surface.kind == ObjectKind::Table {
                self.conference_props(&surface, rng);
                continue;
            }
            let count = 1;
            for i in 0..count {
                let z = (i as f32 - (count - 1) as f32 * 0.5) * 1.04;
                if rng.random_bool((0.35 + self.density * 0.6) as f64) {
                    let kind = if self.layout == IndoorLayout::OpenOffice
                        && surface.kind == ObjectKind::Desk
                        && rng.random_bool(0.55)
                    {
                        ObjectKind::Monitor
                    } else {
                        ObjectKind::Laptop
                    };
                    self.prop(
                        &surface,
                        kind,
                        Vec3::new(0.0, 0.0, z - 0.08),
                        if kind == ObjectKind::Monitor {
                            Vec3::new(
                                rng.random_range(0.38..0.66),
                                rng.random_range(0.30..0.48),
                                0.30,
                            )
                        } else {
                            Vec3::new(
                                rng.random_range(0.27..0.43),
                                rng.random_range(0.23..0.32),
                                rng.random_range(0.30..0.39),
                            )
                        },
                        rng.random_range(-0.12..0.12),
                        rng,
                    );
                }
                if self
                    .objects
                    .iter()
                    .any(|o| o.support == Some(surface.id) && o.kind == ObjectKind::Monitor)
                {
                    self.prop(
                        &surface,
                        ObjectKind::Keyboard,
                        Vec3::new(0.0, 0.0, 0.23),
                        Vec3::new(0.33, 0.022, 0.13),
                        rng.random_range(-0.035..0.035),
                        rng,
                    );
                    self.prop(
                        &surface,
                        ObjectKind::Mouse,
                        Vec3::new(0.26, 0.0, 0.23),
                        Vec3::new(0.065, 0.035, 0.105),
                        rng.random_range(-0.12..0.12),
                        rng,
                    );
                }
                if rng.random_bool((0.20 + self.density * 0.55) as f64) {
                    let kind = if rng.random_bool(0.55) {
                        ObjectKind::WaterBottle
                    } else {
                        ObjectKind::PenHolder
                    };
                    self.prop(
                        &surface,
                        kind,
                        Vec3::new(surface.size.x * 0.29, 0.0, -surface.size.z * 0.29),
                        Vec3::new(
                            0.08,
                            if kind == ObjectKind::WaterBottle {
                                0.24
                            } else {
                                0.17
                            },
                            0.08,
                        ),
                        0.0,
                        rng,
                    );
                }
                self.prop(
                    &surface,
                    ObjectKind::Notebook,
                    Vec3::new(-surface.size.x * 0.27, 0.0, z + 0.03),
                    Vec3::new(0.20, 0.016, 0.27),
                    rng.random_range(-0.18..0.18),
                    rng,
                );
                if rng.random_bool((0.4 + self.density * 0.55) as f64) {
                    let (kind, size) = decor::drink(rng);
                    self.prop(
                        &surface,
                        kind,
                        Vec3::new(surface.size.x * 0.30, 0.0, z),
                        size,
                        rng.random_range(-1.2..1.2),
                        rng,
                    );
                }
            }
        }
    }

    fn conference_props(&mut self, surface: &IndoorObject, rng: &mut ChaCha8Rng) {
        use std::f32::consts::{FRAC_PI_2, PI};
        let seats: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(surface.id))
            .map(|o| {
                surface
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(o.position)
            })
            .collect();
        if rng.random_bool((0.2 + self.density * 0.65) as f64) {
            for x in [-0.18, 0.18] {
                self.prop(
                    surface,
                    ObjectKind::WaterBottle,
                    Vec3::new(x, 0.0, 0.0),
                    Vec3::new(0.08, 0.24, 0.08),
                    0.0,
                    rng,
                );
            }
        }
        for seat in seats {
            // Screens face the person using them, including the two end seats.
            let (centre, yaw) = if seat.x.abs() > surface.size.x * 0.5 {
                (
                    Vec3::new(seat.x.signum() * (surface.size.x * 0.5 - 0.23), 0.0, seat.z),
                    seat.x.signum() * FRAC_PI_2,
                )
            } else {
                (
                    Vec3::new(0.0, 0.0, seat.z.signum() * (surface.size.z * 0.5 - 0.23)),
                    if seat.z > 0.0 { 0.0 } else { PI },
                )
            };
            let orientation = Quat::from_rotation_y(yaw);
            if rng.random_bool((0.22 + self.density * 0.58) as f64) {
                self.prop(
                    surface,
                    ObjectKind::Laptop,
                    centre,
                    Vec3::new(
                        rng.random_range(0.27..0.43),
                        rng.random_range(0.23..0.32),
                        rng.random_range(0.30..0.39),
                    ),
                    yaw,
                    rng,
                );
            }
            self.prop(
                surface,
                ObjectKind::Notebook,
                centre + orientation * Vec3::new(-0.31, 0.0, 0.025),
                Vec3::new(0.20, 0.016, 0.27),
                yaw + rng.random_range(-0.08..0.08),
                rng,
            );
            if rng.random_bool((0.3 + self.density * 0.5) as f64) {
                let (kind, size) = decor::drink(rng);
                self.prop(
                    surface,
                    kind,
                    centre + orientation * Vec3::new(0.30, 0.0, 0.04),
                    size,
                    yaw + rng.random_range(-1.1..1.1),
                    rng,
                );
            }
        }
        if rng.random_bool(0.6) {
            self.prop(
                surface,
                ObjectKind::Microphone,
                Vec3::new(0., 0., surface.size.z * 0.12),
                Vec3::new(0.13, rng.random_range(0.17..0.30), 0.16),
                rng.random_range(-0.4..0.4),
                rng,
            );
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
        let pos = support
            .transform()
            .transform_point(offset + Vec3::Y * support.size.y);
        let mut obj = self.candidate(kind, pos, size, support.yaw + yaw, rng);
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
                let delta = (chair.position - pos).with_y(0.0);
                obj.yaw = delta.x.atan2(delta.z);
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
        if matches!(
            support.kind,
            ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable
        ) && !super::objects::tables::supports(support, lo, hi, margin)
        {
            return false;
        }
        !self
            .objects
            .iter()
            .filter(|o| o.support == Some(support.id) && o.id != object.id)
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
            .map(IndoorObject::bounds)
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
                .map(IndoorObject::bounds)
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
            .map(IndoorObject::bounds)
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
        let mut obstacles = self.columns();
        if let Some(program) = &self.program {
            for p in &program.partitions {
                obstacles.extend(p.portal_obstacles(self.room_size.y));
            }
        }
        obstacles
    }

    pub(super) fn columns(&self) -> Vec<(Vec3, Vec3)> {
        if let Some(envelope) = &self.envelope {
            let mut boxes = envelope.structural_boxes(self.room_size);
            boxes.extend(
                super::architecture::under_mezzanine_fixtures(self)
                    .into_iter()
                    .map(|p| {
                        let half = super::architecture::fixture_size(self) * 0.5;
                        (p - half - Vec3::Y * 0.008, p + half + Vec3::Y * 0.075)
                    }),
            );
            boxes.extend(super::floorplan::obstacles(self));
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
