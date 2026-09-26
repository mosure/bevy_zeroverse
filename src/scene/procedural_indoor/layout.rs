//! Seeded, renderer-independent layout grammar. All lengths are metres.
use bevy::prelude::*;
use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const GENERATOR_VERSION: u32 = 4;
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
            Self::Notebook => "paper",
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
}

impl IndoorCamera {
    /// Local pose at normalized progress, matching the runtime's linear position
    /// and roll-free spherical orientation interpolation.
    pub fn transform_at(&self, progress: f32) -> Transform {
        let progress = progress.clamp(0.0, 1.0);
        let start = Transform::from_translation(self.start).looking_at(self.target, Vec3::Y);
        let end = Transform::from_translation(self.end).looking_at(self.target, Vec3::Y);
        let rotation = start.rotation.slerp(end.rotation, progress);
        let (yaw, pitch, _) = rotation.to_euler(EulerRot::YXZ);
        Transform::from_translation(self.start.lerp(self.end, progress))
            .with_rotation(Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0).normalize())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingMood {
    Daylight,
    Overcast,
    Evening,
}

#[derive(Debug, Clone, Resource, Serialize, Deserialize, PartialEq)]
pub struct IndoorManifest {
    pub generator_version: u32,
    pub seed: u64,
    pub layout: IndoorLayout,
    pub world_yaw: f32,
    pub room_size: Vec3,
    pub palette: u32,
    pub furniture_style: u32,
    pub floor_style: u32,
    pub ceiling_style: u32,
    #[serde(default)]
    pub architecture_style: ArchitectureStyle,
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
    pub density: f32,
    pub objects: Vec<IndoorObject>,
    #[serde(default)]
    pub human_density: f32,
    #[serde(default)]
    pub humans: Vec<super::humans::IndoorHuman>,
    #[serde(default)]
    pub rejected_human_placements: usize,
    pub cameras: Vec<IndoorCamera>,
    pub rejected_placements: usize,
}

/// Separate streams keep camera count and material changes from perturbing the layout.
pub fn stream(seed: u64, domain: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(domain);
    rng
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
            [
                IndoorLayout::Conference,
                IndoorLayout::OpenOffice,
                IndoorLayout::Lounge,
                IndoorLayout::Training,
            ][rng.random_range(0..4)]
        } else {
            layout
        };
        let room_size = Vec3::new(
            rng.random_range(7.0..13.5),
            rng.random_range(2.8..4.05),
            rng.random_range(7.2..12.5),
        );
        let mut scene = Self {
            generator_version: GENERATOR_VERSION,
            seed,
            layout,
            world_yaw: 0.0,
            room_size,
            palette: rng.random_range(0..6),
            furniture_style: rng.random_range(0..3),
            floor_style: rng.random_range(0..3),
            ceiling_style: rng.random_range(0..4),
            architecture_style: [
                ArchitectureStyle::Contemporary,
                ArchitectureStyle::Timber,
                ArchitectureStyle::Industrial,
                ArchitectureStyle::Classic,
            ][stream(seed, 20).random_range(0..4)],
            window_bays: rng.random_range(2..7),
            window_sill: rng.random_range(0.32..0.85),
            glazing_height: room_size.y - 0.35,
            blinds: rng.random_bool(0.4),
            door_x: room_size.x * 0.5 - 1.05,
            column_width: rng.random_range(0.22..0.42),
            lighting: [
                LightingMood::Daylight,
                LightingMood::Overcast,
                LightingMood::Evening,
            ][rng.random_range(0..3)],
            sun_elevation: rng.random_range(0.35..0.82),
            sun_azimuth: rng.random_range(-0.7..0.7),
            light_kelvin: rng.random_range(3000.0..4700.0),
            density,
            objects: Vec::new(),
            human_density,
            humans: Vec::new(),
            rejected_human_placements: 0,
            cameras: Vec::new(),
            rejected_placements: 0,
        };
        scene.furnish(&mut rng);
        scene.decorate(&mut rng);
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
            position: pos,
            size,
            yaw,
            variant: rng.random_range(
                0..if kind == ObjectKind::Plant {
                    super::plants::SPECIES
                } else {
                    3
                },
            ),
            seed: rng.random(),
            solid: true,
            support: None,
            neighbor: false,
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
        if matches!(
            kind,
            ObjectKind::Chair | ObjectKind::Desk | ObjectKind::Table
        ) {
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
        let (lo, hi) = object.bounds();
        let half = self.room_size * 0.5;
        if lo.x < -half.x + 0.30
            || hi.x > half.x - 0.30
            || lo.z < -half.z + 0.30
            || hi.z > half.z - 0.30
        {
            return false;
        }
        // A clear 1.3 m door approach is reserved across every grammar.
        if hi.x > self.door_x - 0.70 && lo.x < self.door_x + 0.70 && hi.z > half.z - 1.45 {
            return false;
        }
        if self.columns().iter().any(|(a, b)| {
            lo.x < b.x + margin && hi.x + margin > a.x && lo.z < b.z + margin && hi.z + margin > a.z
        }) {
            return false;
        }
        !self
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor && o.id != object.id)
            .any(|other| {
                let (a, b) = other.bounds();
                lo.x < b.x + margin
                    && hi.x + margin > a.x
                    && lo.z < b.z + margin
                    && hi.z + margin > a.z
            })
    }

    fn furnish(&mut self, rng: &mut ChaCha8Rng) {
        use std::f32::consts::{FRAC_PI_2, PI};
        let w = self.room_size.x;
        let d = self.room_size.z;
        match self.layout {
            IndoorLayout::Conference => {
                let length = d * rng.random_range(0.43..0.52);
                let width = rng.random_range(1.25..1.60);
                self.add(
                    ObjectKind::Table,
                    Vec3::ZERO,
                    Vec3::new(width, 0.75, length),
                    0.0,
                    rng,
                );
                let rows = ((length - 0.65) / 0.85).floor() as usize + 1;
                for side in [-1.0, 1.0] {
                    for row in 0..rows {
                        let z = (row as f32 - (rows - 1) as f32 * 0.5) * (length - 0.65)
                            / (rows - 1) as f32;
                        self.chair(
                            Vec3::new(side * (width * 0.5 + 0.57), 0.0, z),
                            side * FRAC_PI_2,
                            rng,
                        );
                    }
                }
                self.chair(Vec3::new(0.0, 0.0, length * 0.5 + 0.60), 0.0, rng);
                self.chair(Vec3::new(0.0, 0.0, -length * 0.5 - 0.60), PI, rng);
            }
            IndoorLayout::OpenOffice => {
                let rows = if d > 9.0 { 3 } else { 2 };
                for row in 0..rows {
                    for side in [-1.0, 1.0] {
                        let x = side * w * 0.235;
                        let z = -d * 0.27 + row as f32 * 2.35;
                        self.add(
                            ObjectKind::Desk,
                            Vec3::new(x, 0.0, z),
                            Vec3::new(rng.random_range(1.35..1.7), 0.74, 0.76),
                            0.0,
                            rng,
                        );
                        self.chair(
                            Vec3::new(x + rng.random_range(-0.08..0.08), 0.0, z + 0.94),
                            rng.random_range(-0.12..0.12),
                            rng,
                        );
                    }
                }
            }
            IndoorLayout::Lounge => {
                self.add(
                    ObjectKind::Sofa,
                    Vec3::new(0.0, 0.0, -1.45),
                    Vec3::new(rng.random_range(2.2..2.85), 0.88, 0.91),
                    PI,
                    rng,
                );
                self.add(
                    ObjectKind::CoffeeTable,
                    Vec3::ZERO,
                    Vec3::new(1.7, 0.40, 0.80),
                    0.0,
                    rng,
                );
                for side in [-1.0, 1.0] {
                    self.chair(Vec3::new(side * 1.55, 0.0, 0.20), side * FRAC_PI_2, rng);
                }
                self.fixture(
                    ObjectKind::Rug,
                    Vec3::new(0.0, 0.002, -0.2),
                    Vec3::new(4.3, 0.008, 3.5),
                    0.0,
                    rng,
                );
                // A separate collaboration nook provides a second functional zone.
                self.add(
                    ObjectKind::Desk,
                    Vec3::new(-w * 0.24, 0.0, d * 0.27),
                    Vec3::new(1.40, 0.74, 0.70),
                    PI,
                    rng,
                );
                self.chair(Vec3::new(-w * 0.24, 0.0, d * 0.27 - 0.95), PI, rng);
            }
            IndoorLayout::Training => {
                let rows = if d > 9.0 { 3 } else { 2 };
                for row in 0..rows {
                    for side in [-1.0, 1.0] {
                        let x = side * w * 0.215;
                        let z = -d * 0.23 + row as f32 * 2.1;
                        self.add(
                            ObjectKind::Desk,
                            Vec3::new(x, 0.0, z),
                            Vec3::new(1.8, 0.74, 0.65),
                            0.0,
                            rng,
                        );
                        for offset in [-0.44, 0.44] {
                            self.chair(Vec3::new(x + offset, 0.0, z + 0.83), 0.0, rng);
                        }
                    }
                }
            }
            IndoorLayout::Mixed => unreachable!(),
        }
        self.add(
            ObjectKind::Cabinet,
            Vec3::new(0.0, 0.0, -d * 0.5 + 0.64),
            Vec3::new(w * 0.29, 0.82, 0.55),
            0.0,
            rng,
        );
        self.add(
            ObjectKind::Bookcase,
            Vec3::new(w * 0.5 - 0.65, 0.0, -d * 0.20),
            Vec3::new(1.50, 1.85, 0.36),
            -FRAC_PI_2,
            rng,
        );
        for (x, z) in [
            (-w * 0.5 + 0.88, -d * 0.5 + 0.92),
            (-w * 0.5 + 0.95, d * 0.5 - 0.96),
            (w * 0.5 - 0.90, -d * 0.5 + 0.88),
        ] {
            if rng.random_bool((0.65 + self.density * 0.3) as f64) {
                let h = rng.random_range(1.15..1.95);
                self.add(
                    ObjectKind::Plant,
                    Vec3::new(x, 0.0, z),
                    Vec3::new(0.9, h, 0.9),
                    rng.random_range(0.0..std::f32::consts::TAU),
                    rng,
                );
            }
        }
        self.add(
            ObjectKind::TrashCan,
            Vec3::new(w * 0.5 - 0.62, 0.0, d * 0.5 - 2.0),
            Vec3::new(0.37, 0.55, 0.37),
            0.0,
            rng,
        );
        if self.layout == IndoorLayout::Lounge || rng.random_bool(0.4) {
            self.add(
                ObjectKind::FloorLamp,
                Vec3::new(-w * 0.5 + 0.70, 0.0, -d * 0.18),
                Vec3::new(0.46, 1.65, 0.46),
                0.0,
                rng,
            );
        }
        // Furnished neighboring office, visible through a genuine glass partition.
        for (kind, p, size) in [
            (
                ObjectKind::Desk,
                Vec3::new(-w * 0.22, 0.0, d * 0.5 + 2.0),
                Vec3::new(1.5, 0.74, 0.75),
            ),
            (
                ObjectKind::Chair,
                Vec3::new(-w * 0.22, 0.0, d * 0.5 + 1.08),
                Vec3::new(0.68, 1.02, 0.68),
            ),
            (
                ObjectKind::Cabinet,
                Vec3::new(w * 0.08, 0.0, d * 0.5 + 2.65),
                Vec3::new(1.6, 1.2, 0.40),
            ),
            (
                ObjectKind::Plant,
                Vec3::new(-w * 0.5 + 0.70, 0.0, d * 0.5 + 2.3),
                Vec3::new(0.8, 1.7, 0.8),
            ),
        ] {
            let mut obj = self.candidate(kind, p, size, PI, rng);
            obj.neighbor = true;
            self.objects.push(obj);
        }
    }

    fn chair(&mut self, pos: Vec3, yaw: f32, rng: &mut ChaCha8Rng) {
        let height = rng.random_range(0.92..1.12);
        self.add(
            ObjectKind::Chair,
            pos,
            Vec3::new(0.68, height, 0.68),
            yaw + rng.random_range(-0.055..0.055),
            rng,
        );
    }

    fn decorate(&mut self, rng: &mut ChaCha8Rng) {
        let w = self.room_size.x;
        let d = self.room_size.z;
        self.fixture(
            ObjectKind::Display,
            Vec3::new(-0.50, 1.27, -d * 0.5 + 0.17),
            Vec3::new(1.65, 0.96, 0.07),
            0.0,
            rng,
        );
        self.fixture(
            ObjectKind::Whiteboard,
            Vec3::new(w * 0.5 - 0.16, 1.02, 0.90),
            Vec3::new(1.65, 1.13, 0.05),
            -std::f32::consts::FRAC_PI_2,
            rng,
        );
        self.fixture(
            ObjectKind::WallArt,
            Vec3::new(w * 0.25, 1.43, -d * 0.5 + 0.15),
            Vec3::new(0.82, 0.95, 0.045),
            0.0,
            rng,
        );
        self.fixture(
            ObjectKind::Clock,
            Vec3::new(1.4, 2.30, -d * 0.5 + 0.16),
            Vec3::splat(0.31).with_z(0.045),
            0.0,
            rng,
        );
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
                    {
                        ObjectKind::Monitor
                    } else {
                        ObjectKind::Laptop
                    };
                    self.prop(
                        &surface,
                        kind,
                        Vec3::new(0.0, 0.0, z - 0.08),
                        Vec3::new(0.38, 0.28, 0.26),
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
                    self.prop(
                        &surface,
                        ObjectKind::Mug,
                        Vec3::new(surface.size.x * 0.30, 0.0, z),
                        Vec3::new(0.12, 0.105, 0.09),
                        0.0,
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
            .filter(|o| o.kind == ObjectKind::Chair && !o.neighbor)
            .map(|o| o.position - surface.position)
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
                    Vec3::new(0.38, 0.28, 0.26),
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
                self.prop(
                    surface,
                    ObjectKind::Mug,
                    centre + orientation * Vec3::new(0.30, 0.0, 0.04),
                    Vec3::new(0.12, 0.105, 0.09),
                    yaw,
                    rng,
                );
            }
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
        let pos = support
            .transform()
            .transform_point(offset + Vec3::Y * support.size.y);
        let mut obj = self.candidate(kind, pos, size, support.yaw + yaw, rng);
        obj.solid = false;
        obj.support = Some(support.id);
        obj.neighbor = support.neighbor;
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
            || p.x.abs() > half.x - 0.50
            || p.z.abs() > half.z - 0.50
            || p.y < 0.70
            // Suspended luminaire housings reach 0.284m below the ceiling.
            // Reserve their depth as well as the 0.28m lens clearance.
            || p.y > self.room_size.y - 0.57
        {
            return false;
        }
        if self.columns().iter().any(|(a, b)| {
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
            && !self.columns().iter().any(|(a, b)| {
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
            .chain(self.columns())
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

    pub(super) fn columns(&self) -> Vec<(Vec3, Vec3)> {
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
        boxes
    }

    fn sample_cameras(&mut self, count: usize) -> Result<(), String> {
        let mut rng = stream(self.seed, 3);
        let half = self.room_size * 0.5;
        for index in 0..count {
            let mut found = None;
            for attempt in 0..2048 {
                // Mix eye-level, seated, low and elevated viewpoints; stratify room edges.
                let height = match self.seed.wrapping_add(index as u64) % 5 {
                    0 | 1 => rng.random_range(1.45..1.80),
                    2 => rng.random_range(1.05..1.35),
                    3 => rng.random_range(0.78..1.02),
                    _ => rng.random_range(1.85..2.25),
                };
                let mut p = Vec3::new(
                    rng.random_range(-half.x + 0.65..half.x - 0.65),
                    height,
                    rng.random_range(-half.z + 0.65..half.z - 0.65),
                );
                if attempt < 256 {
                    match (self.seed / 5).wrapping_add(index as u64) % 4 {
                        0 => p.z = half.z - 0.85,
                        1 => p.x = -half.x + 0.85,
                        2 => p.z = -half.z + 1.00,
                        _ => p.x = half.x - 0.85,
                    }
                }
                if !self.camera_clear(p) {
                    continue;
                }
                let target = Vec3::new(
                    rng.random_range(-half.x * 0.28..half.x * 0.28),
                    rng.random_range(0.85..1.4),
                    rng.random_range(-half.z * 0.32..half.z * 0.25),
                );
                if !self.camera_view_clear(p, target) {
                    continue;
                }
                // Metric baselines span short stereo captures through walking
                // motion. Every family remains subject to full-path rejection.
                let forward = (target - p).with_y(0.0).normalize();
                let right = forward.cross(Vec3::Y);
                let family = (self.seed / 7).wrapping_add(index as u64) % 3;
                let (direction, distance) = match family {
                    0 => (right, rng.random_range(0.18..0.40)),
                    1 => (forward, rng.random_range(0.40..0.85)),
                    _ => ((forward + right).normalize(), rng.random_range(0.70..1.20)),
                };
                let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                let delta = direction * distance * sign + Vec3::Y * rng.random_range(-0.06..0.06);
                let end = p + delta;
                if !self.camera_trajectory_clear(p, end, target) {
                    continue;
                }
                if self
                    .cameras
                    .iter()
                    .any(|c| c.start.distance(p) < 0.18 && c.target.distance(target) < 0.5)
                {
                    continue;
                }
                found = Some(IndoorCamera {
                    start: p,
                    end,
                    target,
                    fov_degrees: rng.random_range(48.0..74.0),
                });
                break;
            }
            self.cameras
                .push(found.ok_or_else(|| {
                    format!("seed {}: unable to place camera {index}", self.seed)
                })?);
        }
        Ok(())
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
