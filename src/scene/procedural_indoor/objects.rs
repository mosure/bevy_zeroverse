pub mod accessories;
pub mod audit;
pub mod bags;
pub mod chairs;
pub mod clocks;
pub mod computers;
pub mod containers;
pub mod desktop;
pub mod equipment;
pub mod fans;
pub mod floor;
pub mod hospitality;
pub mod kitchen;
pub mod literature;
pub mod luminaires;
pub mod presentation;
pub mod public;
pub mod seating;
pub mod services;
pub mod standing_lamp;
pub mod storage;
pub mod tables;
pub mod utilities;
pub mod vessels;
pub mod wall;
use super::{
    geometry::Geometry,
    layout::{stream, IndoorObject, ObjectKind},
    materials::{IndoorMaterials, Surface},
};
use crate::{
    annotation::obb::{ObbClass, ObbTracked},
    ovoxel::{OvoxelExcluded, OvoxelTracked},
    render::semantic::SemanticLabel,
};
use bevy::{camera::primitives::Aabb, light::NotShadowCaster, prelude::*};
use rand::Rng;
use std::{
    collections::BTreeMap,
    f32::consts::{FRAC_PI_2, PI, TAU},
};

/// Every assembly batches equal material/semantic parts while preserving object identity.
#[derive(Default)]
pub struct Assembly {
    pub(crate) motion_policy: Option<super::motion::IndoorMotionConfig>,
    pub(crate) joints: Vec<super::motion::Joint>,
    pub parts: BTreeMap<(Surface, String), Geometry>,
}

#[derive(Component, Debug)]
pub struct IndoorInstance {
    pub id: usize,
    pub kind: ObjectKind,
}

#[derive(Component, Debug)]
pub struct IndoorSurface(pub Surface);

/// Assembly keys can distinguish material variants while preserving one semantic class.
pub(super) fn part_label(label: &str) -> &str {
    label.split('#').next().unwrap_or(label)
}

impl Assembly {
    pub fn part(&mut self, surface: Surface, label: &str) -> &mut Geometry {
        self.parts.entry((surface, label.to_owned())).or_default()
    }
    pub fn box_part(&mut self, surface: Surface, label: &str, pos: Vec3, size: Vec3, bevel: f32) {
        self.part(surface, label)
            .cuboid(size, bevel, Transform::from_translation(pos));
    }
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in self.parts.values().flat_map(|g| &g.positions) {
            lo = lo.min(Vec3::from_array(*p));
            hi = hi.max(Vec3::from_array(*p));
        }
        (lo, hi)
    }
    pub fn spawn(
        self,
        parent: Entity,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &IndoorMaterials,
    ) {
        for ((surface, label), geometry) in self.parts {
            if geometry.indices.is_empty() {
                continue;
            }
            let mut entity = commands.spawn((
                Name::new(format!("{label}/{surface:?}")),
                Mesh3d(meshes.add(geometry.into_mesh())),
                MeshMaterial3d(materials.for_part(surface, &label)),
                SemanticLabel::from_label(part_label(&label))
                    .expect("indoor semantic vocabulary is valid"),
                IndoorSurface(surface),
                OvoxelTracked,
                ChildOf(parent),
            ));
            if label.split('#').any(|tag| tag == "exterior") {
                entity.insert(OvoxelExcluded);
            }
            if matches!(
                surface,
                Surface::Glass
                    | Surface::GlassInterior
                    | Surface::ContainerGlass
                    | Surface::Liquid
                    | Surface::Light
            ) {
                entity.insert(NotShadowCaster);
            }
        }
    }
}

pub fn spawn_object(
    object: &IndoorObject,
    parent: Entity,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &IndoorMaterials,
) {
    let assembly = build_object(object);
    let (lo, hi) = assembly.bounds();
    let root = commands
        .spawn((
            Name::new(format!("{:?}_{}", object.kind, object.id)),
            crate::render::semantic::PreserveSemanticParts,
            object.transform(),
            Visibility::default(),
            ChildOf(parent),
            IndoorInstance {
                id: object.id,
                kind: object.kind,
            },
            ObbTracked,
            ObbClass(object.kind.class_name().to_owned()),
            SemanticLabel::from_label(object.kind.class_name()).expect("indoor object class"),
            Aabb::from_min_max(lo, hi),
        ))
        .id();
    assembly.spawn(root, commands, meshes, materials);
}

pub fn build_object(o: &IndoorObject) -> Assembly {
    build_object_with_motion(o, None)
}

pub(crate) fn build_object_with_motion(
    o: &IndoorObject,
    policy: Option<&super::motion::IndoorMotionConfig>,
) -> Assembly {
    let mut a = Assembly {
        motion_policy: policy.cloned(),
        ..default()
    };
    match o.kind {
        ObjectKind::CeilingFan => fans::build(&mut a, o),
        ObjectKind::ExitSign
        | ObjectKind::FireAlarm
        | ObjectKind::FireDetector
        | ObjectKind::FirePullStation
        | ObjectKind::StringLights => equipment::signals::build(&mut a, o),
        ObjectKind::Elevator
        | ObjectKind::ServiceCounter
        | ObjectKind::DisplayCase
        | ObjectKind::TrayReturn
        | ObjectKind::DrinkDispenser
        | ObjectKind::QueueBarrier
        | ObjectKind::Lectern
        | ObjectKind::Locker
        | ObjectKind::TeachingBoard
        | ObjectKind::ExhibitPlinth
        | ObjectKind::Sculpture
        | ObjectKind::InformationStand
        | ObjectKind::ExhibitionPanel
        | ObjectKind::ExtractionHood
        | ObjectKind::Banquette => public::build(&mut a, o),
        ObjectKind::KitchenCounter
        | ObjectKind::KitchenSink
        | ObjectKind::KitchenCupboard
        | ObjectKind::Pantry
        | ObjectKind::Cooktop
        | ObjectKind::Range
        | ObjectKind::Oven
        | ObjectKind::Microwave
        | ObjectKind::Dishwasher
        | ObjectKind::Refrigerator
        | ObjectKind::DryingRack
        | ObjectKind::SoapDispenser
        | ObjectKind::Sponge
        | ObjectKind::KnifeBlock
        | ObjectKind::UtensilHolder
        | ObjectKind::PaperTowel
        | ObjectKind::KitchenTowel
        | ObjectKind::Dishware
        | ObjectKind::CuttingBoard
        | ObjectKind::FruitBowl
        | ObjectKind::Bread => kitchen::build(&mut a, o),
        ObjectKind::Projector
        | ObjectKind::ProjectionScreen
        | ObjectKind::PowerStrip
        | ObjectKind::Cable
        | ObjectKind::WallShelf
        | ObjectKind::NoticeBoard
        | ObjectKind::WallSconce
        | ObjectKind::CeilingPendant => equipment::build(&mut a, o),
        ObjectKind::Printer | ObjectKind::StorageBox | ObjectKind::CoatRack | ObjectKind::Bag => {
            utilities::build(&mut a, o)
        }
        ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable | ObjectKind::SideTable => {
            tables::build(&mut a, o)
        }
        ObjectKind::MobileCart
        | ObjectKind::DrawerUnit
        | ObjectKind::Bench
        | ObjectKind::Ottoman
        | ObjectKind::RoomDivider => services::build(&mut a, o),
        ObjectKind::DeskLamp
        | ObjectKind::Speaker
        | ObjectKind::Headphones
        | ObjectKind::Stapler
        | ObjectKind::TapeDispenser
        | ObjectKind::DocumentTray
        | ObjectKind::DeskOrganizer
        | ObjectKind::TissueBox
        | ObjectKind::DeskFan => accessories::build(&mut a, o),
        ObjectKind::CoffeeMaker | ObjectKind::Kettle | ObjectKind::Vase | ObjectKind::Bowl => {
            hospitality::build(&mut a, o)
        }
        ObjectKind::Chair => chairs::build(&mut a, o),
        ObjectKind::Sofa => seating::build(&mut a, o),
        ObjectKind::Cabinet | ObjectKind::Bookcase => storage::build(&mut a, o),
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => wall::fixture(&mut a, o),
        ObjectKind::Keyboard | ObjectKind::Mouse => desktop::build(&mut a, o),
        ObjectKind::WaterBottle
        | ObjectKind::Mug
        | ObjectKind::CoffeeCup
        | ObjectKind::SodaCan
        | ObjectKind::Notepad
        | ObjectKind::Pencil
        | ObjectKind::Microphone
        | ObjectKind::Phone
        | ObjectKind::PenHolder => super::clutter::build(&mut a, o),
        ObjectKind::Plant => super::plants::build(&mut a, o),
        ObjectKind::TrashCan => containers::bin(&mut a, o),
        ObjectKind::Display | ObjectKind::Monitor => computers::display(&mut a, o),
        ObjectKind::Laptop => computers::laptop(&mut a, o),
        ObjectKind::Whiteboard | ObjectKind::WallArt => wall_panel(&mut a, o),
        ObjectKind::Notebook | ObjectKind::Books => literature::build(&mut a, o),
        ObjectKind::Clock => clocks::build(&mut a, o),
        ObjectKind::FloorLamp => standing_lamp::build(&mut a, o),
        ObjectKind::Rug | ObjectKind::FloorOutlet => floor::build(&mut a, o),
    }
    if matches!(o.kind, ObjectKind::CoffeeMaker | ObjectKind::DrinkDispenser) {
        a.box_part(
            Surface::Rubber,
            o.kind.class_name(),
            Vec3::new(0., o.size.y * 0.10, -o.size.z * 0.5 + 0.004),
            Vec3::new(0.025, 0.020, 0.008),
            0.002,
        );
    }
    // Flat printed pages and phones use their horizontal face dimensions;
    // normalizing all six cuboid faces would crop portrait content horizontally.
    for ((surface, label), geometry) in &mut a.parts {
        if super::materials::rugs::key(label).is_some_and(|(_, binding)| !binding) {
            // Whole-rug coordinates are already normalized and outline-aligned.
            continue;
        } else if matches!(surface, Surface::PhoneScreen | Surface::PrintedPaper) {
            let mut lo = Vec2::splat(f32::INFINITY);
            let mut hi = Vec2::splat(f32::NEG_INFINITY);
            for p in &geometry.positions {
                let p = Vec2::new(p[0], p[2]);
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let extent = (hi - lo).max(Vec2::splat(0.0001));
            for (uv, p) in geometry.uvs.iter_mut().zip(&geometry.positions) {
                *uv = ((Vec2::new(p[0], p[2]) - lo) / extent).to_array();
            }
        } else if matches!(
            surface,
            Surface::Screen | Surface::Whiteboard | Surface::Television
        ) {
            // Upright display faces receive the complete generated image.
            let mut lo = Vec2::splat(f32::INFINITY);
            let mut hi = Vec2::splat(f32::NEG_INFINITY);
            for uv in &geometry.uvs {
                let uv = Vec2::from_array(*uv);
                lo = lo.min(uv);
                hi = hi.max(uv);
            }
            let extent = (hi - lo).max(Vec2::splat(0.0001));
            for uv in &mut geometry.uvs {
                *uv = ((Vec2::from_array(*uv) - lo) / extent).to_array();
            }
        } else if matches!(
            surface,
            Surface::Wood
                | Surface::WoodEdge
                | Surface::Fabric
                | Surface::FabricAlt
                | Surface::Plastic
                | Surface::Metal
                | Surface::Concrete
        ) {
            let mut rng = stream(o.seed, 120 + *surface as u64);
            let phase = Vec2::new(rng.random_range(0.0..4.0), rng.random_range(0.0..4.0));
            for uv in &mut geometry.uvs {
                *uv = (Vec2::from_array(*uv) + phase).to_array();
            }
        }
    }
    // Object identities and pixels use the same taxonomy. Preserve explicit
    // contents (books/paper) on storage furniture, but replace builder-era
    // coarse furniture/prop classes before material batching.
    let class = o.kind.class_name();
    let mut labeled = BTreeMap::<(Surface, String), Geometry>::new();
    for ((surface, label), geometry) in a.parts {
        let old = part_label(&label);
        let label = if !o.kind.is_kitchen()
            && matches!(
                old,
                "other_prop" | "chair" | "sofa" | "table" | "cabinet" | "television"
            ) {
            label.replacen(old, class, 1)
        } else {
            label
        };
        // Several older part labels may converge on the same object class.
        match labeled.entry((surface, label)) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(geometry);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry.get_mut().append(geometry)
            }
        }
    }
    a.parts = labeled;
    // Keep a bounded finish palette, shared maps and separate semantic labels.
    // Independent object seeds avoid uniform mug/chassis/cover colors in a room.
    a.parts = a
        .parts
        .into_iter()
        .map(|((surface, label), geometry)| {
            let label = if super::materials::variants::supports(surface)
                && !label.contains("#finish")
                && !label.contains("#rug")
            {
                format!(
                    "{label}#finish{}",
                    super::materials::variants::object_slot(surface, o.seed, o.finish_seed)
                )
            } else {
                label
            };
            let label = if surface == Surface::Light && super::photometry::is_emitter(o.kind) {
                format!("{label}#source{}", o.id)
            } else {
                label
            };
            ((surface, label), geometry)
        })
        .collect();
    super::motion::construction::movable(&mut a, o);
    // Finish relabeling must keep articulation references aligned with meshes.
    for (id, joint) in a.joints.iter_mut().enumerate() {
        joint.parts = a
            .parts
            .keys()
            .filter(|(_, label)| super::motion::tag(label) == Some(id))
            .cloned()
            .collect();
    }
    a
}

fn wall_panel(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let label = o.kind.class_name();
    // Printed/painted content is a single front-facing image domain. Metric
    // cuboid UVs are correct for timber, but crop images when the panel changes
    // aspect. The physical backing closes the solid mount behind this face.
    let image_face = |a: &mut Assembly, surface: Surface, size: Vec2, z: f32| {
        let g = a.part(surface, label);
        let start = g.positions.len() as u32;
        let y0 = (s.y - size.y) * 0.5;
        let y1 = y0 + size.y;
        g.positions.extend([
            [-size.x * 0.5, y0, z],
            [size.x * 0.5, y0, z],
            [size.x * 0.5, y1, z],
            [-size.x * 0.5, y1, z],
        ]);
        g.normals.extend([[0., 0., 1.]; 4]);
        g.uvs.extend([[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
        g.indices
            .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
    };
    if o.kind == ObjectKind::Whiteboard {
        presentation::whiteboard(a, o);
        return;
    }
    let p = super::materials::art::frame_parameters(o);
    let border = p.border_width;
    let inner = Vec2::new(s.x - border * 2., s.y - border * 2.);
    let image_size = inner - Vec2::splat((p.mat_width + p.shadow_gap) * 2.);
    // All front surfaces have explicit depth separation. In particular, the
    // paper mat cannot share a plane with the picture or the frame front.
    let front = s.z * 0.5 - p.picture_inset;
    let backing_depth = (s.z - p.picture_inset - 0.0015).max(s.z * 0.3);
    a.box_part(
        if p.style == 4 {
            Surface::Plastic
        } else {
            Surface::Paper
        },
        label,
        Vec3::new(0., s.y * 0.5, -s.z * 0.5 + backing_depth * 0.5),
        Vec3::new(inner.x, inner.y, backing_depth),
        0.001,
    );
    if p.style != 0 {
        let surface = match p.style {
            1 | 4 => Surface::Wood,
            2 => Surface::Plastic,
            _ => Surface::Metal,
        };
        let bevel = (border * 0.12).min(s.z * 0.1);
        let recess = if p.stepped {
            s.z * p.recess_fraction
        } else {
            0.
        };
        for y in [border * 0.5, s.y - border * 0.5] {
            a.box_part(
                surface,
                label,
                Vec3::new(0., y, -recess * 0.5),
                Vec3::new(s.x, border, s.z - recess),
                bevel,
            );
            if p.stepped {
                let sign = if y < s.y * 0.5 { -1. } else { 1. };
                a.box_part(
                    surface,
                    label,
                    Vec3::new(
                        0.,
                        y + sign * border * (1. - p.outer_fraction) * 0.5,
                        (s.z - recess) * 0.5,
                    ),
                    Vec3::new(s.x, border * p.outer_fraction, recess),
                    bevel.min(recess * 0.15),
                );
            }
        }
        for x in [-s.x * 0.5 + border * 0.5, s.x * 0.5 - border * 0.5] {
            a.box_part(
                surface,
                label,
                Vec3::new(x, s.y * 0.5, -recess * 0.5),
                Vec3::new(border, s.y - border * 2., s.z - recess),
                bevel,
            );
            if p.stepped {
                a.box_part(
                    surface,
                    label,
                    Vec3::new(
                        x + x.signum() * border * (1. - p.outer_fraction) * 0.5,
                        s.y * 0.5,
                        (s.z - recess) * 0.5,
                    ),
                    Vec3::new(border * p.outer_fraction, s.y - border * 2., recess),
                    bevel.min(recess * 0.15),
                );
            }
        }
    }
    image_face(a, Surface::Art, image_size, front);
}

#[cfg(test)]
mod catalog_tests;
#[cfg(test)]
mod coplanar_tests;
#[cfg(test)]
mod display_tests;
#[cfg(test)]
mod review_tests;
