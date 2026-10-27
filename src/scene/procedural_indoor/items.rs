//! Continuous task mixtures for supported items. Each surface has its own prior,
//! independent of camera, lighting and architectural RNG streams.
use super::layout::{stream, IndoorLayout, IndoorManifest, IndoorObject, ObjectKind};
use bevy::prelude::*;
use rand::Rng;
use serde::Serialize;

pub mod composition;
pub mod distribution;
pub(crate) mod sizing;

pub const TASKS: [&str; 6] = [
    "work",
    "reading",
    "refreshment",
    "technical",
    "personal",
    "storage",
];

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceProgram {
    pub task_weights: [f32; 6],
    pub occupancy: f32,
    pub clustering: f32,
    pub alignment: f32,
    pub focal_point: Vec2,
    pub rotation: f32,
    pub handedness: f32,
}

pub fn parameters(scene: &IndoorManifest, support: &IndoorObject) -> SurfaceProgram {
    let mut rng = stream(support.seed, 5500);
    let mut weights: [f32; 6] = std::array::from_fn(|_| rng.random_range(0.04..1.0_f32).powi(2));
    let prior = match support.kind {
        ObjectKind::Desk => [2.4, 0.8, 0.8, 1.8, 1.1, 0.5],
        ObjectKind::Table => [1.6, 1.1, 1.4, 0.7, 0.6, 0.3],
        ObjectKind::Cabinet | ObjectKind::MobileCart => [0.25, 0.6, 2.3, 0.7, 0.6, 2.0],
        ObjectKind::DrawerUnit => [0.6, 0.4, 0.3, 0.3, 1.5, 1.8],
        _ => [0.25, 2.3, 1.5, 0.3, 2.0, 0.25],
    };
    for (weight, prior) in weights.iter_mut().zip(prior) {
        *weight *= prior;
    }
    let activity = scene
        .program
        .as_ref()
        .and_then(|p| {
            p.zones.iter().find(|z| {
                let q = support.position.xz();
                q.cmpge(z.min).all() && q.cmple(z.max).all()
            })
        })
        .map_or(scene.layout, |z| z.activity);
    let favored = match activity {
        IndoorLayout::Breakroom
        | IndoorLayout::Kitchen
        | IndoorLayout::Kitchenette
        | IndoorLayout::Cafe
        | IndoorLayout::Cafeteria => 2,
        IndoorLayout::Library | IndoorLayout::Lounge => 1,
        IndoorLayout::Workshop | IndoorLayout::Studio => 3,
        IndoorLayout::Reception | IndoorLayout::ElevatorLanding => 4,
        IndoorLayout::Classroom | IndoorLayout::LectureHall => 1,
        _ => 0,
    };
    weights[favored] *= rng.random_range(1.5..3.5);
    let sum: f32 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    // Reserve the amount draw for every surface. Changing the clutter policy
    // must not also resample its task alignment, focal point or handedness.
    let occupancy = 0.18 + 2.42 * rng.random_range(0.0..1.0_f32).powf(1.25);
    SurfaceProgram {
        task_weights: weights,
        occupancy: if distribution::is_table(support.kind) {
            2.6 * scene.clutter.sample(support.seed)
        } else {
            occupancy
        },
        clustering: rng.random_range(0.15..0.82),
        alignment: rng.random_range(0.0..0.95),
        focal_point: Vec2::new(rng.random_range(-0.30..0.30), rng.random_range(-0.24..0.24)),
        rotation: rng.random_range(-0.48..0.48),
        handedness: if rng.random_bool(0.5) { -1. } else { 1. },
    }
}

impl SurfaceProgram {
    pub fn dominant_task(&self) -> &'static str {
        TASKS[self
            .task_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0]
    }

    /// Metric sizes keep task objects recognizable across all support dimensions.
    pub fn item(&self, support: &IndoorObject, rng: &mut impl Rng) -> (ObjectKind, Vec3) {
        use ObjectKind::*;
        let mut t = rng.random_range(0.0..1.0);
        let task = self
            .task_weights
            .iter()
            .position(|&w| {
                t -= w;
                t <= 0.
            })
            .unwrap_or(5);
        let kind = match task {
            0 => [
                Stapler,
                TapeDispenser,
                DocumentTray,
                DeskOrganizer,
                Pencil,
                Notepad,
                Notebook,
                Phone,
                PenHolder,
            ][rng.random_range(0..9)],
            1 => [Books, Notebook, Plant, Vase, DeskLamp, Notepad][rng.random_range(0..6)],
            2 => [
                Mug,
                CoffeeCup,
                WaterBottle,
                SodaCan,
                TissueBox,
                Bowl,
                Kettle,
                CoffeeMaker,
            ][rng.random_range(0..8)],
            3 => [Speaker, Headphones, DeskFan, Phone, Printer, StorageBox][rng.random_range(0..6)],
            4 => [TissueBox, Vase, Bowl, Phone, Plant, Bag][rng.random_range(0..6)],
            _ => [
                DocumentTray,
                StorageBox,
                Books,
                Printer,
                Notebook,
                DeskOrganizer,
            ][rng.random_range(0..6)],
        };
        let service = matches!(support.kind, Cabinet | MobileCart);
        let kind = if matches!(kind, CoffeeMaker | Kettle) && !service {
            [Mug, CoffeeCup, WaterBottle][rng.random_range(0..3)]
        } else if matches!(kind, Printer | DeskFan | DocumentTray) && support.size.y < 0.60 {
            [Books, Plant, TissueBox][rng.random_range(0..3)]
        } else {
            kind
        };
        if let Some(size) = sizing::loose(kind, rng) {
            return (kind, size);
        }
        let size = match kind {
            DeskLamp => Vec3::new(
                rng.random_range(0.17..0.30),
                rng.random_range(0.30..0.55),
                rng.random_range(0.19..0.30),
            ),
            CoffeeMaker => Vec3::new(
                rng.random_range(0.24..0.38),
                rng.random_range(0.28..0.44),
                rng.random_range(0.29..0.43),
            ),
            Kettle => Vec3::new(
                rng.random_range(0.15..0.22),
                rng.random_range(0.19..0.28),
                rng.random_range(0.22..0.31),
            ),
            DocumentTray => Vec3::new(
                rng.random_range(0.23..0.28),
                rng.random_range(0.07..0.16),
                rng.random_range(0.30..0.36),
            ),
            DeskOrganizer => Vec3::new(
                rng.random_range(0.13..0.23),
                rng.random_range(0.10..0.19),
                rng.random_range(0.09..0.16),
            ),
            TissueBox => Vec3::new(
                rng.random_range(0.14..0.24),
                rng.random_range(0.08..0.14),
                rng.random_range(0.10..0.16),
            ),
            Vase => {
                let d = rng.random_range(0.08..0.18);
                Vec3::new(d, rng.random_range(0.14..0.30), d)
            }
            DeskFan => Vec3::new(
                rng.random_range(0.18..0.28),
                rng.random_range(0.27..0.42),
                rng.random_range(0.15..0.23),
            ),
            Plant => sizing::loose(Plant, rng).expect("plant sizing"),
            Printer => Vec3::new(
                rng.random_range(0.36..0.53),
                rng.random_range(0.19..0.29),
                rng.random_range(0.30..0.40),
            ),
            _ => unreachable!("unsupported task item"),
        };
        (kind, size)
    }
}
