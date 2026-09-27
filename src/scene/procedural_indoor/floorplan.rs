//! One floor-plan definition drives architecture, furnishing zones and collision.
use super::{layout::IndoorManifest, materials::Surface, objects::Assembly};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloorPlan {
    #[default]
    OpenHall,
    CornerCore,
    WindowGallery,
    DividedSuite,
}

/// Primary furniture zone, leaving circulation and ancillary rooms accessible.
pub fn work_area(scene: &IndoorManifest) -> (Vec3, Vec2) {
    if let Some(area) = super::program::primary_zone(scene) {
        return area;
    }
    let w = scene.room_size.x;
    let d = scene.room_size.z;
    match scene.floor_plan {
        FloorPlan::OpenHall => (Vec3::ZERO, Vec2::new(w, d)),
        FloorPlan::CornerCore => (Vec3::new(-1.15, 0.0, 0.0), Vec2::new(w - 2.3, d)),
        FloorPlan::WindowGallery => (Vec3::new(1.0, 0.0, 0.0), Vec2::new(w - 2.0, d)),
        FloorPlan::DividedSuite => (Vec3::new(-1.25, 0.0, 0.0), Vec2::new(w - 2.5, d)),
    }
}

/// Solid reserved volumes include an enclosed service core; glazed partitions
/// have real 1.25 m openings, shared by navigation and rendered construction.
fn nominal_volumes(scene: &IndoorManifest) -> Vec<(Vec3, Vec3)> {
    let half = scene.room_size * 0.5;
    let h = scene.room_size.y;
    match scene.floor_plan {
        FloorPlan::OpenHall => vec![],
        FloorPlan::CornerCore => vec![(
            Vec3::new(half.x - 2.2, 0.0, -half.z),
            Vec3::new(half.x, h, -half.z + 2.65),
        )],
        FloorPlan::WindowGallery | FloorPlan::DividedSuite => {
            let x = if scene.floor_plan == FloorPlan::WindowGallery {
                -half.x + 1.85
            } else {
                half.x - 2.35
            };
            vec![
                (
                    Vec3::new(x - 0.045, 0.0, -half.z),
                    Vec3::new(x + 0.045, h, -0.625),
                ),
                (
                    Vec3::new(x - 0.045, 0.0, 0.625),
                    Vec3::new(x + 0.045, h, half.z - 1.6),
                ),
            ]
        }
    }
}

pub fn obstacles(scene: &IndoorManifest) -> Vec<(Vec3, Vec3)> {
    if let Some(program) = &scene.program {
        return program
            .partitions
            .iter()
            .flat_map(|p| p.obstacles(scene.room_size.y))
            .collect();
    }
    let trim = match scene.floor_plan {
        FloorPlan::CornerCore => Vec3::new(0.17, 0.0, 0.10),
        _ => Vec3::new(0.0, 0.0, 0.02),
    };
    nominal_volumes(scene)
        .into_iter()
        .map(|(lo, hi)| (lo - trim, hi + trim))
        .collect()
}

pub fn build(scene: &IndoorManifest, a: &mut Assembly) {
    if let Some(program) = &scene.program {
        for partition in &program.partitions {
            // Branches end at the face of the receiving wall. Building them to
            // its center creates intersecting rails/plaster at T junctions.
            let mut fitted = partition.clone();
            for other in &program.partitions {
                if other.axis == partition.axis
                    || partition.coordinate < other.start
                    || partition.coordinate > other.end
                {
                    continue;
                }
                if (partition.start - other.coordinate).abs() < 0.001 {
                    fitted.start += other.thickness * 0.5;
                }
                if (partition.end - other.coordinate).abs() < 0.001 {
                    fitted.end -= other.thickness * 0.5;
                }
            }
            fitted.build(scene.room_size.y, a);
        }
        return;
    }
    if scene.floor_plan == FloorPlan::CornerCore {
        let half = scene.room_size * 0.5;
        let x = half.x - 2.2;
        let z = -half.z + 2.65;
        // Two external facades enclose the service core against the perimeter.
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new(x, half.y, z - 1.325),
            Vec3::new(0.16, scene.room_size.y, 2.65),
            0.005,
        );
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new(x + 1.1, half.y, z),
            Vec3::new(2.2, scene.room_size.y, 0.16),
            0.005,
        );
        // Recessed closed service door, separate frame, leaf and handle.
        a.box_part(
            Surface::WoodEdge,
            "door",
            Vec3::new(x - 0.086, 1.12, z - 1.25),
            Vec3::new(0.025, 2.24, 1.06),
            0.003,
        );
        a.box_part(
            Surface::Wood,
            "door",
            Vec3::new(x - 0.103, 1.085, z - 1.25),
            Vec3::new(0.014, 2.14, 0.91),
            0.005,
        );
        a.part(Surface::Chrome, "door").rod(
            Vec3::new(x - 0.14, 1.03, z - 0.91),
            Vec3::new(x - 0.14, 1.03, z - 1.04),
            0.008,
        );
        for y in [0.07, scene.room_size.y - 0.04] {
            a.box_part(
                Surface::WoodEdge,
                "wall",
                Vec3::new(x - 0.085, y, z - 1.325),
                Vec3::new(0.018, 0.10, 2.65),
                0.002,
            );
            a.box_part(
                Surface::WoodEdge,
                "wall",
                Vec3::new(x + 1.1, y, z + 0.085),
                Vec3::new(2.2, 0.10, 0.018),
                0.002,
            );
        }
    } else {
        for (lo, hi) in nominal_volumes(scene) {
            let x = (lo.x + hi.x) * 0.5;
            let z = (lo.z + hi.z) * 0.5;
            let height = hi.y;
            let length = hi.z - lo.z;
            let bays = (length / 1.15).ceil() as usize;
            a.box_part(
                Surface::Paint,
                "wall",
                Vec3::new(x, 0.22, z),
                Vec3::new(0.09, 0.44, length),
                0.0,
            );
            a.box_part(
                Surface::Glass,
                "window",
                Vec3::new(x, (height + 0.44) * 0.5, z),
                Vec3::new(0.014, height - 0.50, length),
                0.0,
            );
            for i in 0..=bays {
                a.box_part(
                    Surface::Metal,
                    "window",
                    Vec3::new(x, height * 0.5, lo.z + length * i as f32 / bays as f32),
                    Vec3::new(0.075, height, 0.035),
                    0.002,
                );
            }
            for y in [0.46, height - 0.04] {
                a.box_part(
                    Surface::Metal,
                    "window",
                    Vec3::new(x, y, z),
                    Vec3::new(0.075, 0.035, length),
                    0.002,
                );
            }
        }
    }
}
