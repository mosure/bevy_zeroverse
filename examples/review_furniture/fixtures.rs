//! Illustrative coverage of the independently seeded fixture programs.
use bevy::prelude::*;
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorObject, ObjectKind},
    objects::{containers, wall},
};
pub fn objects(template: &IndoorObject) -> Vec<IndoorObject> {
    (0..24)
        .map(|i| {
            let mut o = template.clone();
            o.id = i;
            o.position = Vec3::X * i as f32 * 4.5;
            o.yaw = 0.;
            o.support = None;
            o.neighbor = false;
            o.interaction_target = None;
            let v = i % 6;
            (o.kind, o.size) = match i / 6 {
                0 => (
                    ObjectKind::TissueBox,
                    Vec3::new(
                        if v % 2 == 0 { 0.23 } else { 0.14 },
                        0.10 + v as f32 * 0.01,
                        0.13,
                    ),
                ),
                1 => (
                    ObjectKind::TrashCan,
                    Vec3::new(
                        0.28 + v as f32 * 0.026,
                        0.40 + v as f32 * 0.032,
                        0.29 + v as f32 * 0.014,
                    ),
                ),
                2 => (
                    ObjectKind::LightSwitch,
                    Vec3::new(0.075 * (1 + v % 3) as f32, 0.115, 0.018),
                ),
                _ => (
                    ObjectKind::WallOutlet,
                    Vec3::new(0.075 * (1 + v % 3) as f32, 0.115, 0.018),
                ),
            };
            o.seed = (0..1000)
                .find(|&seed| {
                    o.seed = seed;
                    match o.kind {
                        ObjectKind::TrashCan => containers::bin_parameters(&o).lid == v as u8 % 5,
                        ObjectKind::LightSwitch | ObjectKind::WallOutlet => {
                            wall::parameters(&o).style == v as u8
                        }
                        _ => seed == v as u64 * 71,
                    }
                })
                .unwrap();
            o
        })
        .collect()
}
