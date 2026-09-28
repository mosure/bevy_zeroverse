use super::*;
use bevy_zeroverse::scene::procedural_indoor::layout::IndoorObject;

pub fn objects(template: &IndoorObject) -> Vec<IndoorObject> {
    let cases = [
        (ObjectKind::Mug, Vec3::new(0.12, 0.10, 0.09), 3),
        (ObjectKind::CoffeeCup, Vec3::new(0.09, 0.13, 0.09), 2),
        (ObjectKind::WaterBottle, Vec3::new(0.075, 0.25, 0.075), 4),
        (ObjectKind::SodaCan, Vec3::new(0.065, 0.12, 0.065), 2),
        (ObjectKind::Notepad, Vec3::new(0.16, 0.012, 0.22), 2),
        (ObjectKind::Pencil, Vec3::new(0.008, 0.009, 0.17), 1),
        (ObjectKind::Microphone, Vec3::new(0.13, 0.24, 0.16), 2),
        (ObjectKind::Phone, Vec3::new(0.075, 0.009, 0.15), 2),
        (ObjectKind::Laptop, Vec3::new(0.36, 0.29, 0.37), 3),
        (ObjectKind::Monitor, Vec3::new(0.56, 0.42, 0.30), 3),
        (ObjectKind::Clock, Vec3::new(0.35, 0.35, 0.045), 4),
    ];
    let mut result = Vec::new();
    for (kind, size, count) in cases {
        for variant in 0..count {
            let mut o = template.clone();
            o.id = result.len();
            o.kind = kind;
            o.size = size;
            o.seed = variant as u64 * 179 + 21;
            o.variant = variant;
            o.support = None;
            o.neighbor = false;
            o.interaction_target = None;
            o.position = Vec3::new(o.id as f32 * 4.5, 0., 0.);
            o.yaw = if kind == ObjectKind::Microphone {
                std::f32::consts::PI
            } else {
                0.
            };
            result.push(o);
        }
    }
    result
}
