//! Curated, fully recorded product examples, separate from population audits.
use bevy::prelude::*;
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorObject, ObjectKind},
    objects,
};

pub fn objects() -> Vec<IndoorObject> {
    let mut result = Vec::new();
    for kind in [
        ObjectKind::FireAlarm,
        ObjectKind::FireDetector,
        ObjectKind::FirePullStation,
    ] {
        for seed in 0..4 {
            let mut o = objects::audit::fixture(kind, seed);
            if kind == ObjectKind::FireAlarm && o.variant == 3 {
                o.size.y = o.size.x;
            }
            result.push(o);
        }
    }
    for suspended in [false, true] {
        for colorful in [false, true] {
            let o = (0..1000)
                .find_map(|seed| {
                    let mut o = objects::audit::fixture(ObjectKind::StringLights, seed);
                    o.variant = u32::from(suspended);
                    o.size = if suspended {
                        Vec3::new(4.8, 0.85, 0.95)
                    } else {
                        Vec3::new(2.6, 0.44, 0.14)
                    };
                    let p = objects::equipment::signals::string_program(&o);
                    ((p.colors_srgb.len() > 1) == colorful && (!suspended || p.bays > 1))
                        .then_some(o)
                })
                .unwrap();
            result.push(o);
        }
    }
    result.push(objects::audit::fixture(ObjectKind::ExitSign, 0));
    for (id, o) in result.iter_mut().enumerate() {
        o.id = id;
        o.position = Vec3::new(
            id as f32 * 9.,
            if o.kind == ObjectKind::FireDetector {
                1.
            } else {
                0.
            },
            0.,
        );
    }
    result
}
