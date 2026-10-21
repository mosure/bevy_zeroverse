use super::*;
use bevy_zeroverse::scene::procedural_indoor::layout::IndoorObject;
use bevy_zeroverse::scene::procedural_indoor::objects::{
    audit,
    bags::{parameters, BagStyle},
};

pub fn objects() -> Vec<IndoorObject> {
    let mut result = Vec::new();
    for style in [
        BagStyle::Briefcase,
        BagStyle::Backpack,
        BagStyle::Tote,
        BagStyle::Satchel,
        BagStyle::Duffel,
    ] {
        for variation in 0..3 {
            let mut o = audit::fixture(ObjectKind::Bag, 0);
            o.seed = (variation * 100..(variation + 1) * 100)
                .find(|&seed| {
                    o.seed = seed;
                    parameters(&o).style == style
                })
                .unwrap();
            o.size = match style {
                BagStyle::Briefcase => Vec3::new(0.42, 0.34, 0.13),
                BagStyle::Backpack => Vec3::new(0.34, 0.47, 0.23),
                BagStyle::Tote => Vec3::new(0.37, 0.46, 0.22),
                BagStyle::Satchel => Vec3::new(0.36, 0.40, 0.17),
                BagStyle::Duffel => Vec3::new(0.52, 0.37, 0.28),
            } * (0.86 + variation as f32 * 0.12);
            o.id = result.len();
            // Geometry fixtures pin finishes for distance studies. This visual
            // gallery should show the normal independently sampled bag palette.
            o.finish_seed = None;
            o.position = Vec3::X * o.id as f32 * 4.5;
            if style == BagStyle::Backpack && variation == 2 {
                o.yaw = std::f32::consts::PI;
            }
            result.push(o);
        }
    }
    result
}
