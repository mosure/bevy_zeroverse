use super::*;
use bevy_zeroverse::scene::procedural_indoor::layout::IndoorObject;

pub fn objects(template: &IndoorObject) -> Vec<IndoorObject> {
    (0..8)
        .map(|i| {
            let mut o = template.clone();
            o.id = i;
            o.kind = ObjectKind::Table;
            o.position = Vec3::X * (i as f32 * 12.);
            o.size = if i < 6 {
                Vec3::new(1.7 + i as f32 * 0.12, 0.75, 4.2 + i as f32 * 0.8)
            } else {
                Vec3::new(2.1, 0.75, 2.1 + (i - 6) as f32 * 0.9)
            };
            o.yaw = 0.;
            o.neighbor = false;
            o.support = None;
            o.interaction_target = None;
            o.seed = (0..2000)
                .find(|&seed| {
                    o.seed = seed;
                    let p = objects::tables::parameters(&o);
                    p.bay_style == (i % 6) as u8
                        && if i.is_multiple_of(2) {
                            p.boat_blend > 0.4
                        } else {
                            p.boat_blend == 0.
                        }
                })
                .unwrap();
            o.finish_seed = Some(o.seed);
            o
        })
        .collect()
}
