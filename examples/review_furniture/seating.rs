use super::*;
use bevy_zeroverse::scene::procedural_indoor::layout::IndoorObject;
use bevy_zeroverse::scene::procedural_indoor::materials::{screens, variants};

pub fn objects(template: &IndoorObject, scene_seed: u64) -> Vec<IndoorObject> {
    let mut cases = vec![
        (ObjectKind::Sofa, Vec3::new(2.85, 0.94, 1.85), 0),
        (ObjectKind::Sofa, Vec3::new(1.55, 0.85, 0.90), 1),
        (ObjectKind::Sofa, Vec3::new(2.20, 1.04, 0.96), 2),
        (ObjectKind::Sofa, Vec3::new(3.1, 0.87, 1.72), 3),
    ];
    for variant in 0..objects::chairs::FAMILIES {
        cases.push((
            ObjectKind::Chair,
            if variant == 8 {
                Vec3::new(0.86, 0.95, 0.90)
            } else {
                Vec3::new(
                    0.68,
                    if variant == 6 {
                        0.48
                    } else if variant == 7 {
                        0.79
                    } else if matches!(variant, 2..=4) {
                        1.03
                    } else {
                        1.38
                    },
                    0.68,
                )
            },
            variant,
        ));
    }
    for (kind, size, count) in [
        (ObjectKind::Bookcase, Vec3::new(1.15, 1.85, 0.34), 4),
        (ObjectKind::Whiteboard, Vec3::new(1.65, 1.12, 0.055), 4),
        (ObjectKind::Display, Vec3::new(1.55, 1.05, 0.075), 3),
        (ObjectKind::WallOutlet, Vec3::new(0.15, 0.115, 0.018), 3),
        (ObjectKind::LightSwitch, Vec3::new(0.15, 0.115, 0.018), 3),
    ] {
        for i in 0..count {
            cases.push((kind, size, i));
        }
    }
    let mut display_layouts = std::collections::BTreeSet::new();
    let mut board_layouts = std::collections::BTreeSet::new();
    cases
        .into_iter()
        .enumerate()
        .map(|(index, (kind, size, variant))| {
            let mut o = template.clone();
            o.id = index;
            o.kind = kind;
            o.size = size;
            o.variant = variant;
            o.seed = 17 + variant as u64 * 197;
            if kind == ObjectKind::Chair && matches!(variant, 0 | 1 | 5 | 9) {
                o.seed = (0..1000)
                    .find(|&seed| {
                        o.seed = seed;
                        let p = objects::chairs::parameters(&o);
                        let target_back = match variant {
                            0 => 0,
                            1 => 2,
                            _ => 3,
                        };
                        p.headrest && p.back_construction == target_back
                    })
                    .unwrap();
            }
            if kind == ObjectKind::Sofa && size.z > 1.25 {
                o.seed = (0..1000)
                    .find(|&seed| {
                        o.seed = seed;
                        objects::seating::parameters(&o).left_return == (variant == 0)
                    })
                    .unwrap();
            }
            // This is an illustrative gallery, so exercise distinct content
            // programs from the same bounded finish palette used in rooms.
            if matches!(kind, ObjectKind::Display | ObjectKind::Whiteboard) {
                o.seed = (0..2000)
                    .find(|&seed| {
                        let image_seed = variants::screen_seed(scene_seed, variants::slot(seed));
                        if kind == ObjectKind::Display {
                            let layout = screens::parameters(image_seed).layout;
                            layout != 7 && display_layouts.insert(layout)
                        } else {
                            board_layouts.insert(image_seed % 8)
                        }
                    })
                    .expect("scene finish palette must cover the requested gallery layouts");
            }
            o.position = Vec3::new(index as f32 * 4.5, 0., 0.);
            o.yaw = 0.;
            o.support = None;
            o.neighbor = false;
            o.interaction_target = None;
            o
        })
        .collect()
}
