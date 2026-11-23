//! Inspect installed stock from unobstructed positions in a production room.
use super::*;
use bevy_zeroverse::scene::procedural_indoor::{
    layout::segment_hits_box,
    materials::Surface,
    objects::{services::bench, storage},
};

pub(super) fn targets(scene: &IndoorManifest) -> Result<[(Vec3, Vec3); 3]> {
    let structure = scene
        .envelope
        .as_ref()
        .map(|e| e.structural_boxes(scene.room_size))
        .unwrap_or_default();
    for o in scene.objects.iter().filter(|o| !o.neighbor) {
        let dims = o.size;
        let level = match o.kind {
            ObjectKind::Bench => {
                let p = bench::parameters(o);
                if !p.slats {
                    continue;
                }
                p.seat_height
            }
            ObjectKind::Bookcase => {
                let p = storage::bookcase::parameters(o);
                if p.shelves != Surface::Wood {
                    continue;
                }
                let levels = &p.bays[0].levels;
                levels[levels.len() / 2] + p.panel_m * 0.5
            }
            ObjectKind::Cabinet if storage::parameters(o).shelves == Surface::Wood => dims.y,
            _ => continue,
        };
        let frame =
            Transform::from_translation(o.position).with_rotation(Quat::from_rotation_y(o.yaw));
        let mut views = [(Vec3::ZERO, Vec3::ZERO); 3];
        for (i, (height, distance)) in [(0.45, 0.62), (0.32, 1.0), (0.62, 2.15)]
            .into_iter()
            .enumerate()
        {
            let focus = if i == 2 { dims.y * 0.55 } else { level };
            let target = frame.transform_point(Vec3::new(-dims.x * 0.19, focus, dims.z * 0.42));
            let eye = [-0.24, 0.24, -0.58, 0.58, 0.]
                .into_iter()
                .map(|x| {
                    frame.transform_point(Vec3::new(
                        x * dims.x,
                        focus + height,
                        dims.z * 0.5 + distance,
                    ))
                })
                .find(|&eye| {
                    scene.in_primary_room(eye, 0.2)
                        && scene.camera_clear(eye)
                        && !structure
                            .iter()
                            .any(|&(lo, hi)| segment_hits_box(eye, target, lo, hi))
                        && !scene
                            .objects
                            .iter()
                            .filter(|other| other.id != o.id && other.solid)
                            .any(|other| {
                                let (lo, hi) = other.bounds();
                                segment_hits_box(eye, target, lo, hi)
                            })
                });
            let Some(eye) = eye else { break };
            views[i] = (eye, target);
            if i == 2 {
                return Ok(views);
            }
        }
    }
    anyhow::bail!(
        "no accessible timber bench or shelves in room {}",
        scene.seed
    )
}
