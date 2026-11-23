//! Inspect a reachable window, both stiles and the surrounding opening.
//! Use the aperture and room obstacles to keep the poses comparable across joinery changes.
use super::*;
use bevy_zeroverse::scene::procedural_indoor::{
    architecture::facade::{PaneInstallation, SashHinge, Shade},
    layout::segment_hits_box,
};

fn details(pane: &PaneInstallation, hardware: bool) -> Option<[(f32, f32, f32); 2]> {
    let (lo, hi) = (pane.min, pane.max);
    if hardware {
        let latch = pane.handle?;
        let pin = lo.lerp(hi, 0.18);
        let hinge = match pane.hinge {
            SashHinge::LeftCasement => Vec2::new(lo.x, pin.y),
            SashHinge::RightCasement => Vec2::new(hi.x, pin.y),
            SashHinge::BottomHopper => Vec2::new(pin.x, lo.y),
            SashHinge::TopAwning => Vec2::new(pin.x, hi.y),
            SashHinge::Fixed => return None,
        };
        return Some([(latch.x, latch.y, 0.48), (hinge.x, hinge.y, 0.58)]);
    }
    let span = hi - lo;
    if !(0.34..1.4).contains(&span.x)
        || !(0.44..2.05).contains(&span.y)
        || lo.y > 1.45
        || hi.y < 1.05
    {
        return None;
    }
    let level = 1.18_f32.clamp(lo.y + 0.08, hi.y - 0.08);
    Some([(lo.x + 0.03, level, 0.7), (hi.x - 0.03, level, 0.9)])
}

pub(super) fn targets(scene: &IndoorManifest, hardware: bool) -> Result<[(Vec3, Vec3); 3]> {
    let e = scene
        .envelope
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing envelope"))?;
    let mut inspection = scene.clone();
    inspection.camera_settings.primary_room = false;
    let structures = e.structural_boxes(scene.room_size);
    for wall in &e.walls {
        let Some(f) = wall.facade.as_ref().filter(|f| f.shade == Shade::None) else {
            continue;
        };
        let tf = e.wall_transform(wall.edge);
        for o in f.openings.iter().filter(|o| o.operable) {
            for pane in f.pane_installations(o) {
                let Some([first, second]) = details(&pane, hardware) else {
                    continue;
                };
                let mut views = [(Vec3::ZERO, Vec3::ZERO); 3];
                for (view, (x, y, distance)) in [
                    first,
                    second,
                    ((o.min.x + o.max.x) * 0.5, (o.min.y + o.max.y) * 0.5, 2.8),
                ]
                .into_iter()
                .enumerate()
                {
                    let target = tf.transform_point(Vec3::new(x, y, -f.recess));
                    // Casework can occupy the immediate inspection position.
                    // Step back or lower the eye while retaining line of sight.
                    let standpoints: &[(f32, f32)] = if hardware {
                        &[
                            (0., 0.14),
                            (0.35, 0.25),
                            (0.7, 0.48),
                            (0.55, -0.23),
                            (1.15, 0.35),
                        ]
                    } else {
                        &[(0., 0.14)]
                    };
                    for &(back, rise) in standpoints {
                        let eye = [0.24, -0.24, 0., 0.6, -0.6]
                            .into_iter()
                            .map(|offset| {
                                tf.transform_point(Vec3::new(x + offset, y + rise, distance + back))
                            })
                            .find(|&eye| {
                                inspection.camera_clear(eye)
                                    && !scene.objects.iter().filter(|o| o.solid).any(|o| {
                                        let (a, b) = o.bounds();
                                        segment_hits_box(eye, target, a, b)
                                    })
                                    && !structures
                                        .iter()
                                        .any(|&(a, b)| segment_hits_box(eye, target, a, b))
                            });
                        if let Some(eye) = eye {
                            views[view] = (eye, target);
                            break;
                        }
                    }
                }
                if views.iter().all(|(p, t)| p.distance_squared(*t) > 0.01) {
                    return Ok(views);
                }
            }
        }
    }
    anyhow::bail!(
        "no unobstructed window detail views for scene {}",
        scene.seed
    )
}
