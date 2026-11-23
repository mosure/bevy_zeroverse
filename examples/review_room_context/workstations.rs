//! Inspection viewpoints of installed desks in their original room and light.
use super::*;
use bevy_zeroverse::scene::procedural_indoor::layout::{segment_hits_box, IndoorObject};

pub(super) fn targets(s: &IndoorManifest) -> [(Vec3, Vec3); 3] {
    // A coworking suite's primary reconstruction region can be its lounge.
    // A detail camera may enter the neighboring work zone to inspect its desk.
    let mut inspection = s.clone();
    inspection.camera_settings.primary_room = false;
    std::array::from_fn(|view| {
        // Operator's view, cable routing from the rear, and raised/shared desk context.
        let mut desks: Vec<_> = s
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Desk && !o.neighbor)
            .filter_map(|d| {
                let count = |k| {
                    s.objects
                        .iter()
                        .filter(|p| p.support == Some(d.id) && p.kind == k)
                        .count()
                };
                let monitors = count(ObjectKind::Monitor);
                if monitors == 0 {
                    return None;
                }
                let shared = s
                    .assembly_diagnostics
                    .accepted
                    .iter()
                    .any(|g| g.surface == d.id && g.seats.len() == 2);
                let score = match view {
                    0 => {
                        monitors * 2
                            + count(ObjectKind::Keyboard)
                            + count(ObjectKind::Mouse)
                            + count(ObjectKind::PcTower) * 2
                    }
                    1 => count(ObjectKind::Cable) * 3 + count(ObjectKind::PcTower),
                    _ => {
                        usize::from(shared) * 6
                            + usize::from(d.size.y >= 0.94) * 5
                            + usize::from(monitors >= 2)
                    }
                };
                Some((score, d))
            })
            .collect();
        desks.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        desks
            .into_iter()
            .find_map(|(_, d)| vantage(&inspection, d, view))
            .unwrap_or((s.cameras[view].start, s.cameras[view].target))
    })
}

fn vantage(s: &IndoorManifest, desk: &IndoorObject, view: usize) -> Option<(Vec3, Vec3)> {
    let monitor = s
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::Monitor && o.support == Some(desk.id))?;
    let target = if view == 1 {
        monitor.position + Vec3::Y * 0.065
    } else {
        desk.position + Vec3::Y * (desk.size.y + if view == 0 { 0.13 } else { 0.035 })
    };
    let rear = if view == 1 { -1. } else { 1. };
    for radius in [1.55, 1.90, 2.30, 2.70] {
        for x in [0.56, -0.56, 0.28, -0.28, 0.85, -0.85, 1.6, -1.6] {
            let bearing = Vec3::new(x, 0., rear).normalize();
            let p = target
                + monitor.transform().rotation * bearing * radius
                + Vec3::Y * radius * if view == 1 { 0.50 } else { 0.46 };
            if !s.camera_clear(p) || !s.camera_view_clear(p, target) {
                continue;
            }
            // camera_view_clear covers one metre. Walk the remaining ray up to
            // the workstation, so a partition cannot hide the diagnostic target.
            let direction = (target - p).normalize();
            let last = (p.distance(target) - 1.07).max(0.);
            let steps = last.ceil().max(1.) as usize;
            if (1..=steps).any(|step| {
                let origin = p + direction * (last * step as f32 / steps as f32);
                !s.camera_view_clear(origin, origin + direction * 2.)
            }) {
                continue;
            }
            let blocked = s
                .objects
                .iter()
                .filter(|o| o.solid && o.id != desk.id)
                .any(|o| {
                    let inverse = o.transform().compute_affine().inverse();
                    let half = o.size * 0.5;
                    segment_hits_box(
                        inverse.transform_point3(p),
                        inverse.transform_point3(target),
                        Vec3::new(-half.x, 0., -half.z),
                        Vec3::new(half.x, o.size.y, half.z),
                    )
                });
            if !blocked {
                return Some((p, target));
            }
        }
    }
    None
}
