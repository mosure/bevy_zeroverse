//! First-surface proxies for a corridor's repeated frontages. Navigation keeps
//! its conservative installation volumes. An oblique view of an open leaf sees
//! the actual opening, and a glazed band contributes the window class.
use super::{Collider, IndoorManifest};
use bevy::prelude::*;

pub(super) fn append(scene: &IndoorManifest, colliders: &mut Vec<Collider>) {
    let program = scene.program.as_ref().unwrap();
    let h = scene.room_size.y;
    for w in &program.partitions {
        let normal = if w.axis == 0 { Vec3::X } else { Vec3::Z };
        let mut slab = |lo: f32, hi: f32, low: f32, high: f32, depth: f32, label| {
            if high <= low || hi <= lo {
                return;
            }
            let a = w.position(lo, low) - normal * (depth * 0.5);
            let b = w.position(hi, high) + normal * (depth * 0.5);
            colliders.push(Collider {
                triangle: None,
                center: (a + b) * 0.5,
                half: (b - a) * 0.5,
                inverse: Quat::IDENTITY,
                label,
            });
        };
        let glass = w.glazing_height(h);
        for (lo, hi) in w.spans() {
            if glass > 0.01 {
                slab(lo, hi, 0., w.sill, w.thickness, "wall");
                slab(lo, hi, w.sill, w.sill + glass, 0.010, "window");
                slab(lo, hi, w.sill + glass, h, w.thickness, "wall");
            } else {
                slab(lo, hi, 0., h, w.thickness, "wall");
            }
        }
        for (a, b) in w.portal_obstacles(h) {
            colliders.push(Collider {
                triangle: None,
                center: (a + b) * 0.5,
                half: (b - a) * 0.5,
                inverse: Quat::IDENTITY,
                label: "wall",
            });
        }
    }
    for door in scene.door_installations() {
        for leaf in door.leaves() {
            colliders.push(Collider {
                triangle: None,
                center: leaf.transform.translation,
                half: leaf.size * 0.5,
                inverse: leaf.transform.rotation.inverse(),
                label: "door",
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::IndoorLayout, program::hallway::front_normal};

    #[test]
    fn hallway_optical_coverage_distinguishes_the_glass_band_from_its_opaque_sill() {
        let mut checks = 0;
        for seed in 0..12 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Hallway, 0.4, 0, 0.)
                    .unwrap();
            let program = scene.program.as_ref().unwrap();
            let mut colliders = vec![];
            append(&scene, &mut colliders);
            for wall in &program.partitions {
                let Some(n) = front_normal(&program.zones[0], wall) else {
                    continue;
                };
                let height = wall.glazing_height(scene.room_size.y);
                if height < 0.25 {
                    continue;
                }
                let along = wall
                    .spans()
                    .into_iter()
                    .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
                    .unwrap();
                let direction = if wall.axis == 0 { Vec3::X } else { Vec3::Z };
                for (y, label, depth) in [
                    (wall.sill * 0.5, "wall", wall.thickness),
                    (wall.sill + height * 0.5, "window", 0.010),
                ] {
                    let o = wall.position((along.0 + along.1) * 0.5, y) + direction * n * 0.65;
                    let (dist, c) = colliders
                        .iter()
                        .filter_map(|c| c.hit(o, -n * direction).map(|t| (t, c)))
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .unwrap();
                    assert_eq!(c.label, label, "seed {seed}");
                    assert!((dist - (0.65 - depth * 0.5)).abs() < 0.001);
                }
                checks += 1;
            }
        }
        assert!(checks >= 10);
    }
}
