//! Sparse ceiling assemblies are optical surfaces, not solid reservation boxes.
//! Reuse the coverage BVH; build once per search, with no GPU or material work.
use super::*;

pub(super) fn append_ceiling(scene: &IndoorManifest, colliders: &mut Vec<Collider>) {
    let Some(ceiling) = &scene.ceiling_installation else {
        return;
    };
    for module in &ceiling.modules {
        for geometry in module.geometry(scene).parts.into_values() {
            for indices in geometry.indices.as_chunks::<3>().0 {
                let p = indices.map(|i| Vec3::from_array(geometry.positions[i as usize]));
                colliders.push(Collider {
                    center: (p[0].min(p[1]).min(p[2]) + p[0].max(p[1]).max(p[2])) * 0.5,
                    half: (p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])) * 0.5,
                    inverse: Quat::IDENTITY,
                    label: "ceiling",
                    triangle: Some(p),
                });
            }
        }
    }
}

impl Collider {
    pub(super) fn hit(&self, origin: Vec3, ray: Vec3) -> Option<f32> {
        let Some([a, b, c]) = self.triangle else {
            return ray_box(
                self.inverse * (origin - self.center),
                self.inverse * ray,
                self.half,
            );
        };
        let ab = b - a;
        let ac = c - a;
        let p = ray.cross(ac);
        let det = ab.dot(p);
        if det.abs() < 1e-9 {
            return None;
        }
        let t = origin - a;
        let u = t.dot(p) / det;
        let q = t.cross(ab);
        let v = ray.dot(q) / det;
        if !(0.0..=1.).contains(&u) || v < 0. || u + v > 1. {
            return None;
        }
        let distance = ac.dot(q) / det;
        (distance > 0.0002).then_some(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        architecture::ceiling::{
            CeilingDetail, CeilingInstallation, CeilingModule, CeilingShape, CeilingSystem,
        },
        layout::IndoorLayout,
    };

    #[test]
    fn actual_ceiling_geometry_occludes_but_empty_reservation_space_does_not() {
        let mut scene =
            IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let module = CeilingModule {
            system: CeilingSystem::Conduit,
            center: Vec3::new(0., 2.7, 0.),
            size: Vec3::new(3., 0.15, 0.8),
            support_top_m: 3.,
            pitch_m: 0.5,
            thickness_m: 0.01,
            seed: 3,
            detail: Some(CeilingDetail {
                finish: 0,
                shape: CeilingShape::Conduit {
                    count: 1,
                    radius_m: 0.012,
                    bend: 0.,
                    rise: 0.,
                    junction_pitch_m: 0.75,
                },
            }),
        };
        let bounds = module.bounds();
        scene.ceiling_installation = Some(CeilingInstallation {
            exposed_slab: true,
            modules: vec![module],
        });
        let mut colliders = Vec::new();
        append_ceiling(&scene, &mut colliders);
        let index = index::Index::new(&colliders);
        // Both rays cross the conservative reservation; only one hits the pipe.
        for (origin, occupied) in [
            (Vec3::new(0.1, 2., 0.), true),
            (Vec3::new(0.1, 2., 0.3), false),
        ] {
            assert!(ray_box(
                origin - (bounds.0 + bounds.1) * 0.5,
                Vec3::Y,
                (bounds.1 - bounds.0) * 0.5
            )
            .is_some());
            let hit = index.closest(&colliders, origin, Vec3::Y, 2., false);
            assert_eq!(hit.is_some(), occupied, "origin={origin:?}");
        }
        // BVH decisions must match a direct all-triangle oracle, including thin
        // suspension rods, grazing rays and the backsides seen by a ceiling lens.
        for x in 0..24 {
            for z in 0..17 {
                for y in [2., 4.] {
                    let origin =
                        Vec3::new((x as f32 / 23. - 0.5) * 3.4, y, (z as f32 / 16. - 0.5) * 1.);
                    let ray = if y < 3. { Vec3::Y } else { Vec3::NEG_Y };
                    let scalar = colliders
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| c.hit(origin, ray).filter(|&d| d < 3.).map(|d| (i, d)))
                        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                    assert_eq!(index.closest(&colliders, origin, ray, 3., false), scalar);
                }
            }
        }
    }
}
