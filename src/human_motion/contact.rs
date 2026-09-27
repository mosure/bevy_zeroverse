//! Contact probes against the generated supporting chair, including its actual
//! curved back, armrests and frame. This is an admission filter, not a solver.
use crate::scene::procedural_indoor::{geometry::Geometry, objects::Assembly};
use bevy::prelude::*;
use std::collections::HashMap;

const CELL: f32 = 0.08;
const TOLERANCE: f32 = 0.012;

struct Triangle {
    p: [Vec3; 3],
    normal: Vec3,
}

pub(super) struct SupportSurface {
    triangles: Vec<Triangle>,
    cells: HashMap<IVec3, Vec<usize>>,
    local_from_scene: Mat4,
}

fn cell(p: Vec3) -> IVec3 {
    (p / CELL).floor().as_ivec3()
}

impl SupportSurface {
    pub fn new(assembly: &Assembly, transform: Transform) -> Self {
        let mut result = Self {
            triangles: Vec::new(),
            cells: HashMap::new(),
            local_from_scene: transform.to_matrix().inverse(),
        };
        for geometry in assembly.parts.values() {
            result.add(geometry);
        }
        result
    }

    fn add(&mut self, geometry: &Geometry) {
        for face in geometry.indices.as_chunks::<3>().0 {
            let p = std::array::from_fn(|i| Vec3::from_array(geometry.positions[face[i] as usize]));
            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
            if cross.length_squared() < 1e-14 {
                continue;
            }
            let index = self.triangles.len();
            self.triangles.push(Triangle {
                p,
                normal: cross.normalize(),
            });
            // Chair components are thin. Every interior point is within one
            // cell of a surface; expanding triangle bins avoids per-frame BVH
            // construction or a scan of the whole furniture mesh.
            let lo = cell(p[0].min(p[1]).min(p[2]) - Vec3::splat(CELL));
            let hi = cell(p[0].max(p[1]).max(p[2]) + Vec3::splat(CELL));
            for x in lo.x..=hi.x {
                for y in lo.y..=hi.y {
                    for z in lo.z..=hi.z {
                        self.cells
                            .entry(IVec3::new(x, y, z))
                            .or_default()
                            .push(index);
                    }
                }
            }
        }
    }

    pub fn penetrates(&self, scene_point: Vec3) -> bool {
        let p = self.local_from_scene.transform_point3(scene_point);
        let Some(candidates) = self.cells.get(&cell(p)) else {
            return false;
        };
        let mut closest = (f32::INFINITY, 0.0);
        for &i in candidates {
            let triangle = &self.triangles[i];
            let q = triangle.closest(p);
            let distance = p.distance_squared(q);
            if distance < closest.0 {
                closest = (distance, (p - q).dot(triangle.normal));
            }
        }
        closest.0 < CELL * CELL && closest.1 < -TOLERANCE
    }
}

impl Triangle {
    fn closest(&self, p: Vec3) -> Vec3 {
        let projected = p - self.normal * (p - self.p[0]).dot(self.normal);
        let edges = [
            (self.p[0], self.p[1]),
            (self.p[1], self.p[2]),
            (self.p[2], self.p[0]),
        ];
        if edges
            .iter()
            .all(|(a, b)| (b - a).cross(projected - a).dot(self.normal) >= -1e-7)
        {
            return projected;
        }
        edges
            .map(|(a, b)| {
                let d = b - a;
                a + d * ((p - a).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0)
            })
            .into_iter()
            .min_by(|a, b| p.distance_squared(*a).total_cmp(&p.distance_squared(*b)))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::materials::Surface;

    #[test]
    fn thin_supports_allow_contact_but_reject_penetration_after_rotation() {
        let mut chair = Assembly::default();
        chair.box_part(
            Surface::Wood,
            "seat",
            Vec3::Y * 0.44,
            Vec3::new(0.48, 0.06, 0.44),
            0.0,
        );
        let transform =
            Transform::from_xyz(2.0, 0.0, -1.0).with_rotation(Quat::from_rotation_y(0.7));
        let support = SupportSurface::new(&chair, transform);
        for point in [
            Vec3::new(0.0, 0.48, 0.0),
            Vec3::new(0.0, 0.468, 0.0),
            Vec3::new(0.3, 0.44, 0.0),
        ] {
            assert!(!support.penetrates(transform.transform_point(point)));
        }
        assert!(support.penetrates(transform.transform_point(Vec3::new(0.0, 0.44, 0.0))));
    }
}
