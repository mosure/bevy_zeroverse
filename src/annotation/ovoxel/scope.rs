//! Clip architecture in scene-local coordinates before world-space voxelization.
use super::Triangle;
use bevy::{math::Affine3A, prelude::*};

/// Exclude an entire subtree, overriding inherited OvoxelTracked markers.
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OvoxelExcluded;

/// Shared scene-local reconstruction bounds for the scene AABB and O-voxel crop.
/// Includes enclosing wall/glazing thickness, not neighboring rooms or context.
#[derive(Component, Debug, Clone, Reflect, serde::Serialize)]
#[reflect(Component)]
pub struct OvoxelRegion {
    pub min: Vec3,
    pub max: Vec3,
}
impl OvoxelRegion {
    pub fn primary_room(scene: &crate::scene::procedural_indoor::layout::IndoorManifest) -> Self {
        let (lo, hi) = scene.primary_room_bounds();
        // Preserve exterior wall/window insets, floor and ceiling slabs. Object
        // membership uses the unpadded room; only architecture gets this shell.
        Self {
            min: Vec3::new(
                lo.x - 0.20,
                scene.envelope.as_ref().map_or(0., |e| e.minimum_floor()) - 0.20,
                lo.y - 0.20,
            ),
            max: Vec3::new(hi.x + 0.20, scene.room_size.y + 0.14, hi.y + 0.20),
        }
    }
    pub(crate) fn world_bounds(&self, transform: Affine3A) -> [[f32; 3]; 2] {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for x in [self.min.x, self.max.x] {
            for y in [self.min.y, self.max.y] {
                for z in [self.min.z, self.max.z] {
                    let p = transform.transform_point3(Vec3::new(x, y, z));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        [lo.to_array(), hi.to_array()]
    }

    pub(super) fn clip(
        &self,
        triangles: Vec<Triangle>,
        world_from_local: Affine3A,
    ) -> Vec<Triangle> {
        let local_from_world = world_from_local.inverse();
        let mut out = Vec::with_capacity(triangles.len());
        // Reuse fixed arrays: convex triangle clipped to six planes has <=9 vertices.
        for triangle in triangles {
            let points =
                [triangle.a, triangle.b, triangle.c].map(|v| local_from_world.transform_point3(v));
            let lo = points[0].min(points[1]).min(points[2]);
            let hi = points[0].max(points[1]).max(points[2]);
            if lo.cmpgt(self.max).any() || hi.cmplt(self.min).any() {
                continue;
            }
            if lo.cmpge(self.min).all() && hi.cmple(self.max).all() {
                out.push(triangle);
                continue;
            }
            let mut input = [Vec3::ZERO; 12];
            input[..3].copy_from_slice(&points);
            let mut count = 3;
            for axis in 0..3 {
                for (edge, sign) in [(self.min[axis], 1.0), (self.max[axis], -1.0)] {
                    if count == 0 {
                        break;
                    }
                    let mut output = [Vec3::ZERO; 12];
                    let mut n = 0;
                    let mut a = input[count - 1];
                    let mut da = (a[axis] - edge) * sign;
                    for b in input[..count].iter().copied() {
                        let db = (b[axis] - edge) * sign;
                        if (da >= 0.0) != (db >= 0.0) {
                            let mut p = a.lerp(b, da / (da - db));
                            p[axis] = edge;
                            if n == 0 || output[n - 1].distance_squared(p) > 1e-12 {
                                output[n] = p;
                                n += 1;
                            }
                        }
                        if db >= 0.0 && (n == 0 || output[n - 1].distance_squared(b) > 1e-12) {
                            output[n] = b;
                            n += 1;
                        }
                        a = b;
                        da = db;
                    }
                    input = output;
                    count = n;
                }
            }
            for i in 1..count.saturating_sub(1) {
                let t = Triangle {
                    a: world_from_local.transform_point3(input[0]),
                    b: world_from_local.transform_point3(input[i]),
                    c: world_from_local.transform_point3(input[i + 1]),
                    ..triangle
                };
                if !t.is_degenerate() {
                    out.push(t);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crossing_floor_is_clipped_not_dropped_or_clamped_as_a_whole() {
        let region = OvoxelRegion {
            min: Vec3::splat(-1.0),
            max: Vec3::ONE,
        };
        let transform =
            Affine3A::from_rotation_translation(Quat::from_rotation_y(0.7), Vec3::new(3., 1., -2.));
        let tri = Triangle {
            a: transform.transform_point3(Vec3::new(-5., 0., -5.)),
            b: transform.transform_point3(Vec3::new(5., 0., -5.)),
            c: transform.transform_point3(Vec3::new(0., 0., 5.)),
            color: Vec4::ONE,
            semantic_id: 2,
        };
        let clipped = region.clip(vec![tri], transform);
        assert!(!clipped.is_empty());
        let mut area = 0.;
        for t in clipped {
            area += (t.b - t.a).cross(t.c - t.a).length() * 0.5;
            for p in [t.a, t.b, t.c] {
                let local = transform.inverse().transform_point3(p);
                assert!(local.abs().max_element() <= 1.000001);
            }
            assert_eq!(t.semantic_id, 2);
        }
        assert!((area - 4.).abs() < 1e-5);
        let exterior = Triangle {
            a: Vec3::splat(4.),
            b: Vec3::splat(4.) + Vec3::X,
            c: Vec3::splat(4.) + Vec3::Z,
            ..tri
        };
        assert!(region.clip(vec![exterior], Affine3A::IDENTITY).is_empty());
    }
}
