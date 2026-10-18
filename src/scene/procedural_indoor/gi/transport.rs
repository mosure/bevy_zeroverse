//! Compact shadow geometry. Surface attributes remain in the authoritative
//! median tree and are resolved only when serializing the GPU transport input.
use super::{Node, Triangle, Vec2, Vec3};
mod packet;

/// Forty bytes instead of a second eighty-byte complete triangle. The source
/// index refers to the finalized median order, never an assembly or asset ID.
#[derive(Clone, Copy)]
pub(super) struct TransportTriangle {
    pub(super) a: Vec3,
    pub(super) ab: Vec3,
    pub(super) ac: Vec3,
    pub(super) source: u32,
}

impl TransportTriangle {
    pub(super) fn new(source: usize, triangle: &Triangle) -> Self {
        Self {
            a: triangle.a,
            ab: triangle.ab,
            ac: triangle.ac,
            source: u32::try_from(source).expect("transport triangle index fits u32"),
        }
    }

    fn hit(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec2)> {
        super::triangle_hit(self.a, self.ab, self.ac, origin, direction, max)
    }
}

pub(super) struct TransportTree {
    pub(super) triangles: Vec<TransportTriangle>,
    pub(super) nodes: Vec<Node>,
    pub(super) preparation_ms: f64,
}

impl TransportTree {
    pub(super) fn surfaces<'a>(
        &'a self,
        scene: &'a super::BakeScene,
    ) -> impl ExactSizeIterator<Item = &'a Triangle> {
        self.triangles
            .iter()
            .map(|triangle| &scene.triangles[triangle.source as usize])
    }

    pub(super) fn occluded(&self, origin: Vec3, direction: Vec3, max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inverse = direction.recip();
        let mut stack = [0usize; 64];
        let mut len = 1;
        while len > 0 {
            len -= 1;
            let index = stack[len];
            let node = &self.nodes[index];
            if !node.intersects(origin, inverse, max) {
                continue;
            }
            if node.count > 0 {
                let (packets, remainder) =
                    self.triangles[node.start..node.start + node.count].as_chunks::<4>();
                for triangles in packets {
                    if packet::occluded(triangles, origin, direction, max) {
                        return true;
                    }
                }
                for triangle in remainder {
                    if triangle.hit(origin, direction, max).is_some() {
                        return true;
                    }
                }
            } else {
                let (near, far) = if direction[node.axis] >= 0.0 {
                    (index + 1, node.right)
                } else {
                    (node.right, index + 1)
                };
                stack[len] = far;
                stack[len + 1] = near;
                len += 2;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_records_preserve_all_sources_and_bound_attributes() {
        assert_eq!(std::mem::size_of::<TransportTriangle>(), 40);
        assert_eq!(std::mem::size_of::<Triangle>(), 80);
        let triangles = (0..4097)
            .map(|i| Triangle {
                a: Vec3::new((i % 31) as f32, (i % 11) as f32, (i / 31) as f32),
                ab: Vec3::X * 0.13,
                ac: Vec3::Y * 0.27,
                normal: if i % 2 == 0 { Vec3::Z } else { -Vec3::Z },
                uv: [Vec2::ZERO, Vec2::X, Vec2::new(i as f32, 1.)],
                material: i,
            })
            .collect();
        let mut scene = super::super::BakeScene {
            triangles,
            nodes: Vec::new(),
            transport_tree: Default::default(),
            materials: Vec::new(),
            lights: Vec::new(),
            sun_direction: Vec3::Y,
            sun: Vec3::ZERO,
            sky: Vec3::ZERO,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ONE,
            preparation_ms: 0.,
            world_rotation: bevy::prelude::Quat::IDENTITY,
        };
        scene.build_bvh();
        scene.prepare_gpu_transport();
        // Attributes bind after tree construction. GPU serialization must see
        // those bindings, while shadow geometry retains the exact coordinates.
        for triangle in &mut scene.triangles {
            triangle.material += 7;
            triangle.uv[0] = Vec2::new(0.17, -0.25);
        }
        let tree = scene.gpu_transport();
        let mut sources: Vec<_> = tree.triangles.iter().map(|t| t.source).collect();
        sources.sort_unstable();
        assert_eq!(
            sources,
            (0..scene.triangles.len() as u32).collect::<Vec<_>>()
        );
        for (compact, surface) in tree.triangles.iter().zip(tree.surfaces(&scene)) {
            assert_eq!(
                compact.a.to_array().map(f32::to_bits),
                surface.a.to_array().map(f32::to_bits)
            );
            assert_eq!(
                compact.ab.to_array().map(f32::to_bits),
                surface.ab.to_array().map(f32::to_bits)
            );
            assert_eq!(
                compact.ac.to_array().map(f32::to_bits),
                surface.ac.to_array().map(f32::to_bits)
            );
            assert_eq!(surface.uv[0], Vec2::new(0.17, -0.25));
            assert!(surface.material >= 7);
        }
        scene.build_bvh();
        assert!(
            scene.transport_tree.get().is_none(),
            "reordering invalidates source indices"
        );
        scene.prepare_gpu_transport();
        assert_eq!(
            scene.gpu_transport().surfaces(&scene).len(),
            scene.triangles.len()
        );
    }
}
