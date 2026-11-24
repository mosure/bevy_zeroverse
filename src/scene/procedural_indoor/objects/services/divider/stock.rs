//! Extruded rectangular frame stock with four small edge chamfers.
//! V follows the member in metres, including on the bevels and rotated rails.
use super::*;

pub(super) fn member(g: &mut Geometry, size: Vec3, tf: Transform, origin: Vec2) {
    let Vec3 { x, y, z } = size * 0.5;
    let c = 0.0015_f32.min(x.min(z) * 0.30);
    let ring = [
        Vec2::new(-x + c, -z),
        Vec2::new(x - c, -z),
        Vec2::new(x, -z + c),
        Vec2::new(x, z - c),
        Vec2::new(x - c, z),
        Vec2::new(-x + c, z),
        Vec2::new(-x, z - c),
        Vec2::new(-x, -z + c),
    ];
    let ends = ring.map(|v| tf.transform_point(Vec3::new(v.x, -y, v.y)));
    let along = tf.rotation * Vec3::Y * size.y;
    let mut u = origin.x;
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        let edge = ring[j] - ring[i];
        let width = edge.length();
        let normal = tf.rotation * Vec3::new(edge.y, 0., -edge.x) / width;
        let v = origin.y;
        g.vertex_quad(
            [ends[i], ends[i] + along, ends[j] + along, ends[j]],
            [normal; 4],
            [
                [u, v],
                [u, v + size.y],
                [u + width, v + size.y],
                [u + width, v],
            ],
        );
        u += width;
    }
    for h in [-y, y] {
        g.slab_cap(&ring, 1., Vec2::ZERO, h, &tf, |point| point + origin);
    }
}
