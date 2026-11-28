use super::*;
use crate::scene::procedural_indoor::envelope::construction::triangle;

/// A flat-bottomed dry-goods bag. The side folds deepen as the paper
/// converges into the sealed strip. The printed panels are the bag itself.
pub(super) fn build(a: &mut Assembly, size: Vec3, tf: Transform, print: &str) {
    let fullness = (size.z / size.x).clamp(0.30, 0.95);
    let shoulder = (0.52 + 0.16 * fullness) * size.y;
    let seal = (0.036 * size.y).clamp(0.004, 0.009);
    let ring = |y: f32, width: f32, depth: f32, fold: f32| {
        let x = 0.5 * size.x * width;
        let z = 0.5 * size.z * depth;
        let panel = x * 0.85;
        let tucked = x * (1. - fold);
        [
            (-panel, -z),
            (panel, -z),
            (x, -0.76 * z),
            (tucked, 0.),
            (x, 0.76 * z),
            (panel, z),
            (-panel, z),
            (-x, 0.76 * z),
            (-tucked, 0.),
            (-x, -0.76 * z),
        ]
        .map(|(dx, dz)| Vec3::new(dx, y, dz))
    };
    let rings = [
        ring(0., 0.88, 0.88, 0.06),
        ring(shoulder, 1., 1., 0.12),
        ring(size.y - seal, 0.92, 0.065, 0.26),
        ring(size.y, 0.92, 0.065, 0.26),
    ];
    for (band, section) in rings.windows(2).enumerate() {
        let g = a.part(Surface::Paper, if band < 2 { print } else { "food" });
        for j in 0..10 {
            let k = (j + 1) % 10;
            let p = [section[0][j], section[1][j], section[1][k], section[0][k]];
            let n = tf.rotation * (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            // Wrap the front artwork and rear information block over the
            // shoulder at the same vertical coordinate as the straight sides.
            let (left, right) = match j {
                5 => (0.03, 0.62),
                0 => (0.65, 0.99),
                _ => (0.62, 0.65),
            };
            let uv =
                std::array::from_fn(|i| [if i < 2 { right } else { left }, 1. - p[i].y / size.y]);
            g.vertex_quad(p.map(|v| tf.transform_point(v)), [n; 4], uv);
        }
    }
    let g = a.part(Surface::Paper, "food");
    for (ring, normal) in [(rings[0], Vec3::NEG_Y), (rings[3], Vec3::Y)] {
        let center = Vec3::Y * ring[0].y;
        for j in 0..10 {
            triangle(
                g,
                [center, ring[j], ring[(j + 1) % 10]].map(|p| tf.transform_point(p)),
                tf.rotation * normal,
            );
        }
    }
}
