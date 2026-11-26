//! A formed, constant-gauge strip. End-grain and broad faces have separate
//! normals, while the longitudinal material chart and bend highlights are continuous.
use super::*;

pub(super) fn build(g: &mut Geometry, path: &[Vec3; 8], width: f32, thickness: f32) {
    let ring = |i: usize| {
        let tangent = (path[(i + 1).min(7)] - path[i.saturating_sub(1)]).normalize();
        let depth = tangent.cross(Vec3::X).normalize();
        let across = depth.cross(tangent);
        let w = across * (width * 0.5);
        let h = depth * (thickness * 0.5);
        let c = path[i];
        (
            [c - w - h, c + w - h, c + w + h, c - w + h],
            [-depth, across, depth, -across],
        )
    };
    let (mut a, mut old_normals) = ring(0);
    g.metric_quad(a, path[0] - path[1]);
    let mut distance = 0.;
    for i in 1..path.len() {
        let (b, normals) = ring(i);
        let next = distance + path[i].distance(path[i - 1]);
        for (j, span) in [width, thickness, width, thickness].into_iter().enumerate() {
            let k = (j + 1) % 4;
            g.vertex_quad(
                [a[j], a[k], b[k], b[j]],
                [old_normals[j], old_normals[j], normals[j], normals[j]],
                [[0., distance], [span, distance], [span, next], [0., next]],
            );
        }
        a = b;
        old_normals = normals;
        distance = next;
    }
    g.metric_quad(a, path[7] - path[6]);
}
