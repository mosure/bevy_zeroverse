//! A sewn shade panel spanning two battens. The skins, edges and selvedges
//! share an outline; texture coordinates follow the length of the hanging cloth.
use super::*;

pub(super) const GAUGE: f32 = 0.004;
const STEPS: usize = 6;

pub(super) fn panel(
    g: &mut Geometry,
    tf: &Transform,
    x0: f32,
    x1: f32,
    half_depth: f32,
    height: f32,
    sag: f32,
) {
    let mut arc = 0.;
    let mut previous = Vec2::new(x0, height);
    let base = g.positions.len() as u32;
    let mut last = [Vec3::ZERO; 4];
    for i in 0..=STEPS {
        let t = i as f32 / STEPS as f32;
        let at = Vec2::new(x0 + (x1 - x0) * t, height - 4. * sag * t * (1. - t));
        arc += at.distance(previous);
        previous = at;
        let normal =
            tf.rotation * Vec3::new(4. * sag * (1. - 2. * t) / (x1 - x0), 1., 0.).normalize();
        let points = [
            (0., -half_depth),
            (0., half_depth),
            (-GAUGE, -half_depth),
            (-GAUGE, half_depth),
        ]
        .map(|(dy, z)| tf.transform_point(Vec3::new(at.x, at.y + dy, z)));
        // Keep both broad skins contiguous even though edge geometry is appended later.
        for (index, p) in points.into_iter().enumerate() {
            g.positions.push(p.to_array());
            g.normals
                .push(if index < 2 { normal } else { -normal }.to_array());
            g.uvs
                .push([arc, if index % 2 == 0 { 0. } else { 2. * half_depth }]);
        }
        if i > 0 {
            let j = base + ((i - 1) * 4) as u32;
            g.indices.extend([
                j,
                j + 1,
                j + 5,
                j,
                j + 5,
                j + 4,
                j + 2,
                j + 6,
                j + 7,
                j + 2,
                j + 7,
                j + 3,
            ]);
        }
        last = points;
    }
    let at =
        |g: &Geometry, i: usize, k: usize| Vec3::from_array(g.positions[base as usize + i * 4 + k]);
    for i in 0..STEPS {
        for (i0, i1, sign) in [(0, 2, -1.), (1, 3, 1.)] {
            let p = [
                at(g, i, i0),
                at(g, i + 1, i0),
                at(g, i + 1, i1),
                at(g, i, i1),
            ];
            g.metric_quad(p, tf.rotation * (sign * Vec3::Z));
        }
    }
    let first = [at(g, 0, 0), at(g, 0, 1), at(g, 0, 3), at(g, 0, 2)];
    g.metric_quad(first, tf.rotation * Vec3::NEG_X);
    g.metric_quad([last[0], last[1], last[3], last[2]], tf.rotation * Vec3::X);
}
