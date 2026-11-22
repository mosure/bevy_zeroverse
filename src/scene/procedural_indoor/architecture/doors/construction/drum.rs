//! Stationary curved glass and channels surrounding the revolving rotor.
use super::*;
use crate::scene::procedural_indoor::{geometry::Geometry, materials::glass::SheetGauge};

pub(super) fn build(a: &mut Assembly, d: &DoorInstallation, frame: &str) {
    let radius = d.width * 0.5 - 0.015;
    let tf = d.transform();
    let gauge = SheetGauge::Eight;
    let glass = format!("door_frame#portal{}#finish{}", d.id, gauge.slot());
    for side in [0., PI] {
        screen(
            a.part(Surface::GlassInterior, &glass),
            tf,
            radius,
            d.height,
            side,
            gauge.metres(),
        );
        for y in [0.018, d.height - 0.018] {
            let rail: [_; 33] = std::array::from_fn(|i| {
                let (s, c) = (side - 0.63 + 1.26 * i as f32 / 32.).sin_cos();
                tf.transform_point(Vec3::new(radius * c, y, radius * s))
            });
            a.part(d.program.frame_finish, frame).tube(&rail, 0.017, 12);
        }
    }
    a.part(
        d.program.hardware_finish,
        &format!("door_hardware#portal{}", d.id),
    )
    .cylinder(
        ROTOR_HEEL_RADIUS_M + d.program.thickness * 0.17,
        d.height - 0.025,
        tf.mul_transform(Transform::from_xyz(0., d.height * 0.5, 0.)),
    );
}

fn screen(g: &mut Geometry, tf: Transform, radius: f32, height: f32, side: f32, gauge: f32) {
    // One bent sheet, with shared stations and radial shading normals. Only
    // the two free ends have cut faces. Both horizontal edges enter a channel.
    const SEGMENTS: usize = 20;
    let radial: [_; SEGMENTS + 1] = std::array::from_fn(|i| {
        let (s, c) = (side - 0.61 + 1.22 * i as f32 / SEGMENTS as f32).sin_cos();
        Vec3::new(c, 0., s)
    });
    let lo = 0.028;
    let hi = height - lo;
    let point =
        |n: Vec3, s: f32, y: f32| tf.transform_point(n * (radius + s * gauge * 0.5) + Vec3::Y * y);
    for s in [-1., 1.] {
        let base = g.positions.len() as u32;
        for (i, &n) in radial.iter().enumerate() {
            for y in if s > 0. { [lo, hi] } else { [hi, lo] } {
                g.positions.push(point(n, s, y).to_array());
                g.normals.push((tf.rotation * (s * n)).to_array());
                g.uvs.push([1.22 * radius * i as f32 / SEGMENTS as f32, y]);
            }
        }
        for i in 0..SEGMENTS as u32 {
            let a = base + 2 * i;
            g.indices.extend([a, a + 1, a + 2, a + 2, a + 1, a + 3]);
        }
    }
    for arc in radial.windows(2) {
        for (y, sign) in [(lo, -1.), (hi, 1.)] {
            g.metric_quad(
                [
                    point(arc[0], -1., y),
                    point(arc[0], 1., y),
                    point(arc[1], 1., y),
                    point(arc[1], -1., y),
                ],
                tf.rotation * (sign * Vec3::Y),
            );
        }
    }
    for (n, sign) in [(radial[0], -1.), (radial[SEGMENTS], 1.)] {
        g.metric_quad(
            [
                point(n, -1., lo),
                point(n, 1., lo),
                point(n, 1., hi),
                point(n, -1., hi),
            ],
            tf.rotation * (sign * n.cross(Vec3::Y)),
        );
    }
}
