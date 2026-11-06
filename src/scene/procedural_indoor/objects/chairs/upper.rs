//! Seat, arm and headrest constructions, fitted to a shared occupant datum.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

pub(super) fn seat(a: &mut Assembly, p: &ChairProgram, width: f32, depth: f32, padding: f32) {
    let profile = CushionProfile {
        roundness: p.seat_roundness,
        crown: p.seat_crown,
        dish: p.seat_dish_m,
    };
    let center = Vec3::new(0., 0.47 - padding * 0.5, -0.015);
    let g = a.part(p.seat_surface, "chair");
    if p.seat_style == 2 {
        // Replaceable front bolster, with a real seam and a common support pan.
        let front = depth * p.front_bolster_fraction;
        g.cushion(
            Vec3::new(width, padding, depth - front - 0.003),
            profile,
            Transform::from_translation(center + Vec3::Z * (front + 0.003) * 0.5),
        );
        g.cushion(
            Vec3::new(width, padding * 0.92, front - 0.003),
            CushionProfile {
                roundness: 3.0,
                dish: 0.,
                ..profile
            },
            Transform::from_translation(
                center - Vec3::Z * (depth - front + 0.003) * 0.5 - Vec3::Y * padding * 0.04,
            ),
        );
    } else {
        let first = g.positions.len();
        g.cushion(
            Vec3::new(width, padding, depth),
            profile,
            Transform::from_translation(center),
        );
        if p.seat_style == 1 {
            // Continuous waterfall lip. Inverse-transpose normals follow the
            // same smooth deformation; no displaced surface or duplicate plane.
            for i in first..g.positions.len() {
                let v = Vec3::from_array(g.positions[i]);
                let t = ((center.z - v.z) / depth * 2.).max(0.);
                let slope = if t > 0. {
                    6. * p.waterfall_drop_m * t * t / depth
                } else {
                    0.
                };
                let normal = Vec3::from_array(g.normals[i]);
                g.positions[i][1] -= p.waterfall_drop_m * t * t * t;
                g.normals[i] = Vec3::new(normal.x, normal.y, normal.z - slope * normal.y)
                    .normalize()
                    .to_array();
            }
        }
    }
}

pub(super) fn headrest(a: &mut Assembly, p: &ChairProgram, width: f32, height: f32, tf: Transform) {
    let w = width * p.headrest_width_fraction;
    for side in [-1., 1.] {
        let x = side * w * p.headrest_post_span * 0.5;
        // Terminate the post inside the actual curved shell (or pad), rather
        // than at a fixed depth that can miss a strongly cupped headrest.
        let v = 0.55_f32;
        let end_z = if p.headrest_style == 0 {
            let factor = 1. + 0.05 * (PI * v).sin();
            let t = x / (w * 0.5 * factor);
            p.headrest_curvature_m * (1. - t * t) + p.headrest_height_m * v * 0.05
                - 0.008 * (PI * v).sin()
                + 0.015
        } else {
            0.025
        };
        a.part(Surface::Chrome, "chair").rod(
            tf.transform_point(Vec3::new(x, height - 0.04, 0.045)),
            tf.transform_point(Vec3::new(
                x,
                height + p.headrest_gap_m + p.headrest_height_m * 0.55,
                end_z,
            )),
            0.008,
        );
    }
    if p.headrest_style == 0 {
        a.part(p.back_surface, "chair").chair_back_contour(
            w,
            p.headrest_height_m,
            [p.headrest_curvature_m, 0., 0.05, 0.035, 0.008, 0.05],
            tf.with_translation(tf.transform_point(Vec3::new(0., height + p.headrest_gap_m, 0.))),
        );
    } else {
        let count = if p.headrest_style == 1 { 1 } else { 2 };
        for i in 0..count {
            let x = (i as f32 - (count - 1) as f32 * 0.5) * w / count as f32;
            a.part(p.back_surface, "chair").cushion(
                Vec3::new(w / count as f32 - 0.004, 0.055, p.headrest_height_m),
                CushionProfile {
                    roundness: if count == 1 { 2.2 } else { 3.2 },
                    crown: 2.8,
                    dish: 0.,
                },
                Transform::from_translation(tf.transform_point(Vec3::new(
                    x,
                    height + p.headrest_gap_m + p.headrest_height_m * 0.5,
                    0.025,
                )))
                .with_rotation(tf.rotation * Quat::from_rotation_x(-FRAC_PI_2)),
            );
        }
    }
}

pub(super) fn arms(a: &mut Assembly, p: &ChairProgram, width: f32, family: u32) {
    if !p.armrests {
        return;
    }
    let frame = p.frame_surface;
    for side in [-1., 1.] {
        let x = side * (width * 0.5 + 0.037);
        let top = p.arm_height - 0.013;
        let half = p.arm_pad_length * 0.38;
        match p.arm_support {
            0 => a.part(frame, "chair").rod(
                Vec3::new(x * 0.79, 0.42, 0.09),
                Vec3::new(x, top, 0.07),
                0.012,
            ),
            1 => a.part(frame, "chair").tube(
                &[
                    Vec3::new(x * 0.79, 0.42, -half),
                    Vec3::new(x, top - 0.05, -half),
                    Vec3::new(x, top, -half + 0.045),
                    Vec3::new(x, top, half - 0.045),
                    Vec3::new(x, top - 0.05, half),
                    Vec3::new(x * 0.79, 0.42, half),
                ],
                0.010,
                10,
            ),
            2 => {
                a.box_part(
                    frame,
                    "chair",
                    Vec3::new(x, (0.44 + top) * 0.5, 0.06),
                    Vec3::new(0.028, top - 0.44, 0.044),
                    0.006,
                );
                a.part(frame, "chair").rod(
                    Vec3::new(x * 0.79, 0.425, 0.06),
                    Vec3::new(x, 0.455, 0.06),
                    0.012,
                );
            }
            3 => {
                let junction = Vec3::new(x, (0.44 + top) * 0.5, 0.);
                a.part(frame, "chair")
                    .rod(Vec3::new(x * 0.8, 0.42, 0.), junction, 0.014);
                for end in [-1., 1.] {
                    a.part(frame, "chair")
                        .rod(junction, Vec3::new(x, top, end * half), 0.011);
                }
            }
            4 => {
                // Short pads cannot support a tall sine arch with a 30 mm
                // tube: its crest radius can be smaller than the tube radius.
                // Two explicit quarter bends keep a physically realizable
                // radius, a flat pad carrier, and valid inside triangles.
                let bend = (half * 0.60).min(0.045);
                let mut path = vec![
                    Vec3::new(x * 0.80, 0.42, -half),
                    Vec3::new(x, top - bend, -half),
                ];
                for i in 1..=6 {
                    let t = FRAC_PI_2 * i as f32 / 6.;
                    path.push(Vec3::new(
                        x,
                        top - bend + bend * t.sin(),
                        -half + bend * (1. - t.cos()),
                    ));
                }
                path.push(Vec3::new(x, top, half - bend));
                for i in 1..=6 {
                    let t = FRAC_PI_2 * i as f32 / 6.;
                    path.push(Vec3::new(
                        x,
                        top - bend + bend * t.cos(),
                        half - bend + bend * t.sin(),
                    ));
                }
                path.push(Vec3::new(x * 0.80, 0.42, half));
                a.part(frame, "chair").tube(&path, 0.012, 10);
            }
            _ => {
                // Upholstered side panel, recessed inside the arm pad.
                a.box_part(
                    p.seat_surface,
                    "chair",
                    Vec3::new(x, (0.425 + top) * 0.5, 0.),
                    Vec3::new(0.030, top - 0.425, p.arm_pad_length * 0.84),
                    0.012,
                );
                a.part(frame, "chair").rod(
                    Vec3::new(x * 0.80, 0.42, 0.),
                    Vec3::new(x, 0.435, 0.),
                    0.013,
                );
            }
        }
        a.part(
            if family == 9 {
                p.back_surface
            } else {
                Surface::Plastic
            },
            "chair",
        )
        .cushion(
            Vec3::new(p.arm_pad_width_m, 0.025, p.arm_pad_length),
            CushionProfile {
                roundness: p.arm_pad_roundness,
                crown: 3.5,
                dish: 0.,
            },
            Transform::from_xyz(x, p.arm_height, 0.),
        );
    }
}
