use super::*;
use crate::scene::procedural_indoor::geometry::housing_front_z;

fn disk(g: &mut Geometry, center: Vec2, z: f32, r: f32, h: f32) {
    g.lathe(
        &[
            (r * 0.92, 0.),
            (r, h * 0.18),
            (r * 0.95, h * 0.83),
            (r * 0.83, h),
            (0., h),
        ],
        24,
        Transform::from_translation(center.extend(z))
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
}
pub(crate) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let class = o.kind.class_name();
    let case = Housing::for_object(o);
    let p = sample(o, &case);
    let label = case.case_label(o);
    let trim = case.trim_label(o);
    let label_face = p.film.label(class);
    let half = s.truncate() * 0.5;
    let tf = Transform::from_xyz(0., half.y, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    // Outside air can enter the recessed side seam between shrouds. A core
    // behind the openings and molded bridges join the backplate to the cover.
    let rear = -0.25 * s.z;
    let front = rear + p.vent_gap_m;
    a.part(case.surface, &label).housing(
        &[
            (-0.5 * s.z, half * 0.88),
            (rear - 0.002, half * 0.96),
            (rear, half * 0.96),
        ],
        p.shoulder,
        tf,
    );
    a.part(Surface::Plastic, &trim).housing(
        &[(rear - 0.0005, half * 0.86), (front + 0.0005, half * 0.86)],
        p.shoulder,
        tf,
    );
    a.part(case.surface, &label).housing(
        &[
            (front, half * 0.97),
            (front + 0.001, half),
            (0.19 * s.z, half),
            (0.28 * s.z, half * 0.945),
        ],
        p.shoulder,
        tf,
    );
    for i in 0..p.vents {
        let along = ((i as f32 + 0.5) / p.vents as f32 - 0.5) * 1.14;
        let x = 0.935 * housing_front_z(half.yx(), p.shoulder, along * half.y).abs();
        for side in [-1., 1.] {
            a.part(case.surface, &label).cuboid(
                Vec3::new(0.07 * half.x, 0.030 * half.y, p.vent_gap_m + 0.001),
                0.,
                Transform::from_xyz(side * x, half.y * (1. + along), (rear + front) * 0.5),
            );
        }
    }
    if let Some((pos, sz)) = p.screen(o) {
        // The bezel ends below the printed face. Three adjoining UV cells
        // cover the entire active window, including the temperature unit.
        a.part(Surface::Plastic, &trim).cuboid(
            (sz * 1.08).extend(s.z * 0.045),
            0.001,
            Transform::from_translation(pos - Vec3::Z * s.z * 0.027),
        );
        let face = a.part(Surface::Ink, &label_face);
        let n = p.display_value();
        let edges = [-0.5, -0.146, 0.208, 0.5];
        for (i, pair) in edges.windows(2).enumerate() {
            let center = pos + Vec3::X * sz.x * ((pair[0] + pair[1]) * 0.5);
            let cell = Vec2::new(sz.x * (pair[1] - pair[0]), sz.y);
            if i == 2 {
                face::unit(face, center, cell, p.fahrenheit);
            } else {
                face::tile(
                    face,
                    center,
                    cell,
                    Glyph::Number(if i == 0 { n / 10 } else { n % 10 }),
                );
            }
        }
    }
    let z = s.z * 0.28;
    match p.style {
        Style::Dial => {
            let radius = half.min_element() * 0.475;
            let center = Vec2::new(0., half.y);
            // The calibration ring has its own thin molded bezel. The base cap
            // is recessed under the print, and the knob covers its central bore.
            let base = z + s.z * 0.033;
            a.part(Surface::Plastic, &trim).lathe(
                &[
                    (radius * 1.45, 0.),
                    (radius * 1.47, s.z * 0.009),
                    (radius * 1.45, s.z * 0.023),
                    (0., s.z * 0.023),
                ],
                48,
                Transform::from_translation(center.extend(z))
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
            face::dial(
                a.part(Surface::Ink, &label_face),
                center,
                base,
                radius,
                p.fahrenheit,
            );
            let height = s.z * 0.169;
            disk(
                a.part(case.surface, &label),
                center,
                z + s.z * 0.033,
                radius,
                height,
            );
            let angle = Quat::from_rotation_z(p.pointer_angle());
            a.part(Surface::Plastic, &trim).cuboid(
                Vec3::new(radius * 0.104, radius * 0.48, s.z * 0.023),
                0.0003,
                Transform::from_translation(
                    center.extend(z + height + s.z * 0.037) + angle * Vec3::Y * (radius * 0.55),
                )
                .with_rotation(angle),
            );
        }
        Style::Encoder => {
            let side = if p.mirrored_controls { -1. } else { 1. };
            let center = Vec2::new(side * s.x * 0.273, half.y);
            let radius = (half.x * 0.245).min(half.y * 0.37);
            disk(
                a.part(Surface::Plastic, &trim),
                center,
                z,
                radius,
                s.z * 0.102,
            );
            disk(
                a.part(case.surface, &label),
                center,
                z + s.z * 0.09,
                radius * 0.88,
                s.z * 0.116,
            );
        }
        Style::PushButton => {
            let w = s.x * 0.17;
            let h = s.y * 0.136;
            for (i, x) in [-0.135, 0.135].into_iter().enumerate() {
                let xy = Vec2::new(x * s.x, 0.256 * s.y);
                a.part(Surface::Plastic, &trim).cuboid(
                    Vec3::new(w, h, s.z * 0.13),
                    h * 0.12,
                    Transform::from_translation(xy.extend(z + s.z * 0.052)),
                );
                let g = a.part(case.surface, &label);
                let half_stroke = h * 0.0585;
                let reach = h * 0.275;
                let face_z = z + s.z * 0.117 + 0.00024;
                let mut print = |x0: f32, x1: f32, y0: f32, y1: f32| {
                    g.metric_quad(
                        [
                            Vec2::new(x0, y0),
                            Vec2::new(x1, y0),
                            Vec2::new(x1, y1),
                            Vec2::new(x0, y1),
                        ]
                        .map(|offset| (xy + offset).extend(face_z)),
                        Vec3::Z,
                    );
                };
                print(-reach, reach, -half_stroke, half_stroke);
                if (i == 1) != p.mirrored_controls {
                    print(-half_stroke, half_stroke, half_stroke, reach);
                    print(-half_stroke, half_stroke, -reach, -half_stroke);
                }
            }
        }
    }
}
