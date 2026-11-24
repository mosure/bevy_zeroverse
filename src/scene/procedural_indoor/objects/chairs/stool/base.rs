use super::*;

pub(super) fn build(a: &mut Assembly, p: &StoolProgram, radius: f32, underside: f32, version: u32) {
    let foot_radius = radius * p.foot_spread;
    match p.base {
        0 => {
            for i in 0..p.legs {
                let t = p.phase + i as f32 * TAU / p.legs as f32;
                let dir = Vec3::new(t.cos(), 0., t.sin());
                let foot = dir * foot_radius + Vec3::Y * 0.016;
                let top = dir * radius * 0.60 + Vec3::Y * underside;
                if version >= FIT_VERSION && p.frame == Surface::WoodEdge {
                    let d = top - foot;
                    // Solid turned stock flares below its seat socket. The foot
                    // and ferrule stay narrower than the load-bearing shoulder.
                    a.part(p.frame, "chair").lathe(
                        &[
                            (0., 0.),
                            (0.014, 0.),
                            (0.019, d.length() * 0.22),
                            (0.026, d.length() * 0.86),
                            (0.023, d.length()),
                            (0., d.length()),
                        ],
                        12,
                        Transform::from_translation(foot)
                            .with_rotation(Quat::from_rotation_arc(Vec3::Y, d.normalize())),
                    );
                } else {
                    a.part(p.frame, "chair").rod(foot, top, 0.016);
                }
                a.part(Surface::Rubber, "chair").cylinder(
                    0.020,
                    0.016,
                    Transform::from_translation(foot.with_y(0.008)),
                );
            }
            // The ring meets the actual splayed legs at this height.
            let t = (p.footrest_height_m - 0.016) / (underside - 0.016);
            ring(
                a,
                p.frame,
                foot_radius * (1. - t) + radius * 0.60 * t,
                p.footrest_height_m,
            );
        }
        1 => {
            let sleeve = if version >= FIT_VERSION {
                p.footrest_height_m + 0.035
            } else {
                underside
            };
            a.part(p.frame, "chair").lathe(
                &[
                    (0., 0.),
                    (foot_radius, 0.),
                    (foot_radius, 0.015),
                    (foot_radius * 0.70, 0.035),
                    (0.032, 0.085),
                    (0.032, sleeve),
                    (0., sleeve),
                ],
                32,
                Transform::IDENTITY,
            );
            if version >= FIT_VERSION {
                // The narrower moving piston emerges through the fixed sleeve.
                // The mounting flange reaches into the underside of the seat.
                a.part(Surface::Chrome, "chair").lathe(
                    &[
                        (0.021, sleeve - 0.004),
                        (0.021, underside - 0.018),
                        (radius * 0.31, underside - 0.010),
                        (radius * 0.31, underside + 0.003),
                    ],
                    20,
                    Transform::IDENTITY,
                );
            }
            let rr = radius * 0.76;
            ring(a, p.frame, rr, p.footrest_height_m);
            for i in 0..3 {
                let angle = p.phase + i as f32 * TAU / 3.;
                a.part(p.frame, "chair").rod(
                    Vec3::Y * p.footrest_height_m,
                    Vec3::new(angle.cos() * rr, p.footrest_height_m, angle.sin() * rr),
                    0.010,
                );
            }
        }
        _ => {
            for side in [-1., 1.] {
                let x = side * radius * 0.66;
                let z = radius * p.seat_aspect * 0.65;
                let path = [
                    Vec3::new(x, underside, -z),
                    Vec3::new(x, 0.022, -z),
                    Vec3::new(x, 0.022, z),
                    Vec3::new(x, underside, z),
                ];
                a.part(p.frame, "chair").tube(&path, 0.016, 10);
                for zz in [-z, z] {
                    a.box_part(
                        Surface::Rubber,
                        "chair",
                        Vec3::new(x, 0.006, zz),
                        Vec3::new(0.042, 0.012, 0.042),
                        0.004,
                    );
                }
            }
            let front =
                if version >= FIT_VERSION { -1. } else { 1. } * (radius * p.seat_aspect * 0.65);
            a.part(p.frame, "chair").rod(
                Vec3::new(-radius * 0.66, p.footrest_height_m, front),
                Vec3::new(radius * 0.66, p.footrest_height_m, front),
                0.012,
            );
        }
    }
}

fn ring(a: &mut Assembly, material: Surface, r: f32, h: f32) {
    let path: Vec<_> = (0..=40)
        .map(|i| {
            let t = (i % 40) as f32 * TAU / 40.;
            Vec3::new(t.cos() * r, h, t.sin() * r)
        })
        .collect();
    a.part(material, "chair").tube(&path, 0.009, 8);
}
