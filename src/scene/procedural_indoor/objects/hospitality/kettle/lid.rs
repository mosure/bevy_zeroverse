//! A plug seats inside the rim. The three lid cuts accept different hardware.
use super::*;

pub(super) fn build(a: &mut Assembly, label: &str, p: &KettleProgram, hw: &Hardware) {
    let seat = p.radius * p.neck - hw.wall_m * 0.5;
    let h = hw.lid_rise_m;
    let y = p.base + p.height;
    let tf = Transform::from_xyz(0., y, 0.);
    let shoulder = seat * 0.68;
    let button = hw.grip_span_m * 0.35;
    let top_r = if hw.lid_grip == LidGrip::PressButton {
        button
    } else {
        0.
    };
    let skirt = seat - hw.wall_m;
    a.part(p.finish, label).lathe(
        &[
            (0., -hw.wall_m * 1.4),
            (skirt, -hw.wall_m * 1.4),
            (seat, 0.),
            (seat * 0.97, h * 0.35),
            (shoulder, h),
            (top_r, h),
        ],
        32,
        tf,
    );
    let g = a.part(p.grip, label);
    let mount = Transform::from_xyz(0., y + h, 0.);
    match hw.lid_grip {
        LidGrip::PressButton => {
            // The button closes the opening in the lid, with a bevel at its edge.
            g.lathe(
                &[(button, 0.), (button * 0.90, hw.wall_m), (0., hw.wall_m)],
                32,
                mount,
            );
        }
        LidGrip::LiftKnob => {
            let r = hw.grip_span_m * 0.35;
            let lift = hw.grip_height_m;
            g.lathe(
                &[
                    (r * 0.49, 0.),
                    (r * 0.48, lift * 0.46),
                    (r, lift * 0.72),
                    (r * 0.89, lift),
                    (0., lift),
                ],
                20,
                mount,
            );
        }
        LidGrip::Bridge => {
            let r = hw.grip_height_m.min(hw.grip_span_m) * 0.18;
            // Both tube ends terminate below the lid crown. The arch is graspable.
            let path: [Vec3; 13] = std::array::from_fn(|i| {
                let (s, c) = (PI * i as f32 / 12.).sin_cos();
                Vec3::new(
                    c * hw.grip_span_m * 0.5,
                    y + h - r * 0.35 + s * hw.grip_height_m,
                    0.,
                )
            });
            g.tube(&path, r, 8);
        }
    }
}
