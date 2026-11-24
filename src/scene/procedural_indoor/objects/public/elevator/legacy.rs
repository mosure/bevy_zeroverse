//! Archived hall hardware for manifests through generation 228.
use super::*;

pub(super) fn hall(a: &mut Assembly, o: &IndoorObject, p: &ElevatorProgram, threshold_back: f32) {
    let cx = (p.portal_left + p.portal_right) * 0.5;
    let width = p.portal_right - p.portal_left;
    panel(
        a,
        o,
        Surface::Metal,
        "elevator",
        [cx, p.threshold * 0.5, (threshold_back + 0.5) * 0.5],
        [width, p.threshold, 0.5 - threshold_back],
    );
    for i in 0..p.leaves_per_side {
        panel(
            a,
            o,
            Surface::Rubber,
            "elevator",
            [cx, p.threshold + 0.0007, 0.46 - p.recess - i as f32 * 0.026],
            [width, 0.001, 0.004],
        );
    }
    let side = p.control_side;
    let jamb = if side < 0. {
        (-0.5 + p.portal_left) * 0.5
    } else {
        (0.5 + p.portal_right) * 0.5
    };
    panel(
        a,
        o,
        Surface::Metal,
        "control_panel",
        [jamb, p.control_height, 0.493],
        [0.040, 0.10, 0.006],
    );
    for delta in [-0.023, 0.023] {
        a.part(Surface::Chrome, "control_panel").cylinder(
            o.size.x * 0.009,
            o.size.z * 0.0015,
            Transform::from_xyz(
                jamb * o.size.x,
                (p.control_height + delta) * o.size.y,
                o.size.z * 0.497,
            )
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
    let indicator = if p.indicator_side {
        [jamb, p.portal_top * 0.84]
    } else {
        [cx, (1. + p.portal_top) * 0.5]
    };
    panel(
        a,
        o,
        Surface::Ink,
        "control_panel",
        [indicator[0], indicator[1], 0.493],
        [0.054, 0.045, 0.006],
    );
    const DIGITS: [u8; 10] = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f];
    for (i, digit) in [p.floor_number / 10, p.floor_number % 10]
        .into_iter()
        .enumerate()
    {
        for (bit, (x, y, w, h)) in [
            (0., 1., 0.010, 0.0015),
            (1., 0.5, 0.0015, 0.011),
            (1., -0.5, 0.0015, 0.011),
            (0., -1., 0.010, 0.0015),
            (-1., -0.5, 0.0015, 0.011),
            (-1., 0.5, 0.0015, 0.011),
            (0., 0., 0.010, 0.0015),
        ]
        .into_iter()
        .enumerate()
        {
            if DIGITS[digit as usize] & (1 << bit) != 0 {
                panel(
                    a,
                    o,
                    Surface::Light,
                    "control_panel",
                    [
                        indicator[0] + (i as f32 - 0.5) * 0.022 + x * 0.006,
                        indicator[1] + y * 0.013,
                        0.497,
                    ],
                    [w, h, 0.001],
                );
            }
        }
    }
}
