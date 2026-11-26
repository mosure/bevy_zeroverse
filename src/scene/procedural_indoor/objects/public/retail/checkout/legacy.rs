//! Geometry of untagged checkout counters in archived manifests.
use super::*;

pub(in super::super) fn build(
    a: &mut Assembly,
    s: Vec3,
    top: f32,
    belt: bool,
    side: f32,
    frame: Surface,
) {
    const COUNTER: &str = "counter";
    let base = 0.085;
    board(
        a,
        Surface::Rubber,
        COUNTER,
        Vec3::Y * (base * 0.5),
        Vec3::new(s.x - 0.15, base, s.z - 0.12),
    );
    board(
        a,
        frame,
        COUNTER,
        Vec3::Y * ((top + base - 0.045) * 0.5),
        Vec3::new(s.x - 0.07, top - base - 0.045, s.z - 0.045),
    );
    // A belt and bagging deck occupy separate bays of the same worktop.
    let split = if belt { s.x * 0.57 } else { 0. };
    if belt {
        board(
            a,
            Surface::Rubber,
            COUNTER,
            Vec3::new(-side * (s.x - split) * 0.5, top - 0.0225, 0.),
            Vec3::new(split, 0.045, s.z),
        );
    }
    board(
        a,
        Surface::Metal,
        COUNTER,
        Vec3::new(side * split * 0.5, top - 0.0225, 0.),
        Vec3::new(s.x - split, 0.045, s.z),
    );
    let x = side * s.x * 0.27;
    let width = (s.x * 0.26).min(0.38);
    let terminal_h = (s.y - top) * 0.54;
    let post = s.y - top - terminal_h;
    board(
        a,
        Surface::Metal,
        COUNTER,
        Vec3::new(x, top + post * 0.5, -s.z * 0.11),
        Vec3::new(0.056, post, 0.05),
    );
    let screen = Vec3::new(x, s.y - terminal_h * 0.5, -s.z * 0.11);
    board(
        a,
        Surface::Plastic,
        "control_panel",
        screen,
        Vec3::new(width, terminal_h, 0.041),
    );
    // Only the cashier side carries the display. The shopper's reader faces +Z.
    board(
        a,
        Surface::Ink,
        "control_panel",
        screen - Vec3::Z * 0.0215,
        Vec3::new(width - 0.024, terminal_h - 0.024, 0.002),
    );
    let reader = Vec3::new(x, top + 0.035, s.z * 0.30);
    a.box_part(
        Surface::Plastic,
        "control_panel",
        reader,
        Vec3::new(0.102, 0.07, 0.158),
        0.007,
    );
    board(
        a,
        Surface::Ink,
        "control_panel",
        reader + Vec3::new(0., 0.036, -0.02),
        Vec3::new(0.078, 0.002, 0.069),
    );
}
