use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::Locker => {
            panel(a, o, p.finish, label, [0., 0.02, 0.], [1., 0.04, 0.98]);
            panel(a, o, p.finish, label, [0., 0.98, 0.], [1., 0.04, 0.98]);
            panel(a, o, p.finish, label, [0., 0.5, -0.48], [1., 0.92, 0.04]);
            for i in 0..=p.bays {
                let x = -0.49 + 0.98 * i as f32 / p.bays as f32;
                panel(a, o, p.finish, label, [x, 0.5, 0.], [0.02, 0.92, 0.94]);
            }
            for i in 0..p.bays {
                let rows = p.shelves.min((o.size.y / 0.38).floor().max(1.) as usize);
                let row_h = 0.94 / rows as f32;
                for row in 0..rows {
                    let w = 0.94 / p.bays as f32;
                    let x = -0.49 + 0.98 * (i as f32 + 0.5) / p.bays as f32;
                    let y = 0.03 + (row as f32 + 0.5) * row_h;
                    panel(
                        a,
                        o,
                        p.finish,
                        label,
                        [x, y - row_h * 0.47, 0.],
                        [w, 0.015, 0.94],
                    );
                    // Split sliding leaves provide a bounded open/closed cupboard state.
                    for side in [-1., 1.] {
                        panel(
                            a,
                            o,
                            p.finish,
                            label,
                            [x + side * w * (0.25 + p.opening * 0.19), y, 0.465],
                            [w * (0.48 - p.opening * 0.38), row_h * 0.94, 0.023],
                        );
                        panel(
                            a,
                            o,
                            Surface::Metal,
                            label,
                            [x + side * w * (0.07 + p.opening * 0.38), y, 0.486],
                            [0.018, 0.046, 0.012],
                        );
                    }
                    for vent in 0..3 {
                        panel(
                            a,
                            o,
                            Surface::Rubber,
                            label,
                            [x - w * 0.29, y + row_h * (0.25 + vent as f32 * 0.04), 0.479],
                            [w * 0.20, 0.004, 0.003],
                        );
                    }
                    if p.contents && p.opening > 0.25 {
                        panel(
                            a,
                            o,
                            Surface::Paper,
                            "books",
                            [x, y - row_h * 0.30, 0.],
                            [w * 0.60, row_h * 0.22, 0.55],
                        );
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}
