use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    if o.kind == ObjectKind::QueueBarrier {
        let edge = 0.5 - o.size.z * 0.50 / o.size.x;
        for x in [-edge, edge] {
            a.part(Surface::Metal, "queue_barrier").cylinder(
                o.size.z * 0.47,
                0.025,
                Transform::from_xyz(x * o.size.x, 0.0125, 0.),
            );
            rod(
                a,
                o,
                Surface::Metal,
                "queue_barrier",
                [x, 0.03, 0.],
                [x, 0.96, 0.],
                0.018,
            );
        }
        let sag = p.opening * 0.10;
        for i in 0..12 {
            let t = i as f32 / 12.;
            let u = (i + 1) as f32 / 12.;
            rod(
                a,
                o,
                Surface::Fabric,
                "queue_barrier",
                [-edge + 2. * edge * t, 0.87 - sag * (PI * t).sin(), 0.],
                [-edge + 2. * edge * u, 0.87 - sag * (PI * u).sin(), 0.],
                0.017,
            );
        }
        return;
    }
    // A bounded car behind a recessed telescoping door bank, never an empty shaft.
    // The car is context: visible to depth/semantic capture, excluded from ovoxels.
    for x in [-0.48, 0.48] {
        panel(a, o, p.finish, "elevator", [x, 0.5, 0.], [0.04, 1., 1.]);
    }
    panel(
        a,
        o,
        Surface::Metal,
        "elevator#exterior",
        [0., 0.48, -0.48],
        [0.92, 0.88, 0.04],
    );
    panel(
        a,
        o,
        Surface::Concrete,
        "floor#exterior",
        [0., 0.012, 0.],
        [0.92, 0.024, 0.96],
    );
    panel(
        a,
        o,
        Surface::Metal,
        "ceiling#exterior",
        [0., 0.96, 0.],
        [0.92, 0.08, 0.96],
    );
    panel(
        a,
        o,
        Surface::Light,
        "lamp#exterior",
        [0., 0.915, -0.12],
        [0.46, 0.012, 0.40],
    );
    // Front jambs/head leave a genuine aperture; no full box behind the doors.
    for x in [-0.43, 0.43] {
        panel(
            a,
            o,
            p.finish,
            "elevator",
            [x, 0.46, 0.46],
            [0.14, 0.88, 0.08],
        );
    }
    panel(
        a,
        o,
        p.finish,
        "elevator",
        [0., 0.94, 0.46],
        [0.72, 0.12, 0.08],
    );
    for side in [-1., 1.] {
        for leaf in 0..2 {
            let j = leaf as f32;
            let x = side * (0.0725 + 0.145 * j + (0.29 - 0.145 * j) * p.opening);
            panel(
                a,
                o,
                Surface::Metal,
                "door",
                [x, 0.455, 0.40 - leaf as f32 * 0.025],
                [0.143, 0.856, 0.014],
            );
            panel(
                a,
                o,
                Surface::Metal,
                "door",
                [x, 0.455, 0.411 - leaf as f32 * 0.025],
                [0.012, 0.82, 0.004],
            );
        }
    }
    panel(
        a,
        o,
        Surface::Metal,
        "elevator",
        [0., 0.022, 0.44],
        [0.72, 0.008, 0.12],
    );
    for z in [0.40, 0.44, 0.48] {
        panel(
            a,
            o,
            Surface::Rubber,
            "elevator",
            [0., 0.027, z],
            [0.70, 0.002, 0.008],
        );
    }
    panel(
        a,
        o,
        Surface::Screen,
        "control_panel",
        [0., 0.94, 0.505 - 0.011],
        [0.14, 0.035, 0.006],
    );
    panel(
        a,
        o,
        Surface::Metal,
        "control_panel",
        [0.43, 0.48, 0.505 - 0.011],
        [0.055, 0.14, 0.006],
    );
    for y in [0.45, 0.51] {
        panel(
            a,
            o,
            Surface::Light,
            "control_panel",
            [0.43, y, 0.499],
            [0.020, 0.015, 0.002],
        );
    }
    for side in [-1., 1.] {
        rod(
            a,
            o,
            Surface::Metal,
            "elevator#exterior",
            [side * 0.435, 0.38, -0.38],
            [side * 0.435, 0.38, 0.28],
            0.022,
        );
    }
    // Fine rear wall seams and floor tile divisions remain separated in depth.
    for i in 1..p.bays {
        panel(
            a,
            o,
            Surface::Rubber,
            "elevator#exterior",
            [-0.46 + 0.92 * i as f32 / p.bays as f32, 0.48, -0.455],
            [0.003, 0.84, 0.003],
        );
    }
}
