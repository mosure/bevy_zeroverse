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
    super::elevator::build(a, o);
}
