use super::*;

pub(super) fn content_size(kind: ObjectKind, seed: u64, r: &mut impl Rng) -> Vec3 {
    if matches!(kind, ObjectKind::Bread | ObjectKind::Dishware) {
        super::super::kitchen::dimensions(kind, seed)
    } else if kind == ObjectKind::Sculpture {
        // Display-case miniatures, distinct from floor/plinth-scale exhibits.
        Vec3::new(
            r.random_range(0.12..0.28),
            r.random_range(0.15..0.45),
            r.random_range(0.10..0.26),
        )
    } else {
        super::super::super::items::sizing::loose(kind, r).expect("case content dimensions")
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    use ObjectKind::*;
    match o.kind {
        Banquette => {
            // Overall height describes the back, not the sitting height. Tall
            // backs must not lift a cafe seat above its adjacent table.
            let seat_top = 0.46 + p.curvature * 0.025;
            let padding = 0.065 + (1. - p.taper) * 0.08;
            let base = seat_top - padding;
            let back_bottom = seat_top - padding * 0.40;
            a.box_part(
                p.finish,
                "bench",
                Vec3::Y * base * 0.5,
                Vec3::new(o.size.x * 0.96, base, o.size.z * 0.92),
                0.004,
            );
            for i in 0..p.bays {
                let x = -0.5 + (i as f32 + 0.5) / p.bays as f32;
                a.box_part(
                    Surface::Fabric,
                    "bench",
                    Vec3::new(x * o.size.x, seat_top - padding * 0.5, o.size.z * 0.055),
                    Vec3::new(o.size.x * 0.97 / p.bays as f32, padding, o.size.z * 0.88),
                    0.045,
                );
                a.box_part(
                    Surface::Fabric,
                    "bench",
                    Vec3::new(
                        x * o.size.x,
                        (o.size.y + back_bottom) * 0.5,
                        -o.size.z * 0.37,
                    ),
                    Vec3::new(
                        o.size.x * 0.97 / p.bays as f32,
                        o.size.y - back_bottom,
                        o.size.z * 0.23,
                    ),
                    0.045,
                );
            }
        }
        _ => unreachable!(),
    }
}
