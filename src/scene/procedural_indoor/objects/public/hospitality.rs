use super::*;

pub(super) fn content_size(kind: ObjectKind, seed: u64, r: &mut impl Rng) -> Vec3 {
    if matches!(kind, ObjectKind::Bread | ObjectKind::Dishware) {
        super::super::kitchen::dimensions(kind, seed)
    } else {
        let w = r.random_range(0.075..0.22);
        Vec3::new(w, w * r.random_range(1.1..1.7), w)
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    use ObjectKind::*;
    let label = o.kind.class_name();
    match o.kind {
        ServiceCounter => {
            panel(
                a,
                o,
                Surface::Rubber,
                label,
                [0., 0.04, 0.],
                [0.90, 0.08, 0.84],
            );
            panel(a, o, p.finish, label, [0., 0.51, -0.02], [0.96, 0.86, 0.88]);
            panel(
                a,
                o,
                Surface::Ceramic,
                label,
                [0., 0.975, 0.],
                [1., 0.05, 1.],
            );
            for i in 0..p.bays {
                let x = -0.48 + 0.96 * (i as f32 + 0.5) / p.bays as f32;
                panel(
                    a,
                    o,
                    p.finish,
                    label,
                    [x, 0.51, 0.434],
                    [0.94 / p.bays as f32, 0.80, 0.018],
                );
                panel(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [x, 0.78, 0.455],
                    [0.09, 0.012, 0.025],
                );
            }
        }
        DisplayCase => {
            panel(a, o, p.finish, label, [0., 0.20, 0.], [0.96, 0.40, 0.96]);
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.015, 0.],
                [0.90, 0.03, 0.88],
            );
            for x in [-0.48, 0.48] {
                for z in [-0.48, 0.48] {
                    panel(
                        a,
                        o,
                        Surface::Metal,
                        label,
                        [x, 0.68, z],
                        [0.025, 0.64, 0.025],
                    );
                }
            }
            for x in [-0.49, 0.49] {
                panel(
                    a,
                    o,
                    Surface::ContainerGlass,
                    label,
                    [x, 0.70, 0.],
                    [0.006, 0.58, 0.95],
                );
            }
            for z in [-0.49, 0.49] {
                panel(
                    a,
                    o,
                    Surface::ContainerGlass,
                    label,
                    [0., 0.70, z],
                    [0.96, 0.58, 0.006],
                );
            }
            panel(
                a,
                o,
                Surface::ContainerGlass,
                label,
                [0., 0.994, 0.],
                [0.98, 0.008, 0.98],
            );
            for i in 0..p.shelves {
                let y = 0.42 + i as f32 * 0.48 / p.shelves as f32;
                panel(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [0., y, 0.],
                    [0.92, 0.012, 0.90],
                );
                if p.contents {
                    use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};
                    let mut r = stream(o.seed, 8210 + i as u64);
                    let pattern = SurfacePattern::sample(&mut r);
                    let mut occupied: Vec<Footprint> = Vec::with_capacity(p.bays);
                    for j in 0..p.bays {
                        let mut child = o.clone();
                        child.kind = if o.variant >= 16 {
                            if r.random_bool(0.5) {
                                Sculpture
                            } else {
                                Vase
                            }
                        } else if r.random_bool(0.5) {
                            Bread
                        } else {
                            Dishware
                        };
                        child.seed = o.seed.wrapping_add((i * 8 + j) as u64);
                        child.size = content_size(child.kind, child.seed, &mut r);
                        let headroom = (0.48 / p.shelves as f32 - 0.015) * o.size.y;
                        if child.size.y > headroom {
                            continue;
                        }
                        child.position = Vec3::ZERO;
                        let half = o.size.xz() * Vec2::new(0.44, 0.43);
                        for _ in 0..16 {
                            let yaw = pattern.yaw(&mut r);
                            let Some(proposal) =
                                pattern.propose((-half, half), child.size.xz(), yaw, 0.015, &mut r)
                            else {
                                continue;
                            };
                            if occupied.iter().any(|q| proposal.overlaps(*q, 0.012)) {
                                continue;
                            }
                            occupied.push(proposal);
                            append_at(
                                a,
                                super::super::build_object(&child),
                                Transform::from_xyz(
                                    proposal.center.x,
                                    (y + 0.006) * o.size.y,
                                    proposal.center.y,
                                )
                                .with_rotation(Quat::from_rotation_y(yaw)),
                            );
                            break;
                        }
                    }
                }
            }
        }
        TrayReturn => {
            panel(a, o, p.finish, label, [0., 0.03, 0.], [0.96, 0.06, 0.94]);
            for x in [-0.47, 0.47] {
                panel(a, o, p.finish, label, [x, 0.51, 0.], [0.06, 0.98, 1.]);
            }
            panel(a, o, p.finish, label, [0., 0.51, -0.47], [0.88, 0.98, 0.06]);
            panel(a, o, p.finish, label, [0., 0.98, 0.], [0.88, 0.04, 1.]);
            for i in 0..=p.shelves {
                let y = 0.16 + i as f32 * 0.64 / p.shelves as f32;
                panel(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [0., y, 0.],
                    [0.86, 0.012, 0.90],
                );
                if p.contents {
                    panel(
                        a,
                        o,
                        Surface::Plastic,
                        "dishware",
                        [0., y + 0.015, 0.03],
                        [0.70, 0.016, 0.68],
                    );
                }
            }
        }
        DrinkDispenser => {
            panel(
                a,
                o,
                Surface::Metal,
                "appliance",
                [0., 0.12, 0.],
                [0.90, 0.24, 0.9],
            );
            a.part(Surface::ContainerGlass, "appliance").cylinder(
                o.size.x.min(o.size.z) * 0.39,
                o.size.y * 0.60,
                Transform::from_xyz(0., o.size.y * 0.56, -o.size.z * 0.06),
            );
            a.part(Surface::Liquid, "appliance").cylinder(
                o.size.x.min(o.size.z) * 0.37,
                o.size.y * (0.26 + 0.28 * p.opening),
                Transform::from_xyz(0., o.size.y * (0.39 + 0.14 * p.opening), -o.size.z * 0.06),
            );
            panel(
                a,
                o,
                Surface::Metal,
                "appliance",
                [0., 0.92, -0.06],
                [0.86, 0.06, 0.84],
            );
            rod(
                a,
                o,
                Surface::Metal,
                "faucet",
                [0., 0.31, 0.28],
                [0., 0.31, 0.45],
                0.012,
            );
            rod(
                a,
                o,
                Surface::Metal,
                "faucet",
                [0., 0.31, 0.45],
                [0., 0.25, 0.45],
                0.012,
            );
            panel(
                a,
                o,
                Surface::Rubber,
                "appliance",
                [0., 0.19, 0.27],
                [0.54, 0.015, 0.42],
            );
        }
        ExtractionHood => {
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0., 0.10, 0.),
                Vec3::new(o.size.x, 0.16, o.size.z),
                0.012,
            );
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0., (0.18 + o.size.y) * 0.5, -o.size.z * 0.14),
                Vec3::new(o.size.x * 0.32, o.size.y - 0.18, o.size.z * 0.36),
                0.006,
            );
            for i in 0..12 {
                a.box_part(
                    Surface::Rubber,
                    label,
                    Vec3::new(o.size.x * (-0.42 + i as f32 * 0.84 / 11.), 0.018, 0.),
                    Vec3::new(0.012, 0.004, o.size.z * 0.76),
                    0.,
                );
            }
            for x in [-0.32, 0.32] {
                a.box_part(
                    Surface::Light,
                    "lamp",
                    Vec3::new(o.size.x * x, 0.016, o.size.z * 0.28),
                    Vec3::new(0.05, 0.003, 0.05),
                    0.,
                );
            }
        }
        Banquette => {
            panel(a, o, p.finish, "bench", [0., 0.20, 0.], [0.96, 0.40, 0.92]);
            for i in 0..p.bays {
                let x = -0.5 + (i as f32 + 0.5) / p.bays as f32;
                a.box_part(
                    Surface::Fabric,
                    "bench",
                    Vec3::new(x * o.size.x, o.size.y * 0.44, o.size.z * 0.055),
                    Vec3::new(
                        o.size.x * 0.97 / p.bays as f32,
                        o.size.y * 0.12,
                        o.size.z * 0.88,
                    ),
                    0.045,
                );
                a.box_part(
                    Surface::Fabric,
                    "bench",
                    Vec3::new(x * o.size.x, o.size.y * 0.74, -o.size.z * 0.37),
                    Vec3::new(
                        o.size.x * 0.97 / p.bays as f32,
                        o.size.y * 0.50,
                        o.size.z * 0.23,
                    ),
                    0.045,
                );
            }
        }
        _ => unreachable!(),
    }
}
