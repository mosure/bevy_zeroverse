//! Actual worktop holes, inset basins, independent taps and washing accessories.
use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram) {
    let s = o.size;
    let top = s.y * 0.69;
    let (d, z) = body(o, p);
    cabinet::open_shell(
        a,
        Vec3::new(s.x - 0.02, top - 0.11, d),
        0.11,
        z,
        p.finish,
        "cabinet",
    );
    a.box_part(
        Surface::Rubber,
        "cabinet",
        Vec3::new(0., 0.055, z - 0.015),
        Vec3::new(s.x - 0.09, 0.11, d - 0.05),
        0.004,
    );
    cabinet::doors(
        a,
        s.x - 0.035,
        0.12,
        top - 0.15,
        z + d * 0.5 - 0.012,
        p,
        "cabinet",
    );
    let width = if p.draining_board {
        s.x * 0.59
    } else {
        s.x * 0.78
    };
    let cx = if p.draining_board { -s.x * 0.105 } else { 0. };
    let depth = d * 0.58;
    let cz = z + d * 0.035;
    let lo = Vec2::new(cx - width * 0.5, cz - depth * 0.5);
    let hi = Vec2::new(cx + width * 0.5, cz + depth * 0.5);
    // Four disjoint slabs replace the countertop around the rectangular aperture.
    for (x0, x1, z0, z1) in [
        (-s.x * 0.5, lo.x, z - d * 0.5, z + d * 0.5),
        (hi.x, s.x * 0.5, z - d * 0.5, z + d * 0.5),
        (lo.x, hi.x, z - d * 0.5, lo.y),
        (lo.x, hi.x, hi.y, z + d * 0.5),
    ] {
        a.box_part(
            p.top,
            "counter",
            Vec3::new((x0 + x1) * 0.5, top - 0.017, (z0 + z1) * 0.5),
            Vec3::new(x1 - x0, 0.034, z1 - z0),
            0.002,
        );
    }
    let floor = top - s.y * 0.16;
    let material = if p.faucet.is_multiple_of(2) {
        Surface::Metal
    } else {
        Surface::Ceramic
    };
    let wall = 0.016;
    let unit = width / p.basins as f32;
    for basin in 0..p.basins {
        let x = lo.x + unit * (basin as f32 + 0.5);
        a.box_part(
            material,
            "sink",
            Vec3::new(x, floor, cz),
            Vec3::new(unit, 0.014, depth),
            0.006,
        );
        for side in [-1., 1.] {
            a.box_part(
                material,
                "sink",
                Vec3::new(x + side * (unit - wall) * 0.5, (top + floor) * 0.5, cz),
                Vec3::new(wall, top - floor, depth),
                0.006,
            );
            a.box_part(
                material,
                "sink",
                Vec3::new(x, (top + floor) * 0.5, cz + side * (depth - wall) * 0.5),
                Vec3::new(unit - 2. * wall, top - floor, wall),
                0.006,
            );
        }
        // Drain is above the bowl floor, not coplanar with it.
        a.part(Surface::Chrome, "sink").cylinder(
            unit.min(depth) * 0.065,
            0.004,
            Transform::from_xyz(x, floor + 0.009, cz),
        );
        for k in -1..=1 {
            a.box_part(
                Surface::Ink,
                "sink",
                Vec3::new(x + k as f32 * 0.006, floor + 0.0115, cz),
                Vec3::new(0.002, 0.001, 0.027),
                0.0001,
            );
        }
        if p.sink_dishes > 0 {
            props::dish(
                a,
                Vec3::new(x, floor + 0.010, cz),
                Vec3::new(unit * 0.70, (top - floor) * 0.66, depth * 0.68),
                if basin == 0 { 0 } else { 1 },
                p.sink_dishes,
            );
        }
    }
    for side in [-1., 1.] {
        a.box_part(
            material,
            "sink",
            Vec3::new(cx + side * width * 0.5, top + 0.003, cz),
            Vec3::new(0.018, 0.006, depth + 0.026),
            0.002,
        );
        a.box_part(
            material,
            "sink",
            Vec3::new(cx, top + 0.003, cz + side * depth * 0.5),
            Vec3::new(width - 0.018, 0.006, 0.018),
            0.002,
        );
    }
    let mount = Vec3::new(cx, top + 0.008, z - d * 0.38);
    let fh = s.y * if p.faucet == 2 { 0.15 } else { 0.23 };
    let reach = d * 0.36;
    let points = match p.faucet {
        0 => (0..=20)
            .map(|i| {
                let t = i as f32 / 20.;
                mount + Vec3::new(0., fh * (PI * t * 0.82).sin(), reach * t)
            })
            .collect::<Vec<_>>(),
        1 => vec![
            mount,
            mount + Vec3::Y * fh,
            mount + Vec3::new(0., fh, reach),
            mount + Vec3::new(0., fh * 0.88, reach),
        ],
        2 => vec![
            mount,
            mount + Vec3::Y * fh * 0.65,
            mount + Vec3::new(0., fh, reach * 0.8),
            mount + Vec3::new(0., fh * 0.82, reach),
        ],
        _ => (0..=20)
            .map(|i| {
                let t = i as f32 / 20.;
                mount
                    + Vec3::new(
                        (t * PI).sin() * s.x * 0.06,
                        fh * (PI * t * 0.83).sin(),
                        reach * t,
                    )
            })
            .collect(),
    };
    pipe(a, Surface::Chrome, "faucet", &points, 0.012);
    a.part(Surface::Chrome, "faucet")
        .cylinder(0.026, 0.018, Transform::from_translation(mount));
    let handle_count = if p.faucet_handles == 1 { 2 } else { 1 };
    for i in 0..handle_count {
        let h = mount + Vec3::new(if i == 0 { -0.075 } else { 0.075 }, 0., 0.);
        a.part(Surface::Chrome, "faucet").cylinder(
            0.018,
            0.035,
            Transform::from_translation(h + Vec3::Y * 0.014),
        );
        if p.faucet_handles == 1 {
            pipe(
                a,
                Surface::Chrome,
                "faucet",
                &[
                    h + Vec3::new(-0.025, 0.034, 0.),
                    h + Vec3::new(0.025, 0.034, 0.),
                ],
                0.004,
            );
        } else if p.faucet_handles == 2 {
            pipe(
                a,
                Surface::Chrome,
                "faucet",
                &[h + Vec3::Y * 0.034, h + Vec3::new(0.045, 0.034, 0.)],
                0.007,
            );
        } else {
            pipe(
                a,
                Surface::Chrome,
                "faucet",
                &[h + Vec3::Y * 0.03, h + Vec3::new(0., 0.075, 0.015)],
                0.006,
            );
        }
    }
    let soap_size = Vec3::new(0.065, (s.y - top) * 0.44, 0.075);
    if p.hand_soap {
        props::soap(
            a,
            Vec3::new(-s.x * 0.405, top + 0.001, z - d * 0.35),
            soap_size,
            true,
        );
    }
    if p.dish_soap {
        props::soap(
            a,
            Vec3::new(s.x * 0.405, top + 0.001, z - d * 0.35),
            soap_size * Vec3::new(1., 1.10, 1.),
            false,
        );
    }
    if p.sponge {
        a.box_part(
            Surface::Fabric,
            "cleaning_supply",
            Vec3::new(s.x * 0.40, top + 0.013, z + d * 0.35),
            Vec3::new(s.x * 0.10, 0.024, d * 0.13),
            0.004,
        );
    }
    if p.draining_board {
        props::rack(
            a,
            Vec3::new(s.x * 0.35, top + 0.001, cz),
            Vec3::new(s.x * 0.23, (s.y - top) * 0.58, depth * 0.82),
            p.sink_dishes,
        );
    }
}
