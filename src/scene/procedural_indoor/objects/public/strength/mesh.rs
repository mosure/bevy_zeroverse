//! Open square-section uprights, their through-bores, bar fittings and annular plates.
use super::*;

pub(super) fn upright(g: &mut Geometry, rack: PowerRack, x: f32) {
    let w = rack.post * 0.5;
    let z = rack.work_z();
    let shell = |u: f32, y: f32, v: f32| Vec3::new(x + u, y, z + v);
    let bottom = 0.066;
    let top = rack.size.y;
    for sign in [-1., 1.] {
        let a = sign * w;
        g.metric_quad(
            [
                shell(a, bottom, -w),
                shell(a, top, -w),
                shell(a, top, w),
                shell(a, bottom, w),
            ],
            Vec3::X * sign,
        );
    }
    for (y, out) in [(bottom, -Vec3::Y), (top, Vec3::Y)] {
        g.metric_quad(
            [
                shell(-w, y, -w),
                shell(w, y, -w),
                shell(w, y, w),
                shell(-w, y, w),
            ],
            out,
        );
    }
    let h = rack.pitch * 0.5;
    for face in [-1., 1.] {
        let p = |u, y| shell(u, y, face * w);
        for (a, b) in [
            (bottom, rack.first_hole - h),
            (rack.pin_y(rack.holes - 1) + h, top),
        ] {
            g.metric_quad([p(-w, a), p(w, a), p(w, b), p(-w, b)], face * Vec3::Z);
        }
    }
    // The inside walls connect both faces. The front/rear rings tile the post
    // exactly, so neither decals nor overlapping faces are needed for the holes.
    const DIRECTIONS: [Vec2; 8] = [
        Vec2::X,
        Vec2::new(1., 1.),
        Vec2::Y,
        Vec2::new(-1., 1.),
        Vec2::NEG_X,
        Vec2::new(-1., -1.),
        Vec2::NEG_Y,
        Vec2::new(1., -1.),
    ];
    let outer = DIRECTIONS.map(|v| v * Vec2::new(w, h));
    let inner = DIRECTIONS.map(|v| v.normalize() * 0.009);
    for i in 0..rack.holes {
        let y = rack.pin_y(i);
        let at = |v: Vec2, face: f32| shell(v.x, y + v.y, face * w);
        for j in 0..8 {
            let k = (j + 1) % 8;
            for f in [-1., 1.] {
                g.metric_quad(
                    [
                        at(outer[j], f),
                        at(outer[k], f),
                        at(inner[k], f),
                        at(inner[j], f),
                    ],
                    Vec3::Z * f,
                );
            }
            g.metric_quad(
                [
                    at(inner[j], -1.),
                    at(inner[j], 1.),
                    at(inner[k], 1.),
                    at(inner[k], -1.),
                ],
                -(inner[j] + inner[k]).extend(0.),
            );
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let q = PowerRack::for_object(o);
    let (coat, label) = Finish::Coat.key();
    q.frame(|p, s| a.box_part(coat, label, p, s, 0.003));
    let bar = q.bar_center();
    for side in [-1., 1.] {
        let x = side * q.upright_span * 0.5;
        upright(a.part(coat, label), q, x);
        let face = q.work_z() + q.post * 0.5;
        let y = q.pin_y(q.cup_pin);
        // Collar, pan, replaceable contact liner and forward keeper.
        let (rubber, liner) = Finish::Rubber.key();
        a.box_part(
            coat,
            label,
            Vec3::new(x, y, face + 0.009),
            Vec3::new(q.post + 0.012, 0.108, 0.018),
            0.002,
        );
        a.box_part(
            coat,
            label,
            Vec3::new(x, y + 0.013, face + 0.064),
            Vec3::new(q.post * 0.82, 0.020, 0.092),
            0.002,
        );
        a.box_part(
            rubber,
            liner,
            Vec3::new(x, y + 0.0265, face + 0.064),
            Vec3::new(q.post * 0.76, 0.007, 0.088),
            0.001,
        );
        a.box_part(
            coat,
            label,
            Vec3::new(x, y + 0.049, face + 0.116),
            Vec3::new(q.post * 0.82, 0.080, 0.012),
            0.002,
        );
        // Retaining pins pass through the bored uprights. Their outer heads
        // sit against the back of the post rather than floating on a decal.
        let (steel, pin) = Finish::Steel.key();
        let back = q.work_z() - q.post * 0.5;
        for slot in [q.cup_pin, q.safety_pin] {
            let y = q.pin_y(slot);
            a.part(steel, pin).rod(
                Vec3::new(x, y, back - 0.009),
                Vec3::new(x, y, face + 0.008),
                0.0075,
            );
            a.part(steel, pin).rod(
                Vec3::new(x, y, back - 0.012),
                Vec3::new(x, y, back - 0.003),
                0.012,
            );
        }
        if !q.full_cage {
            a.part(coat, label).rod(
                Vec3::new(x, q.pin_y(q.safety_pin) - 0.20, face),
                Vec3::new(x, q.pin_y(q.safety_pin) - 0.026, q.guard_end_z() - 0.11),
                0.017,
            );
        }
    }
    let cross_z = if q.full_cage {
        q.opposite_z()
    } else {
        q.work_z()
    };
    let edge = (q.upright_span - q.post) * 0.5;
    a.part(coat, label).rod(
        Vec3::new(-edge, q.size.y - 0.12, cross_z),
        Vec3::new(edge, q.size.y - 0.12, cross_z),
        0.016,
    );
    // A single continuous turned shaft. Grip bands share end rings with the
    // smooth sections and retain metric UVs for the milled crosshatch.
    let axis = Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);
    let tf = Transform::from_translation(bar).with_rotation(axis);
    let rr = q.shaft_radius();
    let (metal, smooth) = Finish::Steel.key();
    let (milled, grip) = Finish::Grip.key();
    let mut last = -INNER_SLEEVE;
    for (end, is_grip) in [
        (-0.44, true),
        (-0.429, false),
        (-0.16, true),
        (0.16, false),
        (0.429, true),
        (0.44, false),
        (INNER_SLEEVE, true),
    ] {
        let g = if is_grip {
            a.part(milled, grip)
        } else {
            a.part(metal, smooth)
        };
        g.lathe(&[(rr, last), (rr, end)], 16, tf);
        last = end;
    }
    for side in [-1., 1.] {
        let end = q.size.x * 0.5;
        let mount = Transform::from_translation(bar)
            .with_rotation(Quat::from_rotation_z(-side * std::f32::consts::FRAC_PI_2));
        a.part(metal, smooth).lathe(
            &[
                (rr, INNER_SLEEVE),
                (0.041, INNER_SLEEVE),
                (0.041, INNER_SLEEVE + 0.028),
                (0.039, INNER_SLEEVE + 0.030),
                (0.025, INNER_SLEEVE + 0.030),
                (0.025, end - 0.002),
                (0.023, end),
                (0., end),
            ],
            20,
            mount,
        );
        let next = q.discs(|x, r, w, stock| {
            let (s, l) = stock.key();
            let at = Transform::from_translation(bar + Vec3::X * (side * x))
                .with_rotation(mount.rotation);
            let h = w * 0.5;
            let lip = (w * 0.13).min(0.006);
            // A flat stacking back and a dished, lipped outer face, facing away
            // from the rack on both sleeves. Small change plates need fewer segments.
            a.part(s, l).lathe(
                &[
                    (BORE, -h),
                    (r - lip, -h),
                    (r, -h + lip),
                    (r, h - lip),
                    (r - lip, h),
                    (r * 0.90, h),
                    (r * 0.86, h - lip),
                    (r * 0.36, h - lip),
                    (r * 0.33, h),
                    (BORE, h),
                    (BORE, -h),
                ],
                if r < 0.16 { 20 } else { 24 },
                at,
            );
        });
        if q.plate_pairs_kg[0] != 0 {
            let (rubber, l) = Finish::Rubber.key();
            a.part(rubber, l).lathe(
                &[
                    (BORE, next),
                    (0.043, next),
                    (0.046, next + 0.006),
                    (0.046, next + 0.027),
                    (0.041, next + 0.033),
                    (BORE, next + 0.033),
                    (BORE, next),
                ],
                12,
                mount,
            );
        }
    }
}
