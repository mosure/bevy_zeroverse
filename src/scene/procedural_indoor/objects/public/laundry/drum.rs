//! The sill, seal, glass, drum and vented back form one continuous sightline.
use super::*;
const N: u32 = cabinet::SIDES;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, m: &Machine, p: &Pocket, steel: &str) {
    let c = o.kind.class_name();
    let r = p.radius_m;
    let tf = Transform::from_xyz(0., p.center_y_m, m.front_z_m)
        .with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let zback = -p.depth_m;
    let g = a.part(Surface::Metal, steel);
    g.lathe(
        &[
            (r, 0.),
            (r, 0.019),
            (r * 0.97, 0.029),
            (r * 0.875, 0.032),
            (r * 0.854, 0.026),
        ],
        N,
        tf,
    );
    a.part(Surface::Rubber, c).lathe(
        &[
            (r * 0.854, 0.026),
            (r * 0.82, 0.021),
            (r * 0.795, 0.018),
            (r * 0.785, -0.019),
            (r * 0.798, -0.043),
        ],
        N,
        tf,
    );
    let glass = format!("{c}#finish65");
    // Pressed door glass dishes into the boot. A single optical interface uses
    // the renderer's five-millimetre stock thickness and avoids double refraction.
    a.part(Surface::ContainerGlass, &glass).lathe(
        &[
            (r * 0.795, 0.018),
            (r * 0.74, 0.014),
            (r * 0.63, -0.004),
            (r * 0.49, -0.023),
            (r * 0.29, -0.034),
            (0., -0.036),
        ],
        N,
        tf,
    );
    let inner = r * 1.08;
    let g = a.part(Surface::Metal, steel);
    g.lathe(
        &[
            (r * 0.798, -0.043),
            (r * 0.84, -0.055),
            (inner, -0.091),
            (inner, zback),
            (inner * 0.72, zback),
        ],
        N,
        tf,
    );
    g.lathe(
        &[
            (inner * 0.53, zback),
            (inner * 0.25, zback),
            (inner * 0.19, zback + 0.012),
            (0., zback + 0.014),
        ],
        N,
        tf,
    );
    let at = |x, y, z| Vec3::new(x, p.center_y_m + y, m.front_z_m + z);
    let xy = |i: u32| {
        let (s, c) = (TAU * (i % N) as f32 / N as f32).sin_cos();
        Vec2::new(c, -s)
    };
    for i in (0..N).step_by(2) {
        let u = xy(i);
        let v = xy(i + 1);
        let ring = |u: Vec2, rr: f32| at(u.x * inner * rr, u.y * inner * rr, zback);
        let q = [ring(u, 0.53), ring(u, 0.72), ring(v, 0.72), ring(v, 0.53)];
        g.metric_quad(q, Vec3::Z);
        for edge in [[0, 1], [2, 3]] {
            let (s, t) = (q[edge[0]], q[edge[1]]);
            let dir = if edge[0] == 0 { u.perp() } else { -v.perp() };
            g.metric_quad(
                [s, t, t - Vec3::Z * 0.007, s - Vec3::Z * 0.007],
                dir.extend(0.),
            );
        }
        // The next sector is the vent. Its inside and outside cut edges
        // remain visible on oblique rays through the door.
        let end = xy(i + 2);
        for (radius, sign) in [(0.53, 1.), (0.72, -1.)] {
            let (s, t) = (ring(v, radius), ring(end, radius));
            g.metric_quad(
                [s, t, t - Vec3::Z * 0.007, s - Vec3::Z * 0.007],
                ((v + end) * sign).extend(0.),
            );
        }
    }
    a.part(Surface::Rubber, c).lathe(
        &[(inner * 0.73, zback - 0.015), (inner * 0.52, zback - 0.015)],
        N,
        tf,
    );
    // Three longitudinal lifters are joined to the inside of the shell.
    for i in 0..3 {
        let (sy, cx) = (p.phase_rad + i as f32 * TAU / 3.).sin_cos();
        let mid = at(
            (inner - 0.012) * cx,
            (inner - 0.012) * sy,
            (zback - 0.091) * 0.5,
        );
        a.part(Surface::Metal, steel).cuboid(
            Vec3::new(0.030, inner * 0.19, p.depth_m - 0.131),
            0.006,
            Transform::from_translation(mid).with_rotation(Quat::from_rotation_z(sy.atan2(cx))),
        );
    }
    if p.load_fraction > 0. {
        load(a, p, m.front_z_m, inner);
    }
    let sign = if m.hinge_right { 1. } else { -1. };
    for dy in [-0.17, 0.17] {
        a.box_part(
            Surface::Metal,
            steel,
            at(sign * r * 0.974, dy * r, 0.012),
            Vec3::new(0.030, 0.029, 0.028),
            0.003,
        );
    }
    let grip_z = 0.039;
    a.part(Surface::Rubber, c).tube(
        &[
            at(-sign * r * 0.939, -r * 0.30, 0.018),
            at(-sign * r * 0.94, -r * 0.23, grip_z),
            at(-sign * r * 0.94, r * 0.23, grip_z),
            at(-sign * r * 0.939, r * 0.30, 0.018),
        ],
        0.005,
        8,
    );
}

fn load(a: &mut Assembly, p: &Pocket, front: f32, r: f32) {
    // A gravity-settled pile follows the lower drum. Pleats are continuous over
    // the crest and exposed front fold; they change with each load's phase.
    let g = a.part(Surface::Fabric, &format!("clothes#finish{}", p.fabric_slot));
    let level = r * (-0.75 + 0.76 * p.load_fraction);
    let half = (r * r - level * level).sqrt() * 0.985;
    let z0 = front - p.depth_m + 0.018;
    let z1 = front - 0.108;
    let sample = |i: usize, j: usize| {
        let u = i as f32 / 12. * 2. - 1.;
        let t = j as f32 / 6.;
        let x = half * u;
        // Interpolate the actual 32-sided lower drum rather than a circle that
        // would pass through its facets. The fabric rests on its inner wall.
        let theta = (x / r).asin();
        let step = TAU / N as f32;
        let a = (theta / step).floor() * step;
        let b = a + step;
        let (sa, ca) = a.sin_cos();
        let (sb, cb) = b.sin_cos();
        let floor = -r * (ca + (cb - ca) * (x / r - sa) / (sb - sa));
        let envelope = 1. - u * u;
        let ripple = (u * 12. + t * 3.7 + p.phase_rad).sin() * 0.072
            + (u * 7. - t * 4.3 + p.phase_rad * 0.6).cos() * 0.047;
        let y = level + envelope * r * (ripple + 0.065 * (PI * t).sin());
        let z = z0 + (z1 - z0) * t + 0.014 * envelope * (u * 9. + p.phase_rad).sin() * t;
        [
            Vec3::new(x, p.center_y_m + floor, z),
            Vec3::new(x, p.center_y_m + y.max(floor + 0.002), z),
        ]
    };
    let mut old: [_; 13] = std::array::from_fn(|i| sample(i, 0));
    for j in 1..=6 {
        let row: [_; 13] = std::array::from_fn(|i| sample(i, j));
        for i in 0..12 {
            g.metric_quad(
                [old[i][1], row[i][1], row[i + 1][1], old[i + 1][1]],
                Vec3::Y,
            );
            if j == 1 {
                g.metric_quad(
                    [old[i][0], old[i][1], old[i + 1][1], old[i + 1][0]],
                    Vec3::NEG_Z,
                );
            }
            if j == 6 {
                g.metric_quad(
                    [row[i][0], row[i + 1][0], row[i + 1][1], row[i][1]],
                    Vec3::Z,
                );
            }
        }
        for (i, x) in [(0, -1.), (12, 1.)] {
            g.metric_quad([old[i][0], row[i][0], row[i][1], old[i][1]], Vec3::X * x);
        }
        old = row;
    }
}
