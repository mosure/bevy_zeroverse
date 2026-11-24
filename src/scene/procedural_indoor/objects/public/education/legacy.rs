//! Reconstruction of banks archived before fitted locker hardware.
use super::*;
#[allow(clippy::too_many_arguments)]
pub(super) fn cell(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &LockerProgram,
    b: &LockerBay,
    row: usize,
    (lo, hi): (f32, f32),
    r: &mut ChaCha8Rng,
) {
    let t = p.panel_m;
    let l = o.kind.class_name();
    let rear = -o.size.z * 0.5;
    let front = rear + p.body_depth_m;
    if (matches!(p.front, 2 | 3) || b.open[row]) && r.random_bool(p.contents_probability as f64) {
        use super::super::super::storage::bookcase::contents::{self, Content};
        // A few independently bound school/library volumes, not a
        // recursively generated pile of spiral notebooks in every cell.
        let count = r.random_range(1..=3);
        let start = b.left + b.width * r.random_range(0.10..0.30);
        for i in 0..count {
            let size = Vec3::new(
                (b.width * 0.12).min(0.045),
                ((hi - lo) * r.random_range(0.30..0.70)).min(0.34),
                (p.body_depth_m * r.random_range(0.45..0.66)).min(0.28),
            );
            contents::build(
                a,
                o,
                &Content::upright_book(
                    r.random(),
                    size,
                    Vec3::new(
                        start + size.x * (i as f32 + 0.5) * 1.1,
                        lo,
                        (rear + front) * 0.5 + r.random_range(-0.025..0.025),
                    ),
                    0.,
                ),
            );
        }
    }
    if p.front == 3 {
        return;
    }
    let mut door = Assembly::default();
    let w = b.width - 0.006;
    let h = hi - lo - 0.008;
    let center = Vec3::new(b.left + b.width * 0.5, (lo + hi) * 0.5, front + t * 0.5);
    if p.front == 2 {
        let rail = 0.025_f32.min(w * 0.14);
        for side in [-1., 1.] {
            door.box_part(
                p.finish,
                l,
                center + Vec3::X * side * (w - rail) * 0.5,
                Vec3::new(rail, h, t),
                0.002,
            );
            door.box_part(
                p.finish,
                l,
                center + Vec3::Y * side * (h - rail) * 0.5,
                Vec3::new(w - 2. * rail, rail, t),
                0.002,
            );
        }
        door.box_part(
            Surface::ContainerGlass,
            l,
            center,
            Vec3::new(w - 2. * rail - 0.002, h - 2. * rail - 0.002, 0.004),
            0.0005,
        );
    } else {
        door.box_part(p.finish, l, center, Vec3::new(w, h, t), 0.003);
    }
    if p.front == 0 {
        for sign in [-1., 1.] {
            for i in 0..p.vent_rows {
                door.box_part(
                    Surface::Ink,
                    l,
                    center + Vec3::new(0., sign * (h * 0.33 - i as f32 * 0.009), t * 0.5 + 0.001),
                    Vec3::new(w * 0.54, 0.0035, 0.001),
                    0.,
                );
            }
        }
    }
    let hc = center + Vec3::new(w * 0.32, 0., t * 0.5 + 0.005);
    match p.handle {
        0 => door.box_part(
            p.accent,
            l,
            hc,
            Vec3::new(0.018, 0.052_f32.min(h * 0.18), 0.010),
            0.003,
        ),
        1 => {
            for sign in [-1., 1.] {
                door.box_part(
                    p.accent,
                    l,
                    hc + Vec3::Y * sign * 0.028,
                    Vec3::new(0.012, 0.008, 0.012),
                    0.001,
                );
            }
            door.part(p.accent, l).rod(
                hc + Vec3::new(0., -0.028, 0.008),
                hc + Vec3::new(0., 0.028, 0.008),
                0.004,
            );
        }
        _ => door.box_part(Surface::Ink, l, hc, Vec3::new(0.026, 0.046, 0.010), 0.004),
    }
    let hinge = Vec3::new(b.left + 0.003, 0., front);
    let q = Quat::from_rotation_y(-b.angles[row]);
    append_at(
        a,
        door,
        Transform::from_translation(hinge - q * hinge).with_rotation(q),
    );
}
