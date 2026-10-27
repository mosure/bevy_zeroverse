//! Metric locker compartments, manufactured fronts and individually sampled use states.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct LockerBay {
    pub left: f32,
    pub width: f32,
    pub levels: Vec<f32>,
    pub open: Vec<bool>,
    pub angles: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LockerProgram {
    /// Steel vented, laminate flush, framed glazed, or open cubbies.
    pub front: u8,
    pub handle: u8,
    pub feet_m: f32,
    pub panel_m: f32,
    pub body_depth_m: f32,
    pub vent_rows: usize,
    pub contents_probability: f32,
    pub bays: Vec<LockerBay>,
    pub finish: Surface,
    pub accent: Surface,
}

pub fn parameters(o: &IndoorObject) -> LockerProgram {
    let mut r = stream(o.seed, 0x4c4f_434b_4552_4745);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4c4f_434b_4552_4649);
    let front = r.random_range(0..4);
    let t = r.random_range(0.014..0.023);
    let feet = r.random_range(0.035..0.13);
    let n = (o.size.x / r.random_range(0.25..0.44))
        .round()
        .clamp(2., 6.) as usize;
    // Equal modules remain common in manufactured lockers, mixed-width banks
    // and independently tiered compartments also occur. Never perturb joins.
    let mixed = r.random_bool(0.48);
    let weights: Vec<f32> = (0..n)
        .map(|_| if mixed { r.random_range(0.8..1.25) } else { 1. })
        .collect();
    let sum: f32 = weights.iter().sum();
    let mut x = -o.size.x * 0.5 + t;
    let depth = o.size.z * r.random_range(0.72..0.84);
    let rows = r.random_range(1..=4);
    let openness = r.random_range(0.0..0.48);
    let mut bays = Vec::with_capacity(n);
    for weight in weights {
        let width = (o.size.x - t * (n + 1) as f32) * weight / sum;
        let rows = if mixed { r.random_range(1..=4) } else { rows };
        let rows = rows.min(((o.size.y - feet) / 0.31).floor().max(1.) as usize);
        let levels: Vec<f32> = (0..=rows)
            .map(|i| feet + t + (o.size.y - feet - t * 2.) * i as f32 / rows as f32)
            .collect();
        let open: Vec<bool> = (0..rows)
            .map(|_| front != 3 && r.random_bool(openness))
            .collect();
        // Opening reserve belongs to the admitted envelope. A single hinged
        // leaf replaces the previous pair of shrinking sliding panels.
        let limit = ((o.size.z - depth - 0.030) / width)
            .clamp(0.02, 0.65)
            .asin();
        let angles = open
            .iter()
            .map(|&open| {
                if open {
                    r.random_range(0.35..1.) * limit
                } else {
                    0.
                }
            })
            .collect();
        bays.push(LockerBay {
            left: x,
            width,
            levels,
            open,
            angles,
        });
        x += width + t;
    }
    LockerProgram {
        front,
        handle: r.random_range(0..3),
        feet_m: feet,
        panel_m: t,
        body_depth_m: depth,
        vent_rows: r.random_range(2..=6),
        contents_probability: r.random_range(0.15..0.8),
        bays,
        finish: if front == 0 {
            Surface::Plastic
        } else {
            [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)]
        },
        accent: [Surface::Metal, Surface::Chrome, Surface::Plastic][f.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let t = p.panel_m;
    let rear = -s.z * 0.5;
    let front = rear + p.body_depth_m;
    let low = p.feet_m;
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., low * 0.5, (rear + front) * 0.5),
        Vec3::new(s.x * 0.88, low, p.body_depth_m * 0.85),
        0.003,
    );
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., (low + s.y) * 0.5, rear + t * 0.5),
        Vec3::new(s.x, s.y - low, t),
        0.002,
    );
    for y in [low + t * 0.5, s.y - t * 0.5] {
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., y, (rear + t + front) * 0.5),
            Vec3::new(s.x, t, p.body_depth_m - t),
            0.002,
        );
    }
    for x in std::iter::once(-s.x * 0.5 + t * 0.5)
        .chain(p.bays.iter().map(|b| b.left + b.width + t * 0.5))
    {
        a.box_part(
            p.finish,
            l,
            Vec3::new(x, (low + s.y) * 0.5, (rear + t + front) * 0.5),
            Vec3::new(t, s.y - low - 2. * t, p.body_depth_m - t),
            0.002,
        );
    }
    let mut r = stream(o.seed, 0x4c4f_434b_434f_4e54);
    for b in &p.bays {
        for (row, limits) in b.levels.windows(2).enumerate() {
            let bottom = limits[0];
            let top = limits[1];
            if row > 0 {
                a.box_part(
                    p.finish,
                    l,
                    Vec3::new(b.left + b.width * 0.5, bottom, (rear + t + front) * 0.5),
                    Vec3::new(b.width, t, p.body_depth_m - t),
                    0.002,
                );
            }
            let lo = bottom + if row > 0 { t * 0.5 } else { 0. };
            let hi = top - if row + 1 < b.open.len() { t * 0.5 } else { 0. };
            if (matches!(p.front, 2 | 3) || b.open[row])
                && r.random_bool(p.contents_probability as f64)
            {
                use super::super::storage::bookcase::contents::{self, Content};
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
                        &Content {
                            kind: ObjectKind::Books,
                            upright: true,
                            seed: r.random(),
                            size,
                            position: Vec3::new(
                                start + size.x * (i as f32 + 0.5) * 1.1,
                                lo,
                                (rear + front) * 0.5 + r.random_range(-0.025..0.025),
                            ),
                            yaw: 0.,
                        },
                    );
                }
            }
            if p.front == 3 {
                continue;
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
                            center
                                + Vec3::new(
                                    0.,
                                    sign * (h * 0.33 - i as f32 * 0.009),
                                    t * 0.5 + 0.001,
                                ),
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
    }
}
