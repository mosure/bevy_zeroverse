//! Dimensioned shelf bays with independently sized upright books and flat stacks.
use super::*;

#[derive(Debug, serde::Serialize)]
pub struct ShelfProgram {
    pub levels: usize,
    pub columns: usize,
    pub open_back: bool,
    pub metal: bool,
    pub fill: f32,
    pub panel_m: f32,
}
pub fn parameters(o: &IndoorObject) -> ShelfProgram {
    let mut rng = stream(o.seed, 3050);
    ShelfProgram {
        levels: (o.size.y / rng.random_range(0.27..0.43))
            .round()
            .clamp(2., 8.) as usize,
        columns: (o.size.x / rng.random_range(0.40..0.85))
            .round()
            .clamp(1., 4.) as usize,
        open_back: rng.random_bool(0.38),
        metal: rng.random_bool(0.32),
        fill: rng.random_range(0.25..0.94),
        panel_m: rng.random_range(0.015..0.032),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let frame = if p.metal {
        Surface::Metal
    } else {
        Surface::Wood
    };
    let t = p.panel_m;
    let bottom = 0.065;
    let top = s.y - t * 0.5;
    let pitch = (top - bottom) / p.levels as f32;
    if !p.open_back || o.kind == ObjectKind::Cabinet {
        a.box_part(
            Surface::WoodEdge,
            label,
            Vec3::new(0., s.y * 0.5, -s.z * 0.5 + t * 0.5),
            Vec3::new(s.x, s.y, t),
            0.002,
        );
    }
    for col in 0..=p.columns {
        let x = -s.x * 0.5 + t * 0.5 + col as f32 * (s.x - t) / p.columns as f32;
        a.box_part(
            frame,
            label,
            Vec3::new(x, s.y * 0.5, 0.),
            Vec3::new(t, s.y, s.z),
            0.003,
        );
    }
    for level in 0..=p.levels {
        let y = bottom + level as f32 * pitch;
        a.box_part(
            frame,
            label,
            Vec3::new(0., y, 0.),
            Vec3::new(s.x - t * 2., t, s.z),
            0.003,
        );
        if level == p.levels || o.kind == ObjectKind::Cabinet {
            continue;
        }
        let mut rng = stream(o.seed, 3060 + level as u64);
        for col in 0..p.columns {
            let bay = (s.x - t) / p.columns as f32;
            let left = -s.x * 0.5 + t + col as f32 * bay;
            let mut x = left + 0.012;
            let end = left + bay - t - 0.01;
            while x + 0.085 < end {
                if rng.random_bool(1. - p.fill as f64) {
                    x += rng.random_range(0.05..0.14);
                    continue;
                }
                let cover = [
                    Surface::Art,
                    Surface::FabricAlt,
                    Surface::WoodEdge,
                    Surface::Paper,
                ][rng.random_range(0..4)];
                let book_label = format!(
                    "books#finish{}",
                    rng.random_range(0..super::super::materials::variants::COUNT)
                );
                let flat = rng.random_bool(0.16) && x + 0.22 < end;
                let w = if flat {
                    rng.random_range(0.14..0.21)
                } else {
                    rng.random_range(0.025..0.068)
                };
                let h = if flat {
                    rng.random_range(0.02..0.055)
                } else {
                    rng.random_range(0.55..0.90) * (pitch - t)
                };
                let depth = s.z * rng.random_range(0.63..0.87);
                let z = (s.z - depth) * 0.34;
                let base = y + t * 0.5;
                a.box_part(
                    cover,
                    &book_label,
                    Vec3::new(x + w * 0.5, base + h * 0.5, z),
                    Vec3::new(w, h, depth),
                    0.0015,
                );
                // Paper block leaves visible cover boards and a colored spine.
                a.box_part(
                    Surface::Paper,
                    "books",
                    Vec3::new(x + w * 0.5, base + h * 0.5, z - depth * 0.49),
                    Vec3::new(w * 0.82, h * 0.96, 0.004),
                    0.,
                );
                for fraction in [0.18, 0.82] {
                    a.box_part(
                        Surface::Paper,
                        "books",
                        Vec3::new(x + w * 0.5, base + h * fraction, z + depth * 0.5 + 0.0004),
                        Vec3::new(w * 0.70, (h * 0.02).max(0.002), 0.0006),
                        0.,
                    );
                }
                x += w + rng.random_range(0.002..0.025);
            }
        }
    }
    if o.kind == ObjectKind::Cabinet {
        for col in 0..p.columns {
            let w = (s.x - t * 2.) / p.columns as f32;
            let x = (col as f32 - (p.columns - 1) as f32 * 0.5) * w;
            a.box_part(
                frame,
                label,
                Vec3::new(x, s.y * 0.5 + 0.02, s.z * 0.5 - 0.023),
                Vec3::new(w - 0.006, s.y - 0.065, 0.02),
                0.003,
            );
            a.part(Surface::Metal, label).rod(
                Vec3::new(x + w * 0.32, s.y * 0.45, s.z * 0.5 - 0.006),
                Vec3::new(x + w * 0.32, s.y * 0.63, s.z * 0.5 - 0.006),
                0.005,
            );
        }
    }
}
