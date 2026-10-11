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
    /// Hinged doors, drawers, mixed front, open bays, sliding panels, slatted doors.
    pub front: u8,
    pub handle: u8,
    pub open_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> ShelfProgram {
    let mut rng = stream(o.seed, 3050);
    let mut program = ShelfProgram {
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
        front: rng.random_range(0..6),
        handle: rng.random_range(0..4),
        open_fraction: rng.random_range(0.18..0.62),
    };
    // Mixed fronts and sliding leaves need at least two physical bays; a
    // one-bay "mixed" cabinet would be indistinguishable from the drawer front.
    if o.kind == ObjectKind::Cabinet && matches!(program.front, 2 | 4) {
        program.columns = program.columns.max(2);
    }
    program
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
    let closed_back = !p.open_back || o.kind == ObjectKind::Cabinet;
    let frame_depth = s.z - if closed_back { t } else { 0. };
    let frame_z = if closed_back { t * 0.5 } else { 0. };
    if closed_back {
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
            Vec3::new(x, s.y * 0.5, frame_z),
            Vec3::new(t, s.y, frame_depth),
            0.003,
        );
    }
    for level in 0..=p.levels {
        let y = bottom + level as f32 * pitch;
        // Shelves butt between uprights and the rear panel. Overlapping full
        // width boards put competing faces on the exterior/front frame planes.
        let bay = (s.x - t) / p.columns as f32;
        for col in 0..p.columns {
            a.box_part(
                frame,
                label,
                Vec3::new(
                    -s.x * 0.5 + t + col as f32 * bay + (bay - t) * 0.5,
                    y,
                    frame_z,
                ),
                Vec3::new(bay - t, t, frame_depth),
                0.003,
            );
        }
        if level == p.levels || (o.kind == ObjectKind::Cabinet && p.front != 3) {
            continue;
        }
        if o.kind == ObjectKind::Cabinet && y + pitch > s.y * p.open_fraction {
            continue;
        }
        let mut rng = stream(o.seed, 3060 + level as u64);
        for col in 0..p.columns {
            let bay = (s.x - t) / p.columns as f32;
            let left = -s.x * 0.5 + t + col as f32 * bay;
            let base = y + t * 0.5;
            // Shelf contents form distinct functional bays rather than uniformly
            // filling every level with upright books. Keep everything contained.
            let content = rng.random_range(0..10);
            if content <= 2 {
                let w = (bay - 2. * t - 0.035).min(rng.random_range(0.18..0.38));
                let h = (pitch - t - 0.025) * rng.random_range(0.46..0.78);
                let z = s.z * 0.03;
                let x = left + (bay - t) * 0.5;
                if content < 2 {
                    a.box_part(
                        Surface::Paper,
                        "box",
                        Vec3::new(x, base + (h - 0.0195) * 0.5, z),
                        Vec3::new(w, h - 0.0195, s.z * 0.77),
                        0.003,
                    );
                    a.box_part(
                        Surface::WoodEdge,
                        "box",
                        Vec3::new(x, base + h - 0.009, z),
                        Vec3::new(w + 0.008, 0.018, s.z * 0.80),
                        0.002,
                    );
                    a.box_part(
                        Surface::Ink,
                        "box",
                        Vec3::new(x, base + h * 0.62, z + s.z * 0.39),
                        Vec3::new(w * 0.32, 0.011, 0.001),
                        0.001,
                    );
                } else {
                    let r = (w * 0.36).min(s.z * 0.30);
                    a.part(Surface::Ceramic, "vase").lathe(
                        &[
                            (0., 0.),
                            (r * 0.64, 0.),
                            (r, h * 0.43),
                            (r * 0.50, h),
                            (r * 0.44, h),
                            (r * 0.88, h * 0.43),
                            (r * 0.56, h * 0.06),
                            (0., h * 0.06),
                        ],
                        24,
                        Transform::from_xyz(x, base, z),
                    );
                }
                continue;
            }
            if content == 3 {
                continue;
            }
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
                // Covers and spine enclose a real paper block. Painting a thin
                // paper cap over a closed cover box created coincident end faces.
                let thickness = 0.002_f32.min(w * 0.10).min(h * 0.10);
                let center = Vec3::new(x + w * 0.5, base + h * 0.5, z);
                for side in [-1., 1.] {
                    let offset = if flat {
                        Vec3::Y * (h - thickness) * 0.5
                    } else {
                        Vec3::X * (w - thickness) * 0.5
                    };
                    let size = if flat {
                        Vec3::new(w, thickness, depth)
                    } else {
                        Vec3::new(thickness, h, depth)
                    };
                    a.box_part(cover, &book_label, center + side * offset, size, 0.0004);
                }
                let inside = if flat {
                    Vec2::new(w, h - 2. * thickness)
                } else {
                    Vec2::new(w - 2. * thickness, h)
                };
                a.box_part(
                    cover,
                    &book_label,
                    center + Vec3::Z * (depth - thickness) * 0.5,
                    Vec3::new(inside.x, inside.y, thickness),
                    0.0004,
                );
                a.box_part(
                    Surface::Paper,
                    "books",
                    center - Vec3::Z * thickness * 0.5,
                    Vec3::new(
                        inside.x - 0.001,
                        inside.y - 0.001,
                        depth - thickness - 0.001,
                    ),
                    0.0003,
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
        if p.front == 4 {
            for y in [bottom + t * 0.5 + 0.005, s.y - t - 0.005] {
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(0., y, s.z * 0.5 - 0.010),
                    Vec3::new(s.x - 2. * t - 0.004, 0.008, 0.012),
                    0.001,
                );
            }
        }
        for col in 0..p.columns {
            let w = (s.x - t * 2.) / p.columns as f32;
            let x = (col as f32 - (p.columns - 1) as f32 * 0.5) * w;
            if p.front == 3 {
                let open = s.y * p.open_fraction;
                a.box_part(
                    frame,
                    label,
                    Vec3::new(x, (open + s.y - 0.012) * 0.5, s.z * 0.5 - 0.023),
                    Vec3::new(w - 0.006, s.y - open - 0.012, 0.020),
                    0.003,
                );
                handle(
                    a,
                    label,
                    &p,
                    Vec3::new(x + w * 0.28, open + (s.y - open) * 0.5, s.z * 0.5 - 0.006),
                    w.min(0.5) * 0.22,
                );
                continue;
            }
            if p.front == 1 || (p.front == 2 && col % 2 == 0) {
                let n = p.levels.clamp(2, 5);
                let pitch = (s.y - 0.080) / n as f32;
                for row in 0..n {
                    let y = 0.070 + pitch * (row as f32 + 0.5);
                    a.box_part(
                        frame,
                        label,
                        Vec3::new(x, y, s.z * 0.5 - 0.018),
                        Vec3::new(w - 0.006, pitch - 0.006, 0.020),
                        0.003,
                    );
                    handle(
                        a,
                        label,
                        &p,
                        Vec3::new(x, y + pitch * 0.27, s.z * 0.5 - 0.004),
                        w * 0.26,
                    );
                }
                continue;
            }
            let depth_offset = if p.front == 4 && col % 2 == 0 {
                0.027
            } else {
                0.0
            };
            a.box_part(
                frame,
                label,
                Vec3::new(x, s.y * 0.5 + 0.02, s.z * 0.5 - 0.023 - depth_offset),
                Vec3::new(w - 0.006, s.y - 0.065, 0.02),
                0.003,
            );
            if p.front == 5 {
                let n = (w / 0.05).round().clamp(4., 20.) as usize;
                for i in 0..n {
                    a.box_part(
                        Surface::WoodEdge,
                        label,
                        Vec3::new(
                            x + (i as f32 + 0.5) * (w - 0.02) / n as f32 - (w - 0.02) * 0.5,
                            s.y * 0.5 + 0.02,
                            s.z * 0.5 - 0.008,
                        ),
                        Vec3::new(0.012, s.y - 0.082, 0.010),
                        0.002,
                    );
                }
            }
            handle(
                a,
                label,
                &p,
                Vec3::new(x + w * 0.28, s.y * 0.57, s.z * 0.5 - 0.006 - depth_offset),
                w.min(0.5) * 0.22,
            );
        }
    }
}

fn handle(a: &mut Assembly, label: &str, p: &ShelfProgram, center: Vec3, width: f32) {
    match p.handle {
        0 => a.part(Surface::Metal, label).rod(
            center - Vec3::X * width * 0.5,
            center + Vec3::X * width * 0.5,
            0.0025,
        ),
        1 => a.part(Surface::Chrome, label).ellipsoid(
            Vec3::new(0.009, 0.009, 0.004),
            Transform::from_translation(center),
        ),
        2 => a.box_part(
            Surface::Ink,
            label,
            center,
            Vec3::new(width, 0.009, 0.001),
            0.001,
        ),
        _ => {} // Flush, touch-latch panels.
    }
}
