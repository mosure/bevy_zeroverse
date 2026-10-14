//! Mounted presentation surfaces retain their exact wall-facing rear datum.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct BoardProgram {
    pub border_m: f32,
    pub tray_fraction: f32,
    pub tray_side: f32,
    pub marker_count: usize,
    pub mounts: bool,
    pub tray: bool,
}
pub fn parameters(o: &IndoorObject) -> BoardProgram {
    let mut r = stream(o.seed, 7470);
    BoardProgram {
        border_m: r.random_range(0.012..0.038),
        tray_fraction: r.random_range(0.30..0.94),
        tray_side: r.random_range(-0.15..0.15),
        marker_count: r.random_range(1..=5),
        mounts: r.random_bool(0.3),
        tray: r.random_bool(0.8),
    }
}
fn face(a: &mut Assembly, surface: Surface, label: &str, lo: Vec3, hi: Vec3) {
    let g = a.part(surface, label);
    let i = g.positions.len() as u32;
    g.positions.extend([
        [lo.x, lo.y, lo.z],
        [hi.x, lo.y, lo.z],
        [hi.x, hi.y, lo.z],
        [lo.x, hi.y, lo.z],
    ]);
    g.normals.extend([[0., 0., 1.]; 4]);
    g.uvs.extend([[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
    g.indices.extend([i, i + 1, i + 2, i, i + 2, i + 3]);
}
pub(super) fn whiteboard(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let d = (s.z * 0.35).min(0.032);
    let front = -s.z * 0.5 + d;
    a.box_part(
        Surface::Metal,
        l,
        Vec3::new(0., s.y * 0.5, -s.z * 0.5 + d * 0.5),
        s.with_z(d),
        p.border_m * 0.15,
    );
    face(
        a,
        Surface::Whiteboard,
        l,
        Vec3::new(-s.x * 0.5 + p.border_m, p.border_m, front + 0.001),
        Vec3::new(s.x * 0.5 - p.border_m, s.y - p.border_m, front + 0.001),
    );
    if p.mounts {
        for x in [-1., 1.] {
            for y in [p.border_m * 0.5, s.y - p.border_m * 0.5] {
                a.part(Surface::Chrome, l).cylinder(
                    p.border_m * 0.17,
                    0.001,
                    Transform::from_xyz(x * (s.x - p.border_m) * 0.5, y, front + 0.001)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                );
            }
        }
    }
    if p.tray {
        let width = s.x * p.tray_fraction;
        let cx = (s.x - width) * p.tray_side;
        let depth = s.z * 0.5 - front;
        a.box_part(
            Surface::Chrome,
            l,
            Vec3::new(cx, 0.012, (front + s.z * 0.5) * 0.5),
            Vec3::new(width, 0.024, depth),
            0.002,
        );
        a.box_part(
            Surface::Chrome,
            l,
            Vec3::new(cx, 0.029, s.z * 0.5 - 0.003),
            Vec3::new(width, 0.014, 0.006),
            0.001,
        );
        for i in 0..p.marker_count {
            let x = cx + width * ((i as f32 + 0.5) / p.marker_count as f32 - 0.5) * 0.8;
            let length = (width / p.marker_count as f32 * 0.68).min(0.13);
            a.part(
                if i % 2 == 0 {
                    Surface::Ink
                } else {
                    Surface::Plastic
                },
                l,
            )
            .rod(
                Vec3::new(x - length * 0.5, 0.029, front + depth * 0.52),
                Vec3::new(x + length * 0.5, 0.029, front + depth * 0.52),
                0.005,
            );
        }
    }
}
pub(super) fn screen(a: &mut Assembly, o: &IndoorObject) {
    let p = equipment::parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    if p.screen_frame {
        let t = s.x.min(s.y) * 0.035;
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(0., s.y * 0.5, -s.z * 0.43),
            Vec3::new(s.x, s.y, s.z * 0.14),
            0.003,
        );
        face(
            a,
            Surface::Paper,
            l,
            Vec3::new(-s.x * 0.5 + t, t, -s.z * 0.35),
            Vec3::new(s.x * 0.5 - t, s.y - t, -s.z * 0.35),
        );
    } else {
        let case = (s.y * 0.045).clamp(0.036, 0.060).min(s.z * 0.8);
        let bottom = (s.y - case) * (1. - p.screen_drop);
        let height = s.y - case - bottom;
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(0., s.y - case * 0.5, 0.),
            Vec3::new(s.x, case, s.z * 0.82),
            case * 0.28,
        );
        for side in [-1., 1.] {
            a.box_part(
                Surface::Metal,
                l,
                Vec3::new(side * s.x * 0.38, s.y - case * 0.5, -s.z * 0.455),
                Vec3::new(0.03, case * 0.72, s.z * 0.09),
                0.001,
            );
        }
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(0., bottom + height * 0.5, s.z * 0.2),
            Vec3::new(s.x * 0.94, height, 0.004),
            0.001,
        );
        face(
            a,
            Surface::Paper,
            l,
            Vec3::new(-s.x * 0.45, bottom + height * 0.025, s.z * 0.2 + 0.0028),
            Vec3::new(s.x * 0.45, s.y - case - height * 0.025, s.z * 0.2 + 0.0028),
        );
        a.part(Surface::Metal, l).cylinder(
            0.009,
            s.x * 0.94,
            Transform::from_xyz(0., bottom + 0.009, s.z * 0.20)
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
    }
}
