//! Writing boards with a fitted tray and supported loose accessories.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct BoardProgram {
    pub border_m: f32,
    pub tray_fraction: f32,
    pub tray_side: f32,
    pub marker_count: usize,
    pub mounts: bool,
    pub tray: bool,
    pub frame_surface: Surface,
    pub frameless: bool,
    pub corner_fraction: f32,
    pub thickness_m: f32,
    pub face_z_m: f32,
    pub tray_surface: Surface,
    pub tray_lo: Vec2,
    pub tray_hi: Vec2,
    pub tray_base_top_m: f32,
    pub tray_base_thickness_m: f32,
    pub tray_lip_height_m: f32,
    pub tray_end_caps: bool,
    pub contents: Vec<TrayItem>,
    pub magnets: Vec<Magnet>,
}

pub fn parameters(o: &IndoorObject) -> BoardProgram {
    let s = o.size;
    let mut r = stream(o.seed, 7470);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x424f41524446494e);
    let frameless = r.random_bool(0.22);
    let border = if frameless {
        0.002_f32
    } else {
        r.random_range(0.010..0.043)
    }
    .min(s.x.min(s.y) * 0.10);
    let fraction = r.random_range(0.24..0.96);
    let side = r.random_range(-0.46..0.46);
    let thickness = r.random_range(0.008_f32..0.032).min(s.z * 0.38);
    let front = -s.z * 0.5 + thickness;
    let width = s.x * fraction;
    let x = (s.x - width) * side;
    let depth = r
        .random_range(0.023_f32..0.078)
        .min(s.z * 0.5 - front - 0.002);
    let base = r.random_range(0.004..0.010);
    let top = base + r.random_range(0.001..0.014);
    let mut p = BoardProgram {
        border_m: border,
        tray_fraction: fraction,
        tray_side: side,
        marker_count: 0,
        mounts: r.random_bool(0.30),
        tray: r.random_bool(0.80),
        frame_surface: [Surface::Metal, Surface::Wood, Surface::Plastic][f.random_range(0..3)],
        frameless,
        corner_fraction: r.random_range(0.05..0.40),
        thickness_m: thickness,
        face_z_m: front,
        tray_surface: [Surface::Metal, Surface::Chrome, Surface::Plastic][f.random_range(0..3)],
        tray_lo: Vec2::new(x - width * 0.5, front),
        tray_hi: Vec2::new(x + width * 0.5, front + depth),
        tray_base_top_m: top,
        tray_base_thickness_m: base,
        tray_lip_height_m: r.random_range(0.008..0.024),
        tray_end_caps: r.random_bool(0.60),
        contents: Vec::new(),
        magnets: Vec::new(),
    };
    if p.tray {
        p.contents = contents::tray_items(o, &p);
        p.marker_count = p.contents.iter().filter(|v| !v.eraser).count();
    }
    p.magnets = contents::magnets(o, &p);
    p
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    a.box_part(
        p.frame_surface,
        l,
        Vec3::new(0., s.y * 0.5, -s.z * 0.5 + p.thickness_m * 0.5),
        s.with_z(p.thickness_m),
        p.border_m * p.corner_fraction,
    );
    face(
        a,
        Surface::Whiteboard,
        l,
        Vec3::new(-s.x * 0.5 + p.border_m, p.border_m, p.face_z_m + 0.001),
        Vec3::new(s.x * 0.5 - p.border_m, s.y - p.border_m, p.face_z_m + 0.001),
    );
    if p.mounts {
        let inset = p.border_m.max(0.012) * 0.5;
        for x in [-1., 1.] {
            for y in [inset, s.y - inset] {
                a.part(Surface::Chrome, l).cylinder(
                    inset * 0.34,
                    0.001,
                    Transform::from_xyz(x * (s.x * 0.5 - inset), y, p.face_z_m + 0.0015)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                );
            }
        }
    }
    if p.tray {
        let width = p.tray_hi.x - p.tray_lo.x;
        let depth = p.tray_hi.y - p.tray_lo.y;
        let center = (p.tray_hi + p.tray_lo) * 0.5;
        a.box_part(
            p.tray_surface,
            l,
            Vec3::new(
                center.x,
                p.tray_base_top_m - p.tray_base_thickness_m * 0.5,
                center.y,
            ),
            Vec3::new(width, p.tray_base_thickness_m, depth),
            0.001,
        );
        let side = if p.tray_end_caps { 0.004 } else { 0. };
        a.box_part(
            p.tray_surface,
            l,
            Vec3::new(
                center.x,
                p.tray_base_top_m + p.tray_lip_height_m * 0.5,
                p.tray_hi.y - 0.002,
            ),
            Vec3::new(width - 2. * side, p.tray_lip_height_m, 0.004),
            0.0005,
        );
        if p.tray_end_caps {
            for x in [p.tray_lo.x + 0.002, p.tray_hi.x - 0.002] {
                a.box_part(
                    p.tray_surface,
                    l,
                    Vec3::new(x, p.tray_base_top_m + p.tray_lip_height_m * 0.5, center.y),
                    Vec3::new(0.004, p.tray_lip_height_m, depth),
                    0.0005,
                );
            }
        }
        for item in &p.contents {
            contents::build_item(a, o, &p, item);
        }
    }
    for magnet in &p.magnets {
        let label = format!("{l}#finish{}", magnet.finish_slot);
        a.part(Surface::Plastic, &label).cylinder(
            magnet.radius_m,
            0.002,
            Transform::from_xyz(magnet.center.x, magnet.center.y, p.face_z_m + 0.002)
                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
}
