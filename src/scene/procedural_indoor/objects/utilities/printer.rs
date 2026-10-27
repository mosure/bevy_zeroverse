//! Desktop printers with a real output cavity, fitted trays and variable scanners.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrinterProgram {
    pub body: Vec3,
    pub lower_top: f32,
    pub slot_height: f32,
    pub slot_fraction: f32,
    pub slot_center: f32,
    pub scanner: bool,
    pub feeder: bool,
    pub control_side: f32,
    pub controls: usize,
    pub screen: bool,
    pub sheet_count: usize,
    pub sheet_yaw: f32,
    pub finish: Surface,
    pub panel: Surface,
    pub bevel: f32,
}
pub fn parameters(o: &IndoorObject) -> PrinterProgram {
    let s = o.size;
    let mut r = stream(o.seed, 0x5052494e54455250);
    let scanner = r.random_bool(0.62);
    let body = s * Vec3::new(
        r.random_range(0.82..0.98),
        r.random_range(0.59..0.78),
        r.random_range(0.62..0.82),
    );
    PrinterProgram {
        body,
        lower_top: body.y * r.random_range(0.24..0.41),
        slot_height: body.y * r.random_range(0.12..0.24),
        slot_fraction: r.random_range(0.61..0.83),
        slot_center: r.random_range(-0.035..0.035),
        scanner,
        feeder: scanner && r.random_bool(0.55),
        control_side: if r.random_bool(0.5) { -1. } else { 1. },
        controls: r.random_range(2..=6),
        screen: r.random_bool(0.63),
        sheet_count: if r.random_bool(0.72) {
            r.random_range(1..=5)
        } else {
            0
        },
        sheet_yaw: r.random_range(-0.06..0.06),
        finish: [Surface::Plastic, Surface::Metal][usize::from(r.random_bool(0.23))],
        panel: [Surface::Plastic, Surface::Metal][usize::from(r.random_bool(0.40))],
        bevel: r.random_range(0.003..0.013),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = p.body;
    let l = o.kind.class_name();
    let front = s.z * 0.5;
    let slot_w = s.x * p.slot_fraction;
    let cx = s.x * p.slot_center;
    a.box_part(
        p.finish,
        l,
        Vec3::Y * p.lower_top * 0.5,
        Vec3::new(s.x, p.lower_top, s.z),
        p.bevel,
    );
    // Cavity side walls and rear are distinct solids; the front is open.
    for side in [-1., 1.] {
        let edge = side * s.x * 0.5;
        let inner = cx + side * slot_w * 0.5;
        a.box_part(
            p.finish,
            l,
            Vec3::new((edge + inner) * 0.5, p.lower_top + p.slot_height * 0.5, 0.),
            Vec3::new((edge - inner).abs(), p.slot_height, s.z),
            p.bevel.min(0.004),
        );
    }
    let back = s.z * 0.16;
    a.box_part(
        p.panel,
        l,
        Vec3::new(
            cx,
            p.lower_top + p.slot_height * 0.5,
            -s.z * 0.5 + back * 0.5,
        ),
        Vec3::new(slot_w, p.slot_height, back),
        0.003,
    );
    let upper = s.y - p.lower_top - p.slot_height;
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., s.y - upper * 0.5, 0.),
        Vec3::new(s.x, upper, s.z),
        p.bevel,
    );
    // Extend a physical ledge from the output opening; paper rests on its top.
    let tray_depth = (o.size.z * 0.5 - front) + s.z * 0.28;
    let tray_z = (front + o.size.z * 0.5) * 0.5 - s.z * 0.14;
    let tray_h = o.size.y * 0.022;
    let tray_top = p.lower_top + 0.001;
    a.box_part(
        p.panel,
        l,
        Vec3::new(cx, tray_top - tray_h * 0.5, tray_z),
        Vec3::new(slot_w * 0.97, tray_h, tray_depth * 0.98),
        0.002,
    );
    let paper_w = slot_w * 0.78;
    let paper_d = tray_depth * 0.82;
    for i in 0..p.sheet_count {
        a.part(Surface::Paper, "paper").cuboid(
            Vec3::new(paper_w, 0.00025, paper_d),
            0.,
            Transform::from_xyz(cx, tray_top + 0.000125 + i as f32 * 0.00028, tray_z)
                .with_rotation(Quat::from_rotation_y(p.sheet_yaw)),
        );
    }
    let lid_h = o.size.y * if p.scanner { 0.075 } else { 0.025 };
    a.box_part(
        p.panel,
        l,
        Vec3::new(0., s.y + lid_h * 0.5, -s.z * 0.025),
        Vec3::new(s.x * 0.95, lid_h, s.z * 0.90),
        p.bevel.min(lid_h * 0.24),
    );
    if p.scanner {
        for side in [-1., 1.] {
            a.part(p.panel, l).cylinder(
                lid_h * 0.45,
                s.x * 0.08,
                Transform::from_xyz(side * s.x * 0.32, s.y + lid_h * 0.5, -s.z * 0.455)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        }
    }
    if p.feeder {
        let h = (o.size.y - s.y - lid_h) * 0.78;
        a.box_part(
            p.panel,
            l,
            Vec3::new(-s.x * 0.27, s.y + lid_h + h * 0.5, -s.z * 0.04),
            Vec3::new(s.x * 0.32, h, s.z * 0.66),
            p.bevel,
        );
        // Shallow feed tray joins the feeder housing instead of floating above it.
        a.box_part(
            p.panel,
            l,
            Vec3::new(s.x * 0.13, s.y + lid_h + h * 0.22, -s.z * 0.04),
            Vec3::new(s.x * 0.55, lid_h * 0.22, s.z * 0.66),
            0.002,
        );
    }
    let cy = p.lower_top + p.slot_height + upper * 0.56;
    let cwidth = s.x * 0.21;
    let cx = p.control_side * s.x * 0.31;
    let control_h = upper * 0.54;
    a.box_part(
        Surface::Plastic,
        l,
        Vec3::new(cx, cy, front + 0.001),
        Vec3::new(cwidth, control_h, 0.003),
        0.001,
    );
    if p.screen {
        a.box_part(
            Surface::Screen,
            l,
            Vec3::new(cx, cy + control_h * 0.10, front + 0.003),
            Vec3::new(cwidth * 0.79, control_h * 0.51, 0.0007),
            0.0001,
        );
    }
    for i in 0..p.controls {
        a.part(Surface::Rubber, l).cylinder(
            (cwidth / p.controls as f32 * 0.22).min(0.004),
            0.001,
            Transform::from_xyz(
                cx + cwidth * ((i as f32 + 0.5) / p.controls as f32 - 0.5) * 0.82,
                cy - control_h * 0.30,
                front + 0.0035,
            )
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
}
