//! Metric equipment assemblies. Connected cable centerlines are archived rather
//! than inferred from an unrelated decorative shape at render time.
mod fixtures;
pub(crate) mod wires;
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct EquipmentProgram {
    pub housing: Surface,
    pub corner_m: f32,
    pub lens_fraction: f32,
    pub lens_offset: f32,
    pub vent_count: usize,
    pub socket_count: usize,
    pub shelf_brackets: usize,
    pub note_count: usize,
    pub shade_fraction: f32,
    pub diffuser_fraction: f32,
    pub shelf_span: f32,
    pub bracket_style: u8,
    pub screen_drop: f32,
    pub screen_frame: bool,
    pub sconce_width: f32,
    pub sconce_height: f32,
    pub sconce_taper: f32,
    pub board_frame_fraction: f32,
    pub board_divider_fraction: Option<f32>,
}

pub fn parameters(o: &IndoorObject) -> EquipmentProgram {
    let mut rng = stream(o.seed, 5700);
    let mut detail = stream(o.seed, 7460);
    let mut board = stream(o.seed, 7510);
    EquipmentProgram {
        board_frame_fraction: board.random_range(0.018..0.052),
        board_divider_fraction: board
            .random_bool(0.30)
            .then(|| board.random_range(0.35..0.65)),
        shelf_span: detail.random_range(0.24..0.41),
        bracket_style: detail.random_range(0..3),
        screen_drop: detail.random_range(0.64..0.99),
        screen_frame: detail.random_bool(0.30),
        sconce_width: detail.random_range(0.65..0.98),
        sconce_height: detail.random_range(0.46..0.89),
        sconce_taper: detail.random_range(0.65..1.0),
        housing: if rng.random_bool(0.38) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
        corner_m: rng.random_range(0.003..0.017),
        lens_fraction: rng.random_range(0.18..0.33),
        lens_offset: rng.random_range(-0.28..0.25),
        vent_count: rng.random_range(5..=15),
        socket_count: rng.random_range(3..=7),
        shelf_brackets: rng.random_range(2..=3),
        note_count: rng.random_range(2..=13),
        shade_fraction: rng.random_range(0.18..0.48),
        diffuser_fraction: rng.random_range(0.44..0.84),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::Projector => projector(a, o, &p),
        ObjectKind::PowerStrip => power_strip(a, o, &p),
        ObjectKind::Cable => wires::build(a, o),
        ObjectKind::WallShelf => fixtures::shelf(a, o, &p),
        ObjectKind::NoticeBoard => fixtures::notice_board(a, o, &p),
        ObjectKind::ProjectionScreen => fixtures::projection_screen(a, o),
        ObjectKind::WallSconce | ObjectKind::CeilingPendant => fixtures::light(a, o, &p),
        _ => unreachable!("non-equipment object"),
    }
}

fn projector(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let foot = (s.y * 0.09).min(0.013);
    // The lens has its own optical rim, inset glass and recessed barrel; a
    // projector does not emit an imaginary opaque beam or floating image.
    a.box_part(
        p.housing,
        label,
        Vec3::new(0., (s.y + foot) * 0.5, -s.z * 0.025),
        Vec3::new(s.x, s.y - foot, s.z * 0.91),
        p.corner_m,
    );
    for side in [-1., 1.] {
        for front in [-1., 1.] {
            a.part(Surface::Rubber, label).cylinder(
                s.x.min(s.z) * 0.035,
                foot,
                Transform::from_xyz(side * s.x * 0.35, foot * 0.5, front * s.z * 0.31),
            );
        }
    }
    let radius = s.y.min(s.x) * p.lens_fraction;
    let lens = Vec3::new(s.x * p.lens_offset, foot + (s.y - foot) * 0.49, s.z * 0.455);
    let frame = Transform::from_translation(lens).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    a.part(Surface::Rubber, label)
        .cylinder(radius, s.z * 0.065, frame);
    a.part(Surface::Metal, label)
        .cylinder(radius * 0.82, s.z * 0.065 + 0.0008, frame);
    a.part(Surface::Ink, label).cylinder(
        radius * 0.72,
        0.001,
        frame.with_translation(lens + Vec3::Z * (s.z * 0.033 + 0.001)),
    );
    a.part(Surface::ContainerGlass, label).ellipsoid(
        Vec3::new(radius * 0.68, radius * 0.68, 0.0011),
        Transform::from_translation(lens + Vec3::Z * (s.z * 0.033 + 0.0014)),
    );
    // Fan slots and buttons remain millimetric, regardless of enclosure size.
    for i in 0..p.vent_count {
        a.box_part(
            Surface::Ink,
            label,
            Vec3::new(
                -s.x * 0.33 + s.x * 0.25 * i as f32 / p.vent_count as f32,
                s.y * 0.54,
                s.z * 0.4301,
            ),
            Vec3::new(s.x * 0.009, s.y * 0.27, 0.001),
            0.0003,
        );
    }
    for i in 0..4 {
        a.box_part(
            p.housing,
            label,
            Vec3::new(s.x * (0.08 + i as f32 * 0.075), s.y - 0.0006, -s.z * 0.15),
            Vec3::new(s.x * 0.045, 0.001, s.z * 0.06),
            0.0003,
        );
    }
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(s.x * 0.24, s.y * 0.23, -s.z * 0.4801),
        Vec3::new(0.014, 0.008, 0.0008),
        0.001,
    );
}

fn power_strip(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        p.housing,
        label,
        Vec3::Y * s.y * 0.48,
        s * Vec3::new(1., 0.96, 1.),
        p.corner_m.min(s.y * 0.22),
    );
    let count = p.socket_count.min((s.x / 0.039).floor().max(3.) as usize);
    for i in 0..count {
        let x = (i as f32 + 0.5) * s.x * 0.84 / count as f32 - s.x * 0.42;
        a.box_part(
            Surface::Paper,
            label,
            Vec3::new(x, s.y * 0.968, 0.),
            Vec3::new(s.x * 0.72 / count as f32, 0.0008, s.z * 0.77),
            0.002,
        );
        for z in [-1., 1.] {
            let position = Vec3::new(x, s.y * 0.99, z * s.z * 0.17);
            if o.variant.is_multiple_of(4) {
                a.part(Surface::Ink, label).cylinder(
                    0.0026,
                    0.0006,
                    Transform::from_translation(position),
                );
            } else {
                a.box_part(
                    Surface::Ink,
                    label,
                    position,
                    Vec3::new(
                        if o.variant % 4 == 2 { 0.010 } else { 0.007 },
                        0.0006,
                        0.0028,
                    ),
                    0.0002,
                );
            }
        }
        if o.variant % 4 >= 2 {
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(x + 0.008, s.y * 0.99, 0.),
                Vec3::new(0.003, 0.0006, 0.004),
                0.0002,
            );
        }
    }
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(-s.x * 0.5001, s.y * 0.48, 0.),
        Vec3::new(0.0008, 0.008, 0.015),
        0.001,
    );
    // Integrated USB-C/PD supply, distinct from the mains sockets. This permits
    // a real low-voltage laptop lead without inventing an absent power brick.
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(s.x * 0.35, s.y * 0.48, s.z * 0.5001),
        Vec3::new(0.008, 0.003, 0.0008),
        0.001,
    );
}

/// Exact world-space attachment points shared by route generation and audits.
/// Power-strip target ports distinguish USB-C PD from the mains receptacle.
pub(crate) fn power_port(o: &IndoorObject, low_voltage: bool) -> Option<Vec3> {
    let s = o.size;
    let local = match o.kind {
        ObjectKind::Laptop => {
            let p = super::computers::parameters(o);
            let tilt = p.hinge_radians - FRAC_PI_2;
            let screen_height = (s.x / p.aspect_ratio)
                .min((s.y - p.chassis_m - 0.008) / tilt.cos())
                .min((s.z - 0.114) / tilt.sin().max(0.001));
            let rear = tilt.sin().max(0.) * screen_height;
            let base_depth = (s.z - rear - 0.014).max(0.10);
            Vec3::new(
                s.x * 0.5 + 0.001,
                p.chassis_m * 0.55 + 0.003,
                -s.z * 0.5 + rear + 0.008 + base_depth * 0.17,
            )
        }
        ObjectKind::CoffeeMaker | ObjectKind::DrinkDispenser if !low_voltage => {
            Vec3::new(0., s.y * 0.10, -s.z * 0.5 - 0.001)
        }
        ObjectKind::Monitor if !low_voltage => super::computers::monitor_power_port_local(o).0,
        ObjectKind::Projector => Vec3::new(s.x * 0.24, s.y * 0.23, -s.z * 0.48 - 0.001),
        ObjectKind::WallOutlet if !low_voltage => super::wall::outlet_port_local(o),
        ObjectKind::FloorOutlet if !low_voltage && super::floor::is_open(o) => {
            super::floor::port_local(o)
        }
        ObjectKind::PowerStrip if low_voltage => {
            Vec3::new(s.x * 0.35, s.y * 0.48, s.z * 0.5 + 0.001)
        }
        ObjectKind::PowerStrip => {
            let count = parameters(o)
                .socket_count
                .min((s.x / 0.039).floor().max(3.) as usize);
            Vec3::new(s.x * 0.42 / count as f32 - s.x * 0.42, s.y + 0.001, 0.)
        }
        _ => return None,
    };
    Some(o.transform().transform_point(local))
}

/// Exact world-space connector departure normal. Generation, edited-prop
/// validation and archived graph qualification share one physical port frame.
pub(crate) fn power_port_normal(o: &IndoorObject, low_voltage: bool) -> Option<Vec3> {
    let local = match o.kind {
        ObjectKind::Laptop => Vec3::X,
        ObjectKind::Monitor if !low_voltage => super::computers::monitor_power_port_local(o).1,
        ObjectKind::Projector | ObjectKind::CoffeeMaker | ObjectKind::DrinkDispenser => Vec3::NEG_Z,
        ObjectKind::WallOutlet if !low_voltage => Vec3::Z,
        ObjectKind::FloorOutlet if !low_voltage && super::floor::is_open(o) => Vec3::Y,
        ObjectKind::PowerStrip if low_voltage => Vec3::Z,
        ObjectKind::PowerStrip => Vec3::Y,
        _ => return None,
    };
    Some(o.transform().rotation * local)
}

/// Power strips have a mains inlet distinct from their output receptacles.
pub(crate) fn source_power_port(o: &IndoorObject) -> Option<Vec3> {
    if o.kind == ObjectKind::PowerStrip {
        Some(
            o.transform()
                .transform_point(Vec3::new(-o.size.x * 0.5 - 0.001, o.size.y * 0.48, 0.)),
        )
    } else {
        power_port(o, o.kind == ObjectKind::Laptop)
    }
}
pub(crate) fn source_power_normal(o: &IndoorObject) -> Option<Vec3> {
    if o.kind == ObjectKind::PowerStrip {
        Some(o.transform().rotation * Vec3::NEG_X)
    } else {
        power_port_normal(o, o.kind == ObjectKind::Laptop)
    }
}

/// Emitter outside its downward diffuser. The caller uses this same result for
/// direct lighting and diffuse transport, preserving the physical contract.
pub(crate) fn fixture_photometry(o: &IndoorObject) -> (Vec3, Vec3, f32, f32, f32) {
    let p = parameters(o);
    let mut rng = stream(o.seed, 5720);
    let local = if o.kind == ObjectKind::CeilingPendant {
        Vec3::new(0., o.size.y * 0.05 - 0.003, 0.)
    } else {
        Vec3::new(
            0.,
            o.size.y * (0.5 - p.sconce_height * 0.5) - 0.003,
            o.size.z * 0.13,
        )
    };
    (
        o.transform().transform_point(local),
        super::super::materials::kelvin_rgb(rng.random_range(2400.0..6500.0)),
        rng.random_range(350.0..1700.0) * (0.75 + p.diffuser_fraction * 0.5),
        rng.random_range(0.37..0.64),
        rng.random_range(0.82..1.18),
    )
}

#[cfg(test)]
mod tests;
