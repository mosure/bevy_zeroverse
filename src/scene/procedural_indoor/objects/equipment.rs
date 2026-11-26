//! Metric equipment assemblies. Connected cable centerlines are archived rather
//! than inferred from an unrelated decorative shape at render time.
pub mod board_section;
mod fixtures;
pub mod pendant;
pub mod pinups;
pub mod power_strip;
pub mod projector;
pub(crate) mod receptacles;
pub mod sconce;
pub mod signals;
pub mod wall_shelf;
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
    /// Requested postings. Clearance and the exposed-area budget determine the installed count.
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
    parameters_at(o, bevy_zeroverse_capture::GENERATOR_VERSION)
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> EquipmentProgram {
    let mut rng = stream(o.seed, 5700);
    let mut detail = stream(o.seed, 7460);
    let mut board = stream(o.seed, 7510);
    let mut p = EquipmentProgram {
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
    };
    if o.kind == ObjectKind::NoticeBoard && version >= board_section::TACKBOARD_VERSION {
        p.note_count = pinups::requested_count(o);
    } else if o.kind == ObjectKind::PowerStrip {
        let strip = power_strip::parameters(o);
        p.socket_count = strip.socket_count;
        p.housing = strip.housing;
        p.corner_m = o.size.y * strip.height_fraction * strip.bevel_fraction;
    } else if o.kind == ObjectKind::WallShelf {
        let shelf = wall_shelf::parameters_at(o, version);
        p.housing = shelf.board;
        p.shelf_brackets = shelf.bracket_x_m.len();
        p.shelf_span = shelf
            .bracket_x_m
            .iter()
            .map(|x| x.abs() / o.size.x)
            .fold(0., f32::max);
        p.bracket_style = shelf.style;
    } else if o.kind == ObjectKind::ProjectionScreen {
        let screen = super::presentation::projection::parameters(o);
        p.housing = screen.housing;
        p.screen_drop = screen.drop_fraction;
        p.screen_frame = screen.framed;
    } else if o.kind == ObjectKind::CeilingPendant {
        let fixture = pendant::parameters(o);
        p.housing = fixture.shade.finish;
        p.shade_fraction = fixture.shade_height_m / o.size.y;
        p.diffuser_fraction = fixture.diffuser_fraction().unwrap_or(0.);
    } else if o.kind == ObjectKind::WallSconce {
        let fixture = sconce::parameters(o);
        p.housing = fixture.shade.finish;
        p.sconce_width = fixture.head_span.x / o.size.x;
        p.sconce_height = fixture.head_height / o.size.y;
        p.sconce_taper = fixture.taper;
        p.diffuser_fraction = fixture.diffuser_radius * 2. / fixture.head_span.x;
    }
    p
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    match o.kind {
        ObjectKind::Projector => return projector::build(a, o),
        ObjectKind::PowerStrip => return power_strip::build(a, o),
        ObjectKind::Cable => return wires::build(a, o),
        ObjectKind::WallShelf => return wall_shelf::build(a, o, version),
        ObjectKind::CeilingPendant => return pendant::build(a, o),
        ObjectKind::WallSconce => return sconce::build(a, o),
        ObjectKind::ProjectionScreen => return super::presentation::screen(a, o, version),
        _ => {}
    }
    let p = parameters_at(o, version);
    match o.kind {
        ObjectKind::NoticeBoard => fixtures::notice_board(a, o, &p, version),
        _ => unreachable!("non-equipment object"),
    }
}

/// Exact world-space attachment points shared by route generation and audits.
/// Power-strip target ports distinguish USB-C PD from the mains receptacle.
pub(crate) fn power_port(o: &IndoorObject, low_voltage: bool) -> Option<Vec3> {
    let s = o.size;
    let local = match o.kind {
        ObjectKind::Laptop => super::computers::io::charge_port(o),
        ObjectKind::HotFoodWell if !low_voltage => super::public::hot_food::power_port(o),
        ObjectKind::DrinkDispenser if !low_voltage => super::public::dispenser::power_port(o)?,
        ObjectKind::Printer if !low_voltage => super::utilities::printer::power_port(o),
        ObjectKind::CoffeeMaker if !low_voltage => Vec3::new(0., s.y * 0.10, -s.z * 0.5 - 0.001),
        ObjectKind::PcTower if !low_voltage => {
            super::desktop::ports::tower_local(o, super::desktop::ports::TowerJack::Mains)
        }
        ObjectKind::Monitor if !low_voltage => super::computers::monitor_power_port_local(o).0,
        ObjectKind::Projector => projector::parameters(o).power_port(),
        ObjectKind::WallOutlet if !low_voltage => super::wall::outlet_port_local(o),
        ObjectKind::FloorOutlet if !low_voltage => super::floor::port_local(o)?,
        ObjectKind::PowerStrip => power_strip::parameters(o).output_port(s, low_voltage),
        _ => return None,
    };
    Some(o.transform().transform_point(local))
}

/// Exact world-space connector departure normal. Generation, edited-prop
/// validation and archived graph qualification share one physical port frame.
pub(crate) fn power_port_normal(o: &IndoorObject, low_voltage: bool) -> Option<Vec3> {
    let local = match o.kind {
        ObjectKind::Laptop => super::computers::io::charge_normal(o),
        ObjectKind::Monitor if !low_voltage => super::computers::monitor_power_port_local(o).1,
        ObjectKind::DrinkDispenser if !low_voltage => {
            super::public::dispenser::power_port(o)?;
            Vec3::NEG_Z
        }
        ObjectKind::Projector => projector::parameters(o).port_normal(),
        ObjectKind::Printer | ObjectKind::HotFoodWell if !low_voltage => Vec3::NEG_Z,
        ObjectKind::CoffeeMaker | ObjectKind::PcTower => Vec3::NEG_Z,
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
                .transform_point(power_strip::parameters(o).inlet(o.size)),
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

pub(crate) fn fixture_beam(o: &IndoorObject) -> super::super::photometry::FixtureBeam {
    if o.kind == ObjectKind::CeilingPendant {
        return pendant::parameters(o).beam(o);
    }
    let mut rng = stream(o.seed, 0x4245414d434f4e45);
    let p = sconce::parameters(o);
    let world = o.transform();
    super::super::photometry::FixtureBeam {
        position: world.transform_point(p.emitter),
        radius: 0.018,
        direction: world.rotation * (p.mount_rotation * Vec3::NEG_Y),
        inner: rng.random_range(0.37..0.64),
        outer: rng.random_range(0.82..1.18),
    }
}

#[cfg(test)]
mod tests;
