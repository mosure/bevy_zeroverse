//! Laptops are hinged assemblies with keyboards, bezels, ports and rubber feet.
use super::super::materials::input_housing::Housing;
use super::*;
pub mod io;
mod palmrest;

pub const LAPTOP_FAMILIES: u32 = 4;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LaptopProgram {
    pub hinge_radians: f32,
    pub aspect_ratio: f32,
    pub chassis_m: f32,
    pub bezel_m: f32,
    pub keyboard_fraction: f32,
    pub trackpad_fraction: f32,
    pub lid_m: f32,
    pub corner_m: f32,
    pub keyboard_columns: u32,
    /// Requested sockets across a fitted chassis, or marks per side on archived models.
    pub port_count: u32,
    pub speaker_grilles: bool,
}
pub fn parameters(o: &IndoorObject) -> LaptopProgram {
    let mut rng = stream(o.seed, 71);
    let mut program = LaptopProgram {
        hinge_radians: rng.random_range(1.48..2.28),
        aspect_ratio: rng.random_range(1.35..1.9),
        chassis_m: rng.random_range(0.008..0.023),
        bezel_m: rng.random_range(0.004..0.017),
        keyboard_fraction: rng.random_range(0.77..0.91),
        trackpad_fraction: rng.random_range(0.23..0.43),
        lid_m: rng.random_range(0.004..0.011),
        corner_m: rng.random_range(0.0015..0.008),
        keyboard_columns: rng.random_range(11..17),
        port_count: rng.random_range(1..5),
        speaker_grilles: rng.random_bool(0.6),
    };
    if o.variant & io::FITTED_IO != 0 {
        program.port_count += rng.random_range(0..3);
    }
    program.keyboard_columns = super::keyboard::columns(
        o.size.x * program.keyboard_fraction,
        program.keyboard_columns as usize,
        false,
    ) as u32;
    program
}
impl LaptopProgram {
    /// Fitted device sockets, including the primary charge/dock receptacle.
    pub fn io_bank(&self, o: &IndoorObject) -> Option<io::Bank> {
        if o.kind != ObjectKind::Laptop || o.variant & io::FITTED_IO == 0 {
            return None;
        }
        io::Bank::fit(o, self, Deck::fit(o, self))
    }
}

pub fn lid_angle(o: &IndoorObject) -> f32 {
    parameters(o).hinge_radians
}

#[derive(Clone, Copy)]
struct Deck {
    height: f32,
    depth: f32,
    rear: f32,
}

impl Deck {
    fn fit(o: &IndoorObject, p: &LaptopProgram) -> Self {
        let tilt = p.hinge_radians - FRAC_PI_2;
        let height = (o.size.x / p.aspect_ratio)
            .min((o.size.y - p.chassis_m - 0.008) / tilt.cos())
            .min((o.size.z - 0.114) / tilt.sin().max(0.001));
        let sweep = tilt.sin().max(0.) * height;
        Self {
            height,
            depth: (o.size.z - sweep - 0.014).max(0.10),
            rear: -o.size.z * 0.5 + sweep + 0.008,
        }
    }
}

pub(super) fn laptop(a: &mut Assembly, o: &IndoorObject) {
    let label = o.kind.class_name();
    let program = parameters(o);
    let width = o.size.x;
    let chassis = program.chassis_m;
    let stock = Housing::for_object(o);
    let metal = stock.surface;
    let angle = program.hinge_radians;
    // Envelope includes the backward lid sweep; base stays fully on its support.
    let tilt = angle - FRAC_PI_2;
    let deck = Deck::fit(o, &program);
    let screen_height = deck.height;
    let base_depth = deck.depth;
    let hinge_z = deck.rear;
    let base_z = hinge_z + base_depth * 0.5;
    let bank = io::Bank::fit(o, &program, deck);
    palmrest::build(a, o, &program, &stock, base_depth, base_z, bank.as_ref());
    let hinge = Vec3::new(0.0, chassis + 0.004, hinge_z);
    let lid = Transform::from_translation(hinge).with_rotation(Quat::from_rotation_x(-tilt));
    let bezel = program.bezel_m;
    use super::super::motion::{construction as motion, MotionKind};
    let moving = a.motion_selected(MotionKind::Prop, o.seed)
        && a.motion_policy
            .as_ref()
            .is_some_and(|p| p.prop_rotation_degrees > 0.);
    let mut leaf = Assembly::default();
    let dest = if moving { &mut leaf } else { &mut *a };
    dest.part(metal, &stock.case_label(o)).cuboid(
        Vec3::new(width, screen_height, program.lid_m),
        program.corner_m.min(program.lid_m * 0.35),
        lid.with_translation(lid.transform_point(Vec3::Y * screen_height * 0.5)),
    );
    displays::screen_cover(
        dest,
        metal,
        Surface::Screen,
        label,
        Vec3::new(width - bezel * 2., screen_height - bezel * 2., 0.0016),
        0.,
        lid.with_translation(lid.transform_point(Vec3::new(
            0.,
            screen_height * 0.5,
            program.lid_m * 0.5 + 0.0004,
        ))),
    );
    dest.part(Surface::Ink, label).ellipsoid(
        Vec3::splat(0.0018),
        Transform::from_translation(lid.transform_point(Vec3::new(
            0.0,
            screen_height - bezel * 0.45,
            program.lid_m * 0.5 + 0.001,
        ))),
    );
    if moving {
        let mut r = stream(o.seed, 0x4c49444d);
        let limit = a
            .motion_policy
            .as_ref()
            .unwrap()
            .prop_rotation_degrees
            .to_radians();
        let angle = (r.random_range(0.45..1.0) + tilt).min(limit);
        let mut joint = motion::rigid(MotionKind::Prop, o.seed, hinge, Vec3::X, angle, Vec3::ZERO);
        joint.frame = lid;
        joint.attachment = Some("contact_laptop_hinge".into());
        a.articulated(leaf, joint);
    }
    let hinge_label = moving.then(|| format!("{label}#contact_laptop_hinge"));
    for side in [-1.0, 1.0] {
        a.part(Surface::Chrome, hinge_label.as_deref().unwrap_or(label))
            .cylinder(
                0.006,
                0.037,
                Transform::from_translation(hinge + Vec3::X * side * width * 0.33)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        for end in [0.08, 0.87] {
            a.part(Surface::Rubber, label).cylinder(
                0.008,
                0.003,
                Transform::from_xyz(side * width * 0.40, 0.0015, hinge_z + base_depth * end),
            );
        }
        if bank.is_none() {
            for port in 0..program.port_count {
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(
                        side * (width * 0.5 + 0.0001),
                        chassis * 0.55 + 0.003,
                        hinge_z + base_depth * (0.17 + port as f32 * 0.13),
                    ),
                    Vec3::new(0.001, 0.003, 0.011),
                    0.0,
                );
            }
        }
    }
    super::keyboard::laptop(a, o, &program, hinge_z, base_depth);
    if program.speaker_grilles {
        for side in [-1., 1.] {
            for row in 0..18 {
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(
                        side * width * 0.47,
                        chassis + 0.0032,
                        hinge_z + base_depth * (0.14 + row as f32 * 0.026),
                    ),
                    Vec3::new(width * 0.018, 0.0004, 0.0008),
                    0.,
                );
            }
        }
    }
}

pub mod displays;
pub(super) use displays::display;
pub(crate) use displays::low_mount_clearance;
pub(crate) use displays::monitor_power_port_local;
pub use displays::{display_geometry, display_parameters, DisplayGeometry, DisplayProgram};
