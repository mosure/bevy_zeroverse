//! Metric housing proportions vary independently of standardized controls.
use super::*;
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct FixtureProgram {
    pub gangs: usize,
    pub style: u8,
    pub edge_radius: f32,
    pub rocker_angle: f32,
    pub finish: Surface,
    pub screws: bool,
    pub case_fraction: Vec2,
    pub control_fraction: Vec2,
    pub receptacle_height: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank: Option<OutletBank>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub switch_bank: Option<switches::SwitchBank>,
}
pub fn parameters(o: &IndoorObject) -> FixtureProgram {
    let mut rng = stream(o.seed, 3070);
    let mut fit = stream(o.seed, 7490);
    let mut section = stream(o.seed, 0x504c415445534543);
    let case_fraction = Vec2::new(
        section.random_range(0.80..1.),
        section.random_range(0.72..1.),
    );
    let capacity = (o.size.x * case_fraction.x / 0.075).round().clamp(1., 3.) as usize;
    let mut program = FixtureProgram {
        gangs: if capacity > 1 && fit.random_bool(0.35) {
            capacity - 1
        } else {
            capacity
        },
        // Receptacle standard is room-level; optional USB inserts are instance-level.
        style: if o.kind == ObjectKind::WallOutlet {
            (o.variant % 3) as u8 + if rng.random_bool(0.35) { 3 } else { 0 }
        } else {
            rng.random_range(0..6)
        },
        edge_radius: rng.random_range(0.001..0.006),
        rocker_angle: rng.random_range(-0.025..0.025),
        finish: [Surface::Metal, Surface::Plastic, Surface::Ceramic][rng.random_range(0..3)],
        screws: rng.random_bool(0.6),
        case_fraction,
        control_fraction: Vec2::new(
            section.random_range(0.48..0.72),
            section.random_range(0.52..0.70),
        ),
        receptacle_height: section.random_range(0.25..0.32),
        bank: None,
        switch_bank: None,
    };
    if let Some(bank) = OutletBank::fitted(o, &program) {
        // Keep the existing public measurements tied to the realized insert.
        program.control_fraction = bank.module_size
            / Vec2::new(
                bank.case_size.x / bank.plate_slots() as f32,
                bank.case_size.y,
            );
        program.receptacle_height = bank.socket_local(0).unwrap().y / bank.case_size.y;
        program.bank = Some(bank);
    } else if let Some(primary) = switches::primary(o) {
        let bank = switches::SwitchBank::fitted(o, &program, primary);
        program.style = primary;
        program.finish = bank.finish;
        program.edge_radius = bank.corner_m;
        program.screws = bank.screws;
        program.rocker_angle = match bank.modules[0] {
            switches::Actuator::Paddle { angle_rad } => angle_rad,
            switches::Actuator::Split { angles_rad } => angles_rad[0],
            _ => 0.,
        };
        program.switch_bank = Some(bank);
    }
    program
}
/// Lower receptacle of the first gang. Cable endpoints use this exact frame.
pub(crate) fn outlet_port_local(o: &IndoorObject) -> Vec3 {
    let p = parameters(o);
    if let Some(bank) = p.bank.as_ref() {
        return bank.socket_local(0).unwrap();
    }
    let s = case_size(o, &p);
    let gangs = p.gangs as f32;
    Vec3::new(
        -(gangs - 1.) * 0.5 * s.x / gangs,
        s.y * p.receptacle_height,
        o.size.z * 0.52 + 0.001,
    )
}
fn case_size(o: &IndoorObject, p: &FixtureProgram) -> Vec3 {
    o.size * Vec3::new(p.case_fraction.x, p.case_fraction.y, 1.)
}
fn disk(a: &mut Assembly, surface: Surface, label: &str, p: Vec3, r: f32, depth: f32) {
    a.part(surface, label).cylinder(
        r,
        depth,
        Transform::from_translation(p).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
}

fn control_setting(seed: u64, gang: usize) -> f32 {
    stream(seed, 0x5357_4954_4348_0000 + gang as u64).random_range(-1_f32..1.)
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = case_size(o, &p);
    let label = o.kind.class_name();
    if let Some(bank) = p.bank.as_ref() {
        bank.build(a, label);
        return;
    }
    if let Some(bank) = &p.switch_bank {
        bank.build(a, label);
        return;
    }
    // A toggle needs room for its lever. Keep the same wall contact but
    // transfer depth from the faceplate to the projecting actuator.
    let toggle = o.kind == ObjectKind::LightSwitch && p.style == 1;
    let rear = -0.35 * s.z;
    let front = if toggle { -0.04 } else { 0.35 } * s.z;
    a.box_part(
        p.finish,
        label,
        Vec3::new(0., s.y * 0.5, (rear + front) * 0.5),
        s.with_z(front - rear),
        p.edge_radius,
    );
    let pitch = s.x / p.gangs as f32;
    for gang in 0..p.gangs {
        let x = (gang as f32 - (p.gangs - 1) as f32 * 0.5) * pitch;
        let face = if toggle {
            front + (0.07 * s.z).min(0.0012)
        } else {
            s.z * 0.48
        };
        if o.kind == ObjectKind::LightSwitch {
            // The recess and controls are *in front* of the plate, not hidden
            // inside it. Separation remains positive at every rocker angle.
            let w = pitch * p.control_fraction.x;
            let h = s.y * p.control_fraction.y;
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(x, s.y * 0.5, front + s.z * 0.01),
                Vec3::new(w + 0.002, h + 0.002, 0.0003),
                0.0001,
            );
            match p.style {
                0 => {
                    // Broad rocker with sampled on/off angle.
                    a.part(Surface::Plastic, label).cuboid(
                        Vec3::new(w, h, 0.003),
                        0.001,
                        Transform::from_xyz(x, s.y * 0.5, face + 0.0005)
                            .with_rotation(Quat::from_rotation_x(p.rocker_angle)),
                    );
                }
                1 => {
                    // The insulating lever projects from a pivot seated in
                    // the escutcheon. Each gang has its own up/down state.
                    a.box_part(
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        Vec3::new(w, h, 0.002),
                        0.001,
                    );
                    let setting = control_setting(o.seed, gang);
                    let angle = setting.signum() * (0.38 + setting.abs() * 0.27);
                    let rotation = Quat::from_rotation_x(angle);
                    let root = Vec3::new(x, s.y * 0.5, face + (0.02 * s.z).min(0.0004));
                    let half = (s.z * 0.095).min(0.0018);
                    let length = (0.49 * s.z - root.z - half * angle.sin().abs()) / angle.cos();
                    a.part(Surface::Plastic, label).cuboid(
                        Vec3::new((w * 0.24).clamp(0.005, 0.010), 2. * half, length),
                        0.0007,
                        Transform::from_translation(root + rotation * Vec3::Z * (length * 0.5))
                            .with_rotation(rotation),
                    );
                }
                2 => {
                    // Rotary dimmer with visible pointer.
                    let diameter = w.min(h);
                    disk(
                        a,
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        diameter * 0.46,
                        0.004,
                    );
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.5 + diameter * 0.25, face + 0.002),
                        Vec3::new(0.001, diameter * 0.19, 0.0005),
                        0.,
                    );
                }
                3 => {
                    // Split rocker.
                    for side in [-1., 1.] {
                        a.box_part(
                            Surface::Plastic,
                            label,
                            Vec3::new(x + side * w * 0.255, s.y * 0.5, face + 0.001),
                            Vec3::new(w * 0.46, h, 0.004),
                            0.001,
                        );
                    }
                }
                4 => {
                    // Sliding dimmer.
                    a.box_part(
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        Vec3::new(w, h, 0.002),
                        0.001,
                    );
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.5, face + 0.0013),
                        Vec3::new(w * 0.10, h * 0.75, 0.0003),
                        0.,
                    );
                    a.box_part(
                        Surface::Plastic,
                        label,
                        Vec3::new(
                            x,
                            s.y * 0.5 + h * 0.275 * control_setting(o.seed, gang),
                            face + 0.002,
                        ),
                        Vec3::new(w * 0.62, h * 0.20, 0.002),
                        0.0005,
                    );
                }
                _ => {
                    // Two push buttons, distinct status lens.
                    for y in [0.33, 0.67] {
                        disk(
                            a,
                            Surface::Plastic,
                            label,
                            Vec3::new(x, s.y * y, face),
                            (w.min(h) * 0.39).min(s.y * 0.145),
                            0.004,
                        );
                    }
                    disk(
                        a,
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.84, face),
                        0.0012,
                        0.0008,
                    );
                }
            }
        } else {
            for y in [s.y * p.receptacle_height, s.y * (1. - p.receptacle_height)] {
                let center = Vec3::new(x, y, face);
                let w = (pitch * p.control_fraction.x).min(0.050);
                let h = (s.y * p.control_fraction.y * 0.46)
                    .max(0.028)
                    .min(s.y * 0.42);
                if p.style == 1 || p.style == 4 {
                    disk(a, Surface::Plastic, label, center, w.min(h) * 0.49, 0.003);
                    disk(
                        a,
                        Surface::Ink,
                        label,
                        center + Vec3::Z * 0.0018,
                        w.min(h) * 0.37,
                        0.0004,
                    );
                    disk(
                        a,
                        p.finish,
                        label,
                        center + Vec3::Z * 0.0022,
                        w.min(h) * 0.32,
                        0.0004,
                    );
                } else {
                    a.box_part(
                        Surface::Plastic,
                        label,
                        center,
                        Vec3::new(w, h, 0.003),
                        0.002,
                    );
                }
                let z = s.z * 0.52 + 0.001;
                for sign in [-1., 1.] {
                    if p.style == 1 || p.style == 4 {
                        disk(
                            a,
                            Surface::Ink,
                            label,
                            Vec3::new(x + sign * w.min(h) * 0.18, y, z + 0.001),
                            w.min(h) * 0.065,
                            0.0005,
                        );
                    } else {
                        a.box_part(
                            Surface::Ink,
                            label,
                            Vec3::new(x + sign * w * 0.18, y + 0.002, z),
                            Vec3::new(
                                if p.style % 3 == 2 { 0.006 } else { 0.0025 },
                                if p.style % 3 == 2 { 0.003 } else { 0.009 },
                                0.0006,
                            ),
                            0.,
                        );
                    }
                }
                if p.style % 3 != 1 {
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(
                            x,
                            y + if p.style % 3 == 2 { 0.010 } else { -0.008 },
                            z + 0.0002,
                        ),
                        Vec3::new(0.003, 0.004, 0.0006),
                        0.0005,
                    );
                }
            }
            if p.style >= 3 {
                // USB-A / USB-C inserts between the mains receptacles.
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(x, s.y * 0.50, face + 0.0018),
                    Vec3::new(if p.style == 3 { 0.013 } else { 0.008 }, 0.0035, 0.0006),
                    0.0005,
                );
            }
        }
    }
    if p.screws {
        for y in [s.y * 0.075, s.y * 0.925] {
            disk(
                a,
                Surface::Metal,
                label,
                Vec3::new(0., y, front + s.z * 0.02),
                0.0019,
                0.0006,
            );
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(0., y, front + s.z * 0.02 + 0.0005),
                Vec3::new(0.0024, 0.0004, 0.0003),
                0.,
            );
        }
    }
}
