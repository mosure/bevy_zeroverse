use super::*;

pub(super) fn nominal_groups(seed: u64) -> usize {
    match stream(seed, u64::from_le_bytes(*b"BrewSize")).random_range(0..100) {
        0..25 => 1,
        25..79 => 2,
        _ => 3,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Machine {
    pub groups: usize,
    pub group_pitch_m: f32,
    pub cup_clearance_m: f32,
    pub tray_height_m: f32,
    pub spout_z_m: f32,
    pub body: Surface,
    pub roundedness: f32,
    pub wand_sides: [bool; 2],
    pub manual_paddles: bool,
    pub rail_height_m: f32,
    pub handle_yaw: Vec<f32>,
}
impl Machine {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BrewHead"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"BarEnaml"),
        );
        let groups =
            nominal_groups(o.seed).min(((o.size.x - 0.12) / 0.195).floor().max(1.) as usize);
        let left = r.random_bool(0.5);
        let both = r.random_bool(if groups > 1 { 0.70 } else { 0.16 });
        Self {
            groups,
            group_pitch_m: ((o.size.x - 0.17) / groups as f32).min(0.245),
            cup_clearance_m: r.random_range(0.098..0.158),
            tray_height_m: o.size.y * r.random_range(0.142..0.221),
            spout_z_m: o.size.z * r.random_range(0.112..0.185),
            body: [
                Surface::Accent,
                Surface::Metal,
                Surface::Ink,
                Surface::Plastic,
            ][finish.random_range(0..4)],
            roundedness: r.random_range(2.7..8.6),
            wand_sides: [left || both, !left || both],
            manual_paddles: r.random_bool(0.32),
            rail_height_m: r.random_range(0.019..0.041),
            handle_yaw: (0..groups).map(|_| r.random_range(-0.58..0.58)).collect(),
        }
    }
    fn head_x(&self, index: usize) -> f32 {
        (index as f32 - (self.groups - 1) as f32 * 0.5) * self.group_pitch_m
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = Machine::for_object(o);
    let tray = p.tray_height_m;
    let underside = tray + p.cup_clearance_m + 0.073;
    let top = s.y - p.rail_height_m - 0.004;
    let foot = (0.045 * s.y).min(0.026);
    for x in [-s.x * 0.405, s.x * 0.405] {
        for z in [-s.z * 0.365, s.z * 0.365] {
            a.part(Surface::Rubber, CLASS).cylinder(
                0.026,
                foot,
                Transform::from_xyz(x, foot * 0.5, z),
            );
        }
    }
    // The front drip pan is depressed below the grate. Its bars span a real recess.
    let floor = tray - 0.011;
    a.part(Surface::Metal, STEEL).housing(
        &[
            (foot, s.xz() * 0.453),
            (foot + 0.011, s.xz() * 0.483),
            (floor - 0.005, s.xz() * 0.483),
            (floor, s.xz() * 0.47),
        ],
        6.2,
        Transform::IDENTITY,
    );
    let zfront = s.z * 0.403;
    let zback = -0.002;
    let rail_y = tray - 0.003;
    for z in [zback, zfront] {
        a.box_part(
            Surface::Metal,
            STEEL,
            Vec3::new(0., (floor + tray) * 0.5, z),
            Vec3::new(s.x * 0.84, tray - floor, 0.012),
            0.002,
        );
    }
    let n = (s.x / 0.023).round() as usize;
    for i in 0..n {
        let x = ((i as f32 + 0.5) / n as f32 - 0.5) * s.x * 0.827;
        a.part(Surface::Metal, STEEL).rod(
            Vec3::new(x, rail_y, zback + 0.005),
            Vec3::new(x, rail_y, zfront - 0.005),
            0.003,
        );
    }
    let rear = -s.z * 0.218;
    let rear_half = Vec2::new(s.x * 0.452, s.z * 0.218);
    a.part(p.body, CLASS).housing(
        &[
            (floor, rear_half * 0.96),
            (floor + 0.026, rear_half),
            (underside, rear_half * 0.98),
        ],
        p.roundedness,
        Transform::from_xyz(0., 0., rear),
    );
    // An overhanging fascia supports the brewing heads; each spout opens into the cup bay.
    let shell = Vec2::new(s.x * 0.492, s.z * 0.389);
    let top_z = -s.z * 0.055;
    a.part(p.body, CLASS).housing(
        &[
            (underside, shell * 0.955),
            (underside + 0.020, shell),
            (top - 0.026, shell),
            (top - 0.008, shell * 0.971),
        ],
        p.roundedness,
        Transform::from_xyz(0., 0., top_z),
    );
    a.part(Surface::Metal, STEEL).housing(
        &[(top - 0.008, shell * 0.97), (top, shell * 0.958)],
        p.roundedness,
        Transform::from_xyz(0., 0., top_z),
    );
    let fascia_z = top_z + shell.y + 0.003;
    let control_y = underside + (top - underside) * 0.52;
    let fascia_width = (s.x * 0.75).min(p.group_pitch_m * p.groups as f32 + 0.05);
    // Bracket the flat instrument strip into the curved housing at both ends.
    a.box_part(
        Surface::Metal,
        STEEL,
        Vec3::new(0., control_y, fascia_z - 0.008),
        Vec3::new(fascia_width, 0.055, 0.023),
        0.005,
    );
    let control_z = fascia_z + 0.0055;
    for i in 0..p.groups {
        let x = p.head_x(i);
        let z = p.spout_z_m;
        a.part(Surface::Chrome, CLASS).lathe(
            &[
                (0., -0.049),
                (0.031, -0.049),
                (0.043, -0.041),
                (0.042, -0.027),
                (0.038, -0.024),
                (0.043, -0.020),
                (0.043, 0.),
            ],
            24,
            Transform::from_xyz(x, underside, z),
        );
        let handle = Quat::from_rotation_y(p.handle_yaw[i]) * Vec3::Z;
        let root = Vec3::new(x, underside - 0.036, z);
        let shoulder = root + handle * 0.071;
        a.part(Surface::Chrome, CLASS)
            .rod(root + handle * 0.035, shoulder, 0.006);
        a.part(Surface::Rubber, CLASS).rod(
            shoulder,
            root + handle * (s.z * 0.265) + Vec3::NEG_Y * 0.010,
            0.010,
        );
        for sign in [-1., 1.] {
            a.part(Surface::Chrome, CLASS).tube(
                &[
                    Vec3::new(x, underside - 0.047, z),
                    Vec3::new(x + sign * 0.014, underside - 0.056, z),
                    Vec3::new(x + sign * 0.015, underside - 0.073, z),
                ],
                0.004,
                8,
            );
        }
        if p.manual_paddles {
            a.box_part(
                Surface::Ink,
                CLASS,
                Vec3::new(x, control_y, control_z + 0.003),
                Vec3::new(0.052, 0.014, 0.009),
                0.004,
            );
        } else {
            for k in -1..=1 {
                button(
                    a,
                    Vec3::new(x + 0.019 * k as f32, control_y, control_z),
                    0.0065,
                    Surface::Ink,
                );
            }
        }
    }
    for (i, enabled) in p.wand_sides.into_iter().enumerate() {
        if !enabled {
            continue;
        }
        let side = 2. * i as f32 - 1.;
        let x = side * s.x * 0.419;
        let z = s.z * 0.106;
        a.part(Surface::Chrome, CLASS).tube(
            &[
                Vec3::new(x, underside + 0.001, z),
                Vec3::new(x, underside - 0.018, z),
                Vec3::new(x + side * 0.014, underside - 0.043, z + 0.022),
                Vec3::new(x + side * 0.013, tray + 0.041, z + 0.058),
            ],
            0.005,
            10,
        );
        a.part(Surface::Rubber, CLASS).rod(
            Vec3::new(x + side * 0.014, underside - 0.044, z + 0.022),
            Vec3::new(x + side * 0.013, (underside + tray) * 0.5, z + 0.039),
            0.008,
        );
        // Side-mounted rotary valve, outboard of the cup heads.
        a.part(Surface::Rubber, CLASS).cylinder(
            0.020,
            0.023,
            Transform::from_xyz(side * (s.x * 0.5 - 0.0116), control_y, 0.)
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
    }
    let rail = Vec2::new(s.x * 0.366, s.z * 0.227);
    let y = s.y - 0.004;
    for x in [-rail.x, rail.x] {
        for z in [-rail.y, rail.y] {
            a.part(Surface::Chrome, CLASS).rod(
                Vec3::new(x, top, z + top_z),
                Vec3::new(x, y, z + top_z),
                0.0035,
            );
        }
    }
    a.part(Surface::Chrome, CLASS).tube(
        &[
            Vec3::new(-rail.x, y, rail.y + top_z),
            Vec3::new(-rail.x, y, -rail.y + top_z),
            Vec3::new(rail.x, y, -rail.y + top_z),
            Vec3::new(rail.x, y, rail.y + top_z),
        ],
        0.0035,
        9,
    );
}
