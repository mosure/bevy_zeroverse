use super::*;
mod head;
use head::LampHead;
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskLampProgram {
    pub arm: AccessoryProgram,
    /// Linear LED, circular puck, deep reflector, or shaded table lamp.
    pub head: u8,
    pub head_height: f32,
    pub head_tilt_radians: f32,
    pub head_swivel_radians: f32,
    pub elbow_height: f32,
    pub aperture_fraction: f32,
    pub bar_width: f32,
    pub bar_depth: f32,
    pub base_width: f32,
    pub base_depth: f32,
    pub shade: luminaires::ShadeProgram,
}
pub fn lamp_parameters(o: &IndoorObject) -> TaskLampProgram {
    let mut r = stream(o.seed, 7591);
    let head = r.random_range(0..4);
    let mut aim = stream(o.seed, 0x0054_4153_4b41_494d);
    TaskLampProgram {
        arm: parameters(o),
        head,
        head_tilt_radians: if head == 3 {
            0.
        } else {
            aim.random_range(0.0_f32..1.).powf(0.68) * 47_f32.to_radians()
        },
        head_swivel_radians: if head == 3 {
            0.
        } else {
            aim.random_range(-0.78..0.78)
        },
        shade: luminaires::parameters(o.seed),
        head_height: r.random_range(0.74..0.93),
        elbow_height: r.random_range(0.34..0.66),
        aperture_fraction: r.random_range(0.18..0.36),
        bar_width: r.random_range(0.43..0.80),
        bar_depth: r.random_range(0.20..0.43),
        base_width: r.random_range(0.28..0.46),
        base_depth: r.random_range(0.26..0.46),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &AccessoryProgram) {
    let detail = lamp_parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    // Keep the construction radius metric: shrinking a unit cylinder would
    // leave metre UVs ten times too dense even though its geometry fits.
    let base_radius = s.x * detail.base_width;
    a.part(p.finish, label).cylinder(
        base_radius,
        s.y * 0.055,
        Transform::from_xyz(0., s.y * 0.0275, 0.).with_scale(Vec3::new(
            1.,
            1.,
            s.z * detail.base_depth / base_radius,
        )),
    );
    let root = Vec3::new(0., s.y * 0.055, -s.z * 0.16);
    let elbow = Vec3::new(-s.x * p.tilt, s.y * detail.elbow_height, -s.z * 0.24);
    let optics = LampHead::new(o, &detail);
    let head = optics.join_arm(a, o, detail.shade.trim);
    if p.style < 2 {
        // Parallel articulated links with pivot discs and a real spring.
        for sign in [-1., 1.] {
            let shift = Vec3::X * (sign * LampHead::joint_radius(o));
            a.part(p.finish, label)
                .rod(root + shift, elbow + shift, 0.004);
            a.part(p.finish, label)
                .rod(elbow + shift, head + shift, 0.004);
        }
        for joint in [root, elbow] {
            a.part(Surface::Chrome, label).cylinder(
                0.011,
                0.023,
                Transform::from_translation(joint).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        }
        let start = root.lerp(elbow, 0.14) + Vec3::X * 0.012;
        let end = root.lerp(elbow, 0.50) + Vec3::X * 0.012;
        let axis = (end - start).normalize();
        let cross = axis.cross(Vec3::Z).normalize();
        let coils: Vec<_> = (0..=48)
            .map(|i| {
                let t = i as f32 / 48.;
                start.lerp(end, t)
                    + (cross * (TAU * 6. * t).cos() + Vec3::Z * (TAU * 6. * t).sin()) * 0.004
            })
            .collect();
        a.part(Surface::Chrome, label).tube(&coils, 0.0007, 5);
    } else {
        let radius = 0.008;
        a.part(p.finish, label)
            .tube(&gooseneck(root, head, s, radius), radius, 10);
    }
    // Switch and independently sampled head construction. Every aperture uses
    // the same lower datum so direct and indirect emitters cannot become buried.
    a.part(Surface::Rubber, label).cylinder(
        s.x.min(s.z) * 0.045,
        0.003,
        Transform::from_xyz(s.x * 0.17, s.y * 0.055 + 0.001, s.z * 0.12),
    );
    optics.build(a, o, &detail);
}

fn gooseneck(root: Vec3, pin: Vec3, size: Vec3, radius: f32) -> [Vec3; 13] {
    let chord = pin - root;
    let axis = chord.normalize();
    let desired = Vec3::new(0., size.y * 0.15, -size.z * 0.12);
    // A transverse sine bow advances monotonically along the chord. Its
    // maximum curvature is π² × bow / chord², at the middle of the arm.
    let bend = (desired - axis * desired.dot(axis))
        .clamp_length_max(chord.length_squared() / (3. * radius * PI * PI));
    let low = Vec3::new(-0.5 * size.x, 0., -0.5 * size.z) + radius;
    let high = Vec3::new(0.5 * size.x, size.y, 0.5 * size.z) - radius;
    let mut scale = 1.0_f32;
    for i in 0..3 {
        if bend[i] > 0. {
            scale = scale.min((high[i] - root[i].max(pin[i])) / bend[i]);
        } else if bend[i] < 0. {
            scale = scale.min((low[i] - root[i].min(pin[i])) / bend[i]);
        }
    }
    let bow = bend * scale.max(0.);
    std::array::from_fn(|i| {
        let t = i as f32 / 12.;
        root.lerp(pin, t) + bow * (PI * t).sin()
    })
}

pub(crate) fn emitting_area(o: &IndoorObject) -> f32 {
    LampHead::new(o, &lamp_parameters(o)).area()
}

pub(crate) fn beam(o: &IndoorObject) -> crate::scene::procedural_indoor::photometry::FixtureBeam {
    let p = lamp_parameters(o);
    LampHead::new(o, &p).beam(o, &p)
}

pub(crate) fn reflection_aperture(o: &IndoorObject) -> (Geometry, f32) {
    let head = LampHead::new(o, &lamp_parameters(o));
    let mut g = Geometry::default();
    head.aperture(&mut g);
    (g, head.area())
}
