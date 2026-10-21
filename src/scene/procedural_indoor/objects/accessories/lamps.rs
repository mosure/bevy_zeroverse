use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskLampProgram {
    pub arm: AccessoryProgram,
    /// Linear LED, circular puck, deep reflector, or shaded table lamp.
    pub head: u8,
    pub head_height: f32,
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
    TaskLampProgram {
        arm: parameters(o),
        head: r.random_range(0..4),
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
    let head = lamp_head(o, p, &detail);
    if p.style < 2 {
        // Parallel articulated links with pivot discs and a real spring.
        for z in [-0.008, 0.008] {
            let shift = Vec3::Z * z;
            a.part(p.finish, label)
                .rod(root + shift, elbow + shift, 0.004);
            a.part(p.finish, label)
                .rod(elbow + shift, head + shift, 0.004);
        }
        for joint in [root, elbow, head] {
            a.part(Surface::Chrome, label).cylinder(
                0.011,
                0.023,
                Transform::from_translation(joint).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
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
        let points: Vec<_> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.;
                root.lerp(head, t)
                    + Vec3::new(
                        0.,
                        s.y * 0.15 * (PI * t).sin(),
                        -s.z * 0.12 * (PI * t).sin(),
                    )
            })
            .collect();
        a.part(p.finish, label).tube(&points, 0.008, 10);
    }
    // Switch and independently sampled head construction. Every aperture uses
    // the same lower datum so direct and indirect emitters cannot become buried.
    a.part(Surface::Rubber, label).cylinder(
        s.x.min(s.z) * 0.045,
        0.003,
        Transform::from_xyz(s.x * 0.17, s.y * 0.055 + 0.001, s.z * 0.12),
    );
    let bottom = head.y - s.y * 0.065;
    if detail.head >= 1 {
        let radius = head_radius(o, &detail, head);
        if detail.head == 3 {
            luminaires::shade(
                a,
                label,
                &detail.shade,
                head.with_y(bottom),
                radius,
                s.y * 0.14,
                0.65 + detail.shade.shoulder * 0.3,
            );
            luminaires::bulb(
                a,
                label,
                &detail.shade,
                head.with_y(bottom + s.y * 0.07),
                s.y * 0.022,
            );
        } else {
            let top_radius = radius * if detail.head == 2 { 0.45 } else { 1. };
            a.part(detail.shade.trim, label).lathe(
                &[
                    (radius, -s.y * 0.065),
                    (top_radius, s.y * 0.05),
                    (top_radius - 0.003, s.y * 0.05),
                    (radius - 0.003, -s.y * 0.065),
                    (radius, -s.y * 0.065),
                ],
                28,
                Transform::from_translation(head),
            );
        }
        a.part(Surface::Light, label).cylinder(
            radius * 0.82,
            0.001,
            Transform::from_translation(head.with_y(bottom + 0.002)),
        );
        return;
    }
    // A shallow down-facing LED head; the diffuser is separated from its case.
    let width = s.x * detail.bar_width;
    let depth = s.z * detail.bar_depth;
    a.box_part(
        p.finish,
        label,
        head.with_y(bottom + s.y * 0.04),
        Vec3::new(width, s.y * 0.08, depth),
        0.008,
    );
    a.box_part(
        Surface::Light,
        label,
        head.with_y(bottom - 0.001),
        Vec3::new(width * 0.88, 0.001, depth * 0.83),
        0.001,
    );
}

fn lamp_head(o: &IndoorObject, p: &AccessoryProgram, detail: &TaskLampProgram) -> Vec3 {
    Vec3::new(
        o.size.x * p.tilt * 0.45,
        o.size.y * detail.head_height,
        o.size.z * (p.reach - 0.35).clamp(-0.15, 0.27),
    )
}

fn head_radius(o: &IndoorObject, p: &TaskLampProgram, head: Vec3) -> f32 {
    (o.size.x.min(o.size.z) * p.aperture_fraction)
        .min(o.size.x * 0.5 - head.x.abs() - 0.002)
        .min(o.size.z * 0.5 - head.z.abs() - 0.002)
}
pub(crate) fn emitting_area(o: &IndoorObject) -> f32 {
    let p = lamp_parameters(o);
    if p.head == 0 {
        o.size.x * p.bar_width * 0.88 * o.size.z * p.bar_depth * 0.83
    } else {
        PI * (head_radius(o, &p, lamp_head(o, &p.arm, &p)) * 0.82).powi(2)
    }
}

/// The direct renderer and diffuse transport use the same metric emitter. Its
/// position sits just below the inset diffuser, outside the opaque housing.
pub(super) fn photometry(o: &IndoorObject) -> (Vec3, Vec3, f32, f32, f32) {
    let detail = lamp_parameters(o);
    let p = &detail.arm;
    let mut rng = stream(o.seed, 5320);
    let position = o
        .transform()
        .transform_point(lamp_head(o, p, &detail) - Vec3::Y * (o.size.y * 0.065 + 0.003));
    let color =
        crate::scene::procedural_indoor::materials::kelvin_rgb(rng.random_range(2700.0..5800.0));
    (
        position,
        color,
        rng.random_range(150.0..550.0),
        rng.random_range(0.32..0.56),
        rng.random_range(0.82..1.10),
    )
}
