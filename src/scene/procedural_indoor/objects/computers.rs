//! Laptops are hinged assemblies with keyboards, bezels, ports and rubber feet.
use super::*;

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
    pub port_count: u32,
    pub speaker_grilles: bool,
}
pub fn parameters(o: &IndoorObject) -> LaptopProgram {
    let mut rng = stream(o.seed, 71);
    LaptopProgram {
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
    }
}
pub fn lid_angle(o: &IndoorObject) -> f32 {
    parameters(o).hinge_radians
}

pub(super) fn laptop(a: &mut Assembly, o: &IndoorObject) {
    let label = o.kind.class_name();
    let family = o.variant % LAPTOP_FAMILIES;
    let program = parameters(o);
    let width = o.size.x;
    let chassis = program.chassis_m;
    let metal = if family == 1 || family == 3 {
        Surface::Plastic
    } else {
        Surface::Metal
    };
    let angle = lid_angle(o);
    // Envelope includes the backward lid sweep; base stays fully on its support.
    let tilt = angle - FRAC_PI_2;
    let screen_height = (width / program.aspect_ratio)
        .min((o.size.y - chassis - 0.008) / tilt.cos())
        .min((o.size.z - 0.114) / tilt.sin().max(0.001));
    let rear_sweep = tilt.sin().max(0.0) * screen_height;
    let base_depth = (o.size.z - rear_sweep - 0.014).max(0.10);
    let hinge_z = -o.size.z * 0.5 + rear_sweep + 0.008;
    let base_z = hinge_z + base_depth * 0.5;
    a.box_part(
        metal,
        label,
        Vec3::new(0.0, chassis * 0.5 + 0.003, base_z),
        Vec3::new(width, chassis, base_depth),
        program.corner_m,
    );
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
    dest.part(metal, label).cuboid(
        Vec3::new(width, screen_height, program.lid_m),
        program.corner_m.min(program.lid_m * 0.35),
        lid.with_translation(lid.transform_point(Vec3::Y * screen_height * 0.5)),
    );
    dest.part(Surface::Screen, label).cuboid(
        Vec3::new(width - bezel * 2.0, screen_height - bezel * 2.0, 0.001),
        0.0,
        lid.with_translation(lid.transform_point(Vec3::new(
            0.0,
            screen_height * 0.5,
            program.lid_m * 0.5 + 0.0007,
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
    let columns = program.keyboard_columns;
    for row in 0..5 {
        for col in 0..columns {
            // The bottom row has modifiers and a broad space bar, not a fifth
            // identical row of letter keys.
            if row == 4 && (3..columns - 3).contains(&col) {
                continue;
            }
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(
                    (col as f32 - (columns - 1) as f32 * 0.5) * width * program.keyboard_fraction
                        / columns as f32,
                    chassis + 0.004,
                    hinge_z + base_depth * (0.17 + row as f32 * 0.095),
                ),
                Vec3::new(
                    width * program.keyboard_fraction * 0.86 / columns as f32,
                    0.0018,
                    base_depth * 0.075,
                ),
                0.0006,
            );
        }
    }
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(0., chassis + 0.004, hinge_z + base_depth * 0.55),
        Vec3::new(
            width * program.keyboard_fraction * (columns - 6) as f32 / columns as f32 * 0.97,
            0.0018,
            base_depth * 0.075,
        ),
        0.0006,
    );
    a.box_part(
        Surface::Plastic,
        label,
        Vec3::new(0.0, chassis + 0.0032, hinge_z + base_depth * 0.79),
        Vec3::new(width * program.trackpad_fraction, 0.001, base_depth * 0.23),
        0.0,
    );
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

#[derive(Debug, serde::Serialize)]
pub struct DisplayProgram {
    pub aspect: f32,
    pub bezel_m: f32,
    pub panel_m: f32,
    pub stand: u32,
    pub lift_fraction: f32,
    pub curvature_radius_m: f32,
    pub tilt_radians: f32,
    pub chin_fraction: f32,
    pub stand_width: f32,
    pub stand_depth: f32,
    pub mount_offset: f32,
    pub casing: Surface,
}
pub fn display_parameters(o: &IndoorObject) -> DisplayProgram {
    let mut rng = stream(o.seed, 873);
    let lift_fraction = rng.random_range(0.15..0.29);
    let wall = o.kind == ObjectKind::Display;
    let mut detail = stream(o.seed, 875);
    // Honour a portrait or broad-panel proposal; the previous independent
    // landscape-only aspect silently shrank every portrait display.
    let envelope_aspect = o.size.x / (o.size.y * if wall { 1. } else { 1. - lift_fraction });
    DisplayProgram {
        aspect: (envelope_aspect * rng.random_range(0.90..1.12)).clamp(0.60, 3.25),
        bezel_m: rng.random_range(0.003..0.032),
        panel_m: rng.random_range(0.009..0.060),
        stand: rng.random_range(0..4),
        lift_fraction,
        chin_fraction: detail.random_range(0.0..0.065),
        stand_width: detail.random_range(0.30..0.82),
        stand_depth: detail.random_range(0.40..0.85),
        mount_offset: detail.random_range(-0.17..0.17),
        casing: match detail.random_range(0..10) {
            0 => Surface::WoodEdge,
            1 | 2 => Surface::Metal,
            _ => Surface::Plastic,
        },
        curvature_radius_m: if !wall && rng.random_bool(0.45) {
            rng.random_range(0.80..2.4)
        } else {
            0.
        },
        tilt_radians: if wall {
            0.
        } else {
            rng.random_range(-0.055..0.23)
        },
    }
}

/// Closed horizontally curved panel with analytic surface normals. The screen
/// itself uses a separate millimetre layer and continuous image coordinates.
fn panel(
    geometry: &mut Geometry,
    width: f32,
    height: f32,
    thickness: f32,
    radius: f32,
    transform: Transform,
) {
    let columns = if radius > 0. { 24 } else { 1 };
    let profile = |x: f32| {
        if radius > 0. {
            let z = radius - (radius * radius - x * x).max(0.).sqrt();
            let normal = Vec3::new(-x / radius, 0., 1. - z / radius).normalize();
            (z, normal)
        } else {
            (0., Vec3::Z)
        }
    };
    let mut quad = |points: [Vec3; 4], normals: [Vec3; 4], uvs: [Vec2; 4]| {
        let start = geometry.positions.len() as u32;
        for ((point, normal), uv) in points.into_iter().zip(normals).zip(uvs) {
            geometry
                .positions
                .push(transform.transform_point(point).to_array());
            geometry
                .normals
                .push((transform.rotation * normal).to_array());
            geometry.uvs.push(uv.to_array());
        }
        geometry
            .indices
            .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
    };
    let y = height * 0.5;
    let t = thickness * 0.5;
    for i in 0..columns {
        let x0 = -width * 0.5 + width * i as f32 / columns as f32;
        let x1 = -width * 0.5 + width * (i + 1) as f32 / columns as f32;
        let (z0, n0) = profile(x0);
        let (z1, n1) = profile(x1);
        let uv = [
            Vec2::new(x0 + width * 0.5, height),
            Vec2::new(x1 + width * 0.5, height),
            Vec2::new(x1 + width * 0.5, 0.),
            Vec2::new(x0 + width * 0.5, 0.),
        ];
        quad(
            [
                Vec3::new(x0, -y, z0 + t),
                Vec3::new(x1, -y, z1 + t),
                Vec3::new(x1, y, z1 + t),
                Vec3::new(x0, y, z0 + t),
            ],
            [n0, n1, n1, n0],
            uv,
        );
        quad(
            [
                Vec3::new(x1, -y, z1 - t),
                Vec3::new(x0, -y, z0 - t),
                Vec3::new(x0, y, z0 - t),
                Vec3::new(x1, y, z1 - t),
            ],
            [-n1, -n0, -n0, -n1],
            uv,
        );
        quad(
            [
                Vec3::new(x0, y, z0 + t),
                Vec3::new(x1, y, z1 + t),
                Vec3::new(x1, y, z1 - t),
                Vec3::new(x0, y, z0 - t),
            ],
            [Vec3::Y; 4],
            uv,
        );
        quad(
            [
                Vec3::new(x0, -y, z0 - t),
                Vec3::new(x1, -y, z1 - t),
                Vec3::new(x1, -y, z1 + t),
                Vec3::new(x0, -y, z0 + t),
            ],
            [Vec3::NEG_Y; 4],
            uv,
        );
    }
    for side in [-1., 1.] {
        let x = width * 0.5 * side;
        let (z, _) = profile(x);
        let mut points = [
            Vec3::new(x, -y, z - t),
            Vec3::new(x, -y, z + t),
            Vec3::new(x, y, z + t),
            Vec3::new(x, y, z - t),
        ];
        if side > 0. {
            points.reverse();
        }
        quad(points, [Vec3::X * side; 4], [Vec2::ZERO; 4]);
    }
}

/// Effective geometry, including footprint-constrained tilt and curved depth.
/// Export these fields when reporting rendered construction diversity.
#[derive(Debug, Clone, Copy)]
pub struct DisplayGeometry {
    pub width_m: f32,
    pub height_m: f32,
    pub bezel_m: f32,
    pub panel_m: f32,
    pub curvature_radius_m: f32,
    pub tilt_radians: f32,
    pub panel_transform: Transform,
}

pub fn display_geometry(o: &IndoorObject) -> DisplayGeometry {
    geometry_parameters(o, &display_parameters(o))
}

fn geometry_parameters(o: &IndoorObject, p: &DisplayProgram) -> DisplayGeometry {
    let s = o.size;
    let wall = o.kind == ObjectKind::Display;
    let bottom = if wall { 0. } else { s.y * p.lift_fraction };
    let height = (s.x / p.aspect).min(s.y - bottom - if wall { 0. } else { 0.025 });
    let width = (height * p.aspect).min(s.x);
    let thickness = p.panel_m.min(s.z * 0.65);
    let radius = p.curvature_radius_m.max(width * 0.55);
    let radius = if p.curvature_radius_m > 0. {
        radius
    } else {
        0.
    };
    let sag = if radius > 0. {
        radius - (radius * radius - width * width * 0.25).sqrt()
    } else {
        0.
    };
    // Reserve the complete rotated panel, not just its centre. The support
    // footprint, collision mask and screen placement therefore remain valid.
    let available_depth = (s.z - sag - thickness - 0.018).max(0.);
    let tilt = p.tilt_radians.clamp(
        -(available_depth / height).min(0.9).asin(),
        (available_depth / height).min(0.9).asin(),
    );
    let half_depth = height * tilt.sin().abs() * 0.5 + (sag + thickness) * 0.5 * tilt.cos();
    let z = if wall {
        0.
    } else {
        -s.z * 0.5 + half_depth + 0.009 - sag * 0.5
    };
    let y = bottom + height * tilt.cos() * 0.5 + (sag + thickness * 0.5) * tilt.sin().abs();
    DisplayGeometry {
        width_m: width,
        height_m: height,
        bezel_m: p.bezel_m.min(width.min(height) * 0.08),
        panel_m: thickness,
        curvature_radius_m: radius,
        tilt_radians: tilt,
        panel_transform: Transform::from_xyz(0., y, z).with_rotation(Quat::from_rotation_x(tilt)),
    }
}

/// A mains inlet on the display's rear electronics pedestal. Its housing is
/// connected to the stand rather than placing a floating port on the envelope.
pub(crate) fn monitor_power_port_local(o: &IndoorObject) -> (Vec3, Vec3) {
    (Vec3::new(0., 0.022, -o.size.z * 0.5 - 0.001), Vec3::NEG_Z)
}

pub(super) fn display(a: &mut Assembly, o: &IndoorObject) {
    let p = display_parameters(o);
    let d = geometry_parameters(o, &p);
    let s = o.size;
    let label = o.kind.class_name();
    let wall = o.kind == ObjectKind::Display;
    let bottom = if wall { 0. } else { s.y * p.lift_fraction };
    let width = d.width_m;
    let height = d.height_m;
    let thickness = d.panel_m;
    let radius = d.curvature_radius_m;
    let tf = d.panel_transform;
    let z = tf.translation.z;
    let casing = p.casing;
    panel(a.part(casing, label), width, height, thickness, radius, tf);
    let bezel = d.bezel_m;
    panel(
        a.part(
            if wall {
                Surface::Television
            } else {
                Surface::Screen
            },
            label,
        ),
        width - bezel * 2.,
        height - bezel * 2. - height * p.chin_fraction,
        0.001,
        radius,
        tf.with_translation(tf.transform_point(Vec3::new(
            0.,
            height * p.chin_fraction * 0.5,
            thickness * 0.5 + 0.0008,
        ))),
    );
    if !wall {
        if p.stand == 0 {
            a.part(Surface::Metal, label).cylinder(
                width.min(s.z) * 0.43,
                0.012,
                Transform::from_xyz(0., 0.006, 0.),
            );
        } else if p.stand == 1 {
            for side in [-1., 1.] {
                a.part(Surface::Metal, label).rod(
                    Vec3::new(0., 0.012, z),
                    Vec3::new(side * width * 0.40, 0.012, s.z * 0.29),
                    0.01,
                );
            }
        } else {
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0., 0.006, 0.),
                Vec3::new(width * p.stand_width, 0.012, s.z * p.stand_depth),
                0.004,
            );
        }
        // Integrated power electronics at the rear of the base allow a real
        // AC connector to leave the housing without crossing the screen body.
        let (port, normal) = monitor_power_port_local(o);
        let rear = port.z + 0.001;
        a.box_part(
            casing,
            label,
            Vec3::new(0., 0.021, rear + 0.024),
            Vec3::new(0.072, 0.042, 0.048),
            0.003,
        );
        a.box_part(
            casing,
            label,
            Vec3::new(0., 0.013, (rear + 0.048) * 0.5),
            Vec3::new(0.027, 0.020, (rear + 0.048).abs() + 0.002),
            0.002,
        );
        // The socket is one millimetre ahead of the archived connector's tip;
        // first-segment contact is the only allowed housing intersection.
        a.box_part(
            Surface::Ink,
            label,
            port - normal * 0.0009,
            Vec3::new(0.014, 0.009, 0.0008),
            0.0008,
        );
        // A bent riser connects a rear panel mount to the actual floor base.
        // Its radius remains small even for a portrait monitor's tall stand.
        let attachment = tf.transform_point(Vec3::new(
            width * p.mount_offset,
            -height * 0.22,
            -thickness * 0.5,
        ));
        let foot = Vec3::new(0., 0.012, 0.);
        let elbow = Vec3::new(0., attachment.y * 0.42, z);
        a.part(Surface::Metal, label).rod(foot, elbow, 0.009);
        a.part(Surface::Metal, label).rod(elbow, attachment, 0.009);
    } else {
        // A real bracket bridges the panel to the reserved rear mounting face.
        let rear = -thickness * 0.5;
        let depth = rear + s.z * 0.5;
        if depth > 0.001 {
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0., bottom + height * 0.5, rear - depth * 0.5),
                Vec3::new(width * 0.32, height * 0.32, depth),
                0.002,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation::validate_geometry,
    };

    #[test]
    fn curved_portrait_and_broad_displays_keep_closed_normals_and_reserved_bounds() {
        let mut scene =
            IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut template = scene.objects[0].clone();
        scene.objects.clear();
        let mut curved = 0;
        let mut portrait = 0;
        let mut landscape = 0;
        for seed in 0..96 {
            for (kind, size) in [
                (ObjectKind::Monitor, Vec3::new(0.29, 0.56, 0.19)),
                (ObjectKind::Monitor, Vec3::new(0.83, 0.33, 0.19)),
                (ObjectKind::Monitor, Vec3::new(0.46, 0.48, 0.28)),
                (ObjectKind::Display, Vec3::new(0.85, 0.55, 0.07)),
                (ObjectKind::Display, Vec3::new(2.1, 1.18, 0.07)),
            ] {
                template.id = scene.objects.len();
                template.kind = kind;
                template.size = size;
                template.seed = seed;
                template.position = Vec3::ZERO;
                template.support = Some(0);
                template.solid = false;
                template.yaw = 0.;
                let p = display_parameters(&template);
                assert_eq!(
                    serde_json::to_value(&p).unwrap(),
                    serde_json::to_value(display_parameters(&template)).unwrap()
                );
                curved += usize::from(p.curvature_radius_m > 0.);
                portrait += usize::from(p.aspect < 1.);
                landscape += usize::from(p.aspect > 2.);
                let a = build_object(&template);
                for geometry in a.parts.values() {
                    for triangle in geometry.indices.as_chunks::<3>().0 {
                        let p = triangle.map(|i| Vec3::from_array(geometry.positions[i as usize]));
                        let normal = (p[1] - p[0]).cross(p[2] - p[0]);
                        assert!(normal.length_squared() > 1e-20);
                        let n = triangle
                            .iter()
                            .map(|&i| Vec3::from_array(geometry.normals[i as usize]))
                            .sum::<Vec3>();
                        assert!(normal.dot(n) > 0., "inverted {kind:?} panel, seed {seed}");
                    }
                }
                scene.objects.push(template.clone());
            }
        }
        assert!(curved > 96 && portrait > 64 && landscape > 64);
        validate_geometry(&scene).unwrap();
    }
}
