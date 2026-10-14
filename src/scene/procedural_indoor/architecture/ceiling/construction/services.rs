use super::*;

fn route(f: &Frame, t: f32, bend: f32, rise: f32) -> Vec3 {
    let t = t.clamp(0., 1.);
    // Straight ends and a smooth offset in the middle: parallel service runs
    // can step around beams without the sharp joints of a polyline zigzag.
    let u = ((t - 0.28) / 0.44).clamp(0., 1.);
    let offset = u * u * (3. - 2. * u) - 0.5;
    Vec3::new(
        (t - 0.5) * f.length * 0.94,
        f.height * (0.55 + rise * offset * 0.40),
        f.width * bend * offset * 0.40,
    )
}
fn hangers(a: &mut Assembly, f: &Frame, points: &[(Vec3, f32)]) -> Vec<Vec3> {
    let mut anchors = Vec::new();
    for &(p, radius_y) in points {
        let y = p.y - radius_y - 0.007;
        f.block(
            a,
            Surface::Metal,
            Vec3::new(p.x, y, 0.),
            Vec3::new(0.026, 0.016, f.width * 0.91),
            0.002,
        );
        for z in [-f.width * 0.445, f.width * 0.445] {
            anchors.push(Vec3::new(p.x, y + 0.006, z));
        }
    }
    anchors
}
fn grille(a: &mut Assembly, f: &Frame, p: Vec3, length: f32, width: f32) {
    f.block(
        a,
        Surface::Metal,
        p + Vec3::Y * 0.014,
        Vec3::new(length, 0.028, width),
        0.003,
    );
    // Recess, frame and separate angled louvers. The dark recess is raised
    // below the housing bottom instead of competing with its face.
    f.block(
        a,
        Surface::Ink,
        p - Vec3::Y * 0.002,
        Vec3::new(length - 0.012, 0.003, width - 0.012),
        0.,
    );
    let count = (length / 0.025).round().clamp(3., 12.) as usize;
    for i in 0..count {
        let x = (i as f32 + 0.5) / count as f32 * (length - 0.022) - (length - 0.022) * 0.5;
        f.part(a, Surface::Metal).cuboid(
            Vec3::new(0.008, 0.004, width - 0.018),
            0.001,
            Transform::from_translation(p + Vec3::new(x, -0.005, 0.))
                .with_rotation(Quat::from_rotation_z(0.3)),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn duct(
    a: &mut Assembly,
    f: &Frame,
    section: f32,
    end_scale: f32,
    bend: f32,
    rise: f32,
    branches: u8,
    seam_pitch: f32,
) -> Vec<Vec3> {
    let round = f.m.system == CeilingSystem::RoundDuct;
    let radii = if round {
        Vec2::splat((f.height * 0.29).min(f.width * 0.24) * section)
    } else {
        Vec2::new(f.height * 0.25, f.width * 0.27) * section
    };
    let scale = |t: f32| 1. - (1. - end_scale) * ((t - 0.48) / 0.24).clamp(0., 1.);
    let rings: Vec<_> = (0..=16)
        .map(|i| {
            let t = i as f32 / 16.;
            (route(f, t, bend, rise), radii * scale(t))
        })
        .collect();
    loft(f.part(a, Surface::Metal), &rings, round);
    let joints = (f.length / seam_pitch).ceil().clamp(1., 24.) as usize;
    for i in 1..joints {
        let t = i as f32 / joints as f32;
        let p = route(f, t, bend, rise);
        let r = radii * scale(t) + Vec2::splat(0.001);
        let outline: Vec<_> = if round {
            (0..=24)
                .map(|j| {
                    let v = Vec2::from_angle(j as f32 / 24. * std::f32::consts::TAU) * r;
                    p + Vec3::new(0., v.x, v.y)
                })
                .collect()
        } else {
            [
                Vec2::new(1., 1.),
                Vec2::new(-1., 1.),
                Vec2::new(-1., -1.),
                Vec2::new(1., -1.),
                Vec2::new(1., 1.),
            ]
            .map(|v| p + Vec3::new(0., v.x * r.x, v.y * r.y))
            .to_vec()
        };
        f.tube(a, Surface::Chrome, &outline, 0.0025, 6);
    }
    for i in 0..branches {
        let t = (i as f32 + 1.) / (branches as f32 + 1.);
        let p = route(f, t, bend, rise);
        let side = if i % 2 == 0 { 1. } else { -1. };
        let r = (radii.min_element() * scale(t) * 0.42).min(f.height * 0.10);
        let z = side * f.width * 0.39;
        let elbow = (p.y - 0.044).min((z - p.z).abs() * 0.40);
        let (points, target) = if elbow > r * 2.5 {
            let mut points = vec![p];
            for j in 0..=6 {
                let angle = j as f32 / 6. * std::f32::consts::FRAC_PI_2;
                points.push(Vec3::new(
                    p.x,
                    p.y - elbow * (1. - angle.cos()),
                    z - side * elbow * (1. - angle.sin()),
                ));
            }
            // Finish at the tangent point. An almost-zero vertical tail would
            // make transported tube rings fold across one another.
            let target = *points.last().unwrap() - Vec3::Y * 0.026;
            (points, target)
        } else {
            // Shallow plenums cannot hold a full drop elbow: mount the register
            // directly beneath a horizontal takeoff, with no folded-back tube.
            (
                vec![p, p.with_z(z)],
                Vec3::new(p.x, (p.y - r - 0.023).max(0.012), z),
            )
        };
        f.tube(a, Surface::Metal, &points, r, 12);
        grille(a, f, target, 0.27_f32.min(f.length * 0.18), f.width * 0.16);
    }
    if branches == 0 {
        let p = route(f, 0.40, bend, rise);
        // Downward register attached to the belly of the main run.
        let radius = radii.x * scale(0.40);
        grille(
            a,
            f,
            p - Vec3::Y * (radius + 0.015),
            0.36_f32.min(f.length * 0.30),
            radii.y * 1.25,
        );
    }
    hangers(
        a,
        f,
        &[0.14, 0.86].map(|t| (route(f, t, bend, rise), radii.x * scale(t))),
    )
}

fn sprinkler_head(a: &mut Assembly, f: &Frame, p: Vec3, upright: bool) {
    let direction = if upright { Vec3::Y } else { Vec3::NEG_Y };
    f.rod(a, Surface::Chrome, p - direction * 0.012, p, 0.011);
    for side in [-1., 1.] {
        f.rod(
            a,
            Surface::Chrome,
            p + Vec3::Z * side * 0.010,
            p + Vec3::Z * side * 0.010 + direction * 0.025,
            0.002,
        );
    }
    f.rod(
        a,
        Surface::Accent,
        p + direction * 0.002,
        p + direction * 0.020,
        0.003,
    );
    let center = p + direction * 0.027;
    f.part(a, Surface::Chrome)
        .cylinder(0.021, 0.003, Transform::from_translation(center));
    // Radial deflector teeth carry an actual silhouette in close-up annotations.
    for i in 0..8 {
        let d = Vec2::from_angle(i as f32 / 8. * std::f32::consts::TAU);
        f.rod(
            a,
            Surface::Chrome,
            center + Vec3::new(d.x, 0., d.y) * 0.016,
            center + Vec3::new(d.x, 0., d.y) * 0.026,
            0.0015,
        );
    }
}

pub(super) fn sprinkler(
    a: &mut Assembly,
    f: &Frame,
    radius: f32,
    reach: f32,
    pitch: f32,
    upright: bool,
) -> Vec<Vec3> {
    let y = f.height * 0.64;
    let radius = radius.min(f.height * 0.16);
    f.rod(
        a,
        Surface::Metal,
        Vec3::new(-f.length * 0.47, y, 0.),
        Vec3::new(f.length * 0.47, y, 0.),
        radius,
    );
    let count = (f.length / pitch).ceil().clamp(1., 10.) as usize;
    for i in 0..count {
        let x = ((i as f32 + 0.5) / count as f32 - 0.5) * f.length * 0.80;
        let z = if i % 2 == 0 { 1. } else { -1. } * f.width * reach * 0.43;
        let main = Vec3::new(x, y, 0.);
        let branch = main.with_z(z);
        let head = branch.with_y(if upright { f.height - 0.045 } else { 0.045 });
        f.rod(a, Surface::Metal, main, branch, radius * 0.57);
        f.rod(a, Surface::Chrome, branch, head, 0.006);
        f.rod(
            a,
            Surface::Metal,
            main - Vec3::X * 0.020,
            main + Vec3::X * 0.020,
            radius + 0.003,
        );
        sprinkler_head(a, f, head, upright);
    }
    hangers(
        a,
        f,
        &[-0.35, 0.35].map(|x| (Vec3::new(x * f.length, y, 0.), radius)),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn conduit(
    a: &mut Assembly,
    f: &Frame,
    count: u8,
    radius: f32,
    bend: f32,
    rise: f32,
    pitch: f32,
) -> Vec<Vec3> {
    let spacing = f.width * 0.56 / count as f32;
    let radius = radius.min(spacing * 0.32).min(f.height * 0.12);
    for lane in 0..count {
        let z = (lane as f32 + 0.5 - count as f32 * 0.5) * spacing;
        let points: Vec<_> = (0..=16)
            .map(|j| route(f, j as f32 / 16., bend, rise) + Vec3::Z * z)
            .collect();
        f.tube(a, Surface::Metal, &points, radius, 10);
        let joints = (f.length / pitch).ceil().clamp(1., 8.) as usize;
        for j in 0..joints {
            let t = (j as f32 + 0.5) / joints as f32;
            let p = route(f, t, bend, rise) + Vec3::Z * z;
            if lane == 0 && j % 2 == 0 {
                f.block(
                    a,
                    Surface::Metal,
                    p,
                    Vec3::new(0.090, f.height * 0.22, spacing * 0.8),
                    0.004,
                );
                f.block(
                    a,
                    Surface::Chrome,
                    p - Vec3::Y * (f.height * 0.11 + 0.002),
                    Vec3::new(0.083, 0.003, spacing * 0.74),
                    0.001,
                );
            } else {
                let d = (route(f, (t + 0.002).min(1.), bend, rise)
                    - route(f, (t - 0.002).max(0.), bend, rise))
                .normalize();
                f.rod(
                    a,
                    Surface::Chrome,
                    p - d * 0.013,
                    p + d * 0.013,
                    radius + 0.002,
                );
            }
        }
    }
    hangers(
        a,
        f,
        &[0.12, 0.88].map(|t| (route(f, t, bend, rise), radius)),
    )
}

pub(super) fn tray(a: &mut Assembly, f: &Frame, cables: u8, pitch: f32, rise: f32) -> Vec<Vec3> {
    let rail_h = f.height * 0.34;
    let y = |t: f32| f.height * (0.32 + rise * (t - 0.5) * 0.45);
    let half = f.width * 0.43;
    let slope = (y(1.) - y(0.)) / (f.length * 0.94);
    let angle = slope.atan();
    for z in [-half, half] {
        f.part(a, Surface::Metal).cuboid(
            Vec3::new(f.length * 0.94 * (1. + slope * slope).sqrt(), rail_h, 0.012),
            0.002,
            Transform::from_translation(Vec3::new(0., y(0.5) + rail_h * 0.5, z))
                .with_rotation(Quat::from_rotation_z(angle)),
        );
    }
    let count = (f.length / pitch).ceil().clamp(2., 40.) as usize;
    for i in 0..=count {
        let t = i as f32 / count as f32;
        f.block(
            a,
            Surface::Metal,
            Vec3::new((t - 0.5) * f.length * 0.92, y(t) + 0.007, 0.),
            Vec3::new(0.016, 0.014, f.width * 0.86),
            0.002,
        );
    }
    let spacing = f.width * 0.70 / cables as f32;
    let radius = 0.007_f32.min(spacing * 0.3);
    for i in 0..cables {
        let z = (i as f32 + 0.5 - cables as f32 * 0.5) * spacing;
        let points: Vec<_> = (0..=16)
            .map(|j| {
                let t = j as f32 / 16.;
                Vec3::new(
                    (t - 0.5) * f.length * 0.92,
                    y(t) + 0.018 + radius,
                    z + spacing * 0.12 * (t * std::f32::consts::TAU * 2. + i as f32).sin(),
                )
            })
            .collect();
        f.tube(
            a,
            if i % 4 == 0 {
                Surface::Accent
            } else {
                Surface::Ink
            },
            &points,
            radius,
            6,
        );
    }
    hangers(
        a,
        f,
        &[0.15, 0.85].map(|t| (Vec3::new((t - 0.5) * f.length * 0.92, y(t), 0.), 0.)),
    )
}
