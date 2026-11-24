use super::*;

pub(super) fn route(f: &Frame, t: f32, bend: f32, rise: f32) -> Vec3 {
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
/// Candidate admission needs the four saddle rod origins. Only a realized
/// module needs the service mesh and the crossbars underneath it.
pub(super) fn suspended(
    mesh: Option<&mut Assembly>,
    f: &Frame,
    supports: [(Vec3, f32); 2],
    body: impl FnOnce(&mut Assembly),
) -> Vec<Vec3> {
    let saddles = supports.map(|(p, radius)| Vec3::new(p.x, p.y - radius - 0.007, 0.));
    if let Some(a) = mesh {
        body(a);
        for &center in &saddles {
            f.block(
                a,
                Surface::Metal,
                center,
                Vec3::new(0.026, 0.016, f.width * 0.91),
                0.002,
            );
        }
    }
    saddles
        .into_iter()
        .flat_map(|p| [-f.width * 0.445, f.width * 0.445].map(|z| Vec3::new(p.x, p.y + 0.006, z)))
        .collect()
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
        Surface::Paint,
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
    mesh: Option<&mut Assembly>,
    f: &Frame,
    radius: f32,
    reach: f32,
    pitch: f32,
    upright: bool,
    version: u32,
) -> Vec<Vec3> {
    let y = f.height * 0.64;
    let radius = radius.min(f.height * 0.16);
    let supports = [-0.35, 0.35].map(|x| (Vec3::new(x * f.length, y, 0.), radius));
    suspended(mesh, f, supports, |a| {
        f.rod(
            a,
            Surface::Metal,
            Vec3::new(-f.length * 0.47, y, 0.),
            Vec3::new(f.length * 0.47, y, 0.),
            radius,
        );
        if let Some(stock) = f.m.sprinkler_run(version) {
            super::sprinkler::branches(a, f, stock, radius, upright);
            return;
        }
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
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn conduit(
    mesh: Option<&mut Assembly>,
    f: &Frame,
    count: u8,
    radius: f32,
    bend: f32,
    rise: f32,
    pitch: f32,
) -> Vec<Vec3> {
    let spacing = f.width * 0.56 / count as f32;
    let radius = radius.min(spacing * 0.32).min(f.height * 0.12);
    let supports = [0.12, 0.88].map(|t| (route(f, t, bend, rise), radius));
    suspended(mesh, f, supports, |a| {
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
    })
}

pub(super) fn tray(
    mesh: Option<&mut Assembly>,
    f: &Frame,
    cables: u8,
    pitch: f32,
    rise: f32,
) -> Vec<Vec3> {
    let rail_h = f.height * 0.34;
    let y = |t: f32| f.height * (0.32 + rise * (t - 0.5) * 0.45);
    let supports = [0.15, 0.85].map(|t| (Vec3::new((t - 0.5) * f.length * 0.92, y(t), 0.), 0.));
    suspended(mesh, f, supports, |a| {
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
                    Surface::Paint
                } else {
                    Surface::Ink
                },
                &points,
                radius,
                6,
            );
        }
    })
}
