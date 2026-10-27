//! Dimensioned speaker cabinets, driver cells and grille-covered enclosures.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Driver {
    pub low: f32,
    pub high: f32,
    pub center: Vec2,
    pub radius: f32,
    pub depth: f32,
    pub dome: f32,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct SpeakerProgram {
    pub body: Vec3,
    pub wall: f32,
    pub horizontal: bool,
    pub covered: bool,
    pub finish: Surface,
    pub grille: Surface,
    pub drivers: Vec<Driver>,
    pub control_x: f32,
    pub control_radius: f32,
}
fn body_size(size: Vec3, r: &mut impl Rng) -> Vec3 {
    size * Vec3::new(
        r.random_range(0.83..0.99),
        r.random_range(0.79..0.98),
        r.random_range(0.78..0.98),
    )
}
/// Mounting queries share the body draws without allocating driver geometry.
pub(crate) fn case_depth(o: &IndoorObject) -> f32 {
    body_size(o.size, &mut stream(o.seed, 0x535045414b455250)).z
}
pub fn parameters(o: &IndoorObject) -> SpeakerProgram {
    let mut r = stream(o.seed, 0x535045414b455250);
    let body = body_size(o.size, &mut r);
    let wall = body.min_element() * r.random_range(0.025..0.045);
    let horizontal = body.x > body.y * 1.6;
    let length = if horizontal { body.x } else { body.y };
    let width = if horizontal { body.y } else { body.x };
    let count = ((length / width * r.random_range(0.8..1.6)).round() as usize).clamp(1, 6);
    let weights: Vec<_> = (0..count).map(|_| r.random_range(0.65..1.65)).collect();
    let total = weights.iter().sum::<f32>();
    let mut cursor = -length * 0.5 + wall;
    let mut drivers = Vec::new();
    for weight in weights {
        let high = cursor + (length - 2. * wall) * weight / total;
        let radius = (width * 0.5 - wall).min((high - cursor) * 0.5) * r.random_range(0.58..0.87);
        let center = Vec2::new(
            (width * 0.5 - wall - radius) * r.random_range(-0.60..0.60),
            (cursor + high) * 0.5,
        );
        drivers.push(Driver {
            low: cursor,
            high,
            center,
            radius,
            depth: (radius * r.random_range(0.16..0.32)).min(body.z * 0.24),
            dome: r.random_range(0.19..0.38),
        });
        cursor = high;
    }
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x535045414b46494e);
    SpeakerProgram {
        body,
        wall,
        horizontal,
        drivers,
        covered: r.random_bool(0.43),
        finish: [Surface::Wood, Surface::Metal, Surface::Plastic][f.random_range(0..3)],
        grille: if f.random_bool(0.5) {
            Surface::Fabric
        } else {
            Surface::FabricAlt
        },
        control_x: r.random_range(-0.30..0.30),
        control_radius: body.x.min(body.z) * r.random_range(0.035..0.065),
    }
}
fn face_frame(p: &SpeakerProgram) -> Transform {
    Transform::from_xyz(0., p.body.y * 0.5, p.body.z * 0.5).with_rotation(if p.horizontal {
        Quat::from_rotation_z(-FRAC_PI_2)
    } else {
        Quat::IDENTITY
    })
}
/// A rectangular baffle cell with a real circular driver aperture. Corner
/// samples are explicit; adjoining cells meet at their edges without overlap.
fn baffle(g: &mut Geometry, p: &SpeakerProgram, d: &Driver, tf: Transform) {
    let half = if p.horizontal { p.body.y } else { p.body.x } * 0.5 - p.wall;
    // Both contours share rays from the displaced driver center. Connecting a
    // centered rectangle to an offset circle by index can fold the annulus and
    // invert triangles on tall/narrow cells. Include exact rectangle corners so
    // adjacent cells still meet without gaps.
    let mut rays: Vec<(f32, Option<Vec2>)> =
        (0..32).map(|i| (TAU * i as f32 / 32., None)).collect();
    for corner in [
        Vec2::new(-half, d.low),
        Vec2::new(half, d.low),
        Vec2::new(half, d.high),
        Vec2::new(-half, d.high),
    ] {
        let delta = corner - d.center;
        let angle = delta.y.atan2(delta.x).rem_euclid(TAU);
        // Merge near-coincident uniform rays, retaining the exact corner.
        rays.retain(|(a, _)| {
            let distance = (a - angle).abs();
            distance.min(TAU - distance) > 1e-5
        });
        rays.push((angle, Some(corner)));
    }
    rays.sort_by(|a, b| a.0.total_cmp(&b.0));
    let contour = |i: usize, inner: bool| {
        let (angle, corner) = rays[i % rays.len()];
        let v = Vec2::new(angle.cos(), angle.sin());
        let q = if inner {
            // Match the driver's 32-sided rim, including additional rays at
            // rectangle corners, instead of opening slivers beside its chords.
            let step = TAU / 32.;
            let middle = (angle / step).floor() * step + step * 0.5;
            let radius = d.radius * (step * 0.5).cos() / (angle - middle).cos();
            d.center + v * radius
        } else if let Some(corner) = corner {
            corner
        } else {
            let x = (v.x.signum() * half - d.center.x) / v.x;
            let y = (if v.y >= 0. { d.high } else { d.low } - d.center.y) / v.y;
            d.center + v * x.min(y)
        };
        tf.transform_point(q.extend(0.))
    };
    for i in 0..rays.len() {
        let pts = [
            contour(i, false),
            contour(i + 1, false),
            contour(i + 1, true),
            contour(i, true),
        ];
        let start = g.positions.len() as u32;
        for v in pts {
            g.positions.push(v.to_array());
            g.normals.push(Vec3::Z.to_array());
            g.uvs.push(v.xy().to_array());
        }
        g.indices
            .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = p.body;
    let t = p.wall;
    let l = o.kind.class_name();
    // Hollow cabinet: drivers can recess into it instead of lying on a solid
    // front cap. Side panels and top/bottom panels have disjoint interiors.
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., s.y * 0.5, -s.z * 0.5 + t * 0.5),
        Vec3::new(s.x, s.y, t),
        t * 0.15,
    );
    for x in [-1., 1.] {
        a.box_part(
            p.finish,
            l,
            Vec3::new(x * (s.x - t) * 0.5, s.y * 0.5, t * 0.5),
            Vec3::new(t, s.y, s.z - t),
            t * 0.15,
        );
    }
    for y in [t * 0.5, s.y - t * 0.5] {
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., y, t * 0.5),
            Vec3::new(s.x - 2. * t, t, s.z - t),
            t * 0.15,
        );
    }
    let tf = face_frame(&p);
    if p.covered {
        a.box_part(
            p.grille,
            l,
            Vec3::new(0., s.y * 0.5, s.z * 0.5 - t * 0.25),
            Vec3::new(s.x - 2. * t, s.y - 2. * t, t * 0.5),
            t * 0.1,
        );
    } else {
        for d in &p.drivers {
            baffle(a.part(p.finish, l), &p, d, tf);
            let center = tf.transform_point(d.center.extend(0.));
            let local =
                Transform::from_translation(center).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
            let r = d.radius;
            a.part(Surface::Rubber, l).lathe(
                &[
                    (r, 0.),
                    (r * 0.92, -d.depth * 0.05),
                    (r * 0.81, -d.depth * 0.08),
                    (r * d.dome, -d.depth),
                    (0., -d.depth * 0.56),
                ],
                32,
                local,
            );
        }
    }
    a.part(Surface::Plastic, l).cylinder(
        p.control_radius,
        t * 0.50,
        Transform::from_xyz(s.x * p.control_x, s.y - t * 0.20, -s.z * 0.14),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displaced_driver_baffles_tile_the_rectangle_without_folds_or_holes() {
        for aspect in [0.08, 0.25, 1., 4., 12.] {
            for offset in [-0.8, -0.3, 0., 0.3, 0.8] {
                let half: f32 = 0.2;
                let height = half * 2. * aspect;
                let radius = half.min(height * 0.5) * 0.75;
                let mut p =
                    parameters(&super::super::super::audit::fixture(ObjectKind::Speaker, 0));
                p.horizontal = false;
                p.body.x = half * 2. + p.wall * 2.;
                let d = Driver {
                    low: 0.,
                    high: height,
                    center: Vec2::new((half - radius) * offset, height * 0.5),
                    radius,
                    depth: 0.01,
                    dome: 0.3,
                };
                let mut g = Geometry::default();
                baffle(&mut g, &p, &d, Transform::IDENTITY);
                let mut area = 0.;
                for triangle in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let twice_area = (b - a).cross(c - a).z;
                    assert!(
                        twice_area > 0.,
                        "aspect={aspect}, offset={offset}: {a:?} {b:?} {c:?}"
                    );
                    area += twice_area * 0.5;
                }
                let expected = 2. * half * height - 16. * radius * radius * (TAU / 32.).sin();
                assert!((area - expected).abs() < expected * 2e-5);
            }
        }
    }
}
