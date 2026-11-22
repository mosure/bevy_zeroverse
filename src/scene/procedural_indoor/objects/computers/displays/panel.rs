//! Layered flat and curved displays. Only the viewing face carries the image.
use super::*;

pub(crate) fn screen_cover(
    a: &mut Assembly,
    casing: Surface,
    content: Surface,
    label: &str,
    size: Vec3,
    radius: f32,
    transform: Transform,
) {
    let mut front = Geometry::default();
    build(
        a.part(casing, label),
        Some(&mut front),
        size,
        radius,
        transform,
    );
    a.part(content, label).append(front);
}

pub(super) fn build(
    shell: &mut Geometry,
    mut image_face: Option<&mut Geometry>,
    size: Vec3,
    radius: f32,
    transform: Transform,
) {
    let Vec3 {
        x: width,
        y: height,
        z: thickness,
    } = size;
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
    let quad = |geometry: &mut Geometry, points: [Vec3; 4], normals: [Vec3; 4], uvs: [Vec2; 4]| {
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
    let arc = |x: f32| {
        if radius > 0. {
            radius * (x / radius).asin()
        } else {
            x
        }
    };
    let half_arc = arc(width * 0.5);
    let rect = |u0: f32, u1: f32, h: f32| {
        [
            Vec2::new(u0, h),
            Vec2::new(u1, h),
            Vec2::new(u1, 0.),
            Vec2::new(u0, 0.),
        ]
    };
    let y = height * 0.5;
    let t = thickness * 0.5;
    for i in 0..columns {
        let x0 = -width * 0.5 + width * i as f32 / columns as f32;
        let x1 = -width * 0.5 + width * (i + 1) as f32 / columns as f32;
        let (z0, n0) = profile(x0);
        let (z1, n1) = profile(x1);
        let u0 = arc(x0) + half_arc;
        let u1 = arc(x1) + half_arc;
        quad(
            image_face.as_deref_mut().unwrap_or(shell),
            [
                Vec3::new(x0, -y, z0 + t),
                Vec3::new(x1, -y, z1 + t),
                Vec3::new(x1, y, z1 + t),
                Vec3::new(x0, y, z0 + t),
            ],
            [n0, n1, n1, n0],
            rect(u0, u1, height),
        );
        quad(
            shell,
            [
                Vec3::new(x1, -y, z1 - t),
                Vec3::new(x0, -y, z0 - t),
                Vec3::new(x0, y, z0 - t),
                Vec3::new(x1, y, z1 - t),
            ],
            [-n1, -n0, -n0, -n1],
            rect(half_arc * 2. - u1, half_arc * 2. - u0, height),
        );
        quad(
            shell,
            [
                Vec3::new(x0, y, z0 + t),
                Vec3::new(x1, y, z1 + t),
                Vec3::new(x1, y, z1 - t),
                Vec3::new(x0, y, z0 - t),
            ],
            [Vec3::Y; 4],
            rect(u0, u1, thickness),
        );
        quad(
            shell,
            [
                Vec3::new(x0, -y, z0 - t),
                Vec3::new(x1, -y, z1 - t),
                Vec3::new(x1, -y, z1 + t),
                Vec3::new(x0, -y, z0 + t),
            ],
            [Vec3::NEG_Y; 4],
            rect(u0, u1, thickness),
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
        let mut side_uv = rect(0., thickness, height);
        if side > 0. {
            side_uv.reverse();
        }
        quad(shell, points, [Vec3::X * side; 4], side_uv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curved_and_flat_covers_share_one_closed_shell_with_outward_image_normals_and_metric_charts()
    {
        use std::collections::BTreeMap;
        for radius in [0., 0.8, 2.4] {
            let tf = Transform::from_xyz(0.1, 1.2, -0.3).with_rotation(Quat::from_rotation_x(0.23));
            let size = Vec3::new(0.72, 0.31, 0.0016);
            let mut a = Assembly::default();
            screen_cover(
                &mut a,
                Surface::Plastic,
                Surface::Screen,
                "monitor",
                size,
                radius,
                tf,
            );
            let mut edges = BTreeMap::new();
            for ((surface, _), part) in &a.parts {
                for t in part.indices.as_chunks::<3>().0 {
                    let p = t.map(|i| Vec3::from_array(part.positions[i as usize]));
                    let n = (p[1] - p[0]).cross(p[2] - p[0]);
                    let uv = t.map(|i| Vec2::from_array(part.uvs[i as usize]));
                    assert!((uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-8);
                    if *surface == Surface::Screen {
                        assert!(t.iter().all(|&i| (tf.rotation * Vec3::Z)
                            .dot(Vec3::from_array(part.normals[i as usize]))
                            > 0.85));
                        assert!(n.dot(tf.rotation * Vec3::Z) > 0.);
                    }
                    let keys = p.map(|v| v.to_array().map(f32::to_bits));
                    for i in 0..3 {
                        let x = keys[i];
                        let y = keys[(i + 1) % 3];
                        *edges.entry((x.min(y), x.max(y))).or_insert(0) += 1;
                    }
                }
            }
            assert!(edges.values().all(|&n| n == 2));
            let face = &a.parts[&(Surface::Screen, "monitor".into())];
            let max_u = face.uvs.iter().map(|uv| uv[0]).fold(0., f32::max);
            let expected = if radius == 0. {
                size.x
            } else {
                2. * radius * (size.x / (2. * radius)).asin()
            };
            assert!((max_u - expected).abs() < 1e-6);
        }
    }
}
