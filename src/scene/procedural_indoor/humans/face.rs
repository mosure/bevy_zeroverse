//! Surface-fitted brows and eyewear. All dimensions are metric and all pieces
//! share the cranial frame; no facial feature is positioned from a world AABB.
use super::{HumanAssembly, HumanSurface, IndoorHuman};
use bevy::prelude::*;
use rand::Rng;

/// Partition the actual Anny eye surface rather than overlaying another sphere.
/// Every source triangle is conserved across sclera, iris and pupil; this avoids
/// depth fighting with the polygonal eyeball and keeps the source skinning shape.
pub(super) fn iris(seed: u64, eye: Vec3, rotation: Quat, scale: f32, mesh: &mut HumanAssembly) {
    use super::garments::{emit, split, GarmentVertex};
    let Some(sclera) = mesh.parts.remove(&HumanSurface::Eye) else {
        return;
    };
    let mut rng = super::stream(seed, 0x49524953);
    let radius = rng.random_range(0.0049..0.0059) * scale;
    let pupil = radius * rng.random_range(0.26..0.46);
    let inverse = rotation.inverse();
    let directions: [Vec2; 32] =
        std::array::from_fn(|i| Vec2::from_angle(i as f32 * std::f32::consts::TAU / 32.));
    // Linear half-planes give exact shared crossings on the source triangles.
    // A nonlinear radial test would distort the boundary on coarse polygons.
    let partition = |mut polygon: Vec<GarmentVertex>, r: f32, outside, mesh: &mut HumanAssembly| {
        let offset = r * (std::f32::consts::PI / 32.).cos();
        for direction in directions {
            let (inside, rejected) =
                split(&polygon, |p| p.truncate().dot(direction) - offset, None);
            emit(&rejected, outside, mesh);
            polygon = inside;
            if polygon.len() < 3 {
                break;
            }
        }
        polygon
    };
    for indices in sclera.indices.as_chunks::<3>().0 {
        let triangle = indices.map(|i| {
            let i = i as usize;
            let position = Vec3::from_array(sclera.positions[i]);
            GarmentVertex {
                rest: inverse * (position - eye),
                position,
                normal: Vec3::from_array(sclera.normals[i]),
                uv: Vec2::from_array(sclera.uvs[i]),
                skin_field: 0.,
            }
        });
        // The other eye and triangles outside the iris bounds need no cutting.
        if (0..2).any(|axis| {
            triangle.iter().all(|v| v.rest[axis] > radius)
                || triangle.iter().all(|v| v.rest[axis] < -radius)
        }) || triangle.iter().all(|v| v.rest.z >= 0.)
        {
            emit(&triangle, HumanSurface::Eye, mesh);
            continue;
        }
        let (front, back) = split(&triangle, |p| p.z, None);
        emit(&back, HumanSurface::Eye, mesh);
        let iris = partition(front, radius, HumanSurface::Eye, mesh);
        let pupil = partition(iris, pupil, HumanSurface::Iris, mesh);
        emit(&pupil, HumanSurface::Detail, mesh);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    h: &IndoorHuman,
    posed: &[Vec3],
    faces: &[i64],
    eyes: &[Vec3],
    rotation: Quat,
    scale: f32,
    mesh: &mut HumanAssembly,
) {
    let origin = (eyes[0] + eyes[1]) * 0.5;
    let inverse = rotation.inverse();
    let local: Vec<_> = posed.iter().map(|p| inverse * (*p - origin)).collect();
    let triangles: Vec<[Vec3; 3]> = faces
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|q| q.iter().all(|i| local[*i as usize].length() < 0.19 * scale))
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .map(|q| q.map(|i| local[i as usize]))
        .collect();
    let front = |x: f32, y: f32| {
        triangles
            .iter()
            .filter_map(|t| {
                let a = t[0].truncate();
                let b = t[1].truncate() - a;
                let c = t[2].truncate() - a;
                let p = Vec2::new(x, y) - a;
                let det = b.perp_dot(c);
                if det.abs() < 1e-10 {
                    return None;
                }
                let u = p.perp_dot(c) / det;
                let v = b.perp_dot(p) / det;
                (u >= -1e-4 && v >= -1e-4 && u + v <= 1.0001)
                    .then_some(t[0].z + u * (t[1].z - t[0].z) + v * (t[2].z - t[0].z))
            })
            .reduce(f32::min)
            .unwrap_or(-0.014 * scale)
    };
    let world = |p: Vec3| origin + rotation * p;
    let mut rng = super::stream(h.seed, 0xface);
    let program = h
        .appearance
        .as_ref()
        .map(|a| &a.face)
        .cloned()
        .unwrap_or_default();
    let brow_width = program.brow_width * scale;
    let brow_thickness = program.brow_thickness * scale;
    let brow_rise = rng.random_range(0.022..0.028) * scale;
    for &eye in eyes {
        let eye = inverse * (eye - origin);
        let g = mesh.part(HumanSurface::Brow);
        let base = g.positions.len() as u32;
        // One opaque fitted strip per brow. Taper its ends, retain a substantial
        // middle, and keep it clear of the skin with a real millimetre offset.
        for i in 0..=16 {
            let t = i as f32 / 16.0;
            let x = eye.x + (t - 0.5) * brow_width;
            let arch = (t * std::f32::consts::PI).sin().max(0.0);
            let y = eye.y + brow_rise + arch * program.brow_arch * scale;
            for side in [-1.0, 1.0] {
                let y = y + side * brow_thickness * (0.12 + 0.88 * arch.sqrt()) * 0.5;
                let p = Vec3::new(x, y, front(x, y) - 0.0015 * scale);
                g.positions.push(world(p).to_array());
                g.normals.push((rotation * Vec3::NEG_Z).to_array());
                g.uvs.push([t, (side + 1.0) * 0.5]);
            }
        }
        for i in 0..16 {
            let a = base + i * 2;
            g.indices.extend([a, a + 1, a + 2, a + 1, a + 3, a + 2]);
        }
    }
    if !h.glasses {
        return;
    }
    let eye_x = (inverse * (eyes[1] - eyes[0])).x.abs() * 0.5;
    let rx = (rng.random_range(0.023..0.028) * scale).min(eye_x * 0.90);
    let ry = rng.random_range(0.014..0.021) * scale;
    let exponent = rng.random_range(0.56..1.0);
    let thickness = rng.random_range(0.0010..0.0020) * scale;
    // A real nose bridge has to sit in front of the nose, not through it.
    let z = front(0.0, 0.006 * scale) - 0.006 * scale;
    let mut inner = Vec::new();
    for &eye in eyes {
        let eye = inverse * (eye - origin);
        let g = mesh.part(HumanSurface::Lens);
        let base = g.positions.len() as u32;
        g.positions
            .push(world(Vec3::new(eye.x, eye.y, z - 0.0006 * scale)).to_array());
        g.normals.push((rotation * Vec3::NEG_Z).to_array());
        g.uvs.push([0.5, 0.5]);
        let mut rim = Vec::new();
        for i in 0..32 {
            let a = i as f32 * std::f32::consts::TAU / 32.0;
            let shape = |v: f32| v.signum() * v.abs().powf(exponent);
            let p = Vec3::new(eye.x + rx * shape(a.cos()), eye.y + ry * shape(a.sin()), z);
            rim.push(p);
            g.positions.push(world(p).to_array());
            g.normals.push((rotation * Vec3::NEG_Z).to_array());
            g.uvs.push([0.5 + a.cos() * 0.5, 0.5 + a.sin() * 0.5]);
        }
        for i in 0..32_u32 {
            g.indices
                .extend([base, base + 1 + (i + 1) % 32, base + 1 + i]);
        }
        let g = mesh.part(HumanSurface::Eyewear);
        for i in 0..32 {
            g.rod(world(rim[i]), world(rim[(i + 1) % 32]), thickness);
        }
        let sign = eye.x.signum();
        inner.push(Vec3::new(eye.x - sign * rx, eye.y + 0.003 * scale, z));
        let hinge = Vec3::new(eye.x + sign * rx, eye.y + 0.003 * scale, z);
        let temple = Vec3::new(
            sign * (eye_x + rx + 0.011 * scale),
            eye.y + 0.001 * scale,
            0.038 * scale,
        );
        g.rod(world(hinge), world(temple), thickness * 0.85);
        g.rod(
            world(temple),
            world(temple + Vec3::new(0.0, -0.012, 0.018) * scale),
            thickness * 0.85,
        );
    }
    let g = mesh.part(HumanSurface::Eyewear);
    let mid = (inner[0] + inner[1]) * 0.5 + Vec3::new(0.0, 0.006, -0.002) * scale;
    g.rod(world(inner[0]), world(mid), thickness * 0.8);
    g.rod(world(mid), world(inner[1]), thickness * 0.8);
}

/// A fitted rest-space lower-face region. The nose exclusion is tied to the
/// actual vermilion edge, not deformation bone weights or a horizontal eye band.
pub(super) struct BeardRegion {
    eye: Vec3,
    scale: f32,
    upper_lip: f32,
}
impl BeardRegion {
    pub fn new(vertices: &[Vec3], faces: &[i64], eyes: [Vec3; 2]) -> Self {
        let eye = (eyes[0] + eyes[1]) * 0.5;
        let scale = (eyes[0].distance(eyes[1]) / 0.064).clamp(0.65, 1.55);
        let upper_lip = faces
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .filter(|(i, _)| super::anatomy::lip_face(*i))
            .flat_map(|(_, q)| q.iter().map(|&i| vertices[i as usize].y))
            .fold(f32::NEG_INFINITY, f32::max);
        Self {
            eye,
            scale,
            upper_lip,
        }
    }
    fn field(&self, point: Vec3) -> f32 {
        let p = (point - self.eye) / self.scale;
        // A cheek line rises smoothly from the moustache toward the sideburns.
        // A rectangular nose cutout produces an equally artificial square beard.
        let side = ((p.x.abs() - 0.026) / 0.049).clamp(0., 1.);
        let lip = (self.upper_lip - self.eye.y) / self.scale;
        let top = p.y - (lip + 0.006 + 0.054 * side * side * (3. - 2. * side));
        let nose = (0.027 - p.x.abs()).min((point.y - self.upper_lip) / self.scale - 0.007);
        top.max(-0.125 - p.y)
            .max(p.z - 0.025)
            .max(p.x.abs() - 0.075)
            .max(nose)
    }
    pub fn append(&self, triangle: [super::garments::GarmentVertex; 3], mesh: &mut HumanAssembly) {
        let (hair, skin) = super::garments::split(&triangle, |p| self.field(p), None);
        super::garments::emit(&skin, HumanSurface::Skin, mesh);
        super::garments::emit(&hair, HumanSurface::FacialHair, mesh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn irises_partition_the_source_eye_without_intersecting_overlays() {
        use crate::scene::procedural_indoor::geometry::Geometry;
        let area = |g: &Geometry| -> f64 {
            g.indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                    f64::from((b - a).cross(c - a).length()) * 0.5
                })
                .sum()
        };
        for seed in 0..8 {
            for scale in [0.7, 1., 1.4] {
                let rotation = Quat::from_euler(EulerRot::YXZ, 0.7, -0.3, 0.15);
                let origin = Vec3::new(0.1, 1.6, -0.2);
                let eyes = [-0.032, 0.032].map(|x| origin + rotation * Vec3::X * x * scale);
                let mut mesh = HumanAssembly::default();
                for eye in eyes {
                    mesh.part(HumanSurface::Eye).ellipsoid(
                        Vec3::new(0.012, 0.011, 0.011) * scale,
                        Transform::from_translation(eye).with_rotation(rotation),
                    );
                }
                let source = mesh.parts[&HumanSurface::Eye].clone();
                let source_area = area(&source);
                for eye in eyes {
                    iris(seed, eye, rotation, scale, &mut mesh);
                }
                let result_area: f64 = mesh.parts.values().map(area).sum();
                // The shared mesh emitter discards slivers of area <= 5e-10
                // square metres, and positions are
                // f32 at head height. Bound their combined area to 0.005%.
                assert!(
                    (result_area / source_area - 1.).abs() < 5e-5,
                    "seed={seed} scale={scale}: original={source_area}, partition={result_area}"
                );
                for surface in [HumanSurface::Eye, HumanSurface::Iris, HumanSurface::Detail] {
                    let g = &mesh.parts[&surface];
                    assert!(area(g) > 0.);
                    for triangle in g.indices.as_chunks::<3>().0 {
                        let [a, b, c] = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                        let n = (b - a).cross(c - a);
                        assert!(n.length_squared() > 1e-18);
                        // Very small clipped triangles can amplify roundoff; their
                        // normals must still agree with the original eye orientation.
                        assert!(n.dot(Vec3::from_array(g.normals[triangle[0] as usize])) > 0.);
                    }
                }
                let iris = &mesh.parts[&HumanSurface::Iris];
                let pupil = &mesh.parts[&HumanSurface::Detail];
                assert!(
                    pupil
                        .positions
                        .iter()
                        .filter(|p| iris.positions.contains(p))
                        .count()
                        >= 32
                );
                // No extra surface is displaced in front of the original mesh.
                for p in mesh.parts.values().flat_map(|g| &g.positions) {
                    let p = Vec3::from_array(*p);
                    let distance = source
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|t| {
                            let [a, b, c] =
                                t.map(|i| Vec3::from_array(source.positions[i as usize]));
                            let n = (b - a).cross(c - a).normalize_or_zero();
                            (p - a).dot(n).abs()
                        })
                        .fold(f32::INFINITY, f32::min);
                    assert!(distance < 2e-6);
                }
            }
        }
    }
    #[test]
    fn beard_excludes_nose_but_retains_cheek_chin_and_moustache() {
        for scale in [0.65, 1.0, 1.55] {
            let mask = BeardRegion {
                eye: Vec3::ZERO,
                scale,
                upper_lip: -0.070 * scale,
            };
            for x in [-0.024, 0., 0.024] {
                for y in [-0.025, -0.045, -0.060] {
                    assert!(mask.field(Vec3::new(x, y, -0.04) * scale) > 0.);
                }
            }
            for p in [
                Vec3::new(0.045, -0.052, -0.01),
                Vec3::new(0., -0.108, 0.),
                Vec3::new(0., -0.069, -0.015),
            ] {
                assert!(mask.field(p * scale) < 0.);
            }
        }
    }
}
