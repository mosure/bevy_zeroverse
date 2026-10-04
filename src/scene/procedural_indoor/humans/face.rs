//! Surface-fitted brows and eyewear. All dimensions are metric and all pieces
//! share the cranial frame; no facial feature is positioned from a world AABB.
use super::{HumanAssembly, HumanSurface, IndoorHuman};
use bevy::prelude::*;
use rand::Rng;

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

/// Split existing skin faces at a fitted cheek/chin region. Stubble never adds
/// an alpha shell or changes the vermilion mask and cannot shimmer against skin.
pub(super) fn stubble(
    triangle: [super::garments::GarmentVertex; 3],
    eye_origin: Vec3,
    mesh: &mut HumanAssembly,
) {
    let (hair, skin) = super::garments::split(
        &triangle,
        |p| {
            let p = p - eye_origin;
            let top = p.y + 0.030 - 0.012 * (p.x.abs() / 0.075).min(1.0);
            top.max(-0.125 - p.y)
                .max(p.z - 0.025)
                .max(p.x.abs() - 0.075)
        },
        None,
    );
    super::garments::emit(&skin, HumanSurface::Skin, mesh);
    super::garments::emit(&hair, HumanSurface::FacialHair, mesh);
}
