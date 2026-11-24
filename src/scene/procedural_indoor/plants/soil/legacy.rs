//! Archived eighteen-fragment dressing and its canopy RNG consumption.
use super::super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(in super::super) fn dressing(
    g: &mut Geometry,
    seed: u64,
    level: f32,
    radius: f32,
    placement: &mut impl Rng,
) {
    let mut mix = stream(seed, u64::from_le_bytes(*b"soilgrit"));
    let coarseness = mix.random_range(0.055..0.16);
    const FRAGMENTS: usize = 18;
    // Six triangular faces above and six below the irregular outline. The soil
    // already holds the planter fill; grow each buffer once for the dressing.
    let vertices = FRAGMENTS * 12 * 3;
    g.positions.reserve(vertices);
    g.normals.reserve(vertices);
    g.uvs.reserve(vertices);
    g.indices.reserve(vertices);
    for _ in 0..FRAGMENTS {
        // Keep the three placement draws per fragment independent of its shape.
        // Subsequent canopy draws therefore replay the same foliage.
        let dir = radial(placement.random_range(0.0..TAU));
        let fraction = placement.random::<f32>().sqrt();
        let yaw = placement.random_range(0.0..TAU);
        let length = (coarseness * radius).clamp(0.002, 0.021) * mix.random_range(0.37..1.);
        let width = length * mix.random_range(0.38..0.94);
        let height = width * mix.random_range(0.35..0.89);
        let center = dir * (radius - length).max(0.) * fraction + Vec3::Y * level;
        let tf = Transform::from_translation(center).with_rotation(Quat::from_rotation_y(yaw));
        clod(
            g,
            tf,
            Vec3::new(length, height, width),
            mix.random_range(0.22..0.76),
        );
    }
}

fn clod(g: &mut Geometry, tf: Transform, half: Vec3, exposure: f32) {
    // A broken aggregate has irregular shoulders and a buried lower half.
    // Neither half contains a face coplanar with the fill underneath.
    const OUTLINE: [[f32; 2]; 6] = [
        [1., 0.06],
        [0.42, 0.88],
        [-0.55, 0.75],
        [-0.96, -0.13],
        [-0.24, -0.90],
        [0.68, -0.64],
    ];
    let point = |x, y, z| tf.transform_point(Vec3::new(x, y, z) * half);
    let ring = OUTLINE.map(|[x, z]| point(x, exposure - 0.54 + 0.10 * z, z));
    let top = point(-0.16, exposure, 0.12);
    let bottom = point(0.12, exposure - 1., -0.08);
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
        for face in [[top, b, a], [bottom, a, b]] {
            let edge = face[1] - face[0];
            let n = edge.cross(face[2] - face[0]).normalize();
            let u = edge.normalize();
            let v = n.cross(u);
            let start = g.positions.len() as u32;
            g.positions.extend(face.map(|p| p.to_array()));
            g.normals.extend([n.to_array(); 3]);
            g.uvs.extend(face.map(|p| [p.dot(u), p.dot(v)]));
            g.indices.extend([start, start + 1, start + 2]);
        }
    }
}
