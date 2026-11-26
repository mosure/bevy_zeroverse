//! A continuous baked surface with an optional spooned-away edge. The crust
//! follows the tapered inner pan section at its local height.
use super::*;
use crate::scene::procedural_indoor::materials::roast::Stock;

pub(super) fn build(a: &mut Assembly, pan: &Pan, deck: f32, outline: &[Vec2; 32]) {
    let mut r = stream(pan.seed, u64::from_le_bytes(*b"BakeCrst"));
    let direction = Vec2::from_angle(r.random_range(-PI..PI));
    let scoop = direction * r.random_range(0.48..0.95);
    let served = if r.random_bool(0.74) {
        r.random_range(0.40..0.93)
    } else {
        0.
    };
    let width = Vec2::new(r.random_range(0.38..0.80), r.random_range(0.46..0.98));
    let phase = r.random_range(-PI..PI);
    let floor = pan.floor(deck);
    let mesh = a.part(Surface::Terracotta, Stock::Gratin.label());
    let base = mesh.positions.len();
    let start = mesh.indices.len();
    const RINGS: usize = 9;
    let count = outline.len();
    let idx = |ring: usize, col: usize| (base + 1 + (ring - 1) * count + col % count) as u32;
    let point = |uv: Vec2| {
        let parting = ((uv - scoop) / width).length_squared();
        let t = parting.clamp(0., 1.);
        let hollow = 1. - t * t * (3. - 2. * t);
        let furrow = (uv.dot(direction) * 15. + uv.perp_dot(direction).powi(2) * 3. + phase).sin();
        let fraction = (pan.fill_fraction * (1. - served * hollow)
            + 0.037 * furrow
            + 0.035 * (uv.x * 6.4 + phase).sin() * (uv.y * 7.5 - phase).cos())
        .clamp(0.06, 0.92);
        let y = floor + (deck - floor) * fraction;
        // The steel side extends to the top of its rolled lip.
        let cross = pan
            .bottom()
            .lerp(pan.inner(), (y - floor) / (deck + mesh::LIP - floor))
            - Vec2::splat(0.0012);
        let pos = pan.center + cross * uv;
        Vec3::new(pos.x, y, pos.y)
    };
    for uv in std::iter::once(Vec2::ZERO).chain((1..=RINGS).flat_map(|ring| {
        outline
            .iter()
            .map(move |v| *v * (ring as f32 / RINGS as f32))
    })) {
        let p = point(uv);
        mesh.positions.push(p.to_array());
        mesh.normals.push([0.; 3]);
        mesh.uvs.push(p.xz().to_array());
    }
    for col in 0..count {
        mesh.indices
            .extend([base as u32, idx(1, col + 1), idx(1, col)]);
        for ring in 1..RINGS {
            let (a, b, c, d) = (
                idx(ring, col),
                idx(ring, col + 1),
                idx(ring + 1, col + 1),
                idx(ring + 1, col),
            );
            mesh.indices.extend([a, b, c, a, c, d]);
        }
    }
    for tri in mesh.indices[start..].as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
        let n = (b - a).cross(c - a);
        for &k in tri {
            let normal = &mut mesh.normals[k as usize];
            *normal = (Vec3::from_array(*normal) + n).to_array();
        }
    }
    for n in &mut mesh.normals[base..] {
        *n = Vec3::from_array(*n).normalize().to_array();
    }
    let bottom = |v: Vec2| {
        let p = pan.center + (pan.bottom() - Vec2::splat(0.0012)) * v;
        Vec3::new(p.x, floor, p.y)
    };
    for i in 0..count {
        let next = (i + 1) % count;
        let (x, z) = (
            Vec3::from_array(mesh.positions[idx(RINGS, i) as usize]),
            Vec3::from_array(mesh.positions[idx(RINGS, next) as usize]),
        );
        mesh.metric_quad(
            [bottom(outline[i]), x, z, bottom(outline[next])],
            Vec3::new(outline[i].x, 0., outline[i].y),
        );
    }
}
