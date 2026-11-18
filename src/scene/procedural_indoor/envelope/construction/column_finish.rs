//! A closed skin with a metric perimeter chart. Rings meet at each shoe,
//! sleeve and joint, so there are no stacked shells or coplanar decals.
use super::*;
use crate::scene::procedural_indoor::envelope::columns::{ColumnConstruction, ColumnStock};

pub(super) fn build(
    a: &mut Assembly,
    p: &Pillar,
    shape: &ColumnProfile,
    finish: ColumnConstruction,
    section: &[Vec2],
    ceiling: impl Fn(Vec2) -> f32,
) {
    let h = shape.collar_height;
    let skin = finish.stock.surface();
    let top: Vec<_> = section
        .iter()
        .map(|&v| {
            let point = p.center + v * (p.radius * shape.taper);
            Vec3::new(point.x, ceiling(point), point.y)
        })
        .collect();
    let min_top = top.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let ring = |y: f32, offset: f32| -> Vec<Vec3> {
        section
            .iter()
            .zip(&top)
            .map(|(&v, roof)| {
                let t = ((y - h) / (roof.y - h)).clamp(0., 1.);
                let point = p.center + v * (p.radius * (1. + t * (shape.taper - 1.)) + offset);
                Vec3::new(point.x, y, point.y)
            })
            .collect()
    };
    let mut arc = vec![0.];
    for (i, d) in section.iter().enumerate() {
        arc.push(arc[i] + d.distance(section[(i + 1) % section.len()]) * p.radius);
    }
    let projection = finish.base_projection_m;
    let bevel = projection.min(h * 0.16);
    let mut last = ring(0., projection * 0.65);
    face(
        a.part(finish.base_surface, "other_structure"),
        &last,
        Vec3::NEG_Y,
    );
    let mut band = |next: Vec<Vec3>, surface| {
        strip(
            a.part(surface, "other_structure"),
            &last,
            &next,
            &arc,
            p.center,
            p.sides <= 4,
        );
        last = next;
    };
    for (y, radius) in [(bevel, projection), (h - bevel, projection), (h, 0.)] {
        band(ring(y, radius), finish.base_surface);
    }
    if let Some(pitch) = finish.sleeve_pitch_m {
        let half = finish.joint_width_m * 0.5;
        let bed = if finish.stock == ColumnStock::VeneerWrap {
            Surface::WoodEdge
        } else {
            Surface::Rubber
        };
        let mut y = h + pitch;
        while y + half + 0.04 < min_top {
            band(ring(y - half, 0.), skin);
            band(ring(y - half * 0.48, -finish.joint_depth_m), skin);
            band(ring(y + half * 0.48, -finish.joint_depth_m), bed);
            band(ring(y + half, 0.), skin);
            y += pitch;
        }
    }
    band(top, skin);
    face(a.part(skin, "other_structure"), &last, Vec3::Y);
}

fn strip(g: &mut Geometry, lo: &[Vec3], hi: &[Vec3], arc: &[f32], offset: Vec2, faceted: bool) {
    let n = lo.len();
    let vertex = |g: &mut Geometry, p: Vec3, normal: Vec3, u: f32| {
        g.positions.push(p.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push([u + offset.x, p.y + offset.y]);
    };
    if faceted {
        for i in 0..n {
            let j = (i + 1) % n;
            let start = g.positions.len() as u32;
            let normal = (hi[i] - lo[i]).cross(lo[j] - lo[i]).normalize();
            for (p, u) in [
                (lo[i], arc[i]),
                (hi[i], arc[i]),
                (lo[j], arc[i + 1]),
                (hi[j], arc[i + 1]),
            ] {
                vertex(g, p, normal, u);
            }
            g.indices
                .extend([start, start + 1, start + 2, start + 2, start + 1, start + 3]);
        }
        return;
    }
    let start = g.positions.len() as u32;
    for (step, &u) in arc.iter().enumerate() {
        let i = step % n;
        let rise = hi[i] - lo[i];
        for ring in [lo, hi] {
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;
            let tangent = (ring[i] - ring[prev]).normalize() + (ring[next] - ring[i]).normalize();
            vertex(g, ring[i], rise.cross(tangent).normalize(), u);
        }
        if step < n {
            let s = start + 2 * step as u32;
            g.indices.extend([s, s + 1, s + 2, s + 2, s + 1, s + 3]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn wrapped_shafts_are_closed_at_every_joint_and_follow_the_sloped_slab() {
        let center = Vec2::new(2.0, -1.25);
        let ceiling = |p: Vec2| 3.68 + p.dot(Vec2::new(0.12, -0.08));
        for sides in [4, 24, 32] {
            let shape = ColumnProfile::sample(32, 0);
            let mut finish = shape.construction.unwrap();
            finish.stock = ColumnStock::MetalJacket;
            finish.sleeve_pitch_m = Some(1.12);
            let section: Vec<_> = (0..sides)
                .map(|i| Vec2::from_angle(std::f32::consts::TAU * i as f32 / sides as f32))
                .collect();
            let pillar = Pillar {
                center,
                radius: 0.27,
                sides,
                profile: Some(shape.clone()),
            };
            let mut a = Assembly::default();
            build(&mut a, &pillar, &shape, finish, &section, ceiling);
            let mut edges = BTreeMap::new();
            for ((_, label), g) in &a.parts {
                assert_eq!(label, "other_structure");
                for t in g.indices.as_chunks::<3>().0 {
                    let [i, j, k] = [t[0], t[1], t[2]].map(|i| i as usize);
                    let [a, b, c] = [i, j, k].map(|i| Vec3::from_array(g.positions[i]));
                    assert!(
                        (a.xz() - center).length()
                            <= pillar.radius + finish.base_projection_m + 1e-5
                    );
                    assert!(a.y >= 0. && a.y <= ceiling(a.xz()) + 1e-5);
                    let surface_normal = (b - a).cross(c - a).normalize();
                    for i in [i, j, k] {
                        assert!(surface_normal.dot(Vec3::from_array(g.normals[i])) > 0.5);
                    }
                    for (x, y) in [(a, b), (b, c), (c, a)] {
                        let mut edge = [x, y].map(|v| v.to_array().map(f32::to_bits));
                        let sign = if edge[0] > edge[1] {
                            edge.swap(0, 1);
                            -1
                        } else {
                            1
                        };
                        let (count, winding) = edges.entry(edge).or_insert((0, 0));
                        *count += 1;
                        *winding += sign;
                    }
                }
            }
            assert!(edges.values().all(|&pair| pair == (2, 0)));
            let bed = &a.parts[&(Surface::Rubber, "other_structure".into())];
            assert_eq!(bed.indices.len() / 6, 3 * sides as usize);
            for uv in &bed.uvs {
                assert!(uv.iter().all(|v| v.is_finite()));
            }
            let width = bed
                .uvs
                .iter()
                .map(|uv| uv[0])
                .fold(f32::NEG_INFINITY, f32::max)
                - bed.uvs.iter().map(|uv| uv[0]).fold(f32::INFINITY, f32::min);
            let perimeter =
                2. * sides as f32 * (std::f32::consts::PI / sides as f32).sin() * pillar.radius;
            assert!((width - perimeter).abs() < 1e-5);
        }
    }
}
