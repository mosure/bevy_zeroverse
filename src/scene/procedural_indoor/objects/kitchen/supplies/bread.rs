//! Loaf construction and crust cuts follow the same realized mesh.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LoafProgram {
    pub height: f32,
    pub foot: f32,
    pub belly_height: f32,
    pub shoulder_height: f32,
    pub top_width: f32,
    pub roundness: f32,
    pub cuts: u32,
    pub cut_pitch: f32,
    pub cut_width: f32,
    pub cut_reach: f32,
    pub cut_angle: f32,
    pub cut_phase: f32,
}
pub fn parameters(o: &IndoorObject) -> LoafProgram {
    let mut r = stream(o.seed, 0x42524541444c4f41);
    LoafProgram {
        height: r.random_range(0.76..0.98),
        foot: r.random_range(0.54..0.88),
        belly_height: r.random_range(0.22..0.48),
        shoulder_height: r.random_range(0.66..0.85),
        top_width: r.random_range(0.08..0.38),
        roundness: r.random_range(2.0..5.5),
        cuts: r.random_range(1..=6),
        cut_pitch: r.random_range(0.085..0.15),
        cut_width: r.random_range(0.006..0.018),
        cut_reach: r.random_range(0.16..0.34),
        cut_angle: r.random_range(-0.65..0.65),
        cut_phase: r.random_range(-0.045..0.045),
    }
}
pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let half = s.xz() * 0.49;
    let h = s.y * p.height;
    let loaf = a.part(Surface::Terracotta, "food#finish103");
    let crown = 4. * (p.shoulder_height - p.belly_height) / (1. - p.belly_height);
    let rings: Vec<_> = (0..=12)
        .map(|i| {
            let t = i as f32 / 12.;
            let radius = if t < p.belly_height {
                p.foot + (1. - p.foot) * (t / p.belly_height * FRAC_PI_2).sin()
            } else {
                let u = (t - p.belly_height) / (1. - p.belly_height);
                p.top_width + (1. - p.top_width) * (1. - u.powf(crown)).max(0.).sqrt()
            };
            (h * t, half * radius)
        })
        .collect();
    loaf.housing(&rings, p.roundness, Transform::IDENTITY);
    let scoring = bread_scoring(loaf, s, &p);
    a.part(Surface::Paper, "food#finish104").append(scoring);
}

/// Clip thin crust markings from the real loaf triangles. Analytic ellipsoid
/// heights or flat strips can float above its tessellated surface. The 0.3 mm
/// vertical offset keeps the projected cut footprint on that same triangle,
/// including steep shoulders, and avoids coincident surfaces.
fn bread_scoring(loaf: &Geometry, size: Vec3, p: &LoafProgram) -> Geometry {
    let mut out = Geometry::default();
    let (sine, cosine) = p.cut_angle.sin_cos();
    for triangle in loaf.indices.as_chunks::<3>().0 {
        let vertices: Vec<_> = triangle
            .iter()
            .map(|&i| Vec3::from_array(loaf.positions[i as usize]))
            .collect();
        if vertices.iter().any(|v| v.y < size.y * 0.60) {
            continue;
        }
        let normal = (vertices[1] - vertices[0])
            .cross(vertices[2] - vertices[0])
            .normalize();
        for i in 0..p.cuts {
            let x = ((i as f32 - (p.cuts - 1) as f32 * 0.5) * p.cut_pitch + p.cut_phase) * size.x;
            let mut polygon = vertices.clone();
            for (axis, bound, sign) in [
                (0, x - size.x * p.cut_width, 1.),
                (0, x + size.x * p.cut_width, -1.),
                (2, -size.z * p.cut_reach, 1.),
                (2, size.z * p.cut_reach, -1.),
            ] {
                if polygon.is_empty() {
                    break;
                }
                let mut clipped = Vec::with_capacity(polygon.len() + 1);
                let mut previous = *polygon.last().unwrap();
                let coordinate = |v: Vec3| {
                    if axis == 0 {
                        v.x * cosine + v.z * sine
                    } else {
                        v.z * cosine - v.x * sine
                    }
                };
                let mut previous_distance = (coordinate(previous) - bound) * sign;
                for &current in &polygon {
                    let distance = (coordinate(current) - bound) * sign;
                    if (distance >= 0.) != (previous_distance >= 0.) {
                        clipped.push(
                            previous
                                .lerp(current, previous_distance / (previous_distance - distance)),
                        );
                    }
                    if distance >= 0. {
                        clipped.push(current);
                    }
                    previous = current;
                    previous_distance = distance;
                }
                polygon = clipped;
            }
            for j in 1..polygon.len().saturating_sub(1) {
                let points = [polygon[0], polygon[j], polygon[j + 1]];
                if (points[1] - points[0])
                    .cross(points[2] - points[0])
                    .length_squared()
                    < 1e-16
                {
                    continue;
                }
                let start = out.positions.len() as u32;
                for point in points {
                    out.positions.push((point + Vec3::Y * 0.0003).to_array());
                    out.normals.push(normal.to_array());
                    out.uvs.push(point.xz().to_array());
                }
                out.indices.extend([start, start + 1, start + 2]);
            }
        }
    }
    out
}
