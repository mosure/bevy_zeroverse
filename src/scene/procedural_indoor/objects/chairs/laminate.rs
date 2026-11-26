//! A laminated strip keeps its broad veneer faces, sawn edges and grain
//! direction through the bends. Section rings stay normal to the centreline.
use super::*;

pub(super) fn build<const N: usize>(
    a: &mut Assembly,
    path: &[Vec3; N],
    width: f32,
    thickness: f32,
    origin: Vec2,
) {
    let (w, h) = (width * 0.5, thickness * 0.5);
    let bevel = 0.0022;
    let outline = [
        Vec2::new(-w + bevel, -h),
        Vec2::new(w - bevel, -h),
        Vec2::new(w, -h + bevel),
        Vec2::new(w, h - bevel),
        Vec2::new(w - bevel, h),
        Vec2::new(-w + bevel, h),
        Vec2::new(-w, h - bevel),
        Vec2::new(-w, -h + bevel),
    ];
    let mut along = [0.; N];
    for i in 1..N {
        along[i] = along[i - 1] + path[i].distance(path[i - 1]);
    }
    let tangents: [Vec3; N] = std::array::from_fn(|i| {
        let incoming = path[i] - path[i.saturating_sub(1)];
        let outgoing = path[(i + 1).min(N - 1)] - path[i];
        (incoming.normalize_or_zero() + outgoing.normalize_or_zero()).normalize()
    });
    let normals = tangents.map(|t| t.cross(Vec3::X));
    let mut across = 0.;
    for j in 0..outline.len() {
        let left = outline[j];
        let right = outline[(j + 1) % outline.len()];
        let edge = right - left;
        let side_normal = Vec2::new(edge.y, -edge.x).normalize();
        let next = across + edge.length();
        let surface = if j == 0 || j == 4 {
            Surface::Wood
        } else {
            Surface::WoodEdge
        };
        let g = a.part(surface, "chair");
        let base = g.positions.len() as u32;
        for i in 0..N {
            let normal = Vec3::X * side_normal.x + normals[i] * side_normal.y;
            for (p, u) in [(left, across), (right, next)] {
                g.positions
                    .push((path[i] + Vec3::X * p.x + normals[i] * p.y).to_array());
                g.normals.push(normal.to_array());
                g.uvs.push((origin + Vec2::new(u, along[i])).to_array());
            }
            if i > 0 {
                let k = base + (i as u32 - 1) * 2;
                g.indices.extend([k, k + 1, k + 3, k, k + 3, k + 2]);
            }
        }
        across = next;
    }
    let g = a.part(Surface::WoodEdge, "chair");
    for i in [0, N - 1] {
        let base = g.positions.len() as u32;
        let normal = if i == 0 { -tangents[i] } else { tangents[i] };
        for p in outline {
            g.positions
                .push((path[i] + Vec3::X * p.x + normals[i] * p.y).to_array());
            g.normals.push(normal.to_array());
            g.uvs.push((origin + p).to_array());
        }
        for j in 1..outline.len() as u32 - 1 {
            let (b, c) = (base + j, base + j + 1);
            g.indices
                .extend(if i == 0 { [base, c, b] } else { [base, b, c] });
        }
    }
}
