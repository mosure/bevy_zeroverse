//! Indexed cloth cuts with smooth shading and a metric weave chart. Positions
//! on adjoining panels come from the same pattern edges.
use super::*;

pub(super) fn patch<const W: usize, const H: usize>(
    g: &mut Geometry,
    p: [[Vec3; W]; H],
    reverse: bool,
) {
    let base = g.positions.len() as u32;
    let mut v = [0.; W];
    for j in 0..H {
        let mut u = 0.;
        for i in 0..W {
            if i > 0 {
                u += p[j][i].distance(p[j][i - 1]);
            }
            if j > 0 {
                v[i] += p[j][i].distance(p[j - 1][i]);
            }
            let du = p[j][(i + 1).min(W - 1)] - p[j][i.saturating_sub(1)];
            let dv = p[(j + 1).min(H - 1)][i] - p[j.saturating_sub(1)][i];
            let normal = du.cross(dv).normalize();
            g.positions.push(p[j][i].to_array());
            g.normals
                .push((if reverse { -normal } else { normal }).to_array());
            g.uvs.push([u, v[i]]);
            if j + 1 < H && i + 1 < W {
                let a = base + (j * W + i) as u32;
                let b = a + W as u32;
                if reverse {
                    g.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                } else {
                    g.indices.extend([a, a + 1, b, a + 1, b + 1, b]);
                }
            }
        }
    }
}
pub(super) fn band<const N: usize>(g: &mut Geometry, a: [Vec3; N], b: [Vec3; N], out: Vec3) {
    let i = (N - 1) / 2;
    let reverse = (a[i + 1] - a[i]).cross(b[i] - a[i]).dot(out) < 0.;
    patch(g, [a, b], reverse);
}
