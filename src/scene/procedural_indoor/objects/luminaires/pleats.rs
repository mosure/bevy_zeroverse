//! One unfolded, metric fabric chart around the shade. Normals crease at folds
//! while neighbouring panels meet on the same line of the weave.
use super::*;

pub(super) fn build(
    g: &mut Geometry,
    center: Vec3,
    lower: f32,
    upper: f32,
    height: f32,
    wall: f32,
    pleats: u32,
) {
    let sides = 2 * pleats;
    let column = |i: u32| {
        let (sin, cos) = (TAU * (i % sides) as f32 / sides as f32).sin_cos();
        let ray = Vec3::new(cos, 0., sin);
        let fold = if i.is_multiple_of(2) { 1. } else { 0.945 };
        (ray, ray * fold)
    };
    let middle = (lower + upper) * 0.5;
    let mut prev = column(0);
    let mut along = 0.;
    for side in 1..=sides {
        let next = column(side);
        let u = along + middle * prev.1.distance(next.1);
        for inside in [false, true] {
            let inset = if inside { wall } else { 0. };
            let cut = |(axis, folded): (Vec3, Vec3), r, y| {
                center + folded * r - axis * inset + Vec3::Y * y
            };
            let a = cut(prev, lower, wall);
            let b = cut(prev, upper, height - wall);
            let c = cut(next, upper, height - wall);
            let d = cut(next, lower, wall);
            let n = (b - a).cross(d - a).normalize() * if inside { -1. } else { 1. };
            let first = g.positions.len() as u32;
            g.positions.extend([a, b, c, d].map(|p| p.to_array()));
            g.normals.extend([n.to_array(); 4]);
            g.uvs.extend([
                [along, 0.],
                [along, b.distance(a)],
                [u, c.distance(d)],
                [u, 0.],
            ]);
            let indices = if inside {
                [0, 2, 1, 0, 3, 2]
            } else {
                [0, 1, 2, 0, 2, 3]
            };
            g.indices.extend(indices.map(|i| first + i));
        }
        prev = next;
        along = u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloth_coordinates_cross_each_fold_at_its_measured_cut_length() {
        let mut g = Geometry::default();
        let center = Vec3::new(0.15, 1.10, -0.2);
        let (r, top, h, wall, n) = (0.18, 0.14, 0.21, 0.0025, 29);
        build(&mut g, center, r, top, h, wall, n);
        assert_eq!(g.positions.len(), 2 * n as usize * 8);
        let mut prev_end = None;
        for (i, (pts, uvs)) in g
            .positions
            .as_chunks::<8>()
            .0
            .iter()
            .zip(g.uvs.as_chunks::<8>().0)
            .enumerate()
        {
            let points: Vec<_> = pts.iter().copied().map(Vec3::from).collect();
            let pitch = points[0]
                .lerp(points[1], 0.5)
                .distance(points[3].lerp(points[2], 0.5));
            assert!((uvs[3][0] - uvs[0][0] - pitch).abs() < 5e-7);
            if let Some((prev_u, prev_p)) = prev_end {
                assert_eq!(uvs[0], prev_u);
                assert_eq!(points[0], prev_p);
            }
            prev_end = Some((uvs[3], points[3]));
            assert_eq!(uvs[3][0], uvs[2][0]);
            assert!((uvs[1][1] - points[1].distance(points[0])).abs() < 1e-6);
            for j in [0, 4] {
                let v = (points[j] - center).with_y(0.);
                let dot = Vec3::from(g.normals[i * 8 + j]).dot(v);
                assert!(if j == 0 { dot > 0. } else { dot < 0. });
            }
        }
    }
}
