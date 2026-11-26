//! Paint and varnished timber share an edge, a height and the floor semantic.
use super::*;
use crate::scene::procedural_indoor::{materials::sport, program::court::LINE};

pub(super) fn top(a: &mut Assembly, c: VolleyballCourt, size: Vec3) {
    let local = |p: Vec2| if c.long_x { Vec2::new(p.y, p.x) } else { p };
    let lo = local(-size.xz() * 0.5 - c.center);
    let hi = local(size.xz() * 0.5 - c.center);
    let w = c.width * 0.5;
    let l = c.length * 0.5;
    let d = c.attack_distance;
    // Attack lines end at the sideline. The rear edge lies at the attack distance.
    let xs = [lo.x, -w, -w + LINE, w - LINE, w, hi.x];
    let zs = [
        lo.y,
        -l,
        -l + LINE,
        -d,
        -d + LINE,
        -LINE * 0.5,
        LINE * 0.5,
        d - LINE,
        d,
        l - LINE,
        l,
        hi.y,
    ];
    for (ix, x) in xs.windows(2).enumerate() {
        // Consecutive cells of one finish form one quad. Keep the painted
        // ribbons in two batched meshes for the entire hall, including run-off.
        let mut first = 0;
        let ink =
            |iz: usize| (1..=3).contains(&ix) && (1..=9).contains(&iz) && (ix != 2 || iz % 2 == 1);
        for iz in 0..zs.len() - 1 {
            if iz + 2 < zs.len() && ink(iz) == ink(iz + 1) {
                continue;
            }
            let g = if ink(iz) {
                a.part(Surface::Paint, sport::LINE)
            } else {
                a.part(Surface::Floor, "floor")
            };
            let rect = Rect::from_corners(
                c.world(Vec2::new(x[0], zs[first])),
                c.world(Vec2::new(x[1], zs[iz + 1])),
            );
            horizontal_quad(g, rect, 0.);
            first = iz + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::program::court::room_size;

    #[test]
    fn volleyball_gymnasium_floor_partitions_actual_triangles_into_paint_and_timber() {
        for seed in 20..32 {
            let size = room_size(seed);
            let c = VolleyballCourt::fit(seed, size).unwrap();
            let mut a = Assembly::default();
            top(&mut a, c, size);
            assert_eq!(a.parts.len(), 2);
            let mut total = 0.;
            let mut painted = 0.;
            for ((_, key), g) in &a.parts {
                for tri in g.indices.as_chunks::<3>().0 {
                    let [p, q, r] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    assert_eq!([p.y, q.y, r.y], [0.; 3]);
                    let area = (q - p).cross(r - p).y * 0.5;
                    assert!(area > 0.);
                    total += area;
                    if key == sport::LINE {
                        painted += area;
                    }
                }
            }
            let expected = 2. * LINE * c.length + 5. * LINE * (c.width - 2. * LINE);
            assert!((total - size.x * size.z).abs() < 0.001 && (painted - expected).abs() < 0.0001);
            let hits = |x, z| {
                let sample = c.world(Vec2::new(x, z));
                let mut labels = Vec::new();
                for ((_, key), g) in &a.parts {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let [p, q, r] = tri.map(|i| Vec3::from_array(g.positions[i as usize]).xz());
                        let (u, v, w) = (q - p, r - p, sample - p);
                        let det = u.perp_dot(v);
                        let (s, t) = (w.perp_dot(v) / det, u.perp_dot(w) / det);
                        if s >= 0. && t >= 0. && s + t <= 1. {
                            labels.push(key.as_str());
                        }
                    }
                }
                labels
            };
            for (x, z, paint) in [
                (0.137, 0.010, true),
                (0.137, 0.030, false),
                (0.11, c.attack_distance - 0.019, true),
                (0.11, c.attack_distance + 0.009, false),
                (c.width * 0.5 - 0.014, 0.271, true),
                (c.width * 0.5 + 0.014, 0.271, false),
                (0.137, -c.length * 0.5 + 0.018, true),
                (0.137, -c.length * 0.5 - 0.014, false),
            ] {
                assert_eq!(hits(x, z), [if paint { sport::LINE } else { "floor" }]);
            }
        }
    }
}
