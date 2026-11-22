use super::*;
use crate::scene::procedural_indoor::{materials, objects::audit};

#[test]
fn film_is_visible_only_in_compatible_bins_and_fits_the_actual_cavity() {
    let mut over = 0;
    let mut tucked = 0;
    let mut bare = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::TrashCan, seed);
        o.size = dimensions(&mut stream(seed, 3015));
        for role in [0, WASTE_STATION, WASTE_STATION | RECYCLING] {
            o.variant = role;
            let p = parameters(&o);
            assert_eq!(p, parameters(&o));
            let Some(film) = p.liner else {
                bare += 1;
                continue;
            };
            assert!(matches!(p.lid, 0 | 4 | 5));
            let mut g = Geometry::default();
            film.build(&mut g, &p);
            assert!(g.indices.len() / 3 <= 416);
            assert!(materials::finish_slot(Surface::Plastic, film.finish.label()).is_some());
            let mut upper = 0_f32;
            for xyz in &g.positions {
                let v = Vec3::from_array(*xyz);
                upper = upper.max(v.y);
                assert!(v.x.abs() <= o.size.x * 0.5 && v.z.abs() <= o.size.z * 0.5);
                assert!((p.floor..=o.size.y + 1e-6).contains(&v.y));
                if v.y <= p.body.y - p.rim_height {
                    let uv = v.xz().abs() / p.cavity_half(v.y);
                    let exterior = v.xz().abs() / (p.body.xz() * 0.5 * 1.015);
                    let outside_hem = film.mount == LinerMount::OverRim
                        && v.y >= p.body.y - film.hem_drop_m - 1e-6
                        && exterior.x.powf(p.squareness) + exterior.y.powf(p.squareness) > 1.;
                    assert!(
                        uv.x.powf(p.squareness) + uv.y.powf(p.squareness) < 1. || outside_hem,
                        "{seed}: fold protrudes through housing"
                    );
                }
            }
            for normal in &g.normals {
                assert!((Vec3::from_array(*normal).length() - 1.).abs() < 3e-5);
            }
            for tri in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let n = (b - a).cross(c - a);
                assert!(n.length_squared() > 1e-16);
                for &i in tri {
                    assert!(
                        n.dot(Vec3::from_array(g.normals[i as usize])) > 0.,
                        "{seed}: inverted film"
                    );
                }
            }
            if film.mount == LinerMount::OverRim {
                over += 1;
                assert_eq!(p.lid, 0);
                assert!((upper - o.size.y).abs() < 1e-6);
            } else {
                tucked += 1;
                assert!(upper < p.body.y);
            }
        }
    }
    assert!(over > 12 && tucked > 120 && bare > 150);
}
