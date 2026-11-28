use super::*;
use crate::scene::procedural_indoor::{
    materials::glass::carboy,
    objects::{audit, build_object, part_label},
};

fn top_at(mesh: &Assembly, p: Vec2) -> f32 {
    let mut top = f32::NEG_INFINITY;
    for g in mesh.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|j| Vec3::from_array(g.positions[j as usize]));
            let (v, w) = (b.xz() - a.xz(), c.xz() - a.xz());
            let det = v.perp_dot(w);
            if det.abs() < 1e-9 {
                continue;
            }
            let k = p - a.xz();
            let (u, t) = (k.perp_dot(w) / det, v.perp_dot(k) / det);
            if u.min(t) >= 0. && u + t <= 1. {
                top = top.max(a.y + u * (b.y - a.y) + t * (c.y - a.y));
            }
        }
    }
    top
}

#[test]
fn cooler_stock_preserves_cup_space_receiver_and_segmented_optics() {
    let mut loads = [0; 3];
    let mut waters = [0; 3];
    let mut taps = [0; 3];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::WaterCooler, seed);
        let p = Cooler::for_object(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(check.triangles < 9000 && check.min_y_m.abs() < 0.00001);
        let (lo, hi) = mesh.bounds();
        assert!(lo.x >= -o.size.x * 0.5 - 1e-6 && hi.z <= o.size.z * 0.5 + 1e-6);
        assert!((hi.y - o.size.y).abs() < 0.00001);
        loads[p.loading as usize] += 1;
        taps[p.taps - 1] += 1;
        let liquid = mesh.parts.get(&(Surface::Liquid, "drink_container".into()));
        let dry = mesh
            .parts
            .get(&(Surface::ContainerGlass, carboy::DRY.into()));
        let wet = mesh
            .parts
            .get(&(Surface::ContainerGlass, carboy::WET.into()));
        assert_eq!(dry.is_some(), p.loading == Loading::TopBottle);
        assert_eq!(wet.is_some(), liquid.is_some());
        let cap = mesh.parts.get(&(Surface::Plastic, carboy::CAP.into()));
        assert_eq!(cap.is_some(), dry.is_some());
        if let Some(g) = cap {
            let bore = p.bottle.as_ref().unwrap().radius_m * 0.25 + 0.003;
            // The load-bearing cap spans both sides of the gasket's top at
            // the precise socket radius. A floating neck cannot pass this.
            let seats: Vec<_> = g
                .positions
                .iter()
                .filter(|v| v[0].hypot(v[2]) > bore)
                .map(|v| v[1])
                .collect();
            assert!(
                seats.iter().copied().fold(f32::INFINITY, f32::min) < p.body_height_m
                    && seats.iter().any(|h| *h > p.body_height_m + 0.002)
            );
        }
        for (s, l) in mesh.parts.keys() {
            assert_eq!(
                part_label(l),
                if matches!(s, Surface::Liquid | Surface::ContainerGlass) || l == carboy::CAP {
                    "drink_container"
                } else {
                    CLASS
                }
            );
        }
        if let Some(b) = p.bottle {
            assert!([11.356, 18.927]
                .iter()
                .any(|x| (x - b.capacity_l).abs() < 0.003));
            waters[if b.fill_fraction == 0. {
                0
            } else if b.fill_fraction < 0.5 {
                1
            } else {
                2
            }] += 1;
            assert_eq!(liquid.is_some(), b.fill_fraction > 0.);
            if let Some(wet) = wet {
                let surface = liquid
                    .unwrap()
                    .positions
                    .iter()
                    .map(|v| v[1])
                    .fold(0., f32::max);
                // Recover volume directly from the rendered inside radii and heights.
                let mut rings: Vec<_> = wet
                    .positions
                    .iter()
                    .map(|v| (v[1], v[0].hypot(v[2]) - 0.0015))
                    .collect();
                rings.sort_by(|a, b| a.0.total_cmp(&b.0));
                rings.dedup_by(|a, b| a.0 == b.0);
                let vol: f32 = rings
                    .windows(2)
                    .map(|p| {
                        let [(y, r), (z, s)] = [p[0], p[1]];
                        16. * (TAU / 32.).sin() * (z - y) * (r * r + r * s + s * s) / 3.
                    })
                    .sum();
                assert!(
                    (1000. * vol - b.capacity_l * b.fill_fraction).abs() < 0.009,
                    "seed {seed}"
                );
                assert!((rings.last().unwrap().0 - surface).abs() < 0.0001);
            }
        } else {
            let y = top_at(&mesh, Vec2::new(0.019, 0.005));
            let expected = if p.loading == Loading::TopVacant {
                0.954 * p.body_height_m
            } else {
                p.body_height_m
            };
            assert!(
                (y - expected).abs() < 1e-5,
                "wrong receiver on seed {seed}: {y} {expected}"
            );
        }
        let pitch = o.size.x * (if p.taps == 3 { 0.253 } else { 0.32 });
        for i in 0..p.taps {
            let center = Vec3::new(
                (i as f32 - (p.taps - 1) as f32 * 0.5) * pitch,
                p.tray_height_m + 0.5 * p.cup_clearance_m,
                p.outlet_z_m,
            );
            let half = Vec3::new(0.032, p.cup_clearance_m * 0.48, 0.032);
            for g in mesh.parts.values() {
                for tri in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = tri.map(|j| Vec3::from_array(g.positions[j as usize]));
                    assert!(
                        !((center + half).cmpgt(a.min(b).min(c)).all()
                            && (center - half).cmplt(a.max(b).max(c)).all()),
                        "cup obstructed on seed {seed}"
                    );
                }
            }
        }
    }
    assert!(
        loads.iter().all(|&n| n >= 3)
            && waters.iter().all(|&n| n >= 2)
            && taps.iter().all(|&n| n >= 5),
        "{loads:?} {waters:?} {taps:?}"
    );
}
