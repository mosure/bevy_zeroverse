use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};

#[test]
fn dining_linen_has_one_load_bearing_face_and_a_bounded_continuous_hem() {
    for seed in 0..96 {
        let mut table = audit::fixture(ObjectKind::Table, seed);
        table.variant = (seed % 3) as u32;
        table.size = Vec3::new(
            0.66 + (seed % 5) as f32 * 0.35,
            0.77,
            0.75 + (seed % 4) as f32 * 0.30,
        );
        let stock = LinenWeave::ALL[seed as usize % 4];
        let base = parameters(&table);
        let support = super::super::outline(&table);
        let service = DiningService {
            prevalence: 1.,
            weave: stock,
            drop: (seed % 32) as f32,
            powered: true,
        };
        service.lay(&mut table);
        let cloth = dining_linen(&table).unwrap();
        assert_eq!(cloth.weave, stock);
        assert_eq!(base.frame_surface, parameters(&table).frame_surface);
        assert_eq!(support, super::super::outline(&table));
        let assembly = build_object(&table);
        audit::check(&table, &assembly).unwrap();
        let g = &assembly.parts[&(Surface::Fabric, stock.label().into())];
        assert_eq!(part_label(stock.label()), "table");
        // The replaceable fabric has a real inner face for low-angle renders.
        assert!(g
            .positions
            .iter()
            .zip(&g.normals)
            .any(|(p, n)| Vec2::new(p[0], p[2]).normalize().dot(Vec2::new(n[0], n[2])) < -0.7));
        assert!(assembly
            .parts
            .iter()
            .filter(|((s, _), _)| *s != Surface::Fabric)
            .all(|(_, g)| g.positions.iter().all(|p| p[1] < table.size.y - 0.001)));
        let mut top_area = 0.;
        for t in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|k| Vec3::from_array(g.positions[k as usize]));
            if [a, b, c].iter().all(|p| p.y == table.size.y) {
                let twice_area = (b - a).cross(c - a).y;
                assert!(twice_area > 0.);
                top_area += 0.5 * twice_area;
            }
        }
        let outline_area: f32 = support
            .iter()
            .zip(support.iter().cycle().skip(1))
            .map(|(p, q)| 0.5 * p.perp_dot(*q))
            .sum();
        assert!((0.93..1.).contains(&(top_area / outline_area)));
        let clearance = Clearance::new(&table);
        let min_y = g
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, f32::min);
        assert!(min_y >= table.size.y - cloth.max_drop_m - 1e-6);
        assert!(min_y < table.size.y - cloth.max_drop_m * 0.74);
        // Every rendered shoulder and hem remains inside the table's collision envelope.
        for p in g.positions.iter().step_by(7) {
            let p = Vec3::from_array(*p);
            assert!(clearance.hits_box(p, p, 0.0001), "{seed} {p:?}");
        }
        let decoded = serde_json::from_slice(&serde_json::to_vec(&table).unwrap()).unwrap();
        let replay = build_object(&decoded);
        let other = &replay.parts[&(Surface::Fabric, stock.label().into())];
        assert_eq!(g.positions, other.positions);
        assert_eq!(g.uvs, other.uvs);
    }
}

#[test]
fn dining_linen_service_samples_bare_mixed_and_uniform_venues() {
    let mut counts = [0; 3];
    let mut stocks = [0; 4];
    for seed in 0..96 {
        let service = DiningService::sample(seed, 0);
        let mut used = 0;
        for k in 0..12 {
            let mut t = audit::fixture(ObjectKind::Table, seed * 12 + k);
            t.variant = 1;
            assert!(dining_linen(&t).is_none());
            service.lay(&mut t);
            if let Some(cloth) = dining_linen(&t) {
                assert_eq!(cloth.weave, service.weave);
                stocks[cloth.weave as usize] += 1;
                used += 1;
            }
        }
        counts[match used {
            0 => 0,
            12 => 2,
            _ => 1,
        }] += 1;
    }
    assert!(counts.iter().all(|&n| n >= 10), "{counts:?}");
    assert!(stocks.iter().all(|&n| n > 20), "{stocks:?}");
    let full = DiningService {
        prevalence: 1.,
        weave: LinenWeave::BistroCheck,
        drop: 20.,
        powered: true,
    };
    for (kind, v) in [
        (ObjectKind::Table, SHARED_ROW_VARIANT),
        (ObjectKind::Table, BOARDROOM_VARIANT),
        (ObjectKind::Desk, 1),
    ] {
        let mut t = audit::fixture(kind, 0);
        t.variant = v;
        full.lay(&mut t);
        assert_eq!(t.variant, v);
        assert!(dining_linen(&t).is_none());
    }
}
