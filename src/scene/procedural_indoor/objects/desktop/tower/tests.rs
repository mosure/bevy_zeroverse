use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

fn check_apertures(face: Rect, openings: &[Rect]) {
    for (i, r) in openings.iter().enumerate() {
        assert!(face.contains(r.min) && face.contains(r.max) && r.size().min_element() > 0.);
        for other in &openings[..i] {
            assert!(r.intersect(*other).is_empty(), "{r:?} overlaps {other:?}");
        }
    }
    // Integrate the actual uncut sheet triangles. This catches face strips
    // spanning an aperture or being emitted twice at a shared edge.
    let mut sheet = Geometry::default();
    face::Plane::Front(0.).sheet(&mut sheet, face, openings);
    let area: f32 = sheet
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|tri| {
            let p = |k: usize| Vec3::from_array(sheet.positions[tri[k] as usize]).xy();
            let signed = (p(1) - p(0)).perp_dot(p(2) - p(0)) * 0.5;
            assert!(signed > 0.);
            signed
        })
        .sum();
    let expected =
        face.width() * face.height() - openings.iter().map(|r| r.width() * r.height()).sum::<f32>();
    assert!((area - expected).abs() < 2e-6, "{area} vs {expected}");
}

#[test]
fn tower_controls_exhaust_and_cable_jacks_fit_compact_and_upright_sheet_stock() {
    let mut banks = [0usize; 4];
    let mut intakes = [0usize; 3];
    let mut cooling = [0usize; 3];
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::PcTower, seed);
        o.size = dimensions(&mut stream(seed, 79), seed % 2 == 0);
        audit::check(&o, &build_object(&o)).unwrap_or_else(|e| panic!("tower seed {seed}: {e}"));
        let f = Fascia::sample(&o, &parameters(&o));
        banks[f.bank as usize] += 1;
        intakes[f.intake as usize] += 1;
        assert!(f
            .controls
            .iter()
            .any(|s| s.connector == sockets::Connector::Button));
        let front: Vec<_> = f
            .vents
            .iter()
            .copied()
            .chain(f.controls.iter().map(|s| s.aperture))
            .collect();
        check_apertures(f.front_face, &front);
        let sockets = sockets::rear_sockets(&o);
        let mut back = sockets.map(|s| s.aperture).to_vec();
        for (socket, jack) in sockets.iter().zip(ports::TowerJack::ALL) {
            assert_eq!(
                ports::tower_local(&o, jack),
                socket.aperture.center().extend(-o.size.z * 0.5)
            );
        }
        let vent = exhaust::Exhaust::fit(&o, f.rear_face, &sockets)
            .expect("a cooling band fits production cases");
        back.push(vent.opening);
        cooling[usize::from(vent.round)] += 1;
        if let Some(expansion) = vent.expansion {
            assert!(vent.blanking_plates > 0);
            back.push(expansion);
            cooling[2] += 1;
        }
        check_apertures(f.rear_face, &back);
    }
    assert!(banks
        .into_iter()
        .chain(intakes)
        .chain(cooling)
        .all(|n| n > 16));
}
