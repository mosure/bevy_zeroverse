use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn fixed_wall_admits_asymmetric_high_broad_and_private_glazing() {
    let mut scene =
        IndoorManifest::generate_with_humans(11, IndoorLayout::Conference, 0.5, 0, 0.).unwrap();
    let h = partition_height(&scene);
    let mut counts = [0; 4];
    let mut spans = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let installed = InteriorGlazing::sample(seed);
        installed.validate().unwrap();
        let sill = installed.sill_m;
        scene
            .program
            .as_mut()
            .unwrap()
            .finishes
            .as_mut()
            .unwrap()
            .interior_glazing = Some(installed);
        let facade = shared_facade(&scene, h);
        let mut sides = [false; 2];
        let mut area = 0.;
        for o in &facade.openings {
            assert!(o.min.y >= sill && o.max.y <= h && o.max.cmpgt(o.min).all());
            let left = o.max.x <= -scene.door_x - scene.entry_width() * 0.5;
            let right = o.min.x >= -scene.door_x + scene.entry_width() * 0.5;
            assert!(left || right);
            sides[usize::from(right)] = true;
            area += (o.max - o.min).element_product();
        }
        counts[0] += usize::from(!sides[0] && !sides[1]);
        counts[1] += usize::from(sides[0] != sides[1]);
        counts[2] += usize::from(sides[0] && sides[1]);
        counts[3] += usize::from(
            !facade.openings.is_empty() && facade.openings.iter().all(|o| o.min.y > h * 0.48),
        );
        spans.insert((area * 100.).round() as usize);
    }
    assert!(counts.iter().all(|count| *count >= 10), "{counts:?}");
    assert!(spans.len() > 155);
    let mut legacy = serde_json::to_value(InteriorGlazing::sample(42)).unwrap();
    legacy.as_object_mut().unwrap().remove("shared_openings");
    legacy.as_object_mut().unwrap().remove("vision_panels");
    let legacy: InteriorGlazing = serde_json::from_value(legacy).unwrap();
    assert!(legacy.shared_openings.is_none());
    let sill = legacy.sill_m;
    scene
        .program
        .as_mut()
        .unwrap()
        .finishes
        .as_mut()
        .unwrap()
        .interior_glazing = Some(legacy);
    let windows = shared_facade(&scene, h).openings;
    assert!(
        !windows.is_empty()
            && windows
                .iter()
                .all(|o| o.min.y == sill && o.glazing.is_none())
    );
}

#[test]
fn fixed_bays_share_privacy_stock_and_preserve_wall_and_door_clearances() {
    let mut scene =
        IndoorManifest::generate_with_humans(11, IndoorLayout::Conference, 0.5, 0, 0.).unwrap();
    scene.room_size.x = 22.;
    scene.door_x = 0.;
    let height = 3.5;
    let mut treatments = std::collections::BTreeSet::new();
    let mut regular = 0;
    let mut cut = 0;
    for seed in 0..128 {
        let mut p = InteriorGlazing::sample(seed);
        p.shared_openings = Some(SharedOpenings {
            jamb_intervals: [Vec2::Y; 2],
            height_interval: Vec2::Y,
        });
        p.opening_pitch_m = 2.6;
        let stock = p.vision_panels.clone().unwrap();
        let fw = p.frame_width_m;
        treatments.insert(stock.glass.treatment);
        scene
            .program
            .as_mut()
            .unwrap()
            .finishes
            .as_mut()
            .unwrap()
            .interior_glazing = Some(p);
        let f = shared_facade(&scene, height);
        let old = scene
            .program
            .as_mut()
            .unwrap()
            .finishes
            .as_mut()
            .unwrap()
            .interior_glazing
            .as_mut()
            .unwrap();
        old.vision_panels = None;
        let old = shared_facade(&scene, height);
        assert_eq!(f.openings.len(), old.openings.len());
        let area = |f: &Facade| f.openings.iter().map(WindowOpening::area).sum::<f32>();
        assert!((area(&f) - area(&old)).abs() < 3e-5);
        for o in &f.openings {
            assert!(o.min.x >= scene.entry_width() * 0.5 || o.max.x <= -scene.entry_width() * 0.5);
            assert!(o.max.x - o.min.x > 2. * fw);
            assert_eq!(o.glazing.as_ref(), Some(&stock.glass));
            assert!(!o.operable);
            for pane in f.pane_installations(o) {
                assert!(pane.handle.is_none() && pane.swing_radians == 0.);
                assert!((pane.max - pane.min).min_element() >= 0.04);
            }
        }
        for side in [false, true] {
            let run: Vec<_> = f
                .openings
                .iter()
                .filter(|o| (o.min.x > 0.) == side)
                .collect();
            assert!(run.len() >= 3);
            let widths: Vec<_> = run.iter().map(|o| o.max.x - o.min.x).collect();
            let span = widths.iter().copied().reduce(f32::max).unwrap()
                - widths.iter().copied().reduce(f32::min).unwrap();
            regular += usize::from(span < 1e-4);
            cut += usize::from(span > 0.10);
            let pier = run[1].min.x - run[0].max.x;
            assert!(pier > 0.);
            assert!(run
                .windows(2)
                .all(|w| (w[1].min.x - w[0].max.x - pier).abs() < 1e-5));
        }
    }
    assert_eq!(treatments, [0, 1, 2, 3, 5, 6].into_iter().collect());
    assert!(regular > 35 && cut > 100);
}
