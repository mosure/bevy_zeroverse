use super::*;
use crate::scene::procedural_indoor::{
    materials::{self, point_of_sale},
    objects::{audit, build_object, part_label},
};
use std::collections::BTreeSet;

// Raycast the actual assembled triangles to detect a deck covering a flush
// scanner aperture, even if the material key itself is correctly registered.
fn countertop_at(a: &Assembly, x: f32, z: f32) -> (Surface, f32) {
    let mut highest = (Surface::Metal, f32::NEG_INFINITY);
    for ((surface, _), g) in &a.parts {
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, q, r] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let (u, v) = (q - p, r - p);
            let det = u.xz().perp_dot(v.xz());
            if det >= -1e-8 {
                continue;
            }
            let delta = Vec2::new(x, z) - p.xz();
            let k = delta.perp_dot(v.xz()) / det;
            let l = u.xz().perp_dot(delta) / det;
            let h = p.y + k * u.y + l * v.y;
            if k >= 0. && l >= 0. && k + l <= 1. && h > highest.1 {
                highest = (*surface, h);
            }
        }
    }
    highest
}

#[test]
fn checkout_hardware_fits_the_admitted_box_and_faces_its_operator() {
    let mut forms = BTreeSet::new();
    let mut hands = BTreeSet::new();
    let mut cases = BTreeSet::new();
    let mut pages = BTreeSet::new();
    for seed in 3200..3328 {
        let o = audit::fixture(ObjectKind::CheckoutCounter, seed);
        let d = checkout_station(&o).unwrap();
        let archived: IndoorObject =
            serde_json::from_slice(&serde_json::to_vec(&o).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(d).unwrap(),
            serde_json::to_value(checkout_station(&archived).unwrap()).unwrap()
        );
        let a = build_object(&o);
        let stat = audit::check(&o, &a).unwrap();
        assert!(
            stat.batches <= 12 && stat.triangles < 4500 && stat.max_envelope_overrun_m < 0.001,
            "{seed} {stat:?}"
        );
        let (lo, hi) = a.bounds();
        assert!(
            lo.y.abs() < 1e-4 && (hi.y - o.size.y).abs() < 1e-4,
            "{seed} {lo} {hi}"
        );
        for (mat, label) in a.parts.keys() {
            assert!(matches!(part_label(label), "control_panel" | "counter"));
            if let Some(slot) = materials::finish_slot(*mat, label) {
                assert!(materials::registered_finish(*mat, slot));
            }
        }
        let label = point_of_sale::label(&o, d.form);
        let face = &a.parts[&(Surface::Ink, label.clone())];
        assert_eq!(face.indices.len(), 6);
        assert_eq!(face.uvs, [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
        let normal = Vec3::from_array(face.normals[0]);
        let kiosk = d.form == CheckoutForm::SelfCheckout;
        assert!(normal.y > 0. && normal.z * (if kiosk { 1. } else { -1. }) > 0.9);
        let reader = &a.parts[&(Surface::Ink, point_of_sale::READER.to_string())];
        assert!(reader.normals.iter().all(|v| v[1] > 0.7 && v[2] > 0.6));
        let [(cx, _), (wx, _)] = d.bays(o.size.x);
        let hit = countertop_at(&a, cx + 0.012, o.size.z * 0.046);
        if d.form == CheckoutForm::ApparelWrap {
            assert_eq!(hit.0, d.deck);
            assert!((hit.1 - d.worktop_height_m).abs() < 1e-4);
        } else {
            assert_eq!(hit.0, Surface::Ink, "scanner occluded in {seed}");
            assert!((hit.1 - (d.worktop_height_m - 0.0015)).abs() < 1e-4);
        }
        let work = countertop_at(&a, wx + 0.012, o.size.z * 0.056);
        let deck = if kiosk {
            d.bag_platform_height_m
        } else {
            d.worktop_height_m
        };
        assert!((work.1 - deck).abs() < 1e-4);
        assert_eq!(
            work.0,
            match d.form {
                CheckoutForm::SelfCheckout => Surface::Metal,
                CheckoutForm::ConveyorLane => Surface::Rubber,
                _ => d.deck,
            }
        );
        let local = Vec3::new(wx, d.worktop_height_m + 0.12, 0.);
        let mut through = true;
        let mut plate = false;
        d.view_boxes(o.size, |center, half| {
            through &= !(local - center).abs().cmplt(half).all();
            plate |= (Vec3::new(wx, deck - 0.009, 0.) - center)
                .abs()
                .cmplt(half)
                .all();
        });
        assert!(through && plate);
        forms.insert(d.form.tag());
        hands.insert((d.form.tag(), d.terminal_on_right));
        pages.insert(label);
        cases.insert((d.form.tag(), d.cabinet));
    }
    assert_eq!(forms.len(), 4);
    assert_eq!(hands.len(), 8);
    assert_eq!(cases.len(), 12);
    assert_eq!(pages.len(), 5);
}

#[test]
fn untagged_checkout_replays_the_original_station() {
    let mut o = audit::fixture(ObjectKind::CheckoutCounter, 19);
    for variant in 0..3 {
        o.variant = variant;
        assert!(checkout_station(&o).is_none());
        let Installation::Checkout { .. } = parameters(&o) else {
            panic!("untagged station changed")
        };
        let a = build_object(&o);
        assert!(a.parts.contains_key(&(Surface::Ink, CONTROLS.to_string())));
        assert!(a.parts.keys().all(|(_, label)| !label.contains("finish98")));
    }
}
