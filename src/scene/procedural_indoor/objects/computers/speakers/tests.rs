use super::*;
use crate::scene::procedural_indoor::objects::audit;

#[test]
fn openings_fit_the_free_rails_and_cover_each_deck_panel_exactly_once() {
    let mut layouts = [0; 3];
    let mut finishes = [[0; 2]; 2];
    for seed in 0..640 {
        let mut o = audit::fixture(ObjectKind::Laptop, seed);
        o.finish_seed = Some(seed);
        let mut r = stream(seed, 19);
        o.size = Vec3::new(
            r.random_range(0.25..0.44),
            r.random_range(0.205..0.34),
            r.random_range(0.225..0.42),
        );
        let p = parameters(&o);
        assert!(p.speaker_bank(&o, INTRODUCED - 1).is_none());
        let bank = p.speaker_bank(&o, INTRODUCED);
        assert_eq!(bank.is_some(), p.speaker_grilles, "seed {seed}");
        let Some(bank) = bank else { continue };
        let d = Deck::fit(&o, &p);
        let skin = DeckSkin::fit(&o, &p, d.depth);
        assert_eq!(
            bank.second.is_some(),
            bank.layout == GrilleLayout::KeyboardFlanks
        );
        if let Some(other) = bank.second {
            assert_eq!(bank.first.size_xz_m, other.size_xz_m);
            assert_eq!(
                bank.first.deck_center_xz_m,
                other.deck_center_xz_m * Vec2::new(-1., 1.)
            );
        }
        layouts[bank.layout as usize] += 1;
        finishes[usize::from(bank.facing == Surface::Metal)][usize::from(bank.light_finish)] += 1;
        let mut inserts = Assembly::default();
        let y = p.chassis_m * 0.5;
        bank.insert_faces(&mut inserts, &Transform::IDENTITY, y);
        let g = &inserts.parts[&(bank.facing, bank.label().into())];
        assert!(g.positions.iter().all(|p| p[1] == y - bank.recess_m));
        for (idx, opening) in bank.openings().enumerate() {
            let center = opening.deck_center_xz_m;
            let half = opening.size_xz_m * 0.5;
            if bank.layout == GrilleLayout::KeyboardFlanks {
                assert!(center.x.abs() - half.x > o.size.x * p.keyboard_fraction * 0.5 + 0.001);
            } else {
                assert!(center.y - half.y + d.depth * 0.5 >= 0.0075);
                assert!(center.y + half.y + d.depth * 0.5 < d.depth * 0.065 - 0.001);
            }
            let mut metal = Geometry::default();
            let q = skin.panel(opening.panel);
            opening.cut(&mut metal, q, &bank, &Transform::IDENTITY, y);
            let projected = |geom: &Geometry, tri: &[u32]| {
                let v = [0, 1, 2].map(|i| Vec3::from_array(geom.positions[tri[i] as usize]));
                let cross = (v[1] - v[0]).cross(v[2] - v[0]);
                assert!(
                    cross.y > 0.,
                    "overlapping or folded deck at seed {seed}, {:?}, {v:?}",
                    bank.layout
                );
                assert!(cross.dot(Vec3::from_array(geom.normals[tri[0] as usize])) > 0.);
                cross.y
            };
            let area2 = metal
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| projected(&metal, t))
                .sum::<f32>()
                + g.indices[idx * 36..(idx + 1) * 36]
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| projected(g, t))
                    .sum::<f32>();
            let panel_area2 =
                (q[1] - q[0]).perp_dot(q[2] - q[0]) + (q[2] - q[0]).perp_dot(q[3] - q[0]);
            assert!(
                (area2 - panel_area2).abs() < panel_area2 * 0.00002,
                "{seed}"
            );
        }
    }
    assert!(layouts.iter().all(|&n| n > 40), "{layouts:?}");
    assert!(finishes.iter().flatten().all(|&n| n > 6), "{finishes:?}");
}

#[test]
fn changes_only_the_exposed_deck_and_preserves_the_other_device_parts() {
    for seed in 0..28 {
        let o = audit::fixture(ObjectKind::Laptop, seed);
        let mut old = Assembly::default();
        let mut new = Assembly::default();
        laptop(&mut old, &o, INTRODUCED - 1);
        laptop(&mut new, &o, INTRODUCED);
        let p = parameters(&o);
        let bank = p.speaker_bank(&o, INTRODUCED);
        let case = Housing::for_object(&o).case_label(&o);
        for (key, before) in &old.parts {
            if bank.is_some()
                && (key.1 == case || (key.0 == Surface::Ink && key.1 == o.kind.class_name()))
            {
                continue;
            }
            let after = &new.parts[key];
            assert_eq!(before.positions, after.positions);
            assert_eq!(before.normals, after.normals);
            assert_eq!(before.uvs, after.uvs);
            assert_eq!(before.indices, after.indices);
        }
        let old_count = audit::check(&o, &old).unwrap().triangles;
        let new_count = audit::check(&o, &new).unwrap().triangles;
        if let Some(bank) = bank {
            assert_eq!(
                new_count + 36 * 12,
                old_count + 58 * bank.openings().count()
            );
            assert!(new.parts.contains_key(&(bank.facing, bank.label().into())));
        } else {
            assert_eq!(new_count, old_count);
        }
    }
}
