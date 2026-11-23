use super::*;
use crate::scene::procedural_indoor::{
    materials::{finish_slot, registered_finish},
    objects::{audit, build_object},
};

#[test]
fn keyboards_have_complete_disjoint_typing_blocks_and_fitted_secondary_banks() {
    for columns in [15, 18, 19, 22] {
        for function in [false, true] {
            for stagger in [-0.09, 0., 0.09] {
                let mut keys = vec![];
                layout::visit(columns, function, stagger, |k| keys.push(k));
                for letter in b'A'..=b'Z' {
                    assert_eq!(keys.iter().filter(|k| k.legend == letter).count(), 1);
                }
                for (i, k) in keys.iter().enumerate() {
                    let edge = k.at + k.size;
                    if columns == 15 && k.legend != b' ' {
                        assert!(if k.right { k.at.x >= 6. } else { edge.x <= 7.4 });
                    }
                    assert!(k.at.cmpge(Vec2::ZERO).all());
                    assert!(edge
                        .cmple(Vec2::new(units(columns), layout::rows(function)) + 1e-5)
                        .all());
                    for other in &keys[..i] {
                        let overlap = edge.min(other.at + other.size) - k.at.max(other.at);
                        assert!(
                            overlap.min_element() < 0.00001,
                            "overlapping keys {} and {}",
                            k.legend,
                            other.legend
                        );
                    }
                }
                for (a, b) in [(b'1', b'Q'), (b'Q', b'A'), (b'A', b'Z')] {
                    let x = |letter| keys.iter().find(|k| k.legend == letter).unwrap().at.x;
                    assert!((0.1..0.65).contains(&(x(b) - x(a))));
                }
                let key = |code| keys.iter().find(|k| k.legend == code).unwrap();
                assert_eq!(key(b' ').size.x, 6.25);
                assert_eq!(
                    keys.iter()
                        .filter(|k| k.legend == keycaps::legend::NUM)
                        .count(),
                    usize::from(matches!(columns, 19 | 22))
                );
                assert_eq!(
                    keys.iter()
                        .filter(|k| k.legend == keycaps::legend::F1)
                        .count(),
                    usize::from(function)
                );
                assert!(key(keycaps::legend::UP).at.y < key(keycaps::legend::DOWN).at.y);
            }
        }
    }
}

// World-space test against the actual deck triangles, including the splayed halves.
fn deck_at(deck: &Geometry, p: Vec3) -> bool {
    deck.indices.as_chunks::<3>().0.iter().any(|t| {
        let [a, b, c] = t.map(|i| Vec3::from_array(deck.positions[i as usize]));
        let plane = (b - a).cross(c - a);
        if plane.y <= 0.00000001 {
            return false;
        }
        let uv = p.xz() - a.xz();
        let ab = b.xz() - a.xz();
        let ac = c.xz() - a.xz();
        let det = ab.perp_dot(ac);
        let u = uv.perp_dot(ac) / det;
        let v = ab.perp_dot(uv) / det;
        u >= -1e-5
            && v >= -1e-5
            && u + v <= 1.00001
            && ((a.y + u * (b.y - a.y) + v * (c.y - a.y)) - p.y - 0.0003).abs() < 0.00002
    })
}

#[test]
fn installed_caps_keep_their_print_and_are_supported_over_the_sampled_size_range() {
    let mut banks = std::collections::BTreeSet::new();
    let mut split = 0;
    let mut stocks = std::collections::BTreeSet::new();
    for kind in [ObjectKind::Laptop, ObjectKind::Keyboard] {
        for seed in 0..48 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed.rotate_left(19));
            o.size = if kind == ObjectKind::Keyboard {
                Vec3::new(
                    [0.28, 0.34, 0.46][seed as usize % 3],
                    if seed % 2 == 0 { 0.014 } else { 0.033 },
                    if seed % 2 == 0 { 0.105 } else { 0.185 },
                )
            } else {
                Vec3::new(
                    [0.27, 0.34, 0.40][seed as usize % 3],
                    0.27,
                    if seed % 2 == 0 { 0.26 } else { 0.39 },
                )
            };
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let label = keycaps::label(&o);
            let slot = finish_slot(Surface::Plastic, label).unwrap();
            assert!(registered_finish(Surface::Plastic, slot));
            stocks.insert(slot);
            let cap = &a.parts[&(Surface::Plastic, label.to_owned())];
            if kind == ObjectKind::Keyboard {
                let p = desktop::parameters(&o);
                banks.insert(p.keyboard_columns);
                split += usize::from(p.keyboard_split);
            }
            let stock = super::super::super::materials::input_housing::Housing::for_object(&o);
            let deck = &a.parts[&(stock.surface, stock.case_label(&o))];
            let mut alphas = std::collections::BTreeSet::new();
            for (verts, uvs) in cap
                .positions
                .as_chunks::<33>()
                .0
                .iter()
                .zip(cap.uvs.as_chunks::<33>().0)
            {
                let code =
                    |uv: [f32; 2]| (uv[0] * 16.).floor() as u8 + 16 * (uv[1] * 16.).floor() as u8;
                let glyph = code(uvs[24]);
                assert!(uvs[16..25].iter().all(|&v| code(v) == glyph));
                assert!(uvs[..16].iter().chain(&uvs[25..]).all(|&v| code(v) == b' '));
                if glyph.is_ascii_uppercase() {
                    alphas.insert(glyph);
                }
                for &corner in &verts[25..] {
                    let foot = Vec3::from_array(corner);
                    assert!(
                        deck_at(deck, foot),
                        "{kind:?}/{seed}: unsupported {glyph} at {foot:?}"
                    );
                }
                assert!(verts[24][1] < verts[16..24].iter().map(|v| v[1]).sum::<f32>() / 8.);
            }
            assert_eq!(alphas.len(), 26);
            assert_eq!(cap.positions.len() % 33, 0);
        }
    }
    assert_eq!(banks, std::collections::BTreeSet::from([15, 18, 19, 22]));
    assert!(split > 0 && split < 48 && stocks.len() == 2);
}
