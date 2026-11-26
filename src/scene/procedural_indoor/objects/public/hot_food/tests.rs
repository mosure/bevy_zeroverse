use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

/// Vertical ray on the actual top-facing triangles of a material batch.
fn top_at(g: &Geometry, point: Vec2) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let (e, f, v) = ((b - a).xz(), (c - a).xz(), point - a.xz());
            let det = e.perp_dot(f);
            if det >= -1e-12 {
                return None;
            }
            let (u, w) = (v.perp_dot(f) / det, e.perp_dot(v) / det);
            (u >= -1e-5 && w >= -1e-5 && u + w <= 1.00001)
                .then_some(a.y + u * (b.y - a.y) + w * (c.y - a.y))
        })
        .max_by(f32::total_cmp)
}

#[test]
fn hot_food_well_has_metric_openings_recessed_pans_and_aligned_contents() {
    let mut dishes = BTreeSet::new();
    let mut formats = BTreeSet::new();
    let mut shells = BTreeSet::new();
    for seed in 500..596 {
        let o = audit::fixture(ObjectKind::HotFoodWell, seed);
        let well = HotFoodWell::for_object(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(
            check.triangles < 14000
                && check.max_envelope_overrun_m < 0.0002
                && check.min_y_m.abs() < 0.0001
        );
        let steel = &mesh.parts[&(Surface::Metal, mesh::STEEL.into())];
        let guarded = mesh
            .parts
            .keys()
            .any(|(s, _)| *s == Surface::ContainerGlass);
        assert_eq!(guarded, well.guard.is_some());
        shells.insert((guarded, well.wells));
        let area: f32 = well.pans.iter().map(|p| p.size.element_product()).sum();
        assert!((area - GN.element_product() * well.wells as f32).abs() < 1e-6);
        for pan in &well.pans {
            let probe = pan.center + pan.size * Vec2::new(0.071, 0.113);
            let y = top_at(steel, probe).unwrap();
            let bottom = pan.floor(well.deck_height);
            assert!(bottom > 0.019);
            if pan.contents == Contents::Covered {
                assert!(y > well.deck_height && y < well.deck_height + 0.017);
            } else {
                assert!((y - bottom).abs() < 0.00005);
            }
            if pan.contents.soup().is_some() {
                let liquid = &mesh.parts[&(Surface::Ceramic, broth::LABEL.into())];
                let fill = top_at(liquid, probe).unwrap();
                assert!(
                    (fill - bottom - (well.deck_height + mesh::LIP - bottom) * pan.fill_fraction)
                        .abs()
                        < 0.00005
                );
            } else if pan.contents == Contents::PotatoGratin {
                let g = &mesh.parts[&(
                    Surface::Terracotta,
                    crate::scene::procedural_indoor::materials::roast::Stock::Gratin
                        .label()
                        .into(),
                )];
                let ranges: Vec<_> = g
                    .positions
                    .iter()
                    .map(|p| Vec3::from_array(*p))
                    .filter(|v| (v.xz() - pan.center).abs().cmplt(pan.size * 0.5).all())
                    .collect();
                let peak = ranges.iter().map(|p| p.y).fold(bottom, f32::max);
                assert!(peak > bottom + 0.012 && peak < well.deck_height);
                assert!(ranges.iter().any(|p| (p.y - bottom).abs() < 0.00001));
                assert!(
                    top_at(g, probe).is_some_and(|y| y > bottom + 0.002 && y < well.deck_height)
                );
            } else if matches!(
                pan.contents,
                Contents::RoastPotatoes | Contents::GlazedCarrots
            ) {
                let food = mesh
                    .parts
                    .iter()
                    .filter(|((_, l), _)| part_label(l) == "food")
                    .flat_map(|(_, g)| &g.positions)
                    .map(|p| Vec3::from_array(*p))
                    .filter(|v| (v.xz() - pan.center).abs().cmplt(pan.size * 0.5).all())
                    .collect::<Vec<_>>();
                assert!(food.len() >= 20);
                assert!(food
                    .iter()
                    .all(|p| (p.xz() - pan.center).abs().cmplt(pan.bottom()).all()
                        && p.y >= bottom - 0.00001
                        && p.y < well.deck_height));
                assert!(food.iter().any(|p| (p.y - bottom).abs() < 0.00001));
            }
            formats.insert((GN.element_product() / pan.size.element_product()).round() as u8);
            dishes.insert(pan.contents as u8);
        }
        let replay: IndoorObject =
            serde_json::from_slice(&serde_json::to_vec(&o).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&well).unwrap(),
            serde_json::to_value(HotFoodWell::for_object(&replay)).unwrap()
        );
    }
    assert_eq!(dishes.len(), 8);
    assert_eq!(formats, BTreeSet::from([1, 2, 3, 4]));
    assert_eq!(shells.len(), 4);
}
