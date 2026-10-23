use super::*;

#[test]
fn zero_detail_removes_relief_in_authored_programs_and_installation_seams() {
    for floor in 0..3 {
        for seed in [0, 202] {
            let recipes = sample_with_floor(seed, floor).unwrap();
            for mut r in recipes {
                r.detail_scale = Some(0.);
                r.relief_m = 0.;
                r.contrast = 0.;
                for slot in [0, 1, 2] {
                    let r = r.variant(slot);
                    for y in 0..8 {
                        for x in 0..8 {
                            let t = r.texel(x as f32 / 8., y as f32 / 8., floor);
                            assert_eq!(t.height, 0., "{:?} {seed} {floor} {slot}", r.surface);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn interior_palettes_replay_with_bounded_correlated_pigments() {
    let mut dark = 0;
    let mut neutral = 0;
    let mut hues = [false; 12];
    for seed in 0..512 {
        let p = PaletteProgram::sample(seed);
        assert_eq!(p, PaletteProgram::sample(seed));
        dark += usize::from(p.wall_lightness < 0.5);
        neutral += usize::from(p.pigment_strength < 0.07);
        hues[(p.hue / 30.) as usize] = true;
        assert!((16.0..48.0).contains(&p.wood_hue));
        assert!((0.72..0.96).contains(&p.ceiling_lightness));
        for recipe in sample(seed) {
            recipe.validate().unwrap();
        }
        for (h, s, l) in [
            (p.hue, p.pigment_strength, p.wall_lightness),
            (p.hue, p.pigment_strength * 0.16, p.ceiling_lightness),
            (p.wood_hue, p.wood_saturation, p.wood_lightness),
        ] {
            assert!(PaletteProgram::rgb(h, s, l)
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
    }
    assert!(hues.into_iter().all(|v| v));
    assert!((20..65).contains(&dark));
    assert!((220..310).contains(&neutral));
}

#[test]
fn finish_programs_replay_are_periodic_and_validate_their_physical_scales() {
    for seed in 0..256 {
        let rs = sample(seed);
        for surface in [
            Surface::Paint,
            Surface::Accent,
            Surface::Ceiling,
            Surface::Concrete,
            Surface::Ceramic,
        ] {
            let r = &rs[surface as usize];
            r.validate().unwrap();
            assert_eq!(r, &sample(seed)[surface as usize]);
            for p in [0., 0.19, 0.73, 1.] {
                for (a, b) in [
                    (r.texel(0., p, 0), r.texel(1., p, 0)),
                    (r.texel(p, 0., 0), r.texel(p, 1., 0)),
                ] {
                    assert!(
                        a.color
                            .iter()
                            .zip(b.color)
                            .all(|(a, b)| (a - b).abs() < 0.0001),
                        "{surface:?}"
                    );
                    assert!(
                        (a.height - b.height).abs() < 1e-7
                            && (a.roughness - b.roughness).abs() < 0.0001
                    );
                }
            }
        }
        let concrete = rs[Surface::Concrete as usize].mineral.as_ref().unwrap();
        assert_eq!(concrete.marble_mix, 0., "concrete is not a marble slab");
        assert!(concrete.casting.is_some());
        let ceramic = &rs[Surface::Ceramic as usize];
        assert_eq!(
            ceramic.layers.as_ref().unwrap().quarter_turn % 2,
            0,
            "wheel marks must follow circumference"
        );
        assert!(ceramic.coating.as_ref().unwrap().glaze.is_some());
        assert_ne!(ceramic.variant(0).coating, ceramic.variant(1).coating);
        assert_ne!(ceramic.variant(1).coating, ceramic.variant(2).coating);
        assert_eq!(ceramic.variant(0), ceramic.variant(3));
    }
    let mut r = sample(7).remove(Surface::Ceramic as usize);
    r.coating
        .as_mut()
        .unwrap()
        .glaze
        .as_mut()
        .unwrap()
        .body_grain_m = 0.;
    assert!(r.validate().is_err());
    let mut r = sample(7).remove(Surface::Paint as usize);
    r.coating
        .as_mut()
        .unwrap()
        .application
        .as_mut()
        .unwrap()
        .roller_stretch = f32::NAN;
    assert!(r.validate().is_err());
    let mut r = sample(7).remove(Surface::Concrete as usize);
    r.mineral
        .as_mut()
        .unwrap()
        .casting
        .as_mut()
        .unwrap()
        .bughole_depth_m = 0.1;
    assert!(r.validate().is_err());
}

#[test]
fn ceramic_pigment_is_applied_once_and_old_recorded_recipes_remain_readable() {
    let mut r = sample(81).remove(Surface::Ceramic as usize);
    let maps = r.maps(0);
    r.color = [0.08, 0.23, 0.60];
    assert_eq!(r.texture_base_color(0), r.color);
    for (a, b) in maps.iter().zip(r.maps(0)) {
        assert_eq!(
            a.data, b.data,
            "ceramic pigment was baked into a relative map"
        );
    }
    for surface in [Surface::Ceramic, Surface::Paint, Surface::Concrete] {
        let r = sample(81).remove(surface as usize);
        let mut json = serde_json::to_value(r).unwrap();
        if let Some(c) = json
            .get_mut("coating")
            .and_then(serde_json::Value::as_object_mut)
        {
            c.remove("glaze");
            c.remove("application");
            c.remove("wall");
        }
        if let Some(m) = json
            .get_mut("mineral")
            .and_then(serde_json::Value::as_object_mut)
        {
            m.remove("casting");
        }
        let old: MaterialRecipe = serde_json::from_value(json).unwrap();
        old.validate().unwrap();
        assert!(old.absolute_color(0));
        assert!(old.texel(0.12, 0.71, 0).height.is_finite());
    }
}

#[test]
fn mineral_coating_timber_and_foliage_programs_are_bounded_and_seeded() {
    for seed in 0..1024 {
        let recipes = sample(seed);
        assert_eq!(recipes, sample(seed));
        assert_eq!(
            recipes[Surface::Wood as usize].wood,
            recipes[Surface::WoodEdge as usize].wood
        );
        for r in &recipes {
            if let Some(m) = &r.mineral {
                m.validate().unwrap();
            }
            if let Some(c) = &r.coating {
                c.validate().unwrap();
            }
            if let Some(w) = &r.wood {
                w.validate().unwrap();
            }
            if let Some(l) = &r.leaf {
                l.validate().unwrap();
            }
            for (u, v) in [(0.07, 0.13), (0.49, 0.31), (0.53, 0.87)] {
                for floor in 0..3 {
                    let t = r.texel(u, v, floor);
                    assert!(
                        t.color
                            .iter()
                            .all(|c| c.is_finite() && (0.0..=1.).contains(c)),
                        "{:?}",
                        r.surface
                    );
                    assert!(t.height.is_finite() && t.height.abs() < 0.003);
                    assert!((0.045..=1.).contains(&t.roughness));
                    assert!((0.8..=1.).contains(&t.occlusion));
                }
            }
        }
        for slot in 0..12 {
            assert_eq!(
                recipes[Surface::Wood as usize].variant(slot).wood,
                recipes[Surface::WoodEdge as usize].variant(slot).wood
            );
        }
    }
}

#[test]
fn leaf_atlas_keeps_midrib_on_the_geometric_fold_and_preserves_pale_pigment() {
    let mut r = sample(31).remove(Surface::LeafVariegated as usize);
    let peak: f32 = (1..32)
        .map(|i| r.texel(0.5, i as f32 / 32., 0).height)
        .sum();
    let blade: f32 = (1..32)
        .map(|i| r.texel(0.25, i as f32 / 32., 0).height)
        .sum();
    assert!(peak > blade * 1.5);
    let before = r.texel(0.5, 0.31, 0);
    r.phase = [0.17, 0.81];
    r.layers.as_mut().unwrap().quarter_turn = 3;
    let after = r.texel(0.5, 0.31, 0);
    assert_eq!(before.color, after.color);
    assert_eq!(before.height, after.height);
    let l = r.leaf.as_mut().unwrap();
    l.variegation = 1.;
    l.pale_color = [0.85, 0.87, 0.68];
    let brightest = (0..32)
        .flat_map(|y| (0..32).map(move |x| (x, y)))
        .map(|(x, y)| r.texel(x as f32 / 32., y as f32 / 32., 0).color[0])
        .fold(0., f32::max);
    assert!(
        brightest > 0.75,
        "pale foliage was multiplied by a dark green base tint"
    );
}

#[test]
fn malformed_substrate_parameters_are_rejected() {
    let mut rs = sample(7);
    let m = rs[Surface::Concrete as usize].mineral.as_mut().unwrap();
    m.aggregate_cells[0] = 0;
    assert!(m.validate().is_err());
    let c = rs[Surface::Paint as usize].coating.as_mut().unwrap();
    c.gloss = f32::NAN;
    assert!(c.validate().is_err());
    let l = rs[Surface::Leaf as usize].leaf.as_mut().unwrap();
    l.vein_width = 0.;
    assert!(l.validate().is_err());
    let w = rs[Surface::Wood as usize].wood.as_mut().unwrap();
    w.stain_strength = 1.1;
    assert!(w.validate().is_err());
}

#[test]
fn weaving_and_leather_programs_replay_close_at_seams_and_span_parameters() {
    let mut topologies = std::collections::BTreeSet::new();
    for seed in 0..512 {
        let recipes = sample(seed);
        assert_eq!(recipes, sample(seed));
        for s in [Surface::Fabric, Surface::FabricAlt, Surface::Leather] {
            let r = &recipes[s as usize];
            if let Some(t) = &r.textile {
                t.validate().unwrap();
                topologies.insert((t.repeat, t.float_length, t.advance, t.bundle, t.herringbone));
                let spacing = r.period_m / t.yarns[1] as f32;
                // Fine woven yarn through coarse chenille, including the
                // continuously sampled physical feature-scale multiplier.
                assert!((0.0009..0.007).contains(&spacing));
            }
            if let Some(l) = &r.leather {
                l.validate().unwrap();
            }
            for t in [0.0, 0.123, 0.371, 0.825, 1.0] {
                for (a, b) in [
                    (r.texel(0., t, 0), r.texel(1., t, 0)),
                    (r.texel(t, 0., 0), r.texel(t, 1., 0)),
                ] {
                    assert!(
                        a.color
                            .iter()
                            .zip(b.color)
                            .all(|(x, y)| (x - y).abs() < 0.0001),
                        "seed {seed}, {s:?}, seam at {t}"
                    );
                    assert!(
                        (a.height - b.height).abs() < 1e-7
                            && (a.roughness - b.roughness).abs() < 0.0001
                    );
                    assert!(a.height.is_finite() && a.height.abs() < 0.001);
                    // Loop-knit valleys have the existing 0.89 AO floor;
                    // woven cloth/leather retain their stricter bound.
                    let min_ao = if r.textile.as_ref().is_some_and(|t| t.knit) {
                        0.89
                    } else {
                        0.9
                    };
                    assert!(
                        (0.18..=1.).contains(&a.roughness) && (min_ao..=1.).contains(&a.occlusion)
                    );
                }
            }
            assert_eq!(r.variant(0), r.variant(3));
            assert_ne!(r.variant(0), r.variant(1));
            assert_ne!(r.variant(1), r.variant(2));
        }
    }
    assert!(
        topologies.len() > 150,
        "weaving topology collapsed: {}",
        topologies.len()
    );
}

#[test]
fn malformed_weaving_programs_are_rejected_without_overflow() {
    let mut t = super::super::textile::TextileRecipe::sample(8);
    t.repeat = u32::MAX;
    assert!(t.validate().is_err());
    t.repeat = 0;
    assert!(t.validate().is_err());
}

#[test]
fn upholstery_variants_change_structure_not_only_palette() {
    let recipes = sample(81);
    for s in [Surface::Fabric, Surface::FabricAlt, Surface::Leather] {
        let a = recipes[s as usize].maps(0);
        let b = recipes[s as usize].variant(1).maps(0);
        assert_ne!(a[1].data, b[1].data, "{s:?} shares all relief");
        assert_ne!(a[2].data, b[2].data, "{s:?} shares all scattering");
        assert!(a[2]
            .data
            .as_ref()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[2] == 0));
        let pixels = &a[0].data.as_ref().unwrap()[..256 * 256 * 4];
        let codes: std::collections::BTreeSet<_> =
            pixels.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert!(codes.len() > 15, "{s:?} has flat albedo");
    }
}

#[test]
fn timber_cut_spans_growth_rings_and_resolves_millimetre_pores() {
    for seed in 0..128 {
        for r in sample(seed)
            .into_iter()
            .filter(|r| matches!(r.surface, Surface::Wood | Surface::WoodEdge))
        {
            let period = r.period_uv();
            assert_eq!(period.y, r.period_m);
            assert!((0.002..0.009).contains(&(period.x / 109.)));
            assert!(period.x <= period.y);
        }
    }
}

#[test]
fn recorded_floor_recipes_have_physical_scale_and_finish() {
    for seed in 0..128 {
        for style in 0..3 {
            let mut recipes = sample(seed);
            let original = recipes.clone();
            floor_finish(&mut recipes, style);
            for (r, old) in recipes.iter().zip(&original) {
                if r.surface != Surface::Floor {
                    assert_eq!(r, old, "floor specialization modified another substrate");
                }
            }
            let floor = &recipes[Surface::Floor as usize];
            assert!(floor.relief_m > 0. && floor.relief_m < 0.001);
            if style == 1 {
                assert!((2.8..=4.2).contains(&floor.period_m));
                assert!((0.85..=0.98).contains(&floor.roughness));
            } else if floor.flooring.as_ref().unwrap().layout
                == super::super::flooring::FloorLayout::Seamless
            {
                assert!((0.80..1.70).contains(&floor.period_m));
                assert!((0.12..=0.95).contains(&floor.roughness));
            } else {
                assert_eq!(
                    floor.period_m,
                    if style == 0 {
                        original[Surface::Floor as usize].period_m.max(2.8)
                    } else {
                        original[Surface::Floor as usize].period_m
                    }
                );
                if style == 0 {
                    assert!((0.20..=0.85).contains(&floor.roughness));
                } else {
                    assert!((0.12..=0.95).contains(&floor.roughness));
                }
            }
        }
    }
}
