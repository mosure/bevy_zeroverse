use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, validation::validate_layout};

#[test]
fn installation_sections_and_lights_are_continuous_and_bounded() {
    let mut counts = std::collections::BTreeSet::new();
    let mut sash = Vec::new();
    let mut sills = Vec::new();
    for seed in 0..1024 {
        let size = Vec3::new(8., 3.5, 10.);
        let p = ExteriorProgram::sample(seed, size, 0.45);
        p.validate(size)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        for f in p.facades {
            let j = f.joinery.as_ref().unwrap();
            sash.push(j.sash_width_m);
            sills.push(j.sill_thickness_m);
            for o in &f.openings {
                counts.insert(o.rows().len());
                let panes = f.pane_rectangles(o);
                for (i, &(a, b)) in panes.iter().enumerate() {
                    assert!((b - a).min_element() > 0.04);
                    assert!(a.cmpge(o.min).all() && b.cmple(o.max).all());
                    assert!(panes[i + 1..]
                        .iter()
                        .all(|(c, d)| !a.cmplt(*d).all() || !b.cmpgt(*c).all()));
                }
            }
        }
    }
    sash.sort_by(f32::total_cmp);
    sills.sort_by(f32::total_cmp);
    assert_eq!(counts, [0, 1, 2, 3].into_iter().collect());
    for samples in [sash, sills] {
        assert!(samples[samples.len() * 19 / 20] / samples[samples.len() / 20] > 2.7);
    }
}

#[test]
fn invalid_joinery_and_duplicate_transoms_are_rejected() {
    let size = Vec3::new(8., 3., 10.);
    let mut p = ExteriorProgram::sample(27, size, 0.3);
    let original = p.clone();
    p.facades[0].joinery.as_mut().unwrap().sash_depth_m = f32::NAN;
    assert!(p.validate(size).is_err());
    let mut p = original;
    p.facades[0].openings[0].transom = 0.5;
    p.facades[0].openings[0].transoms = vec![0.5];
    assert!(p.validate(size).is_err());
}

#[test]
fn missing_optional_joinery_retains_legacy_serialization() {
    let p = ExteriorProgram::sample(38, Vec3::new(8., 3., 10.), 0.4);
    let mut value = serde_json::to_value(p).unwrap();
    for f in value["facades"].as_array_mut().unwrap() {
        f.as_object_mut().unwrap().remove("joinery");
        for o in f["openings"].as_array_mut().unwrap() {
            o.as_object_mut().unwrap().remove("transoms");
        }
    }
    let old: ExteriorProgram = serde_json::from_value(value).unwrap();
    for f in old.facades {
        assert!(f.joinery.is_none());
        let mut a = Assembly::default();
        glazing_at(&mut a, Transform::IDENTITY, &f);
        assert!(a.parts.keys().any(|(s, _)| *s == Surface::Glass));
    }
}

#[test]
fn opened_sash_moves_all_panes_without_a_hidden_closed_surface() {
    use crate::scene::procedural_indoor::materials::glazing::GlazingProgram;
    let mut recipe = GlazingProgram::sample(3, 0, true);
    recipe.opening = 0.;
    recipe.treatment = 6;
    let mut facade = Facade {
        joinery: None,
        side: FacadeSide::Rear,
        openings: vec![WindowOpening {
            glazing: Some(recipe),
            min: Vec2::new(-1., 0.4),
            max: Vec2::new(1., 2.4),
            columns: 1,
            transoms: Vec::new(),
            transom: 0.,
            operable: true,
        }],
        frame: Surface::Metal,
        frame_width: 0.05,
        frame_depth: 0.08,
        recess: 0.06,
        sill_projection: 0.1,
        treatment: None,
        shade: Shade::None,
        shade_coverage: 0.,
        shade_tilt: 0.,
    };
    let mut closed = Assembly::default();
    geometry::glazing_at(&mut closed, Transform::IDENTITY, &facade);
    facade.openings[0].glazing.as_mut().unwrap().opening = 1.;
    let mut opened = Assembly::default();
    geometry::glazing_at(&mut opened, Transform::IDENTITY, &facade);
    for ((surface, label), g) in &opened.parts {
        assert!(g.positions.iter().all(|p| Vec3::from(*p).is_finite()));
        if *surface == Surface::Glass || label == "window#frit" {
            let old = &closed.parts[&(*surface, label.clone())];
            assert_eq!(old.positions.len(), g.positions.len());
            assert!(g
                .positions
                .iter()
                .zip(&old.positions)
                .any(|(a, b)| Vec3::from(*a).distance(Vec3::from(*b)) > 0.05));
            // Pane fronts are coherently tilted; no unchanged front plane is
            // present behind the articulated assembly.
            assert!(g
                .normals
                .iter()
                .filter(|n| n[2].abs() > 0.9)
                .all(|n| n[1].abs() > 0.03));
        }
    }
    assert!(opened
        .parts
        .contains_key(&(Surface::Paint, "window#frit".into())));
    assert!(opened
        .parts
        .contains_key(&(Surface::Rubber, "window".into())));
    // Shades prevent the articulated unit from intersecting their occupied sweep.
    facade.shade = Shade::Roller;
    facade.shade_coverage = 0.6;
    let mut shaded = Assembly::default();
    geometry::glazing_at(&mut shaded, Transform::IDENTITY, &facade);
    for (key, g) in closed
        .parts
        .iter()
        .filter(|((s, _), _)| *s == Surface::Glass)
    {
        assert_eq!(g.positions, shaded.parts[key].positions);
    }
}

#[test]
fn continuous_facades_cover_all_exposures_and_full_height_openings() {
    let mut combinations = std::collections::BTreeSet::new();
    let mut full_height_rooms = 0;
    let mut widths = (f32::MAX, 0_f32);
    let mut sills = (f32::MAX, 0_f32);
    for seed in 0..512 {
        let size = crate::scene::procedural_indoor::domain::room_size(seed);
        let p = ExteriorProgram::sample(seed, size, 0.55);
        p.validate(size)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_eq!(p, ExteriorProgram::sample(seed, size, 0.55));
        assert_eq!(
            p,
            serde_json::from_str::<ExteriorProgram>(&serde_json::to_string(&p).unwrap()).unwrap()
        );
        combinations.insert(p.facades.iter().map(|f| 1 << f.side as u8).sum::<u8>());
        let mut full = false;
        for o in p.facades.iter().flat_map(|f| &f.openings) {
            full |= o.full_height(size.y);
            widths = (
                widths.0.min(o.max.x - o.min.x),
                widths.1.max(o.max.x - o.min.x),
            );
            sills = (sills.0.min(o.min.y), sills.1.max(o.min.y));
        }
        full_height_rooms += usize::from(full);
    }
    assert_eq!(
        combinations.len(),
        7,
        "single, adjacent/opposite double and triple exposure"
    );
    assert!(
        (100..400).contains(&full_height_rooms),
        "full-height rooms={full_height_rooms}"
    );
    assert!(widths.0 < 0.9 && widths.1 > 4.5, "widths={widths:?}");
    assert!(sills.0 < 0.025 && sills.1 > 2.0, "sills={sills:?}");
}

#[test]
fn wall_complement_preserves_area_without_filling_apertures() {
    for seed in 0..128 {
        let size = crate::scene::procedural_indoor::domain::room_size(seed);
        let p = ExteriorProgram::sample(seed, size, 0.55);
        for f in &p.facades {
            let min = Vec2::new(-f.side.span(size) * 0.5, 0.);
            let max = Vec2::new(-min.x, size.y);
            let solid = solid_rectangles(min, max, &f.openings);
            let area: f32 = solid.iter().map(|(a, b)| (*b - *a).element_product()).sum();
            let glass: f32 = f.openings.iter().map(WindowOpening::area).sum();
            assert!((area + glass - (max - min).element_product()).abs() < 0.0002);
            for (a, b) in solid {
                assert!((b - a).min_element() > 0.);
                for o in &f.openings {
                    let overlap = (b.min(o.max) - a.max(o.min))
                        .max(Vec2::ZERO)
                        .element_product();
                    assert!(overlap < 1e-6);
                }
            }
        }
    }
}

fn intersects(origin: Vec3, dir: Vec3, p: [Vec3; 3], distance: f32) -> bool {
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-7 {
        return false;
    }
    let s = origin - p[0];
    let u = s.dot(h) / det;
    let q = s.cross(e1);
    let v = dir.dot(q) / det;
    let t = e2.dot(q) / det;
    (0.0..=1.0).contains(&u) && v >= 0. && u + v <= 1. && (0.0..=distance).contains(&t)
}

#[test]
fn inset_glass_is_visible_through_actual_meshes_on_every_side() {
    for seed in 0..32 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.5, 0, 0.).unwrap();
        validate_layout(&scene).unwrap();
        let mut bare = scene.clone();
        for f in &mut bare.exterior.as_mut().unwrap().facades {
            f.shade = Shade::None;
            for opening in &mut f.openings {
                if let Some(program) = &mut opening.glazing {
                    program.treatment = 0;
                    program.opening = 0.;
                }
            }
        }
        let mut assembly = Assembly::default();
        shell(&mut assembly, &bare);
        super::super::details::feature_wall(&mut assembly, &bare);
        for f in &bare.exterior.as_ref().unwrap().facades {
            let tf = f.side.transform(bare.room_size);
            for o in &f.openings {
                let (lo, hi) = f.pane_rectangles(o)[0];
                let Vec2 { x, y } = (lo + hi) * 0.5;
                let origin = tf.transform_point(Vec3::new(x, y, 0.25));
                let dir = tf.rotation * -Vec3::Z;
                let mut glass = false;
                for ((surface, label), g) in &assembly.parts {
                    let hit = g.indices.as_chunks::<3>().0.iter().any(|t| {
                        intersects(
                            origin,
                            dir,
                            t.map(|i| Vec3::from_array(g.positions[i as usize])),
                            f.wall_thickness_m() + 0.26,
                        )
                    });
                    if *surface == Surface::Glass {
                        glass |= hit;
                    } else {
                        assert!(
                            !hit,
                            "seed {seed} {:?} covered by {label}/{surface:?} at {origin:?}",
                            f.side
                        );
                    }
                }
                assert!(glass, "missing glass seed {seed} {:?}", f.side);
            }
        }
    }
}

#[test]
fn malformed_openings_are_rejected_and_legacy_shell_still_builds() {
    let size = Vec3::new(8., 3., 10.);
    let original = ExteriorProgram::sample(5, size, 0.4);
    let mut p = original.clone();
    p.facades.push(p.facades[0].clone());
    assert!(p.validate(size).is_err());
    let mut p = original.clone();
    let f = &mut p.facades[0];
    f.openings.push(f.openings[0].clone());
    assert!(p.validate(size).is_err());
    let mut p = original;
    p.facades[0].openings[0].max.x = f32::NAN;
    assert!(p.validate(size).is_err());
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.5, 0, 0.).unwrap();
    scene.exterior = None;
    let mut assembly = Assembly::default();
    shell(&mut assembly, &scene);
    assert!(assembly
        .parts
        .keys()
        .any(|(s, l)| *s == Surface::Glass && l == "window"));
    let mut json = serde_json::to_value(&scene).unwrap();
    json.as_object_mut().unwrap().remove("exterior");
    assert!(serde_json::from_value::<IndoorManifest>(json)
        .unwrap()
        .exterior
        .is_none());
}
