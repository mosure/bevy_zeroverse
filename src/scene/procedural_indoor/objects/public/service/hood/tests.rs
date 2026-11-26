use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version};
use crate::scene::procedural_indoor::{
    appearance::AppearanceSettings,
    layout::{IndoorLayout, IndoorManifest},
    objects::kitchen,
    photometry,
};

#[test]
fn installed_work_light_projects_onto_its_own_cooking_surface() {
    for seed in [5203, 5204, 5211, 5217] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Kitchen, 0.65, 3, 0.25)
                .unwrap();
        let hood = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::ExtractionHood)
            .unwrap();
        let mount = scene
            .assembly_diagnostics
            .mounts
            .iter()
            .find(|m| m.object == hood.id)
            .unwrap();
        let host = &scene.objects[mount.host];
        let (target, _) = kitchen::appliances::cooking_surface(host, &kitchen::parameters(host));
        let deck = FilterDeck::installed(hood);
        let beam = deck.beam(&scene, hood);
        let t = (target.y - beam.position.y) / beam.direction.y;
        assert!(t > 0. && t < 2.5);
        assert!(target.distance(beam.position + t * beam.direction) < 1e-5);
        assert!(
            beam.inner > 0. && beam.inner < beam.outer && beam.outer < std::f32::consts::FRAC_PI_2
        );
        let mut archived = scene.clone();
        archived.generator_version = FITTED_INTAKE;
        assert_eq!(photometry::lamp(&archived, hood).lumens, 0.);
        assert!(!photometry::spot_fixtures(&archived).any(|(o, _)| o.id == hood.id));
        assert_eq!(
            photometry::spot_fixtures(&scene)
                .filter(|(o, _)| o.id == hood.id)
                .count(),
            1
        );
    }
}

#[test]
fn hood_product_and_switch_replay_independently_and_dimming_applies_once() {
    let scene =
        IndoorManifest::generate_with_humans(4, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
    let mut dimmed = scene.clone();
    dimmed
        .apply_appearance(AppearanceSettings {
            illumination_scale: 0.25,
            ..default()
        })
        .unwrap();
    let mut relit = scene.clone();
    relit
        .apply_appearance(AppearanceSettings {
            lighting_seed: Some(890),
            ..default()
        })
        .unwrap();
    let mut off = 0;
    let mut switched = 0;
    for seed in 0..256 {
        let hood = audit::fixture(ObjectKind::ExtractionHood, seed);
        let deck = FilterDeck::installed(&hood);
        let a = deck.power(&scene, &hood);
        let b = deck.power(&relit, &hood);
        assert_eq!(a, photometry::lamp(&scene, &hood));
        assert_eq!((a.kelvin, a.tint), (b.kelvin, b.tint));
        assert_eq!(deck.beam(&scene, &hood), deck.beam(&relit, &hood));
        assert_eq!(deck.power(&dimmed, &hood).lumens, a.lumens * 0.25);
        off += usize::from(a.lumens == 0.);
        switched += usize::from((a.lumens == 0.) != (b.lumens == 0.));
        assert!((2700.0..5300.).contains(&a.kelvin));
        assert!(a.lumens.is_finite() && a.lumens >= 0.);
    }
    assert!(
        (42..88).contains(&off) && switched > 45,
        "off {off}, switched {switched}"
    );
}

// One-sided mesh hits, independent of the fitted deck's collision or metadata.
fn first_face(a: &Assembly, origin: Vec3, direction: Vec3) -> Option<(Surface, &str, f32)> {
    let origin = origin.as_dvec3();
    let direction = direction.as_dvec3();
    let mut hit = None;
    let mut nearest = f64::INFINITY;
    for ((surface, label), g) in &a.parts {
        for t in g.indices.as_chunks::<3>().0 {
            let [p, q, r] = t.map(|i| Vec3::from_array(g.positions[i as usize]).as_dvec3());
            let e = q - p;
            let f = r - p;
            let h = direction.cross(f);
            let det = e.dot(h);
            if det <= 1e-10 {
                continue;
            }
            let v = origin - p;
            let u = v.dot(h) / det;
            let v2 = v.cross(e);
            let w = direction.dot(v2) / det;
            let d = f.dot(v2) / det;
            if u >= -1e-10 && w >= -1e-10 && u + w <= 1. + 1e-10 && d > 0. && d < nearest {
                nearest = d;
                hit = Some((*surface, label.as_str(), d as f32));
            }
        }
    }
    hit
}

#[test]
fn work_lights_are_exposed_and_filter_slots_have_real_depth() {
    let mut shapes = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::ExtractionHood, seed);
        // Include a compact cooktop next to the full catalogue envelopes.
        if seed % 3 == 0 {
            o.size = Vec3::new(0.42, 0.31, 0.34);
        }
        let p = hood_parameters(&o);
        let d = FilterDeck::fit(&o, &p);
        assert_eq!(d, FilterDeck::fit(&o, &p));
        let a = build_object_at_version(&o, None, FITTED_INTAKE);
        audit::check(&o, &a).unwrap();
        for i in 0..d.lamps {
            let c = d.lens_center(i);
            for eye in [
                Vec3::new(0., -0.3, 0.),
                Vec3::new(0.21, -0.30, 0.16),
                Vec3::new(-0.16, -0.30, -0.21),
            ] {
                let (surf, label, distance) = first_face(&a, c + eye, -eye.normalize()).unwrap();
                assert_eq!(
                    (surf, label),
                    (Surface::Light, "lamp"),
                    "seed {seed} lamp {i} from {eye:?}, lens at {}, hit at {distance}",
                    eye.length()
                );
                assert!((distance - eye.length()).abs() < 1e-4);
            }
        }

        let width = d.aperture_size.x / d.cassettes as f32;
        let center = d.aperture_center - Vec2::X * ((d.aperture_size.x - width) * 0.5);
        let inner = Vec2::new(width, d.aperture_size.y) - Vec2::splat(2. * d.rim_m);
        let (metal, gap, name) = match d.filter {
            FilterStyle::FoldedBaffle {
                across_x,
                folds,
                pitch_m,
                ..
            } => {
                let axis = if across_x { Vec2::X } else { Vec2::Y };
                let first = center - axis * ((folds as f32 - 1.) * 0.5 * pitch_m);
                (first, first + axis * pitch_m * 0.5, "baffle")
            }
            FilterStyle::PerimeterPanel { gap_m } => (
                center,
                center + Vec2::X * ((inner.x - gap_m) * 0.5),
                "panel",
            ),
        };
        let upward = |p: Vec2| {
            let (surface, label, distance) =
                first_face(&a, Vec3::new(p.x, -0.1, p.y), Vec3::Y).unwrap();
            assert_eq!(label.split('#').next(), Some("extraction_hood"));
            assert_ne!(
                surface,
                Surface::Glass,
                "aperture through motor case {seed}"
            );
            distance - 0.1
        };
        assert!((0.011..0.018).contains(&upward(metal)), "seed {seed}");
        assert!(
            (upward(gap) - intake::ROOF).abs() < 0.0003,
            "blocked air path seed {seed}"
        );
        shapes.insert((p.construction, name, format!("{:?}", d.lamp)));
    }
    assert_eq!(shapes.len(), 16, "missing canopy/filter/lamp combination");
}

#[test]
fn fitted_canopy_uses_the_flue_to_accommodate_tall_ceilings() {
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::ExtractionHood, seed);
        o.size.y = 0.70;
        let p = hood_parameters(&o);
        let top = p.canopy_top(o.size.y, FITTED_INTAKE);
        let d = FilterDeck::fit(&o, &p);
        o.size.y = 2.40;
        assert_eq!(hood_parameters(&o).canopy_top(o.size.y, FITTED_INTAKE), top);
        assert_eq!(d, FilterDeck::fit(&o, &hood_parameters(&o)));
        assert!((0.09..=0.34).contains(&top));
        let old = build_object_at_version(&o, None, FITTED_INTAKE - 1);
        let new = build_object_at_version(&o, None, FITTED_INTAKE);
        assert!(old
            .parts
            .contains_key(&(Surface::Rubber, "extraction_hood".into())));
        assert!(!new
            .parts
            .contains_key(&(Surface::Rubber, "extraction_hood".into())));
        assert!((new.bounds().1.y - o.size.y).abs() < 1e-5);
        audit::check(&o, &new).unwrap();
    }
}
