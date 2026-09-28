use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, validation::validate_layout};

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

fn intersects(origin: Vec3, dir: Vec3, p: [Vec3; 3]) -> bool {
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
    (0.0..=1.0).contains(&u) && v >= 0. && u + v <= 1. && (0.0..=0.7).contains(&t)
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
        }
        let mut assembly = Assembly::default();
        shell(&mut assembly, &bare);
        super::super::details::feature_wall(&mut assembly, &bare);
        for f in &bare.exterior.as_ref().unwrap().facades {
            let tf = f.side.transform(bare.room_size);
            for o in &f.openings {
                let lo = o.min + Vec2::splat(f.frame_width);
                let hi = o.max - Vec2::splat(f.frame_width);
                let y = lo.y + (hi.y - lo.y) * if o.transom > 0. { o.transom * 0.5 } else { 0.5 };
                let x = lo.x + (hi.x - lo.x) / o.columns as f32 * 0.5;
                let origin = tf.transform_point(Vec3::new(x, y, 0.25));
                let dir = tf.rotation * -Vec3::Z;
                let mut glass = false;
                for ((surface, label), g) in &assembly.parts {
                    let hit = g.indices.as_chunks::<3>().0.iter().any(|t| {
                        intersects(
                            origin,
                            dir,
                            t.map(|i| Vec3::from_array(g.positions[i as usize])),
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
