use super::super::audit::{check, fixture};
use super::*;
use std::collections::BTreeSet;

fn assert_fits(o: &IndoorObject, a: &Assembly) {
    let q = check(o, a).unwrap_or_else(|e| panic!("{:?}/{} {:?}: {e}", o.kind, o.seed, o.size));
    assert!(q.max_envelope_overrun_m < 0.0005, "{q:?}");
    assert!(
        a.bounds().0.y.abs() < 0.0001,
        "{:?}/{} {:?}: {:?}",
        o.kind,
        o.seed,
        o.size,
        a.bounds()
    );
    assert!(q.triangles < 30000);
}
fn receipt(name: &str, value: serde_json::Value) {
    println!("{name}: {value}");
    if let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn served_food_contacts_the_inside_of_the_dish_and_inventory_stays_bare() {
    use super::serving::Serving;
    let mut counts = [0; 5];
    for seed in 0..192 {
        let mut o = fixture(ObjectKind::Dishware, seed);
        assert!(serving::sample(&o).is_none());
        assert!(metadata(&o).get("serving").is_none());
        for (variant, style) in [(dishware::SERVED_PLATE, 0), (dishware::SERVED_BOWL, 1)] {
            o.variant = variant;
            o.size =
                crate::scene::procedural_indoor::items::sizing::dish(style, &mut stream(seed, 321));
            let d = dishware::for_object(&o, &props::program(&o));
            let meal = serving::sample(&o).unwrap();
            assert_eq!(meal, serving::sample(&o).unwrap());
            assert_eq!(metadata(&o)["serving"], serde_json::json!(meal));
            counts[meal as usize] += 1;
            let mut built = Assembly::default();
            props::build(&mut built, &o);
            assert_fits(&o, &built);
            let food: Vec<_> = built
                .parts
                .iter()
                .filter(|((_, label), _)| label.starts_with("food#finish"))
                .collect();
            assert_eq!(food.is_empty(), meal == Serving::Empty);
            assert!(food.iter().map(|(_, g)| g.indices.len() / 3).sum::<usize>() <= 1200);
            if meal == Serving::Empty {
                continue;
            }
            let points = food
                .iter()
                .flat_map(|(_, g)| &g.positions)
                .map(|p| Vec3::from_array(*p));
            if meal == Serving::Soup {
                let liquid = &built.parts[&(Surface::Ceramic, "food#finish105".into())];
                let y = liquid.positions[0][1];
                let middle = d.height * d.shoulder_height + d.wall * 0.25;
                assert!(y > middle && y < d.height && liquid.normals.iter().all(|n| n[1] > 0.999));
                let r = d.radius - d.wall
                    + (d.mouth - 1.) * d.radius * (y - middle) / (d.height - middle);
                for v in &liquid.positions[..d.segments as usize + 1] {
                    assert!(((Vec3::from_array(*v).xz() / d.aspect).length() - r).abs() < 1e-6);
                }
                assert!(points.into_iter().all(|v| v.y > d.wall && v.y < o.size.y));
            } else {
                let foot = d.radius * d.foot - if style == 1 { d.wall } else { 0. };
                let mut bottom = f32::INFINITY;
                for v in points {
                    bottom = bottom.min(v.y);
                    assert!(
                        (v.xz() / d.aspect).length()
                            <= foot * (PI / d.segments as f32).cos() + 1e-6
                    );
                }
                assert!(
                    (bottom - d.interior_floor()).abs() < 1e-6,
                    "floating serving {meal:?} {seed}: {bottom}"
                );
            }
        }
    }
    assert!(counts.iter().all(|&n| n > 15), "{counts:?}");
}
#[test]
fn dishes_have_hollow_interiors_supported_stacks_and_correct_yawed_footprints() {
    let mut materials = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for seed in 0..64 {
        for style in 0..5 {
            for size in [
                Vec3::new(0.12, 0.05, 0.15),
                Vec3::new(0.26, 0.18, 0.22),
                Vec3::new(0.33, 0.07, 0.26),
            ] {
                let mut o = fixture(ObjectKind::Dishware, seed);
                o.size = size;
                let p = dishware::parameters(size, style, 5, seed);
                let mut a = Assembly::default();
                dishware::build(&mut a, Transform::IDENTITY, &p, seed);
                assert_fits(&o, &a);
                // Each foot contacts the preceding well; the central ray hits
                // the uppermost usable floor even when the stack is staggered.
                assert!(
                    (tests::top_at(&a, Vec2::ZERO) - p.interior_floor() * p.count as f32).abs()
                        < 1e-5
                );
                materials.insert(format!("{:?}", p.surface));
                shapes.insert(serde_json::to_string(&p).unwrap());
                for yaw in [0.17, 0.71, 1.43, 2.22] {
                    let mut a = Assembly::default();
                    dishware::build(
                        &mut a,
                        Transform::from_rotation(Quat::from_rotation_y(yaw)),
                        &p,
                        seed,
                    );
                    let (lo, hi) = a.bounds();
                    let half = p.half_extents(yaw) + Vec2::splat(1e-5);
                    assert!(lo.xz().cmpge(-half).all() && hi.xz().cmple(half).all());
                }
                assert!((a.bounds().1.y - p.stack_height()).abs() < 1e-5);
            }
        }
    }
    assert!(materials.len() >= 4 && shapes.len() > 900);
    receipt(
        "dishware",
        serde_json::json!({"meshes":960,"styles":5,"substrates":materials,"yawed_footprint_checks":3840,"hollow_floor_and_stack_checks":960}),
    );
}

#[test]
fn six_package_constructions_fit_their_admission_footprints() {
    let mut styles = BTreeSet::new();
    let mut substrates = BTreeSet::new();
    let output = std::env::var_os("ZEROVERSE_OBJECT_REPORT");
    let mut sheet = image::RgbImage::new(6 * 192, 3 * 192);
    for seed in 0..48 {
        for style in 0..6 {
            for size in [
                Vec3::new(0.075, 0.09, 0.065),
                Vec3::new(0.18, 0.23, 0.15),
                Vec3::new(0.22, 0.12, 0.20),
            ] {
                let p = packaging::parameters(size, style, seed);
                let mut o = fixture(ObjectKind::Dishware, seed);
                o.size = size;
                let mut a = Assembly::default();
                packaging::build(&mut a, Transform::IDENTITY, &p, seed);
                assert_fits(&o, &a);
                styles.insert(style);
                substrates.insert(format!("{:?}", p.finish));
                for yaw in [0.13, 0.91, 1.6, 2.41] {
                    let mut b = Assembly::default();
                    packaging::build(
                        &mut b,
                        Transform::from_rotation(Quat::from_rotation_y(yaw)),
                        &p,
                        seed,
                    );
                    let (lo, hi) = b.bounds();
                    let half = p.half_extents(yaw) + Vec2::splat(1e-5);
                    assert!(lo.xz().cmpge(-half).all() && hi.xz().cmple(half).all());
                }
                if output.is_some() && seed < 3 && size.y == 0.23 {
                    let im = crate::scene::procedural_indoor::geometry::preview::orthographic(
                        &a,
                        Vec3::Y * size.y * 0.5,
                        size.length() * 1.05,
                        192,
                        Vec3::new(1.1, 1.25, 1.6),
                    );
                    image::imageops::replace(
                        &mut sheet,
                        &im,
                        style as i64 * 192,
                        seed as i64 * 192,
                    );
                }
            }
        }
    }
    assert_eq!(styles.len(), 6);
    assert!(substrates.len() >= 4);
    if let Some(path) = output {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        sheet.save(path.join("packaging.png")).unwrap();
    }
    receipt(
        "packaging",
        serde_json::json!({"meshes":864,"styles":styles,"substrates":substrates,"yawed_footprint_checks":3456}),
    );
}

#[test]
fn racks_fit_plates_between_real_wires_and_rest_them_on_the_tray() {
    let mut slots = BTreeSet::new();
    let mut occupancy = BTreeSet::new();
    let mut plates = 0;
    let mut right = 0;
    for seed in 0..96 {
        for size in [
            Vec3::new(0.23, 0.18, 0.26),
            Vec3::new(0.44, 0.22, 0.36),
            Vec3::new(0.60, 0.14, 0.45),
        ] {
            let p = supplies::rack::parameters(size, 5, seed);
            let mut o = fixture(ObjectKind::DryingRack, seed);
            o.size = size;
            let mut a = Assembly::default();
            supplies::rack::build(&mut a, Transform::IDENTITY, &p, seed);
            assert_fits(&o, &a);
            slots.insert(p.rail_x.len() - 1);
            occupancy.insert(p.plates.iter().map(|v| v.slot).collect::<Vec<_>>());
            for plate in &p.plates {
                let mut a = Assembly::default();
                dishware::build(&mut a, plate.transform(), &plate.dish, seed);
                let (lo, hi) = a.bounds();
                assert!((lo.y - p.tray).abs() < 1e-5);
                assert!(lo.x >= p.rail_x[plate.slot] + p.wire + 0.0009);
                assert!(hi.x <= p.rail_x[plate.slot + 1] - p.wire - 0.0009);
                assert!(lo.z >= -p.size.z * 0.37 - 1e-5 && hi.z <= p.size.z * 0.37 + 1e-5);
                for peer in &p.plates {
                    if peer.slot == plate.slot {
                        assert!(std::ptr::eq(plate, peer));
                    }
                }
                plates += 1;
                right += usize::from(plate.position.x > 0.);
            }
            assert!(supplies::rack::parameters(size, 0, seed).plates.is_empty());
        }
    }
    assert!(plates > 450 && slots.len() >= 5 && occupancy.len() > 100);
    assert!((right as f64 / plates as f64 - 0.5).abs() < 0.1);
    receipt(
        "drying_racks",
        serde_json::json!({"meshes":288,"plates":plates,"slot_counts":slots,"occupied_slot_patterns":occupancy.len(),"right_half_fraction":right as f64/plates as f64,"empty_contents_checks":288}),
    );
}

#[test]
fn holder_tools_clear_the_cavity_and_each_other_with_diverse_realized_poses() {
    let mut counts = BTreeSet::new();
    let mut styles = BTreeSet::new();
    let mut tools = 0;
    let mut poses = BTreeSet::new();
    for seed in 0..128 {
        for scale in [
            Vec3::ONE,
            Vec3::new(0.72, 0.65, 1.1),
            Vec3::new(1.4, 1.2, 0.9),
        ] {
            let mut o = fixture(ObjectKind::UtensilHolder, seed);
            o.size *= scale;
            let p = supplies::utensils::parameters(&o);
            assert!(!p.tools.is_empty());
            counts.insert(p.tools.len());
            let a = super::super::build_object(&o);
            assert_fits(&o, &a);
            for (i, t) in p.tools.iter().enumerate() {
                styles.insert(t.style);
                poses.insert(serde_json::to_string(t).unwrap());
                tools += 1;
                let mut a = Assembly::default();
                supplies::utensils::tool(&mut a, t, seed);
                for v in a
                    .parts
                    .values()
                    .flat_map(|g| &g.positions)
                    .map(|v| Vec3::from_array(*v))
                {
                    assert!(v
                        .xz()
                        .cmpge(t.footprint.center - t.footprint.half - Vec2::splat(1e-5))
                        .all());
                    assert!(v
                        .xz()
                        .cmple(t.footprint.center + t.footprint.half + Vec2::splat(1e-5))
                        .all());
                    if v.y < p.height {
                        assert!(v.y >= p.wall - 1e-5, "shaft through base {seed}: {v:?}");
                        let fraction = (v.y - p.wall) / (p.height - p.wall);
                        let inner = p.radius * (p.foot + (1. - p.foot) * fraction) - p.wall;
                        let angle = v.z.atan2(v.x).rem_euclid(TAU / p.segments as f32)
                            - PI / p.segments as f32;
                        let limit = inner * (PI / p.segments as f32).cos() / angle.cos();
                        assert!(
                            v.xz().length() <= limit + 1e-5,
                            "shaft through holder {seed}: {v:?}"
                        );
                    }
                }
                for peer in &p.tools[..i] {
                    assert!(!t.footprint.overlaps(peer.footprint, 0.00149));
                }
            }
        }
    }
    assert_eq!(styles.len(), 4);
    assert!(counts.len() >= 3 && tools > 500 && poses.len() > 500);
    receipt(
        "utensil_holders",
        serde_json::json!({"meshes":384,"tools":tools,"styles":styles,"realized_counts":counts,"unique_tool_programs":poses.len()}),
    );
}

#[test]
fn refrigerator_contents_use_the_same_shapes_for_admission_and_rendering() {
    let mut styles = BTreeSet::new();
    let mut count = 0;
    let mut centered = 0;
    let bounds = (Vec2::new(-0.30, -0.18), Vec2::new(0.30, 0.18));
    for seed in 0..128 {
        let o = fixture(ObjectKind::Refrigerator, seed);
        for level in 0..3 {
            let items = contents::shelf_contents(&o, level, bounds, 0., 0.23);
            assert_eq!(items, contents::shelf_contents(&o, level, bounds, 0., 0.23));
            for (i, item) in items.iter().enumerate() {
                let mut a = Assembly::default();
                contents::build(&mut a, std::slice::from_ref(item));
                let (lo, hi) = a.bounds();
                let f = item.footprint();
                assert!(lo.xz().cmpge(f.center - f.half - Vec2::splat(1e-5)).all());
                assert!(hi.xz().cmple(f.center + f.half + Vec2::splat(1e-5)).all());
                assert!(f.inside(bounds.0, bounds.1, 0.00799));
                assert!(lo.y.abs() < 1e-5 && hi.y < 0.23);
                for peer in &items[..i] {
                    assert!(!f.overlaps(peer.footprint(), 0.00799));
                }
                styles.insert(item.style);
                count += 1;
                centered += usize::from(item.position.xz().length() < 0.01);
            }
        }
    }
    assert_eq!(styles.len(), 6);
    assert!(count > 650 && centered as f64 / (count as f64) < 0.05);
    receipt(
        "refrigerator_contents",
        serde_json::json!({"shelves":384,"items":count,"styles":styles,"within_1cm_center_fraction":centered as f64/count as f64}),
    );
}

#[test]
fn assigning_parent_finishes_never_discards_explicit_child_meshes() {
    let mut merged = 0;
    for seed in 0..128 {
        for kind in [
            ObjectKind::UtensilHolder,
            ObjectKind::DryingRack,
            ObjectKind::Dishware,
            ObjectKind::KitchenCupboard,
            ObjectKind::Refrigerator,
        ] {
            let o = fixture(kind, seed);
            let mut raw = Assembly::default();
            build(&mut raw, &o, bevy_zeroverse_capture::GENERATOR_VERSION);
            let finished = super::super::build_object(&o);
            let vertices =
                |a: &Assembly| a.parts.values().map(|g| g.positions.len()).sum::<usize>();
            let indices = |a: &Assembly| a.parts.values().map(|g| g.indices.len()).sum::<usize>();
            assert_eq!(
                vertices(&raw),
                vertices(&finished),
                "{kind:?}/{seed}: finish assignment lost vertices"
            );
            assert_eq!(
                indices(&raw),
                indices(&finished),
                "{kind:?}/{seed}: finish assignment lost triangles"
            );
            for (key, g) in &raw.parts {
                if key.1.contains("#uv_local") {
                    assert_eq!(
                        g.uvs, finished.parts[key].uvs,
                        "package label unwrap changed"
                    );
                    for uv in &g.uvs {
                        assert!(uv.iter().all(|v| *v >= -1e-5 && *v <= 1.00001));
                    }
                    for axis in 0..2 {
                        let lo = g
                            .uvs
                            .iter()
                            .map(|uv| uv[axis])
                            .fold(f32::INFINITY, f32::min);
                        let hi = g
                            .uvs
                            .iter()
                            .map(|uv| uv[axis])
                            .fold(f32::NEG_INFINITY, f32::max);
                        assert!(hi - lo > 0.999, "label UV axis collapsed");
                    }
                }
            }
            merged += usize::from(raw.parts.len() > finished.parts.len());
        }
    }
    assert!(merged > 0, "exercise actual convergent material keys");
    receipt(
        "finish_preservation",
        serde_json::json!({"objects":640,"objects_with_convergent_keys":merged,"vertices_and_triangles_preserved":true,"label_uvs_preserved":true}),
    );
}

#[test]
fn coordinated_finish_identity_changes_dish_and_rack_finishes_without_shape_changes() {
    for kind in [ObjectKind::Dishware, ObjectKind::DryingRack] {
        let mut changed = 0;
        for seed in 0..64 {
            let o = fixture(kind, seed);
            let mut alternate = o.clone();
            alternate.finish_seed = Some(seed ^ 0xfeed1234);
            let a = super::super::build_object(&o);
            let b = super::super::build_object(&alternate);
            // A rim glaze can split a shared vertex ring between two materials.
            // Compare the actual oriented triangles across all material groups.
            let triangles = |a: &Assembly| {
                let mut p: Vec<_> = a
                    .parts
                    .values()
                    .flat_map(|g| {
                        g.indices
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .map(|t| t.map(|i| g.positions[i as usize].map(f32::to_bits)))
                    })
                    .collect();
                p.sort_unstable();
                p
            };
            assert_eq!(triangles(&a), triangles(&b));
            changed += usize::from(
                a.parts.keys().collect::<Vec<_>>() != b.parts.keys().collect::<Vec<_>>(),
            );
        }
        assert!(
            changed > 20,
            "{kind:?}: collection finish identity was ignored"
        );
    }
}
