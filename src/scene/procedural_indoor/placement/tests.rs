use super::*;
use crate::scene::procedural_indoor::{
    architecture,
    layout::{stream, IndoorLayout, IndoorManifest, ObjectKind},
    objects::{self, audit, kitchen::contents},
    validation::{validate_geometry, validate_layout},
};
use std::collections::BTreeSet;

#[test]
fn rotated_support_proposals_are_bounded_and_have_no_clamped_edge_mass() {
    let mut occupied = BTreeSet::new();
    let lo = Vec2::new(-0.50, -0.25);
    let hi = Vec2::new(0.50, 0.25);
    for seed in 0..128 {
        let mut r = stream(seed, 17);
        let pattern = SurfacePattern::sample(&mut r);
        for _ in 0..32 {
            let yaw = pattern.yaw(&mut r);
            let size = Vec2::new(0.12, 0.18);
            let p = pattern.propose((lo, hi), size, yaw, 0.01, &mut r).unwrap();
            assert!(p.inside(lo, hi, 0.00999));
            // Independently transform every corner, including non-square items.
            let q = Quat::from_rotation_y(yaw);
            for x in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    let corner = p.center + (q * Vec3::new(size.x * x, 0., size.y * z)).xz();
                    assert!(corner.cmpge(lo).all() && corner.cmple(hi).all());
                }
            }
            let uv = (p.center - lo) / (hi - lo);
            occupied.insert(((uv.x * 8.) as u32, (uv.y * 8.) as u32));
            assert!((p.center - p.half - lo).min_element() > 0.01);
            assert!((hi - p.center - p.half).min_element() > 0.01);
        }
        assert!(pattern
            .propose((lo, hi), Vec2::splat(10.), 0., 0.01, &mut r)
            .is_none());
    }
    assert!(occupied.len() >= 35, "support occupancy: {occupied:?}");
}

#[test]
fn shelf_contents_replay_and_break_rows_without_clipping_shelves_or_each_other() {
    let mut yaws = [0_usize; 16];
    let mut asymmetric_yaws = [0_usize; 16];
    let mut asymmetric_items = 0;
    let mut asymmetric_aligned = 0;
    let mut cells = BTreeSet::new();
    let mut counts = BTreeSet::new();
    let mut aligned = 0;
    let mut samples = 0;
    let mut pairs = 0;
    let mut same_row = 0;
    let mut sizes = std::collections::BTreeMap::<String, Vec<f32>>::new();
    let mut mixed_scale_shelves = 0;
    for seed in 0..256 {
        for kind in [ObjectKind::KitchenCupboard, ObjectKind::Pantry] {
            let o = audit::fixture(kind, seed);
            let bounds = (Vec2::new(-0.40, -0.14), Vec2::new(0.40, 0.14));
            for level in 0..3 {
                let y = level as f32 * 0.25;
                let items = contents::shelf_contents(&o, level, bounds, y, 0.23);
                assert_eq!(items, contents::shelf_contents(&o, level, bounds, y, 0.23));
                counts.insert(items.len());
                if let (Some(lo), Some(hi)) = (
                    items.iter().map(|i| i.size.x).min_by(f32::total_cmp),
                    items.iter().map(|i| i.size.x).max_by(f32::total_cmp),
                ) {
                    mixed_scale_shelves += usize::from(hi > lo * 1.5);
                }
                for (index, item) in items.iter().enumerate() {
                    sizes
                        .entry(format!(
                            "{}-{}",
                            if item.pantry { "pantry" } else { "dish" },
                            item.style
                        ))
                        .or_default()
                        .push(item.size.x);
                    let f = item.footprint();
                    assert!(f.inside(bounds.0, bounds.1, 0.00799));
                    let mut a = objects::Assembly::default();
                    contents::build(&mut a, std::slice::from_ref(item));
                    let (lo, hi) = a.bounds();
                    assert!(lo.xz().cmpge(bounds.0).all() && hi.xz().cmple(bounds.1).all());
                    assert!(lo.y >= y - 1e-6 && lo.y < y + 0.003);
                    assert!(hi.y < y + 0.23);
                    for peer in &items[..index] {
                        assert!(!f.overlaps(peer.footprint(), 0.0079));
                        pairs += 1;
                        same_row += usize::from((item.position.z - peer.position.z).abs() < 0.005);
                    }
                    let angle = item.yaw.rem_euclid(std::f32::consts::TAU);
                    let bin = ((angle / std::f32::consts::TAU * 16.) as usize).min(15);
                    yaws[bin] += 1;
                    let near_axis = usize::from(
                        (2. * item.yaw).sin().abs() < (2.0_f32 * 2_f32.to_radians()).sin(),
                    );
                    aligned += near_axis;
                    // A round plate's yaw is invisible. Qualify cartons and
                    // handled cups separately, where orientation changes RGB/geometry.
                    if (item.pantry && item.style == 0) || (!item.pantry && item.style == 2) {
                        asymmetric_yaws[bin] += 1;
                        asymmetric_items += 1;
                        asymmetric_aligned += near_axis;
                    }
                    let uv = (item.position.xz() - bounds.0) / (bounds.1 - bounds.0);
                    cells.insert(((uv.x * 8.) as u32, (uv.y * 8.) as u32));
                    samples += 1;
                }
            }
        }
    }
    assert!(yaws.iter().all(|&n| n > 50));
    assert!(asymmetric_yaws.iter().all(|&n| n > 10));
    assert!(asymmetric_aligned as f64 / (asymmetric_items as f64) < 0.08);
    assert!(counts.len() >= 4 && cells.len() >= 24);
    assert!(
        aligned as f32 / (samples as f32) < 0.08,
        "axis fraction: {aligned}/{samples}"
    );
    assert!(same_row as f32 / (pairs as f32) < 0.12);
    let sizes: std::collections::BTreeMap<_, _> = sizes
        .into_iter()
        .map(|(kind, mut widths)| {
            widths.sort_by(f32::total_cmp);
            let n = widths.len();
            let q05 = widths[n / 20];
            let q95 = widths[n * 19 / 20];
            assert!(
                q95 / q05 > 1.55,
                "{kind}: accepted width spread {q05}..{q95}"
            );
            (
                kind,
                serde_json::json!({"n":n,"p05_width_m":q05,"p95_width_m":q95,"ratio":q95/q05}),
            )
        })
        .collect();
    assert!(mixed_scale_shelves > 300);
    println!(
        "shelf placement audit: {}",
        serde_json::json!({
            "seeds":256,"items":samples,"yaw_bins":yaws,"occupied_8x8_cells":cells.len(),
        "count_values":counts,"within_2deg_of_axis_fraction":aligned as f64/samples as f64,
        "handled_cups_and_cartons":asymmetric_items,"asymmetric_yaw_bins":asymmetric_yaws,
        "asymmetric_within_2deg_of_axis_fraction":asymmetric_aligned as f64/asymmetric_items as f64,
            "pairs":pairs,"within_5mm_same_row_fraction":same_row as f64/pairs as f64,
            "accepted_size_envelopes":sizes,"shelves_with_size_ratio_over_1_5":mixed_scale_shelves
        })
    );
}

#[test]
fn placement_population_preserves_geometry_support_cameras_and_annotation_labels() {
    let start = std::time::Instant::now();
    let mut loose_count = 0;
    let mut loose_yaws = BTreeSet::new();
    let mut fixture_yaws = BTreeSet::new();
    let mut fixture_pairs = 0;
    let mut aligned_fixture_pairs = 0;
    let mut rooms = 0;
    let mut objects = 0;
    let mut fixture_widths = Vec::new();
    let mut rooms_with_varied_fixture_size = 0;
    let mut loose_sizes = std::collections::BTreeMap::<String, Vec<f32>>::new();
    for layout in [
        IndoorLayout::Mixed,
        IndoorLayout::Kitchen,
        IndoorLayout::Kitchenette,
        IndoorLayout::Lounge,
    ] {
        for seed in 0..16 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.75, 3, 0.25)
                .unwrap_or_else(|e| panic!("{layout:?} seed={seed}: {e}"));
            validate_layout(&scene).unwrap();
            validate_geometry(&scene).unwrap_or_else(|e| panic!("{layout:?} seed={seed}: {e}"));
            objects += scene.objects.len();
            rooms += 1;
            for o in &scene.objects {
                if let Some(support) = o.support {
                    if matches!(
                        o.kind,
                        ObjectKind::Dishware
                            | ObjectKind::Mug
                            | ObjectKind::CoffeeCup
                            | ObjectKind::Notepad
                            | ObjectKind::Phone
                    ) {
                        let s = &scene.objects[support];
                        assert!(scene.prop_clear(o, s, 0.009));
                        loose_yaws.insert(
                            ((((o.yaw - s.yaw).rem_euclid(2. * PI)) * 8. / PI) as u32).min(15),
                        );
                        loose_count += 1;
                        loose_sizes
                            .entry(format!("{:?}", o.kind))
                            .or_default()
                            .push(o.size.x);
                    }
                }
            }
            let positions = architecture::fixture_positions(&scene);
            assert!(positions.len() > 1, "room has no primary ceiling fixtures");
            let mut min_size = f32::INFINITY;
            let mut max_size = 0_f32;
            for (index, &p) in positions.iter().enumerate() {
                let a = architecture::fixtures::geometry(&scene, index, p);
                let half = architecture::fixture_half_extents(&scene, p);
                let (lo, hi) = a.bounds();
                assert!(lo.xz().cmpge(p.xz() - half.xz() - Vec2::splat(0.003)).all());
                assert!(hi.xz().cmple(p.xz() + half.xz() + Vec2::splat(0.003)).all());
                assert!(a
                    .parts
                    .keys()
                    .all(|(_, label)| label.starts_with(&format!("lamp#{index}"))));
                let yaw = architecture::fixtures::rotation(&scene, p)
                    .to_euler(EulerRot::YXZ)
                    .0;
                fixture_yaws.insert(((yaw.rem_euclid(2. * PI) * 8. / PI) as u32).min(15));
                if index + 1 < positions.len() {
                    let width = architecture::fixtures::size(&scene, p).x;
                    fixture_widths.push(width);
                    min_size = min_size.min(width);
                    max_size = max_size.max(width);
                    for peer in &positions[..index] {
                        fixture_pairs += 1;
                        aligned_fixture_pairs +=
                            usize::from((p.xz() - peer.xz()).abs().min_element() < 0.05);
                    }
                }
            }
            rooms_with_varied_fixture_size += usize::from(max_size > min_size * 1.4);
        }
    }
    assert_eq!(loose_yaws.len(), 16);
    assert_eq!(fixture_yaws.len(), 16);
    assert!(aligned_fixture_pairs as f64 / (fixture_pairs as f64) < 0.10);
    assert!(rooms_with_varied_fixture_size > 40);
    fixture_widths.sort_by(f32::total_cmp);
    let loose_sizes: std::collections::BTreeMap<_, _> = loose_sizes
        .into_iter()
        .map(|(kind, mut sizes)| {
            sizes.sort_by(f32::total_cmp);
            let n = sizes.len();
            (
                kind,
                serde_json::json!({"n":n,"p05_width_m":sizes[n/20],"p95_width_m":sizes[n*19/20]}),
            )
        })
        .collect();
    println!(
        "room placement audit: {}",
        serde_json::json!({"rooms":rooms,"objects":objects,
        "supported_loose_items":loose_count,"loose_yaw_bins":loose_yaws.len(),
        "fixture_yaw_bins":fixture_yaws.len(),"fixture_pairs":fixture_pairs,
        "fixture_pairs_within_5cm_of_same_row_or_column_fraction":aligned_fixture_pairs as f64/fixture_pairs as f64,
        "rooms_with_fixture_width_ratio_over_1_4":rooms_with_varied_fixture_size,
        "fixture_width_p05_m":fixture_widths[fixture_widths.len()/20],
        "fixture_width_p95_m":fixture_widths[fixture_widths.len()*19/20],
        "supported_loose_size_envelopes":loose_sizes,
        "elapsed_s":start.elapsed().as_secs_f64(),
        "validation":"CPU layout, support, mesh geometry, camera paths and semantic part identities"})
    );
}
