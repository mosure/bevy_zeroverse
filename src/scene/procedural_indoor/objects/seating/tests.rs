use super::*;

#[test]
fn loose_pillows_are_asymmetric_replayable_and_separated() {
    let mut counts = [0_usize; 5];
    let mut cells = std::collections::BTreeSet::new();
    let mut asymmetric = 0;
    let mut diameters = Vec::new();
    for seed in 0..256 {
        let o = object(seed);
        let p = parameters(&o);
        let widths = module_widths(&o);
        let parts = pillows::build(&o, &p, &widths);
        let replay = pillows::build(&o, &p, &widths);
        counts[parts.len()] += 1;
        let bounds: Vec<_> = parts
            .iter()
            .map(|g| {
                g.positions.iter().map(|v| Vec3::from_array(*v)).fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(lo, hi), v| (lo.min(v), hi.max(v)),
                )
            })
            .collect();
        for (i, ((lo, hi), g)) in bounds.iter().zip(&parts).enumerate() {
            assert_eq!(g.positions, replay[i].positions);
            assert_eq!(g.normals, replay[i].normals);
            // Rotation-independent bounding diameter of actual accepted mesh.
            let center = (*lo + *hi) * 0.5;
            diameters.push(
                g.positions
                    .iter()
                    .map(|v| Vec3::from_array(*v).distance(center) * 2.)
                    .fold(0_f32, f32::max),
            );
            assert!(lo.y > p.seat_height - p.cushion);
            assert!(hi.y <= o.size.y && (lo.y - p.seat_height).abs() < 0.05);
            for (peer_lo, peer_hi) in &bounds[..i] {
                assert!(!lo.xz().cmplt(peer_hi.xz()).all() || !hi.xz().cmpgt(peer_lo.xz()).all());
            }
            let uv = ((*lo + *hi).xz() * 0.5 / o.size.xz()) + Vec2::splat(0.5);
            cells.insert(((uv.x * 8.) as u32, (uv.y * 8.) as u32));
        }
        if bounds.len() >= 2 {
            let x_sum = bounds
                .iter()
                .map(|(lo, hi)| (lo.x + hi.x) * 0.5)
                .sum::<f32>();
            asymmetric += usize::from(x_sum.abs() > 0.02);
        }
    }
    assert!(counts[1..].iter().filter(|&&n| n > 0).count() >= 3);
    assert!(asymmetric > 50 && cells.len() > 16);
    diameters.sort_by(f32::total_cmp);
    let q05 = diameters[diameters.len() / 20];
    let q95 = diameters[diameters.len() * 19 / 20];
    assert!(q95 / q05 > 1.45, "accepted pillow size spread {q05}..{q95}");
    println!(
        "pillow placement audit: {}",
        serde_json::json!({"seeds":256,
        "count_histogram":counts,"asymmetric_multi_pillow_sofas":asymmetric,
        "occupied_8x8_cells":cells.len(),"mesh_diameter_p05_m":q05,"mesh_diameter_p95_m":q95})
    );
}

fn object(seed: u64) -> IndoorObject {
    IndoorObject {
        finish_seed: None,
        id: 0,
        kind: ObjectKind::Sofa,
        position: Vec3::ZERO,
        size: Vec3::new(2.8, 0.92, 1.85),
        yaw: 0.,
        variant: 0,
        seed,
        solid: true,
        support: None,
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    }
}
#[test]
fn asymmetric_upholstery_modules_reserve_arms_and_replay_exactly() {
    let mut signatures = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let o = object(seed);
        let p = parameters(&o);
        let widths = module_widths(&o);
        assert_eq!(widths.len(), p.modules);
        assert_eq!(widths, module_widths(&o));
        assert!((widths.iter().sum::<f32>() + p.arm_width * 2. - o.size.x).abs() < 1e-6);
        assert!(widths.iter().all(|&w| w > p.cushion_gap_m + 0.20));
        let smallest = widths.iter().copied().fold(f32::INFINITY, f32::min);
        let largest = widths.iter().copied().fold(0., f32::max);
        assert!(
            largest / smallest < 1.64,
            "realistic module ratio: {widths:?}"
        );
        signatures.insert(widths.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(signatures.len(), 256);
}
#[test]
fn upholstery_profiles_are_closed_and_bounded_for_short_loveseats_and_deep_chaise() {
    let mut scene = super::super::super::layout::IndoorManifest::generate_with_humans(
        17,
        super::super::super::layout::IndoorLayout::Lounge,
        0.6,
        0,
        0.,
    )
    .unwrap();
    scene.objects.clear();
    for seed in 0..64 {
        for (kind, size, variant) in [
            (ObjectKind::Sofa, Vec3::new(1.30, 0.76, 0.82), 0),
            (ObjectKind::Sofa, Vec3::new(2.8, 0.92, 1.85), 0),
            (ObjectKind::Chair, Vec3::new(0.68, 0.95, 0.68), 8),
        ] {
            let mut o = object(seed);
            o.kind = kind;
            o.size = size;
            o.variant = variant;
            o.id = scene.objects.len();
            let assembly = super::super::build_object(&o);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, -0.003, -size.z * 0.5) - Vec3::splat(1e-5))
                    .all(),
                "{kind:?}/{seed}: {lo:?}"
            );
            assert!(
                hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.003))
                    .all(),
                "{kind:?}/{seed}: {hi:?}"
            );
            scene.objects.push(o);
        }
    }
    super::super::super::validation::validate_geometry(&scene).unwrap();
}

#[test]
fn seating_constructions_and_recliners_respect_reserved_envelopes() {
    use super::super::audit;
    let mut arms = std::collections::BTreeSet::new();
    let mut bases = std::collections::BTreeSet::new();
    let mut backs = std::collections::BTreeSet::new();
    let mut seats = std::collections::BTreeSet::new();
    let mut recliners = 0;
    for seed in 0..128 {
        for (kind, size, variant) in [
            (ObjectKind::Chair, Vec3::new(0.84, 1.12, 1.35), 8),
            (ObjectKind::Chair, Vec3::new(1.02, 1.30, 1.65), 8),
            (ObjectKind::Sofa, Vec3::new(1.30, 0.76, 0.82), 0),
            (ObjectKind::Sofa, Vec3::new(3.2, 1.10, 1.85), 0),
        ] {
            let mut o = audit::fixture(kind, seed);
            o.size = size;
            o.variant = variant;
            let p = parameters(&o);
            let a = super::super::build_object(&o);
            let check =
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{size}/{seed}: {e}; {p:?}"));
            assert!(
                check.min_y_m <= 0.003,
                "floor contact {size}/{seed}: {check:?}"
            );
            if p.recliner {
                recliners += 1;
                assert!(!super::super::chairs::supports_upright_occupant(&o));
            } else {
                arms.insert(p.arm_style);
                bases.insert(p.base_style);
                backs.insert(p.back_style);
                seats.insert(p.seat_style);
            }
        }
    }
    assert!(recliners > 60);
    assert_eq!(arms, (0..6).collect());
    assert_eq!(bases, (0..6).collect());
    assert_eq!(backs, (0..5).collect());
    assert_eq!(seats, (0..3).collect());
}

#[test]
fn upholstery_controls_change_actual_surfaces_with_fixed_other_parameters() {
    use super::super::audit;
    let o = audit::fixture(ObjectKind::Sofa, 17);
    for (axis, count) in [("arms", 6), ("bases", 6), ("backs", 5), ("seats", 3)] {
        let mut maps = Vec::new();
        for value in 0..count {
            let mut p = parameters(&o);
            p.pillows = false;
            match axis {
                "arms" => p.arm_style = value,
                "bases" => p.base_style = value,
                "backs" => p.back_style = value,
                _ => p.seat_style = value,
            }
            let mut a = Assembly::default();
            build_program(&mut a, &o, p);
            audit::check(&o, &a).unwrap();
            maps.push(audit::descriptor_at_resolution(&o, &a, 128));
        }
        // This ablation checks that the control is active, including a narrow
        // cushion seam. Population diversity has separate, much coarser distance
        // thresholds; this test must not turn seam visibility into a volume claim.
        for (i, map) in maps.iter().enumerate() {
            for other in &maps[..i] {
                assert!(
                    audit::distance(map, other) > 1e-6,
                    "inactive {axis} control"
                );
            }
        }
    }
}
