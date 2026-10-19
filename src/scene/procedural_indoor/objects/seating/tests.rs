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
