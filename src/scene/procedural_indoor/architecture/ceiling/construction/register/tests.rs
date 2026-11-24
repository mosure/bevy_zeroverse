use super::*;
use crate::scene::procedural_indoor::architecture::ceiling::CeilingDetail;

fn module(seed: u64, system: CeilingSystem, size: Vec3, branches: u8) -> CeilingModule {
    CeilingModule {
        system,
        center: Vec3::ZERO,
        size,
        seed,
        support_top_m: 1.,
        pitch_m: 0.5,
        thickness_m: 0.03,
        detail: Some(CeilingDetail {
            finish: 2,
            shape: CeilingShape::Duct {
                section: 0.92,
                end_scale: 0.58,
                bend: 0.76,
                rise: -0.51,
                branches,
                seam_pitch_m: 0.53,
            },
        }),
        suspension: None,
    }
}

#[test]
fn fitted_registers_clear_saddles_and_one_another_even_on_short_runs() {
    let mut sides = [0; 2];
    let mut patterns = [0; 3];
    let mut coatings = [0; 2];
    let mut opposed_hands = [false; 2];
    for seed in 0..512 {
        let branches = (seed % 5) as u8;
        let size = if seed % 2 == 0 {
            Vec3::new(0.4, 0.10, 0.08)
        } else {
            Vec3::new(0.49, 0.37, 7.3)
        };
        let m = module(seed, CeilingSystem::RoundDuct, size, branches);
        assert!(m.duct_register_stock(191).is_none());
        let stock = m.duct_register_stock(192).unwrap();
        patterns[stock.pattern as usize] += 1;
        coatings[usize::from(stock.coated)] += 1;
        let length = m.size.x.max(m.size.z) * 0.94;
        let width = m.size.x.min(m.size.z);
        let mut right = 0.14 * length + 0.013;
        let mut previous_side = 0.;
        for i in 0..branches.max(1) {
            let site = stock.site(&m, branches, i);
            assert!(site.size.is_finite() && site.size.min_element() > 0.);
            let center = site.station * length;
            assert!(center - site.size.x * 0.5 > right);
            right = center + site.size.x * 0.5;
            assert!(right < 0.86 * length - 0.013);
            assert!(site.lateral_m.abs() + 0.5 * site.size.y < width * 0.5);
            if branches > 0 {
                sides[usize::from(site.lateral_m > 0.)] += 1;
                if i > 0 {
                    opposed_hands[usize::from(previous_side * site.lateral_m < 0.)] = true;
                }
                previous_side = site.lateral_m;
            }
        }
    }
    assert!(patterns.into_iter().all(|n| n > 90));
    assert!(coatings.into_iter().all(|n| n > 130));
    assert!(sides.into_iter().all(|n| n > 300));
    assert_eq!(opposed_hands, [true, true]);
}

/// Rasterize upward-facing rays through the *complete* takeoff and trunk so
/// an intervening solid plate, seam, or adjacent branch cannot masquerade as
/// a properly recessed opening. Each 2D sample records its first surface.
fn cavity_hits(a: &Assembly, rear: &[[f32; 3]]) -> (f32, f32) {
    const N: usize = 29;
    let corners = [rear[0], rear[1], rear[2], rear[3]].map(Vec3::from);
    let lo = corners
        .iter()
        .map(|p| p.xz())
        .fold(Vec2::splat(f32::INFINITY), Vec2::min);
    let hi = corners
        .iter()
        .map(|p| p.xz())
        .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
    let mut hits = [(f32::INFINITY, Surface::Metal); N * N];
    for ((material, _), g) in &a.parts {
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, q, r] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            if p.min(q).min(r).xz().cmpgt(hi).any() || p.max(q).max(r).xz().cmplt(lo).any() {
                continue;
            }
            let u = q - p;
            let v = r - p;
            let area = u.xz().perp_dot(v.xz());
            if area.abs() < 1e-12 {
                continue;
            }
            for (i, hit) in hits.iter_mut().enumerate() {
                let uv = (Vec2::new((i % N) as f32 + 0.43, (i / N) as f32 + 0.61) / N as f32)
                    * (hi - lo)
                    + lo
                    - p.xz();
                let x = uv.perp_dot(v.xz()) / area;
                let y = u.xz().perp_dot(uv) / area;
                let h = p.y + x * u.y + y * v.y;
                if x >= 0. && y >= 0. && x + y <= 1. && h < hit.0 {
                    *hit = (h, *material);
                }
            }
        }
    }
    assert!(hits
        .iter()
        .all(|h| h.0.is_finite() && h.0 <= corners[0].y + 1e-5));
    let open = hits.iter().filter(|h| h.1 == Surface::Ink).count() as f32 / hits.len() as f32;
    let nearest = hits.iter().map(|h| h.0).fold(f32::INFINITY, f32::min);
    (open, corners[0].y - nearest)
}

#[test]
fn real_slots_survive_complete_bent_reducing_ducts_and_planning_skips_meshing() {
    for seed in 0..20 {
        let n = (seed % 5) as u8;
        let round = seed % 2 == 0;
        let depth = if seed % 3 == 0 { 0.10 } else { 0.35 };
        let m = module(
            seed,
            if round {
                CeilingSystem::RoundDuct
            } else {
                CeilingSystem::RectangularDuct
            },
            Vec3::new(3.1, depth, 0.48),
            n,
        );
        let f = Frame::new(&m, 2);
        let emit = |mesh, version| {
            super::super::duct::build(mesh, &f, 0.92, 0.58, 0.76, -0.51, n, 0.53, version)
        };
        let mut a = Assembly::default();
        let anchors = emit(Some(&mut a), 192);
        assert_eq!(anchors, emit(None, 192));
        let mut archive = Assembly::default();
        let prior = super::super::duct::build(
            Some(&mut archive),
            &f,
            0.92,
            0.58,
            0.76,
            -0.51,
            n,
            0.53,
            191,
        );
        assert_eq!(prior, anchors);
        let ink = (Surface::Ink, "ceiling".to_owned());
        assert_eq!(
            archive.parts[&ink].positions.len(),
            24 * usize::from(n.max(1))
        );
        let back = &a.parts[&ink];
        assert_eq!(back.positions.len(), 4 * usize::from(n.max(1)));
        for rear in back.positions.as_chunks::<4>().0 {
            let (open, relief) = cavity_hits(&a, rear);
            assert!(
                (0.15..0.99).contains(&open) && relief > 0.002,
                "seed {seed}: {open} open, {relief} m relief"
            );
        }
    }
    let m = module(14, CeilingSystem::RoundDuct, Vec3::new(4., 0.3, 0.4), 1);
    let f = Frame::new(&m, 2);
    let supports = [
        (Vec3::new(-1., 0.12, 0.), 0.04),
        (Vec3::new(1., 0.14, 0.), 0.03),
    ];
    let anchors = super::super::services::suspended(None, &f, supports, |_| {
        panic!("planning ran the body mesher")
    });
    assert_eq!(anchors.len(), 4);
}
