use super::super::{program, Surface};
use super::*;
use std::collections::BTreeSet;

#[test]
fn floor_products_cover_independent_color_finish_and_scale() {
    for style in 0..3 {
        let mut colors = BTreeSet::new();
        let mut sizes = BTreeSet::new();
        let mut finishes = BTreeSet::new();
        let mut dye_poles = [0; 6];
        let mut factors = Vec::new();
        for seed in 0..1024 {
            let rs = program::sample_with_floor(seed, style).unwrap();
            let r = &rs[Surface::Floor as usize];
            r.validate().unwrap();
            let f = r.flooring.as_ref().unwrap();
            colors.insert(r.color.map(|c| (c * 10.).floor() as u32));
            sizes.insert(f.cells);
            finishes.insert((r.roughness * 20.).floor() as u32);
            factors.push([
                r.color.iter().sum::<f32>() as f64 / 3.,
                r.period_m as f64,
                r.roughness as f64,
            ]);
            assert_eq!(r.map_size(style), 512);
            let json = serde_json::to_string(r).unwrap();
            assert_eq!(
                *r,
                serde_json::from_str::<program::MaterialRecipe>(&json).unwrap()
            );
            if let Some(d) = r.carpet.as_ref().and_then(|c| c.dye.as_ref()) {
                let pole = (0..6)
                    .max_by(|&a, &b| d.weights[a].total_cmp(&d.weights[b]))
                    .unwrap();
                dye_poles[pole] += 1;
            }
        }
        assert!(colors.len() > 50, "color support {style}: {}", colors.len());
        assert!(sizes.len() > 90, "cut sizes {style}: {}", sizes.len());
        assert!(finishes.len() >= if style == 1 { 4 } else { 10 });
        if style == 1 {
            assert!(dye_poles.iter().all(|&n| n > 100), "{dye_poles:?}");
        }
        let mean: [f64; 3] = std::array::from_fn(|i| {
            factors.iter().map(|r| r[i]).sum::<f64>() / factors.len() as f64
        });
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let covariance: f64 = factors
                .iter()
                .map(|r| (r[a] - mean[a]) * (r[b] - mean[b]))
                .sum();
            let variance = |i: usize| {
                factors
                    .iter()
                    .map(|r| (r[i] - mean[i]).powi(2))
                    .sum::<f64>()
            };
            let correlation = covariance / (variance(a) * variance(b)).sqrt();
            assert!(
                correlation.abs() < 0.15,
                "coupled product factors {style}/{a}/{b}: {correlation}"
            );
        }
    }
}

#[test]
fn random_cut_stock_partitions_without_slivers_or_missing_pieces() {
    for seed in 0..32 {
        let mut r = program::sample_with_floor(seed, 0)
            .unwrap()
            .remove(Surface::Floor as usize);
        let f = r.flooring.as_mut().unwrap();
        f.layout = FloorLayout::BrokenBond;
        f.cells = [7, 5];
        f.stock.as_mut().unwrap().cut_jitter = [0.4, 0.4];
        let f = r.flooring.as_ref().unwrap();
        let mut pieces = BTreeSet::new();
        let mut area = std::collections::BTreeMap::<[u32; 2], u32>::new();
        for i in 0..16384 {
            let uv = [
                ((i as f64 + 0.3) * 0.7548776662).fract() as f32,
                ((i as f64 + 0.7) * 0.5698402910).fract() as f32,
            ];
            let c = f.cut(&r, uv);
            assert!(c.distance_m >= 0. && c.distance_m.is_finite());
            assert!(c.uv.into_iter().all(|x| (0.0..=1.).contains(&x)));
            pieces.insert(c.id);
            *area.entry(c.id).or_default() += 1;
            let wrap = f.cut(&r, [uv[0] + 2., uv[1] - 1.]);
            // Discontinuous piece IDs are compared away from exact boundaries.
            if c.distance_m > 1e-4 {
                assert_eq!(c.id, wrap.id);
                assert!((c.distance_m - wrap.distance_m).abs() < 3e-5);
            }
        }
        assert_eq!(pieces.len(), 35);
        assert!(
            area.values().all(|&n| n >= 15),
            "collapsed interval {area:?}"
        );
        assert!(area.values().max().unwrap() > &(area.values().min().unwrap() * 2));
    }
}

#[test]
fn new_floor_programs_reject_invalid_authored_values_and_preserve_legacy_serde() {
    let r = program::sample_with_floor(42, 1)
        .unwrap()
        .remove(Surface::Floor as usize);
    let mut f = r.flooring.clone().unwrap();
    f.stock.as_mut().unwrap().cut_jitter[0] = f32::NAN;
    assert!(f.validate().is_err());
    let mut dye = r.carpet.as_ref().unwrap().dye.clone().unwrap();
    dye.weights = [0.; 6];
    assert!(dye.validate().is_err());
    let mut legacy = r.clone();
    legacy.flooring.as_mut().unwrap().stock = None;
    legacy.carpet.as_mut().unwrap().dye = None;
    let json = serde_json::to_value(&legacy).unwrap();
    assert!(json["flooring"].get("stock").is_none());
    assert!(json["carpet"].get("dye").is_none());
    assert_eq!(legacy, serde_json::from_value(json).unwrap());
}

#[test]
fn continuous_carpet_dye_changes_content_at_fixed_seed() {
    let r = program::sample_with_floor(9, 1)
        .unwrap()
        .remove(Surface::Floor as usize);
    let mut signatures = BTreeSet::new();
    // Six process poles and six intermediate compositions: the same geometry,
    // palette, phase and seed must visibly respond to authored dye parameters.
    for i in 0..12 {
        let mut r = r.clone();
        let d = r.carpet.as_mut().unwrap().dye.as_mut().unwrap();
        d.weights = [0.; 6];
        d.weights[i % 6] = if i < 6 { 1. } else { 0.5 };
        if i >= 6 {
            d.weights[(i + 1) % 6] = 0.5;
        }
        let mut signature = Vec::new();
        for y in 0..24 {
            for x in 0..24 {
                let t = r.texel((x as f32 + 0.13) / 24., (y as f32 + 0.31) / 24., 1);
                signature.extend(t.color.map(|v| (v * 255.).round() as u8));
            }
        }
        assert!(
            signatures.insert(signature),
            "duplicate dye intervention {i}"
        );
    }
}
