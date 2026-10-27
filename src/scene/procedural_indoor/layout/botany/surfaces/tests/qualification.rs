//! Optional local geometry review produced by the same bounded room test.
use super::*;
use crate::scene::procedural_indoor::{geometry::preview, materials::Surface};
use image::{Rgb, RgbImage};
use serde_json::json;

pub(super) fn save(
    rows: &[serde_json::Value],
    supports: &BTreeMap<String, usize>,
    elevated_rooms: usize,
    timings: &[f64],
) {
    let Ok(output) = std::env::var("ZEROVERSE_PLANT_REPORT") else {
        return;
    };
    let output = std::path::Path::new(&output);
    std::fs::create_dir_all(output).unwrap();
    let quantiles = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        [
            v[0],
            v[v.len() / 10],
            v[v.len() / 2],
            v[v.len() * 9 / 10],
            v[v.len() - 1],
        ]
    };
    let mut numeric = BTreeMap::new();
    for key in ["pot_radius_m", "pot_height_m", "pot_height_to_diameter"] {
        numeric.insert(
            key,
            quantiles(rows.iter().map(|r| r[key].as_f64().unwrap()).collect()),
        );
    }
    let report = json!({"provenance":crate::provenance::capture_provenance(),
        "protocol":"64 consecutive Mixed rooms, seeds 0..64, density 0.65, default clutter distribution, no humans/cameras; actual plant mesh audits and full layout validation. CPU generation timings exclude mesh construction and rendering. Geometry previews use flat CPU shading, not PBR or photographic-realism evidence.",
        "rooms":64,"rooms_with_elevated_plants":elevated_rooms,"plants":rows.len(),"plant_support_counts":supports,
        "quantile_order":["minimum","p10","median","p90","maximum"],"numeric":numeric,
        "manifest_cpu_ms":quantiles(timings.to_vec()),"samples":rows});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    let n = 192;
    let mut pots = RgbImage::from_pixel(n * 8, n * 2, Rgb([241, 243, 246]));
    let mut plants_grid = RgbImage::from_pixel(n * 8, n * 4, Rgb([241, 243, 246]));
    for seed in 0..16 {
        let mut o = audit::fixture(ObjectKind::Plant, seed);
        o.size = Vec3::new(0.9, 1.7, 0.9);
        o.variant = 0;
        let mut a = objects::build_object(&o);
        a.parts.retain(|(surface, _), _| {
            !matches!(
                surface,
                Surface::Bark | Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
            )
        });
        let tile = preview::orthographic(&a, Vec3::Y * 0.40, 1.10, n, Vec3::new(0.7, 0.38, 1.));
        image::imageops::replace(
            &mut pots,
            &tile,
            (seed % 8 * n as u64) as i64,
            (seed / 8 * n as u64) as i64,
        );
    }
    for (row, variant) in [0, 4, 7, 11].into_iter().enumerate() {
        for seed in 0..8 {
            let mut o = audit::fixture(ObjectKind::Plant, seed);
            o.size = Vec3::new(
                0.9,
                if variant == 7 {
                    0.55
                } else if variant == 4 {
                    0.9
                } else {
                    1.7
                },
                0.9,
            );
            o.variant = variant;
            let a = objects::build_object(&o);
            let tile = preview::orthographic(
                &a,
                Vec3::Y * o.size.y * 0.5,
                o.size.max_element() * 1.13,
                n,
                Vec3::new(0.65, 0.35, 1.),
            );
            image::imageops::replace(
                &mut plants_grid,
                &tile,
                (seed * n as u64) as i64,
                row as i64 * n as i64,
            );
        }
    }
    pots.save(output.join("pots.png")).unwrap();
    plants_grid.save(output.join("plants.png")).unwrap();
}
