use super::*;
use crate::scene::procedural_indoor::{
    materials,
    objects::{self, audit},
};
use std::collections::BTreeSet;

// Intersections of a Z ray with the actual emitted triangles. Deduplicate shared edges.
fn hits(g: &Geometry, x: f32, y: f32) -> Vec<f32> {
    let mut depths = Vec::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
        let (u, v) = ((b - a).xy(), (c - a).xy());
        let det = u.perp_dot(v);
        if det.abs() < 1e-11 {
            continue;
        }
        let q = Vec2::new(x, y) - a.xy();
        let (s, t) = (q.perp_dot(v) / det, u.perp_dot(q) / det);
        if s >= -1e-5 && t >= -1e-5 && s + t <= 1.00001 {
            depths.push(a.z + s * (b.z - a.z) + t * (c.z - a.z));
        }
    }
    depths.sort_unstable_by(f32::total_cmp);
    depths.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
    depths
}

fn part<'a>(a: &'a Assembly, surface: Surface, label: &str) -> &'a Geometry {
    a.parts
        .iter()
        .find(|((s, l), _)| {
            *s == surface && (l == label || !label.contains('#') && objects::part_label(l) == label)
        })
        .unwrap()
        .1
}

#[test]
fn stamped_headers_tube_openings_and_supply_return_ports_are_real_surfaces() {
    let mut found = BTreeSet::new();
    for seed in 0..160 {
        let o = audit::fixture(ObjectKind::BaseboardHeater, seed);
        let stock = Installation::for_object(&o);
        if !found.insert(stock.emitter.name()) {
            continue;
        }
        let a = objects::build_object(&o);
        let g = part(&a, Surface::Metal, &stock.casing_label());
        let b = match stock.emitter {
            Emitter::Pressed(p) => {
                let x = -p.body.width_m * 0.5 + p.margin_m + p.pitch_m * (p.channels / 2) as f32;
                let y = (p.body.bottom_m + p.body.top_m) * 0.5;
                let valley = hits(g, x, y);
                let crown = hits(g, x + p.pitch_m * 0.52, y);
                assert_eq!(valley.len(), if p.bank_spacing_m.is_some() { 4 } else { 2 });
                assert_eq!(valley.len(), crown.len());
                assert!((crown.last().unwrap() - valley.last().unwrap() - p.relief_m).abs() < 1e-5);
                let header = p.body.bottom_m + 0.62 * p.header_m;
                assert!(
                    (hits(g, x, header).last().unwrap()
                        - hits(g, x + 0.52 * p.pitch_m, header).last().unwrap())
                    .abs()
                        < 1e-5
                );
                if let Some(spacing) = p.bank_spacing_m {
                    assert!((valley[2] - valley[1] - (spacing - p.web_depth_m)).abs() < 1e-5);
                }
                p.body
            }
            Emitter::Tubular(t) => {
                for i in 0..t.rows {
                    let y = t.body.bottom_m + (i as f32 + 0.5) * t.pitch_m;
                    let surfaces = hits(g, 0.11 * t.body.width_m, y);
                    assert_eq!(surfaces.len(), 2);
                    assert!((surfaces[1] - surfaces[0] - 2. * t.tube_radii_m.y).abs() < 1e-5);
                    if i + 1 < t.rows {
                        assert!(hits(g, 0.11 * t.body.width_m, y + t.pitch_m * 0.5).is_empty());
                    }
                }
                t.body
            }
            Emitter::Convector { hydronic } => {
                if hydronic {
                    let x = o.size.x * 0.32;
                    let y = o.size.y * 0.255;
                    let metal = a
                        .parts
                        .iter()
                        .find(|((s, l), _)| *s == Surface::Metal && l != &stock.casing_label())
                        .unwrap()
                        .1;
                    let exchanger = hits(metal, x, y);
                    let pipe = hits(part(&a, Surface::Chrome, "appliance"), x, y);
                    assert!(
                        exchanger.len() >= 2
                            && pipe.len() >= 2
                            && pipe[0] > exchanger[0]
                            && pipe[1] < *exchanger.last().unwrap()
                    );
                }
                continue;
            }
        };
        // Locate each real pipe end 4 mm inside the side header. Its axis
        // must fall inside the measured section of this unit's water panel.
        let plumbing = part(&a, Surface::Chrome, "appliance");
        for sign in [-1., 1.] {
            let x = sign * (b.width_m * 0.5 - 0.004);
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = -lo;
            for v in plumbing
                .positions
                .iter()
                .copied()
                .map(Vec3::from)
                .filter(|p| (p.x - x).abs() < 1e-5)
            {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            assert!(lo.is_finite() && (hi.y - lo.y) > 0.006);
            let center = (lo + hi) * 0.5;
            let shell = hits(g, x, center.y);
            assert!(shell
                .as_chunks::<2>()
                .0
                .iter()
                .any(|pair| center.z > pair[0] && center.z < pair[1]));
        }
    }
    assert_eq!(found.len(), 5);
}

#[test]
fn stock_fits_extreme_bays_and_finish_edits_preserve_geometry() {
    let mut forms = BTreeSet::new();
    for seed in 0..80 {
        let mut o = audit::fixture(ObjectKind::BaseboardHeater, seed);
        for size in [
            Vec3::new(0.55, 0.13, 0.075),
            Vec3::new(1.2, 0.17, 0.099),
            Vec3::new(3.2, 0.46, 0.19),
        ] {
            o.size = size;
            let stock = Installation::for_object(&o);
            let a = objects::build_object(&o);
            let stats = audit::check(&o, &a).unwrap();
            assert!(
                stats.max_envelope_overrun_m < 1e-5 && stats.min_y_m.abs() < 1e-5,
                "{seed} {size:?} {stats:?}"
            );
            assert!(stats.triangles < 8000 && a.parts.len() <= 4);
            assert!(a
                .parts
                .keys()
                .all(|(s, l)| *s != Surface::Paint && objects::part_label(l) == "appliance"));
            let label = stock.casing_label();
            assert!(materials::finish_slot(Surface::Metal, &label)
                .is_some_and(|s| materials::registered_finish(Surface::Metal, s)));
            let mut other = o.clone();
            other.finish_seed = Some(!seed);
            let edited = Installation::for_object(&other);
            assert_eq!(
                serde_json::to_value(stock.emitter).unwrap(),
                serde_json::to_value(edited.emitter).unwrap()
            );
            let new = objects::build_object(&other);
            let (old, new) = (
                part(&a, Surface::Metal, &label),
                part(&new, Surface::Metal, &edited.casing_label()),
            );
            assert_eq!(old.positions, new.positions);
            assert_eq!(old.indices, new.indices);
            forms.insert(stock.emitter.name());
        }
    }
    assert_eq!(forms.len(), 5);
}

#[test]
fn shared_heating_series_preserves_archive_and_individual_dimensions() {
    let series = (0..256).find_map(HeatingSeries::for_room).unwrap();
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::BaseboardHeater, seed);
        o.finish_seed = None;
        let old = objects::build_object_at_version(&o, None, FITTED_HEATING_VERSION - 1);
        let base_variant = o.variant;
        series.install(&mut o);
        assert_eq!(o.variant & 0xff, base_variant);
        let saved = serde_json::to_string(&o).unwrap();
        let replay: IndoorObject = serde_json::from_str(&saved).unwrap();
        let retained = objects::build_object_at_version(&replay, None, FITTED_HEATING_VERSION - 1);
        assert_eq!(old.parts.len(), retained.parts.len());
        for (k, v) in old.parts {
            let copy = &retained.parts[&k];
            assert_eq!(copy.positions, v.positions);
            assert_eq!(copy.normals, v.normals);
            assert_eq!(copy.uvs, v.uvs);
            assert_eq!(copy.indices, v.indices);
        }
        assert_eq!(
            serde_json::to_value(Installation::for_object(&o)).unwrap(),
            serde_json::to_value(Installation::for_object(&replay)).unwrap()
        );
    }
}
