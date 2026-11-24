use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeMap;

fn test_frame(width: f32, height: f32, roundness: f32) -> Frame {
    Frame {
        width,
        height,
        roundness,
        curvature: 0.065,
        lumbar: 0.036,
        transform: Transform::IDENTITY,
    }
}

// Independent orthographic ray intersections through the triangulated stock,
// including the rear skin. Samples on a shared triangle edge count once.
fn intersections(g: &Geometry, xy: Vec2) -> Vec<f32> {
    let mut hits = Vec::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        let (ab, ac, ap) = (b - a, c - a, xy - a.truncate());
        let det = ab.truncate().perp_dot(ac.truncate());
        if det.abs() < 1.0e-10 {
            continue;
        }
        let u = ap.perp_dot(ac.truncate()) / det;
        let v = ab.truncate().perp_dot(ap) / det;
        if u >= 0. && v >= 0. && u + v <= 1. {
            hits.push(a.z + u * ab.z + v * ac.z);
        }
    }
    hits.sort_by(f32::total_cmp);
    hits.dedup_by(|a, b| (*a - *b).abs() < 0.00001);
    hits
}

fn check_stock(g: &Geometry) {
    let mut edge_use = BTreeMap::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        let face = (p[1] - p[0]).cross(p[2] - p[0]);
        let normal: Vec3 = tri
            .iter()
            .map(|&i| Vec3::from_array(g.normals[i as usize]))
            .sum();
        assert!(face.dot(normal) > 0.);
        for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
            let (a, b) = (
                a.to_array().map(f32::to_bits),
                b.to_array().map(f32::to_bits),
            );
            let (count, balance) = edge_use.entry((a.min(b), a.max(b))).or_insert((0_u8, 0_i8));
            *count += 1;
            *balance += if a < b { 1 } else { -1 };
        }
    }
    assert!(edge_use.values().all(|&edge| edge == (2, 0)));
    assert!(g
        .normals
        .iter()
        .all(|n| (Vec3::from(*n).length() - 1.).abs() < 1.0e-5));
    assert!(g.uvs.iter().all(|p| Vec2::from(*p).is_finite()));
}

#[test]
fn fitted_sling_has_two_closed_skins_and_covers_the_opening_up_to_its_rim() {
    for (height, roundness) in [(0.12, 6.8), (0.34, 4.3), (0.67, 2.4)] {
        let f = test_frame(0.42, height, roundness);
        let mut g = Geometry::default();
        cloth::sling(&mut g, &f, 0.0045, 0.014);
        check_stock(&g);
        // Ignore the narrowest 8.5 mm frame and probe up to its inner edge all around.
        for r in 0..=25 {
            let y = 0.0085 + (height - 0.017) * r as f32 / 25.;
            let extent = f.width * 0.5 * f.chord(2. * y / height - 1.) - 0.0085;
            for c in 0..=20 {
                let x = (c as f32 / 10. - 1.) * extent;
                let hits = intersections(&g, Vec2::new(x, y));
                assert_eq!(hits.len(), 2, "x={x} y={y} h={height}");
                assert!(hits[1] - hits[0] >= 0.0038);
            }
        }
    }
}

#[test]
fn webbing_has_metric_stock_and_clear_openings_along_both_axes() {
    for direction in [WebbingDirection::Horizontal, WebbingDirection::Vertical] {
        for (height, roundness) in [(0.12, 6.8), (0.40, 2.5), (0.69, 4.8)] {
            let f = test_frame(0.48, height, roundness);
            let mut g = Geometry::default();
            let width = 0.69;
            let pitch = 0.084;
            cloth::webbing(&mut g, &f, direction, pitch, width, 0.003, 0.01);
            check_stock(&g);
            let horizontal = direction == WebbingDirection::Horizontal;
            let count = ((if horizontal { height } else { f.width }) / pitch)
                .round()
                .clamp(2., 18.) as usize;
            for i in 0..count {
                for (offset, expected) in [(0., 2), (width * 0.4, 2), (0.47, 0)] {
                    let t = (i as f32 + 0.5 + offset) / count as f32;
                    let xy = if horizontal {
                        Vec2::new(0.019, t * height)
                    } else {
                        Vec2::new((t - 0.5) * f.width, height * 0.47)
                    };
                    assert_eq!(
                        intersections(&g, xy).len(),
                        expected,
                        "{direction:?}, h={height}, t={t}"
                    );
                }
            }
        }
    }
}

#[test]
fn selection_replays_archives_and_is_independent_of_finish_and_placement() {
    let mut counts = [0; 4];
    for seed in 0..1024 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 1;
        o.interaction_target = None;
        let p = parameters(&o);
        if let Some(stock) = p.tension_back(&o, TEXTILE_VERSION) {
            assert_eq!(
                p.tension_back(&o, TEXTILE_VERSION - 1)
                    .map(|back| back.infill),
                Some(TensionInfill::OpenCord)
            );
            o.finish_seed = Some(!seed);
            o.position += Vec3::new(1., 0., -4.);
            o.yaw += 0.7;
            assert_eq!(
                parameters(&o).tension_back(&o, TEXTILE_VERSION),
                Some(stock)
            );
            let bin = match stock.infill {
                TensionInfill::OpenCord => 0,
                TensionInfill::WovenSling { .. } => 1,
                TensionInfill::Webbing {
                    direction: WebbingDirection::Horizontal,
                    ..
                } => 2,
                TensionInfill::Webbing {
                    direction: WebbingDirection::Vertical,
                    ..
                } => 3,
            };
            counts[bin] += 1;
        }
    }
    assert!(counts.iter().all(|&n| n > 12), "{counts:?}");
}

#[test]
fn coordinated_chairs_share_an_order_and_replacements_keep_their_own_stock() {
    let mut order = None;
    let mut matched = 0;
    let mut replacements = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 1;
        o.finish_seed = Some(877);
        o.interaction_target = Some(12);
        let mut p = parameters(&o);
        p.back_construction = 0;
        let fitted = p.tension_back(&o, TEXTILE_VERSION).unwrap();
        if fitted.table_order {
            let cut = (fitted.infill, fitted.rail_radius_m);
            assert_eq!(*order.get_or_insert(cut), cut);
            matched += 1;
        } else {
            o.interaction_target = None;
            assert_eq!(p.tension_back(&o, TEXTILE_VERSION), Some(fitted));
            replacements += 1;
        }
    }
    assert!((167..225).contains(&matched) && replacements > 30);
}

#[test]
fn sling_and_straps_remain_on_the_swivelling_upper_with_the_static_rest_geometry() {
    use crate::scene::procedural_indoor::{motion::IndoorMotionConfig, objects};
    let policy = IndoorMotionConfig {
        chair_fraction: 1.,
        ..default()
    };
    let mut checked = [false; 3];
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 0;
        let Some(stock) = parameters(&o).tension_back(&o, TEXTILE_VERSION) else {
            continue;
        };
        let bin = match stock.infill {
            TensionInfill::OpenCord => continue,
            TensionInfill::WovenSling { .. } => 0,
            TensionInfill::Webbing {
                direction: WebbingDirection::Horizontal,
                ..
            } => 1,
            TensionInfill::Webbing {
                direction: WebbingDirection::Vertical,
                ..
            } => 2,
        };
        if checked[bin] {
            continue;
        }
        let still = objects::build_object(&o);
        let moving = objects::build_object_with_motion(&o, Some(&policy));
        let upper = moving.joints.iter().find(|j| j.angle != 0.).unwrap();
        let above_seat = |parts: Vec<&Geometry>| {
            let mut points: Vec<_> = parts
                .into_iter()
                .flat_map(|g| &g.positions)
                .filter(|p| p[1] > 0.70)
                .map(|p| p.map(f32::to_bits))
                .collect();
            points.sort_unstable();
            points
        };
        let expected = above_seat(still.parts.values().collect());
        assert!(!expected.is_empty());
        assert_eq!(
            above_seat(upper.parts.iter().map(|k| &moving.parts[k]).collect()),
            expected
        );
        checked[bin] = true;
        if checked == [true; 3] {
            break;
        }
    }
    assert_eq!(checked, [true; 3]);
}
