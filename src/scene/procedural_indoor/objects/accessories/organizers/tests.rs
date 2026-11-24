use super::*;
#[test]
fn printed_content_rotates_with_its_supported_page() {
    let mut rotated = 0;
    for seed in 0..32 {
        let o = crate::scene::procedural_indoor::objects::audit::fixture(
            ObjectKind::DocumentTray,
            seed,
        );
        let a = crate::scene::procedural_indoor::objects::build_object(&o);
        for ((surface, _), g) in &a.parts {
            if *surface != Surface::PrintedPaper {
                continue;
            }
            assert!(g
                .uvs
                .iter()
                .flatten()
                .all(|&u| (-0.00001..=1.00001).contains(&u)));
            for tri in g.indices.as_chunks::<3>().0 {
                if tri.iter().any(|&i| g.normals[i as usize][1] < 0.99) {
                    continue;
                }
                let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let uv = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                let du = uv[1] - uv[0];
                let dv = uv[2] - uv[0];
                let determinant = du.perp_dot(dv);
                if determinant.abs() < 1e-8 {
                    continue;
                }
                let tangent =
                    (((p[1] - p[0]) * dv.y - (p[2] - p[0]) * du.y) / determinant).normalize();
                rotated += usize::from(tangent.z.abs() > 0.01);
                break;
            }
        }
    }
    assert!(
        rotated > 20,
        "page content snapped back to room axes: {rotated}"
    );
}

use crate::scene::procedural_indoor::objects::{audit, build_object_at_version};
use std::collections::BTreeSet;

#[test]
fn finished_stock_fits_on_the_bed_and_only_the_recto_carries_a_page_uv_chart() {
    let mut media = BTreeSet::new();
    let mut print_faces = [0; 2];
    let mut piles = [0; 3];
    let mut lifted = 0;
    let mut tools = 0;
    let mut empty = 0;
    for kind in [ObjectKind::DocumentTray, ObjectKind::DeskOrganizer] {
        for seed in 0..192 {
            let mut o = audit::fixture(kind, seed);
            let mut sizes = stream(seed, 13);
            o.size *= Vec3::new(
                sizes.random_range(0.86..1.06),
                sizes.random_range(0.69..1.17),
                sizes.random_range(0.86..1.06),
            );
            let p = parameters(&o);
            for b in p.bays(&o, INTRODUCED) {
                let stack = match contents(&o, &p, &b) {
                    Contents::Paper(s) => s,
                    Contents::Tools { .. } => {
                        tools += 1;
                        continue;
                    }
                    Contents::Empty => {
                        empty += 1;
                        continue;
                    }
                };
                let mut a = Assembly::default();
                stack.build(&mut a);
                let check = audit::check(&o, &a).unwrap();
                assert!(check.triangles <= 52, "{check:?}");
                let (lo, hi) = a.bounds();
                assert!((lo.y - (b.base + b.t)).abs() < 1e-6, "floating stock");
                assert!(
                    hi.y <= b.base + b.height
                        && lo.x > b.x - b.inner.x * 0.5
                        && hi.x < b.x + b.inner.x * 0.5
                );
                assert!(lo.z > -b.inner.y * 0.5 && hi.z < b.inner.y * 0.5);
                media.insert(stack.media);
                print_faces[stack.printed_face_up as usize] += 1;
                piles[if stack.sheets == 1 {
                    0
                } else if stack.sheets < 16 {
                    1
                } else {
                    2
                }] += 1;
                lifted += usize::from(stack.curl_m > 0.001);
                let stock = &a.parts[&(Surface::Paper, "paper".to_owned())];
                assert!(stock.normals.iter().any(|n| n[1].abs() < 0.65));
                let ink = a
                    .parts
                    .iter()
                    .find(|((s, _), _)| *s == Surface::PrintedPaper);
                assert_eq!(ink.is_some(), stack.printed_face_up);
                if let Some(((_, label), g)) = ink {
                    assert!(label.starts_with("paper#") && label.ends_with("#uv_local"));
                    let mut uv_area = 0.;
                    for tri in g.indices.as_chunks::<3>().0 {
                        assert!(tri.iter().all(|&i| g.normals[i as usize][1] > 0.66));
                        let [a, b, c] = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                        let area = 0.5 * (b - a).perp_dot(c - a).abs();
                        assert!(area > 0.001, "ink stretched down a cut edge");
                        uv_area += area;
                    }
                    assert!((uv_area - 1_f32).abs() < 1e-5);
                }
                if stack.sheets > 1 {
                    let mut heavy = stack.clone();
                    heavy.sheets = 128;
                    let mut other = Assembly::default();
                    heavy.build(&mut other);
                    assert_eq!(
                        other
                            .parts
                            .values()
                            .map(|g| g.indices.len() / 3)
                            .sum::<usize>(),
                        check.triangles
                    );
                }
            }
        }
    }
    assert!(media.len() >= 8 && print_faces.into_iter().all(|n| n > 35));
    assert!(piles.into_iter().all(|n| n > 45) && lifted > 100 && tools > 65 && empty > 120);
}

#[test]
fn tight_trays_keep_nominal_stock_and_do_not_shrink_oversize_pages_into_cards() {
    use crate::scene::procedural_indoor::objects::literature::media::A4;
    let o = audit::fixture(ObjectKind::DocumentTray, 18);
    let p = parameters(&o);
    let mut b = p.bays(&o, INTRODUCED).next().unwrap();
    b.inner = Vec2::new(0.016, 0.110);
    assert!(sheets::PaperStack::fit(&o, &b).is_none());
    let mut a4_fit = 0;
    b.inner = A4.1 + Vec2::splat(0.002);
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::DocumentTray, seed);
        let stack = sheets::PaperStack::fit(&o, &b).unwrap();
        if stack.media == A4.0 {
            assert_eq!(stack.sheet_m, A4.1);
            assert!(stack.yaw_rad.sin().abs() < 0.0025);
            a4_fit += 1;
        }
    }
    assert!(a4_fit > 15);
}

#[test]
fn compartment_tools_rest_on_the_bed_and_real_rim_inside_their_own_bay() {
    let mut ends = BTreeSet::new();
    let mut inclinations = BTreeSet::new();
    let mut fitted = 0;
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::DeskOrganizer, seed);
        let p = parameters(&o);
        for b in p.bays(&o, INTRODUCED) {
            let Contents::Tools { tools, bay_x_m } = contents(&o, &p, &b) else {
                continue;
            };
            assert!((1..=3).contains(&tools.len()));
            let clear = Vec2::new(b.width * 0.5 - 1.1 * b.t, b.depth * 0.5 - b.t);
            for tool in tools {
                ends.insert(tool.upper_end);
                inclinations.insert((tool.inclination_rad().to_degrees() / 3.) as usize);
                let mut a = Assembly::default();
                tool.draw(&mut a, o.kind.class_name(), Vec3::X * bay_x_m);
                audit::check(&o, &a).unwrap();
                let (lo, hi) = a.bounds();
                assert!((lo.y - b.t).abs() < 2e-6);
                assert!(lo.x >= bay_x_m - clear.x - 1e-6 && hi.x <= bay_x_m + clear.x + 1e-6);
                let mut closest = f32::INFINITY;
                for g in a.parts.values() {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let v = tri
                            .map(|i| Vec3::from_array(g.positions[i as usize]) - Vec3::X * bay_x_m);
                        for (a, c) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                            if a.y <= b.height {
                                assert!((a.xz().abs() - clear).max_element() < 1e-6);
                            }
                            if (a.y < b.height) != (c.y < b.height) {
                                let hit = a.lerp(c, (b.height - a.y) / (c.y - a.y));
                                let slack = clear - hit.xz().abs();
                                assert!(slack.min_element() > -1e-6);
                                closest = closest.min(slack.min_element().abs());
                            }
                        }
                    }
                }
                assert!(closest < 2e-5, "tool misses rim by {closest} m");
                fitted += 1;
            }
        }
    }
    assert!(fitted > 70 && ends.len() >= 4 && inclinations.len() >= 6);
}

#[test]
fn finish_edits_preserve_page_and_tool_poses_and_older_manifests_replay_the_box_stock() {
    let mut shell_finishes = BTreeSet::new();
    for kind in [ObjectKind::DocumentTray, ObjectKind::DeskOrganizer] {
        for seed in 0..32 {
            let mut o = audit::fixture(kind, seed);
            let pose = |o: &IndoorObject| {
                compartments(o, &parameters(o))
                    .into_iter()
                    .map(|b| match b {
                        Contents::Paper(s) => serde_json::json!(s),
                        Contents::Tools { tools, .. } => serde_json::json!(tools
                            .iter()
                            .map(|p| (p.base, p.top, p.contact, p.upper_end))
                            .collect::<Vec<_>>()),
                        Contents::Empty => serde_json::Value::Null,
                    })
                    .collect::<Vec<_>>()
            };
            let expected = pose(&o);
            o.finish_seed = Some(seed + 1471);
            assert_eq!(expected, pose(&o));
            shell_finishes.insert(parameters(&o).finish);
            let old = build_object_at_version(&o, None, INTRODUCED - 1);
            for ((surface, _), g) in &old.parts {
                if *surface == Surface::PrintedPaper {
                    assert!(g.normals.iter().any(|n| n[1] < -0.99));
                }
            }
            audit::check(&o, &build_object_at_version(&o, None, INTRODUCED)).unwrap();
        }
    }
    assert_eq!(shell_finishes.len(), 3);
}
