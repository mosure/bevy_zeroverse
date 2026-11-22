use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn approaches_land_on_thresholds_and_clear_the_real_kerb_and_railing() {
    let room =
        IndoorManifest::generate_with_humans(44214, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let width = District::sample(room.seed).paving_width;
    let sides = strips(&room, width);
    let lip = sides[2].lip();
    let entry = Entrance {
        threshold: lip + Vec2::new(0.23, -3.5),
        outward: Vec2::from_angle(0.13).rotate(Vec2::Y),
        width: 1.14,
    };
    let site = Parcel {
        center: entry.threshold - entry.outward * 1.25,
        size: Vec2::new(3., 2.5),
        yaw: entry.outward.x.atan2(entry.outward.y),
        kind: 0,
        seed: 88,
    };
    let paths = access::plan(&sides, std::slice::from_ref(&site), &[], &[(0, entry)]);
    let path = &paths[0];
    assert_eq!(paths.len(), 1);
    assert_eq!(path.side, 2);
    assert!(((path.start[0] + path.start[1]) * 0.5).distance(entry.threshold) < 0.004);
    assert!(path
        .end
        .iter()
        .all(|v| ((v - lip).dot(sides[2].outward)).abs() < 1e-5));
    let mut assembly = Assembly::default();
    build(
        &mut assembly,
        &room,
        std::slice::from_ref(&site),
        &[],
        &[(0, entry)],
    );
    super::super::qualification::check(&assembly);
    for ((_, label), g) in &assembly.parts {
        if label != "other_structure" {
            continue;
        }
        for triangle in g.indices.as_chunks::<3>().0 {
            let v = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
            if v.iter().all(|p| (p.z - lip.y).abs() < 0.20) {
                assert!(
                    v.iter().all(|p| p.x <= lip.x + path.opening[0] + 0.021)
                        || v.iter().all(|p| p.x >= lip.x + path.opening[1] - 0.021),
                    "path blocked by {label}"
                );
            }
        }
    }
    let obstacle = Parcel {
        center: lip + Vec2::new(0., -1.65),
        size: Vec2::new(9., 1.50),
        yaw: 0.,
        kind: 1,
        seed: 89,
    };
    assert!(access::plan(&sides, &[site, obstacle], &[], &[(0, entry)]).is_empty());
    assert_eq!(
        access::intervals(0., 4., &[[-3., 1.], [0.5, 1.5], [2., 3.]]).collect::<Vec<_>>(),
        [(1.5, 2.), (3., 4.)]
    );
}

#[test]
fn walks_use_emitted_exterior_doors_across_a_population() {
    let mut connected = 0;
    for seed in (57300..57332).chain([44214, 57496]) {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let windows = window_rays(&room);
        let sites = parcels_at(&room, &windows);
        let rows = midground::plan(&room, &windows, &sites);
        let entries: Vec<_> = sites
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == 0)
            .filter_map(|(i, p)| {
                let mut a = Assembly::default();
                let b = Building::sample(p, super::super::super::background::radius(&room));
                let planned = buildings::main_entrance(p, &b);
                let emitted = buildings::build(&mut a, p, &b, None);
                assert_eq!(planned.is_some(), emitted.is_some());
                let entry = emitted?;
                let forecast = planned.unwrap();
                assert!((entry.width - forecast.width).abs() < 1e-5);
                assert!(entry.outward.distance(forecast.outward) < 1e-5);
                assert!(entry.threshold.distance(forecast.threshold) < 1e-5);
                let has_frame = a.parts.keys().any(|(_, l)| l.starts_with("door#"));
                assert!(has_frame);
                Some((i, entry))
            })
            .collect();
        let sides = strips(&room, District::sample(seed).paving_width);
        let walks = access::plan(&sides, &sites, &rows, &entries);
        connected += usize::from(!walks.is_empty());
        assert!(walks.len() <= 5);
        for (i, walk) in walks.iter().enumerate() {
            assert!(walks[..i].iter().all(|p| !walk.span.overlaps(p.span, 0.09)));
            assert!(rows
                .iter()
                .all(|p| !walk.span.overlaps(p.footprint(), 0.11)));
        }
    }
    assert!(
        connected >= 18,
        "only {connected} of 34 rooms had a feasible door approach"
    );
}
