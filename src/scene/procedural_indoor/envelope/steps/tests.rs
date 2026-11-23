use super::*;
use crate::scene::procedural_indoor::{envelope::construction, layout::IndoorLayout};

/// A long tier meets two different neighbors. The ray hits beside the
/// midpoint, exactly where adding a riser would falsely occlude the aisle.
#[test]
fn ray_uses_the_adjacent_support_at_the_hit_point() {
    let mut room =
        IndoorManifest::generate_with_humans(75188, IndoorLayout::LectureHall, 0.65, 0, 0.)
            .unwrap();
    let e = room.envelope.as_mut().unwrap();
    e.footprint = vec![
        Vec2::splat(-5.),
        Vec2::new(5., -5.),
        Vec2::splat(5.),
        Vec2::new(-5., 5.),
    ];
    let patch = |x0, x1, z0, z1, height| FloorPatch {
        min: Vec2::new(x0, z0),
        max: Vec2::new(x1, z1),
        height,
    };
    e.soffits.clear();
    e.floor_patches = vec![patch(-2., 0., -2., 2., 0.12), patch(0., 2., -1., 1., 0.32)];
    let trace = |z| {
        e.ray_hit(
            room.room_size,
            0.,
            1.,
            2.,
            Vec3::new(-0.5, 0.20, z),
            Vec3::X,
        )
        .unwrap()
    };
    // Both rays start in air above the low tier. Only the center ray hits
    // the high one. At either end the next visible surface is the far wall.
    for z in [-1.5, 1.5] {
        let (distance, label) = trace(z);
        assert!((distance - 5.5).abs() < 1e-5 && label == "wall");
    }
    let (distance, label) = trace(0.);
    assert!((distance - 0.5).abs() < 1e-5 && label == "floor");
}

#[test]
fn fitted_aisles_preserve_the_mesh_support_and_floor_class() {
    let mut room =
        IndoorManifest::generate_with_humans(75188, IndoorLayout::LectureHall, 0.65, 0, 0.)
            .unwrap();
    let e = room.envelope.as_mut().unwrap();
    let steps = e.step_installation.as_mut().unwrap();
    steps.nosing = StepNosing::MetalWithInsert;
    let edges = steps.walk_edges.clone();
    let depth = steps.tread_band_m;
    let trim = steps.insert_margin_m;
    assert!(steps.passage.is_none() && edges.len() >= 3);
    let mesh = construction::build(&room);
    let e = room.envelope.as_ref().unwrap();
    assert!(e.step_installation.as_ref().unwrap().valid(e));
    let samples = |x: f32, z: f32, y: f32| {
        let p = Vec2::new(x, z);
        let mut result = Vec::new();
        for ((surface, label), g) in &mesh.parts {
            if label != "floor" {
                continue;
            }
            for t in g.indices.as_chunks::<3>().0 {
                let v = [0, 1, 2].map(|i| Vec3::from_array(g.positions[t[i] as usize]));
                if v.iter().any(|v| (v.y - y).abs() > 1e-5) {
                    continue;
                }
                let sides =
                    [0, 1, 2].map(|i| (v[(i + 1) % 3].xz() - v[i].xz()).perp_dot(p - v[i].xz()));
                if sides.iter().all(|&s| s < -1e-8) {
                    result.push(*surface);
                }
            }
        }
        result
    };
    for edge in edges {
        let x = edge.span_m.x + 0.373 * (edge.span_m.y - edge.span_m.x);
        let z = edge.coordinate_m;
        let sign = (e.floor_height(Vec2::new(x, z + 0.001))
            - e.floor_height(Vec2::new(x, z - 0.001)))
        .signum();
        for (offset, stock) in [
            (trim * 0.41, Surface::Metal),
            (depth * 0.46, Surface::Rubber),
            (depth - trim * 0.43, Surface::Metal),
            (depth + 0.012, Surface::Floor),
        ] {
            let at = z + sign * offset;
            assert_eq!(
                samples(x, at, e.floor_height(Vec2::new(x, at))),
                vec![stock],
                "edge z={z}, offset={offset}"
            );
        }
    }
}

#[test]
fn plain_archive_keeps_the_original_floor_and_round_trips() {
    let room = IndoorManifest::generate_with_humans(75188, IndoorLayout::LectureHall, 0.65, 0, 0.)
        .unwrap();
    let e = room.envelope.unwrap();
    let mut value = serde_json::to_value(&e).unwrap();
    let decoded: EnvelopeProgram = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(decoded, e);
    value.as_object_mut().unwrap().remove("step_installation");
    let old: EnvelopeProgram = serde_json::from_value(value).unwrap();
    assert!(old.step_installation.is_none());
}

#[test]
fn short_flights_have_four_clear_entrances_and_usable_landings() {
    use crate::scene::procedural_indoor::{footprint::Footprint, layout::ObjectKind};
    let mut directions = [0_u16; 4];
    let mut levels = [false; 2];
    for seed in 100..180 {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.60, 0, 0.)
                .unwrap();
        let e = room.envelope.as_ref().unwrap();
        let Some(s) = &e.step_installation else {
            continue;
        };
        let passage = s.passage.unwrap();
        let k = s.run_axis.index();
        let center = |p: &FloorPatch| (p.min[k] + p.max[k]) * 0.5;
        let entering_positive = center(&e.floor_patches[0]) > center(&e.floor_patches[1]);
        directions[k * 2 + usize::from(entering_positive)] += 1;
        levels[usize::from(e.floor_patches[0].height > 0.)] = true;
        assert!(s.valid(e));
        let path = Footprint::oriented(passage.center(), passage.size(), 0.);
        assert!(room
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor && o.position.y < 1.)
            .all(|o| !Footprint::object(o).overlaps(path, 0.)));
        assert!(e.pillars.iter().all(|p| {
            let pad = Vec2::splat(p.radius);
            !passage.min.cmplt(p.center + pad).all() || !passage.max.cmpgt(p.center - pad).all()
        }));
        let Some(table) = room.objects.iter().find(|o| o.solid && !o.neighbor) else {
            continue;
        };
        let mut probe = table.clone();
        probe.kind = ObjectKind::Desk;
        probe.size = Vec3::new(0.25, 0.8, 0.25);
        let top = &e.floor_patches[0];
        let offset = if entering_positive {
            top.min[k] + 0.24
        } else {
            top.max[k] - 0.24
        };
        let mut landing = passage.center();
        landing[k] = offset;
        probe.position = Vec3::new(landing.x, top.height, landing.y);
        let (lo, hi) = probe.bounds();
        assert!(e.support_clear(lo, hi));
        assert!(
            !room.placement_clear(&probe, 0.),
            "seed {seed}: furniture obstructs the landing"
        );
    }
    assert!(levels.into_iter().all(|x| x));
    assert!(directions.into_iter().all(|n| n > 0), "{directions:?}");
}

#[test]
fn overlong_walk_edges_and_invalid_stock_cannot_enter_the_mesh() {
    let room = IndoorManifest::generate_with_humans(75188, IndoorLayout::LectureHall, 0.65, 0, 0.)
        .unwrap();
    let e = room.envelope.unwrap();
    let mut s = e.step_installation.clone().unwrap();
    s.walk_edges[0].span_m = Vec2::new(-30., 30.);
    assert!(!s.valid(&e));
    let mut s = e.step_installation.clone().unwrap();
    s.riser_surface = Surface::Glass;
    assert!(!s.valid(&e));
    s.riser_surface = Surface::Wood;
    s.tread_band_m = f32::NAN;
    assert!(!s.valid(&e));
}
