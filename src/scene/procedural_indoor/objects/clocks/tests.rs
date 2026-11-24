use super::*;
use std::collections::BTreeSet;

#[test]
fn clock_cases_and_relief_fit_varied_aspects_and_shallow_depths() {
    let mut depths = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for seed in 0..96 {
        for size in [
            Vec3::new(0.20, 0.20, 0.032),
            Vec3::new(0.72, 0.32, 0.040),
            Vec3::new(0.25, 0.40, 0.080),
            Vec3::new(0.46, 0.46, 0.032),
        ] {
            let mut o = audit::fixture(ObjectKind::Clock, seed);
            o.size = size;
            let p = parameters(&o);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            let (lo, hi) = a.bounds();
            assert!(lo
                .cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(1e-5))
                .all());
            assert!(hi
                .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(1e-5))
                .all());
            let (back, front) = case_planes(&o);
            assert!((lo.z - back).abs() < 1e-6);
            assert!((hi.z - front).abs() < 1e-6);
            let dial_half = size.xy() * 0.5 * (1. - p.bezel_fraction).min(0.98);
            // Hands turn inside the recessed aperture on separate spindle
            // levels. Printed marks and digital segments have no mesh relief.
            let printed = a
                .parts
                .iter()
                .find(|((_, label), _)| label.contains("#uv_local"))
                .unwrap()
                .1;
            let dial_z = printed.positions[0][2];
            let mut hands = 0;
            for ((surface, _), part) in &a.parts {
                if *surface != Surface::Ink {
                    continue;
                }
                hands += 1;
                for v in &part.positions {
                    let v = Vec3::from_array(*v);
                    assert!(v.z > dial_z && v.z < front);
                    let uv = (v.xy() - Vec2::Y * size.y * 0.5).abs() / dial_half;
                    assert!(
                        uv.x.powf(p.roundness) + uv.y.powf(p.roundness) < 1.001,
                        "unsupported hand {seed}/{size:?}: {v:?}"
                    );
                }
            }
            assert_eq!(hands == 0, p.digital);
            depths.insert(((front - back) * 10000.).round() as i32);
            shapes.insert((
                (p.roundness * 100.).round() as i32,
                (p.case_taper * 100.).round() as i32,
            ));
        }
    }
    assert!(
        depths.len() > 100 && shapes.len() > 60,
        "{} depths, {} profiles",
        depths.len(),
        shapes.len()
    );
}

#[test]
fn clock_metric_sampling_and_finish_identity_are_independent() {
    let mut ratios = BTreeSet::new();
    let mut depths = BTreeSet::new();
    for seed in 0..128 {
        let mut r = stream(seed, 99881);
        let size = dimensions(&mut r);
        assert!((0.20..0.46).contains(&size.y) && (0.032..0.080).contains(&size.z));
        ratios.insert((size.x / size.y * 100.).round() as i32);
        depths.insert((size.z * 10000.).round() as i32);
        let mut o = audit::fixture(ObjectKind::Clock, seed);
        o.size = size;
        let a = audit::descriptor(&o, &build_object(&o));
        o.finish_seed = Some(10000 + seed);
        assert_eq!(a, audit::descriptor(&o, &build_object(&o)));
        let angles = hand_angles(parameters(&o).seconds);
        assert!(angles
            .iter()
            .all(|a| a.is_finite() && (0.0..TAU).contains(a)));
    }
    assert!(ratios.len() > 35 && depths.len() > 90);
}

#[test]
fn room_time_survives_serialization_and_finish_resampling() {
    use crate::scene::procedural_indoor::layout::LightingMood;
    let mut hours = BTreeSet::new();
    for room in 0..128 {
        for mood in [LightingMood::Daylight, LightingMood::Evening] {
            let mut times = Vec::new();
            for seed in [11, 57, 93, 322] {
                let mut o = audit::fixture(ObjectKind::Clock, seed);
                install(&mut o, room, mood);
                let seconds = parameters(&o).seconds;
                let restored: IndoorObject =
                    serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
                assert_eq!(parameters(&restored).seconds, seconds);
                o.finish_seed = Some(994);
                assert_eq!(parameters(&o).seconds, seconds);
                times.push(seconds);
                hours.insert(seconds / 3600);
            }
            for t in &times {
                let d = t.abs_diff(times[0]);
                assert!(d.min(86400 - d) <= 16);
            }
        }
    }
    assert!(hours.len() >= 19);
}

#[test]
fn dial_coordinate_frame_is_upright_and_recessed_and_archive_clocks_replay() {
    let mut stocks = BTreeSet::new();
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::Clock, seed);
        let p = parameters(&o);
        let g = build_object(&o);
        let aperture = o.size.xy() * (0.5 * (1. - p.bezel_fraction));
        let face = &g.parts[&(
            Surface::Paper,
            crate::scene::procedural_indoor::materials::clock_face::face_label(&o),
        )];
        for (pos, uv) in face.positions.iter().zip(&face.uvs) {
            let expected =
                Vec2::splat(0.5) + (Vec2::new(pos[0], o.size.y * 0.5 - pos[1]) / (2. * aperture));
            assert!(expected.abs_diff_eq(Vec2::from_array(*uv), 1e-5));
            assert!(pos[2] < case_planes(&o).1 - 0.0003);
        }
        assert!(face.normals.iter().all(|v| v[2] > 0.9999));
        let triangles: usize = g.parts.values().map(|p| p.indices.len() / 3).sum();
        assert!(triangles < 850);
        stocks.insert(format!("{:?}", p.printed.unwrap().style));
        o.variant &= 7;
        assert!(parameters(&o).printed.is_none());
        let old = build_object(&o);
        assert!(old.parts.keys().all(|(_, l)| !l.contains("uv_local")));
        assert!(
            old.parts
                .values()
                .map(|g| g.indices.len() / 3)
                .sum::<usize>()
                > triangles
        );
    }
    assert_eq!(stocks.len(), 6);
}

#[test]
fn digital_hour_format_wraps_midnight_noon_and_suppresses_leading_zeros() {
    let mut dial = parameters(&audit::fixture(ObjectKind::Clock, 1))
        .printed
        .unwrap();
    dial.hour_24 = false;
    dial.leading_zero = false;
    for s in [0, 12 * 3600] {
        assert_eq!(dial.digits(s), [Some(1), Some(2), Some(0), Some(0)]);
    }
    assert_eq!(
        dial.digits(13 * 3600 + 7 * 60),
        [None, Some(1), Some(0), Some(7)]
    );
    dial.hour_24 = true;
    assert_eq!(
        dial.digits(23 * 3600 + 59 * 60),
        [Some(2), Some(3), Some(5), Some(9)]
    );
    dial.leading_zero = true;
    assert_eq!(dial.digits(5 * 3600), [Some(0), Some(5), Some(0), Some(0)]);
}

#[test]
fn thin_stamped_hands_fit_the_clock_from_room_93496() {
    use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
    let room =
        IndoorManifest::generate_with_humans(93496, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    let clock = room
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::Clock && !parameters(o).digital)
        .unwrap();
    let (back, front) = case_planes(clock);
    assert!(back.abs() < 0.004 && front - back > 0.016);
    // The original thinner stock of this sample exposed the generic clamp.
    let outline = [
        Vec2::new(-0.004, -0.05),
        Vec2::new(0.004, -0.05),
        Vec2::new(0.004, 0.01),
        Vec2::new(-0.004, 0.01),
    ];
    let (bevel, inset) = Geometry::slab_profile(&outline, 0.000159, 0.000022);
    assert!(bevel > 0. && bevel <= 0.000159 * 0.45 && (0.0..1.0).contains(&inset));
    audit::check(clock, &build_object(clock)).unwrap();
}
