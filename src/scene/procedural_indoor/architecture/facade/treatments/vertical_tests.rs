use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::glazing_at, materials::glazing::GlazingProgram,
};

#[test]
fn stacked_vanes_clear_each_other_and_traverse_at_shallow_and_steep_tilts() {
    for seed in 0..64 {
        let mut f = facade(seed, Shade::Vertical, Vec2::new(2.75, 2.4), 0.);
        let p = f.treatment.as_mut().unwrap();
        let stock = p.vertical.unwrap();
        for stack in [Stack::Left, Stack::Pair, Stack::Right] {
            p.stack = stack;
            for tilt in [-1.48, -0.9, -0.16, 0., 0.32, 1.48] {
                let mut previous = 0.;
                for draw in [0., 0.1, 0.5, 0.9, 1.] {
                    let l = stock.layout(p, &f.openings[0], tilt, draw);
                    let spacing = (1..l.count)
                        .map(|i| l.x(i) - l.x(i - 1))
                        .fold(f32::INFINITY, f32::min);
                    assert!(spacing > previous && spacing <= l.pitch + 1e-6);
                    let separation =
                        spacing * l.angle.sin().abs() - stock.crown_m - stock.thickness_m;
                    assert!(
                        separation >= 0.001 - 1e-6,
                        "{seed} {stack:?} {tilt} {draw}: {separation}"
                    );
                    assert!(l.packed < l.pitch * 0.55);
                    previous = spacing;
                }
            }
        }
    }
}

#[test]
fn narrow_two_vane_opening_has_a_working_side_draw() {
    let mut f = facade(13, Shade::Vertical, Vec2::splat(0.35), 0.);
    let p = f.treatment.as_mut().unwrap();
    p.stack = Stack::Pair;
    let stock = p.vertical.unwrap();
    let o = &mut f.openings[0];
    o.max.x = o.min.x + 0.068 + 2. * stock.width_m - stock.overlap_m;
    let parked = stock.layout(p, o, 0.16, 0.);
    let deployed = stock.layout(p, o, 0.16, 1.);
    assert_eq!(parked.count, 2);
    assert!(deployed.x(1) - deployed.x(0) > parked.x(1) - parked.x(0) + 0.025);
}

#[test]
fn moving_vanes_keep_rigid_flow_and_individual_metric_fabric_cuts() {
    let tf = Transform::from_xyz(3., 0., -4.).with_rotation(Quat::from_rotation_y(0.79));
    for seed in 0..12 {
        let mut f = facade(seed, Shade::Vertical, Vec2::new(2.8, 2.1), 0.98);
        f.shade_tilt = [-0.16, 0.59, 1.48][seed as usize % 3];
        let p = f.treatment.as_mut().unwrap();
        p.stack = [Stack::Left, Stack::Pair, Stack::Right][seed as usize % 3];
        let l = p
            .vertical
            .unwrap()
            .layout(p, &f.openings[0], f.shade_tilt, f.shade_coverage);
        let fabric = p.fabric;
        let mut moving = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        build(&mut moving, tf, &f, &f.openings[0]);
        assert_eq!(moving.joints.len(), 1);
        let j = &moving.joints[0];
        assert!(j.attachment.is_some());
        let key = j.parts.iter().find(|k| k.0 == fabric).unwrap();
        let (a, b) = (&moving.parts[key], &j.morph[key]);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.uvs, b.uvs);
        assert_eq!(a.normals, b.normals);
        let per_vane = a.positions.len() / l.count;
        let mut cuts = BTreeSet::new();
        let mut max_travel = 0_f32;
        let mut last_end = f32::NEG_INFINITY;
        for i in 0..l.count {
            let first = i * per_vane;
            assert!(
                cuts.insert(a.uvs[first].map(f32::to_bits)),
                "a cut repeated"
            );
            let local_delta = |k| {
                tf.rotation.inverse() * (Vec3::from(b.positions[k]) - Vec3::from(a.positions[k]))
            };
            let delta = local_delta(first);
            assert!(delta.y.abs() + delta.z.abs() < 2e-6);
            assert!(l.x(i) + delta.x > last_end);
            last_end = l.x(i) + delta.x;
            max_travel = max_travel.max(delta.x.abs());
            for k in first..first + per_vane {
                assert!(local_delta(k).distance(delta) < 3e-6);
            }
            // First row spans the stock width in metres in the weft direction.
            assert!((a.uvs[first + 3][0] - a.uvs[first][0] - l.width).abs() < 2e-6);
        }
        assert!(max_travel > 0.5, "{seed}: travelled {max_travel}m");
    }
}

#[test]
fn frontal_meter_tracks_vanes_glazing_and_frit_at_every_draw() {
    let frit = (0..1024)
        .map(|s| GlazingProgram::sample(s, 0, false))
        .find(|g| g.treatment == 6)
        .unwrap();
    let size = Vec2::new(2.75, 2.4);
    let mut worst = 0_f32;
    for (seed, stack) in [Stack::Left, Stack::Pair, Stack::Right]
        .into_iter()
        .enumerate()
    {
        let mut f = facade(seed as u64, Shade::Vertical, size, 0.);
        f.treatment.as_mut().unwrap().stack = stack;
        f.openings[0].columns = 3;
        f.openings[0].transoms = vec![0.56];
        f.openings[0].operable = false;
        f.openings[0].glazing = (seed == 1).then_some(frit.clone());
        for tilt in [-0.16, 0.74, 1.48] {
            f.shade_tilt = tilt;
            for draw in [0., 0.38, 0.74, 1.] {
                f.shade_coverage = draw;
                let o = &f.openings[0];
                let expected = vertical::visible_area(&f, o, draw).unwrap() / o.area();
                let mut a = Assembly::default();
                glazing_at(&mut a, Transform::IDENTITY, &f);
                let actual = frontal_clear::<619, 223>(&a, o.min, size);
                let error = (actual - expected).abs();
                worst = worst.max(error);
                assert!(
                    error < 0.022,
                    "{stack:?} {tilt} {draw}: {actual} vs {expected}"
                );
            }
        }
    }
    println!("maximum fractional aperture error over 36 metered assemblies: {worst:.6}");
}

#[test]
fn archived_vertical_program_replays_and_new_stock_is_validated() {
    let f = facade(19, Shade::Vertical, Vec2::new(2.75, 2.4), 0.98);
    let mut wire = serde_json::to_value(&f).unwrap();
    let field = wire["treatment"]
        .as_object_mut()
        .unwrap()
        .remove("vertical")
        .unwrap();
    let archived: Facade = serde_json::from_value(wire.clone()).unwrap();
    assert!(archived.treatment.as_ref().unwrap().vertical.is_none());
    assert!(vertical::visible_area(&archived, &archived.openings[0], 0.98).is_none());
    let mut expected = f.clone();
    expected.treatment.as_mut().unwrap().vertical = None;
    let mut a = Assembly::default();
    let mut b = Assembly::default();
    build(
        &mut a,
        Transform::IDENTITY,
        &archived,
        &archived.openings[0],
    );
    build(
        &mut b,
        Transform::IDENTITY,
        &expected,
        &expected.openings[0],
    );
    for (k, g) in a.parts {
        assert_eq!(g.positions, b.parts[&k].positions);
        assert_eq!(g.uvs, b.parts[&k].uvs);
    }
    wire["treatment"]["vertical"] = field;
    wire["treatment"]["vertical"]["carrier_pitch_m"] = serde_json::json!(0.);
    let bad: Facade = serde_json::from_value(wire).unwrap();
    assert!(bad.treatment.unwrap().validate().is_err());
}
