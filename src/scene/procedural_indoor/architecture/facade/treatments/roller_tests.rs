use super::*;

#[test]
fn roller_control_handedness_is_independent_of_the_housing_and_fits_the_clutch() {
    let mut pairs = BTreeMap::<String, usize>::new();
    for seed in 0..2048 {
        let (kind, program) = ShadeProgram::sample(seed);
        let Some(stock) = program.roller else {
            continue;
        };
        assert_eq!(kind, Shade::Roller);
        assert_eq!(program, ShadeProgram::sample(seed).1);
        assert!(stock.valid());
        *pairs
            .entry(format!("{:?}/{:?}", stock.head, stock.drive))
            .or_default() += 1;
        let mut f = facade(seed, kind, Vec2::new(2.05, 1.68), 1.);
        f.treatment = Some(program);
        let o = &f.openings[0];
        let mut a = Assembly::default();
        mesh::build(
            &mut a,
            Transform::IDENTITY,
            &f,
            o,
            f.treatment.as_ref().unwrap(),
            mesh::Draw {
                coverage: 1.,
                fixed_angle: None,
                moving: true,
                other_coverage: Some(0.82),
            },
        );
        let cord = a
            .parts
            .iter()
            .find(|((_, label), _)| label.ends_with("#roller_drive"));
        assert_eq!(cord.is_some(), stock.drive != RollerDrive::Cordless);
        let Some((_, geometry)) = cord else { continue };
        let top = o.max.y - 0.072;
        let x = match stock.drive {
            RollerDrive::RightLoop => o.max.x - 0.0225,
            RollerDrive::LeftLoop => o.min.x + 0.0225,
            RollerDrive::Cordless => unreachable!(),
        };
        let drop = stock.loop_drop(o.max.y - o.min.y - 0.123).unwrap();
        let apex = Vec3::new(x, top + stock.wheel_radius_m, 0.101);
        let bight = apex - Vec3::Y * (drop + 2. * stock.wheel_radius_m);
        for point in [apex, bight] {
            assert!(geometry
                .positions
                .iter()
                .any(|&p| Vec3::from(p).distance(point) < 0.0015));
        }
        assert!(geometry.positions.iter().all(|p| (p[0] - x).abs() < 0.0014));
        assert!(a
            .parts
            .iter()
            .any(|((_, key), g)| !key.ends_with("#roller_drive")
                && g.positions
                    .iter()
                    .any(|&p| Vec3::from(p).distance(apex) < 0.0035)));
    }
    assert_eq!(pairs.len(), 9, "{pairs:?}");
    assert!(pairs.values().all(|n| *n >= 8), "{pairs:?}");
}

#[test]
fn straight_web_spans_preserve_yarn_positions_uvs_and_flow_through_the_feed() {
    let mut f = facade(17, Shade::Roller, Vec2::new(2.1, 2.6), 0.41);
    let p = f.treatment.as_mut().unwrap();
    p.roller = Some(RollerStock::sample(17));
    let o = &f.openings[0];
    let tf = Transform::from_xyz(1.4, 0.65, -2.).with_rotation(Quat::from_rotation_y(1.1));
    let mesh = |c, other| {
        let mut a = Assembly::default();
        mesh::build(
            &mut a,
            tf,
            &f,
            o,
            f.treatment.as_ref().unwrap(),
            mesh::Draw {
                coverage: c,
                fixed_angle: None,
                moving: true,
                other_coverage: other,
            },
        );
        a.parts
            .into_iter()
            .find(|((_, key), _)| key.ends_with("#roller"))
            .unwrap()
            .1
    };
    let (lo, hi) = (0.41, 0.60);
    let compact = [mesh(lo, Some(hi)), mesh(hi, Some(lo))];
    let full = [mesh(lo, None), mesh(hi, None)];
    assert_eq!(compact[0].indices, compact[1].indices);
    assert_eq!(compact[0].uvs, compact[1].uvs);
    assert!(compact[0].indices.len() * 3 < full[0].indices.len());
    let feed = mesh::feed_plane(tf, o);
    let n = |g: &crate::scene::procedural_indoor::geometry::Geometry| {
        g.uvs
            .iter()
            .skip(5)
            .step_by(5)
            .position(|uv| uv[1].abs() < 1e-5)
            .unwrap()
            + 2
    };
    let (n_coarse, n_dense) = (n(&compact[0]), n(&full[0]));
    for t in [0., 0.22, 0.5, 0.81, 1.] {
        let at = |pair: &[_; 2], i: usize| {
            let [a, b]: &[crate::scene::procedural_indoor::geometry::Geometry; 2] = pair;
            feed.interpolate(Vec3::from(a.positions[i]), Vec3::from(b.positions[i]), t)
        };
        for j in 0..n_dense {
            let uv = full[0].uvs[j * 5][1];
            let k = (0..n_coarse - 1)
                .find(|k| compact[0].uvs[(k + 1) * 5][1] <= uv + 1e-6)
                .unwrap();
            let (a, b) = (compact[0].uvs[k * 5][1], compact[0].uvs[(k + 1) * 5][1]);
            let alpha = ((a - uv) / (a - b)).clamp(0., 1.);
            for col in 0..5 {
                let p = at(&compact, k * 5 + col).lerp(at(&compact, (k + 1) * 5 + col), alpha);
                assert!(
                    p.distance(at(&full, j * 5 + col)) < 0.00015,
                    "t={t} row={j} col={col}"
                );
            }
        }
    }
}
