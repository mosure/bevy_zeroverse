use super::*;

#[test]
fn installed_screens_fit_the_reserved_footprint_and_clear_neighbouring_bays() {
    let mut counts = [0; 5];
    let mut signed_turns = [0; 2];
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::RoomDivider, seed);
        o.size = if seed % 2 == 0 {
            Vec3::new(0.90, 1.05, 0.49)
        } else {
            Vec3::new(1.60, 1.70, 0.35)
        };
        let p = parameters(&o);
        counts[match p.plan.arrangement {
            Arrangement::Single => 0,
            Arrangement::Straight => 1,
            Arrangement::Angled => 2,
            Arrangement::Curved => 3,
            Arrangement::Zigzag => 4,
        }] += 1;
        let a = build_object(&o);
        let audit = audit::check(&o, &a).unwrap();
        assert!(
            audit.min_y_m.abs() < 2e-6 && audit.max_envelope_overrun_m < 2e-6,
            "seed {seed}"
        );
        assert!(a
            .parts
            .keys()
            .all(|(_, label)| part_label(label) == o.kind.class_name()));
        let depth = p.frame_depth();
        let mut last_turn = 0.;
        for i in 1..p.weights.len() {
            let turn = p.plan.joint_angle(i);
            match p.plan.arrangement {
                Arrangement::Straight => assert!(turn.abs() < 1e-5),
                Arrangement::Curved if i > 1 => assert!(turn * last_turn > 0.),
                Arrangement::Zigzag if i > 1 => assert!(turn * last_turn < 0.),
                _ => {}
            }
            if turn.abs() > 0.2 {
                signed_turns[usize::from(turn > 0.)] += 1;
            }
            last_turn = turn;
            // Separate the square-cut physical corners on their common bisector.
            let axis = (p.plan.axis(i - 1) + p.plan.axis(i)).normalize();
            let corners = |bay: plan::Bay, side: f32| {
                [-1., 1.].map(|z| {
                    bay.frame
                        .transform_point(Vec3::new(side * bay.width * 0.5, 0., z * depth * 0.5))
                        .xz()
                        .dot(axis)
                })
            };
            let [l1, l2] = corners(p.plan.bay(i - 1, depth), 1.);
            let [r1, r2] = corners(p.plan.bay(i, depth), -1.);
            let expected = p.plan.joint_gap_m * (turn * 0.5).cos();
            assert!(
                (r1.min(r2) - l1.max(l2) - expected).abs() < 3e-6,
                "seed {seed}, joint {i}"
            );
        }
        for side in [-1., 1.] {
            let (foot, _) = p
                .plan
                .support(side * o.size.x * p.stand_span, depth, p.border * 1.1);
            assert!((0..p.weights.len()).any(|i| {
                let bay = p.plan.bay(i, depth);
                let rel = foot - bay.frame.translation.xz();
                let axis = p.plan.axis(i);
                rel.perp_dot(axis).abs() < 1e-6
                    && rel.dot(axis).abs() < bay.width * 0.5 - p.border + 1e-6
            }));
        }
    }
    assert!(counts.into_iter().all(|n| n >= 10), "{counts:?}");
    assert!(signed_turns.into_iter().all(|n| n > 24));
}

// Intersect a ray normal to the bay with the exported indexed triangles.
// Return distinct surfaces in the immediate vicinity of the panel.
fn hits(a: &Assembly, frame: Transform, x: f32, y: f32) -> Vec<(f32, Surface)> {
    let inv = frame.rotation.inverse();
    let mut hits = Vec::new();
    for ((surface, _), g) in &a.parts {
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, b, c] =
                tri.map(|i| inv * (Vec3::from(g.positions[i as usize]) - frame.translation));
            let (b, c, q) = (b - p, c - p, Vec2::new(x, y) - p.xy());
            let denom = b.xy().perp_dot(c.xy());
            if denom.abs() < 1e-10 {
                continue;
            }
            let u = q.perp_dot(c.xy()) / denom;
            let v = b.xy().perp_dot(q) / denom;
            let z = p.z + u * b.z + v * c.z;
            if u >= 0. && v >= 0. && u + v <= 1. && z.abs() < 0.06 {
                hits.push((z, *surface));
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits.dedup_by(|a, b| a.1 == b.1 && (a.0 - b.0).abs() < 1e-5);
    hits
}

#[test]
fn glass_at_the_stands_and_the_openings_between_louvres_have_no_intruding_stock() {
    let mut checked = [0; 2];
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::RoomDivider, seed);
        o.finish_seed = Some(seed ^ 0x4d42_4159);
        let p = parameters(&o);
        let a = build_object(&o);
        let lo = p.bottom + p.border;
        let hi = p.top - p.border;
        for i in 0..p.weights.len() {
            let bay = p.plan.bay(i, p.frame_depth());
            let inner = bay.width - 2. * p.border;
            if p.panel == Surface::GlassInterior {
                let y = lo + (hi - lo) * 0.123;
                for side in [-1., 1.] {
                    let (foot, _) = p.plan.support(
                        side * o.size.x * p.stand_span,
                        p.frame_depth(),
                        p.border * 1.1,
                    );
                    let x = (foot - bay.frame.translation.xz()).dot(p.plan.axis(i));
                    if x.abs() > inner * 0.5 - 0.008 {
                        continue;
                    }
                    let crossings = hits(&a, bay.frame, x, y);
                    assert_eq!(crossings.len(), 2, "seed {seed}, bay {i}: {crossings:?}");
                    assert!(crossings.iter().all(|(_, s)| *s == Surface::GlassInterior));
                    assert!((crossings[1].0 - crossings[0].0 - p.insert_thickness_m).abs() < 2e-6);
                    checked[0] += 1;
                }
            }
            if p.slats {
                let slats = p.slat_field(inner);
                let y = lo + (hi - lo) * 0.836;
                let x = -inner * 0.5 + slats.pitch;
                assert!(
                    hits(&a, bay.frame, x, y).is_empty(),
                    "closed slot, seed {seed}, bay {i}"
                );
                let blade = hits(&a, bay.frame, x - slats.pitch * 0.5, y);
                assert_eq!(blade.len(), 2, "missing louvre, seed {seed}, bay {i}");
                assert!(blade.iter().all(|(_, s)| *s == p.frame));
                checked[1] += 1;
            }
        }
    }
    assert!(checked.into_iter().all(|n| n > 40), "{checked:?}");
}

#[test]
fn timber_and_infill_keep_metric_grain_after_folding() {
    let mut g = Geometry::default();
    let size = Vec3::new(0.07, 0.81, 0.023);
    let pose = Transform::from_xyz(0.41, 0.82, -0.16)
        .with_rotation(Quat::from_rotation_y(0.71) * Quat::from_rotation_z(-FRAC_PI_2));
    let origin = Vec2::new(1.56, 0.87);
    stock::member(&mut g, size, pose, origin);
    assert_eq!(g.indices.len() / 3, 32);
    for i in (0..32).step_by(4) {
        let long = Vec3::from(g.positions[i + 1]) - Vec3::from(g.positions[i]);
        let uv = Vec2::from(g.uvs[i + 1]) - Vec2::from(g.uvs[i]);
        assert!(long.distance(pose.rotation * Vec3::Y * size.y) < 2e-6);
        assert!(uv.distance(Vec2::Y * size.y) < 2e-6);
    }
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec2::from(g.uvs[i as usize]));
        assert!((b - a).perp_dot(c - a).abs() > 1e-8);
    }
}

#[test]
fn desk_screens_have_grounded_mounts_metric_faces_and_a_real_cable_gap() {
    let mut mounts = [0; 2];
    let mut finishes = std::collections::BTreeSet::new();
    let mut slopes = [0; 3];
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::RoomDivider, seed);
        o.variant = desktop::VARIANT;
        o.finish_seed = None;
        assert!(!desktop::is_desktop(&o));
        o.support = Some(8);
        o.size = match seed % 3 {
            0 => Vec3::new(0.39, 0.29, 0.067),
            1 => Vec3::new(1.7, 0.57, 0.22),
            _ => Vec3::new(0.98, 0.42, 0.16),
        };
        let p = desktop::parameters(&o);
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(check.min_y_m.abs() < 1e-6);
        assert_eq!(check.batches, 2);
        assert!(check.triangles < 900);
        for (_, label) in a.parts.keys() {
            assert_eq!(part_label(label), o.kind.class_name());
        }
        let tf = Transform::IDENTITY;
        assert!(hits(&a, tf, 0.01, p.cable_gap_m * 0.52).is_empty());
        for x in p.post_x_m {
            assert!(hits(&a, tf, x, p.cable_gap_m * 0.65)
                .iter()
                .any(|h| h.1 == p.hardware));
        }
        let board = hits(&a, tf, 0.013, o.size.y * 0.62);
        assert_eq!(board.len(), 2);
        assert!(board.iter().all(|h| h.1 == p.face));
        assert!((board[1].0 - board[0].0 - p.thickness_m).abs() < 1e-5);
        let face = a
            .parts
            .iter()
            .find(|((surface, _), _)| *surface == p.face)
            .unwrap()
            .1;
        // On each wide face a metre in either direction is a metre in its UV chart.
        let mut phases = face
            .positions
            .iter()
            .zip(&face.uvs)
            .zip(&face.normals)
            .filter(|(_, normal)| normal[2].abs() > 0.999)
            .map(|((position, uv), _)| {
                Vec2::from_array(*uv) - Vec2::new(position[0], -position[1])
            });
        let phase = phases.next().unwrap();
        assert!(phases.all(|sample| sample.abs_diff_eq(phase, 1e-5)));
        mounts[usize::from(p.mount == desktop::Mount::ScrewPlate)] += 1;
        finishes.insert(p.face);
        slopes[if p.top_drop_m[0] > 0. {
            0
        } else if p.top_drop_m[1] > 0. {
            1
        } else {
            2
        }] += 1;
        o.finish_seed = Some(seed.wrapping_mul(51));
        let other = desktop::parameters(&o);
        assert_eq!(other.top_drop_m, p.top_drop_m);
        assert_eq!(other.upper_corner_m, p.upper_corner_m);
        assert_eq!(other.post_x_m, p.post_x_m);
        assert_eq!(other.mount, p.mount);
        assert_eq!(other.thickness_m, p.thickness_m);
    }
    assert!(mounts.iter().all(|n| *n > 60));
    assert!(slopes.iter().all(|n| *n > 20));
    assert_eq!(finishes.len(), 3);
}
