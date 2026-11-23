use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;
use std::collections::BTreeMap;

fn room(seed: u64) -> IndoorManifest {
    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap()
}

fn closed_stock(g: &Geometry) {
    let mut incidence = BTreeMap::new();
    for t in g.indices.as_chunks::<3>().0 {
        let p = t.map(|i| Vec3::from(g.positions[i as usize]));
        let cross = (p[1] - p[0]).cross(p[2] - p[0]);
        assert!(cross.length_squared() > 1e-20);
        for i in t {
            let n = Vec3::from(g.normals[*i as usize]);
            assert!(n.is_finite() && (n.length_squared() - 1.).abs() < 1e-4);
            assert!(
                n.dot(cross.normalize()) > 0.,
                "shading normal faces into stock"
            );
            assert!(Vec2::from(g.uvs[*i as usize]).is_finite());
        }
        let keys = p.map(|p| p.to_array().map(|v| if v == 0. { 0 } else { v.to_bits() }));
        for i in 0..3 {
            let (a, b) = (keys[i], keys[(i + 1) % 3]);
            let (edge, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
            let (count, winding) = incidence.entry(edge).or_insert((0, 0));
            *count += 1;
            *winding += sign;
        }
    }
    assert!(!incidence.is_empty());
    assert!(incidence.values().all(|&edge| edge == (2, 0)));
}

#[test]
fn measured_joint_is_constant_across_profile_depth_and_plan_angles() {
    let mut convex = 0;
    let mut reflex = 0;
    let mut sloped = 0;
    for seed in 100..196 {
        let scene = room(seed);
        let env = scene.envelope.as_ref().unwrap();
        let p = TrimProgram::for_scene(&scene).unwrap();
        let seam = p.miter_joint_m.unwrap();
        let n = env.footprint.len();
        for (crown, section) in [(false, &p.base), (true, &p.crown)] {
            let Some(s) = section else { continue };
            let datum = if crown {
                s.height_m + p.joint_m
            } else {
                p.joint_m
            };
            for edge in 0..n {
                let vertex = (edge + 1) % n;
                let mut left = Run::new(env, scene.room_size, edge, datum, crown);
                let mut right = Run::new(env, scene.room_size, vertex, datum, crown);
                left.fit_ends(s.depth_m, seam, &[]);
                right.fit_ends(s.depth_m, seam, &[]);
                if left.ends[1] != right.ends[0] {
                    continue;
                } // a short square return
                let perpendicular = left.ends[1].perp().normalize();
                for d in [BACK, 0., s.depth_m * 0.41, s.depth_m] {
                    let a = left.point(left.span.y, s.height_m, d);
                    let b = right.point(right.span.x, s.height_m, d);
                    assert!(
                        ((a.xz() - b.xz()).dot(perpendicular).abs() - seam).abs() < 5e-6,
                        "seed {seed}, edge {edge}, depth {d}"
                    );
                    if crown && d >= 0. {
                        for v in [a, b] {
                            assert!(
                                (env.roof_height(scene.room_size, v.xz()) - v.y - p.joint_m).abs()
                                    < 5e-6
                            );
                        }
                    }
                }
                convex += usize::from(left.along.perp_dot(right.along) > 0.);
                reflex += usize::from(left.along.perp_dot(right.along) < 0.);
                sloped += usize::from(crown && env.ceiling_drop.length_squared() > 0.);
            }
        }
    }
    assert!(
        convex > 400 && reflex > 80 && sloped > 150,
        "{convex} {reflex} {sloped}"
    );
}

#[test]
fn cut_stock_keeps_sill_material_closes_reveals_and_has_no_internal_faces() {
    let scene = room(50801);
    let env = scene.envelope.as_ref().unwrap();
    let edge = 0;
    let mut run = Run::new(env, scene.room_size, edge, 0.005, false);
    let profile = [
        Vec2::new(0., 0.045),
        Vec2::new(0.075, 0.045),
        Vec2::new(0.16, 0.018),
    ];
    // A tall casing, a low sill and a second cut overlapping its edge.
    let cuts = [
        Rect::from_corners(Vec2::new(-0.5, 0.), Vec2::new(-0.22, 0.16)),
        Rect::from_corners(Vec2::new(0.1, 0.08), Vec2::new(0.35, 0.16)),
        Rect::from_corners(Vec2::new(0.28, 0.12), Vec2::new(0.55, 0.16)),
    ];
    run.fit_ends(0.045, 0.0014, &cuts);
    let mut g = Geometry::default();
    sweep(&mut g, &run, &profile, &cuts, None);
    closed_stock(&g);
    let mut retained = 0;
    for t in g.indices.as_chunks::<3>().0 {
        let c = t
            .map(|i| Vec3::from(g.positions[i as usize]))
            .into_iter()
            .sum::<Vec3>()
            / 3.;
        let xy = Vec2::new((c.xz() - run.center).dot(run.along), c.y - run.datum);
        for cut in &cuts {
            assert!(
                !(xy.cmpgt(cut.min + Vec2::splat(1e-5)).all()
                    && xy.cmplt(cut.max - Vec2::splat(1e-5)).all())
            );
        }
        retained += usize::from((0.11..0.33).contains(&xy.x) && xy.y < 0.081);
    }
    assert!(retained > 0);
    let mut whole = Geometry::default();
    sweep(&mut whole, &run, &profile, &[], None);
    closed_stock(&whole);
    assert_eq!(whole.indices.len() / 3, 4 + 8 * (profile.len() - 1));
}

#[test]
fn tight_jambs_and_short_acute_returns_cannot_invert_a_saw_cut() {
    let mut scene = room(50802);
    let env = scene.envelope.as_mut().unwrap();
    env.footprint = vec![
        Vec2::ZERO,
        Vec2::new(4., 0.),
        Vec2::new(4., 3.),
        Vec2::new(0., 3.),
    ];
    let section = TrimSection {
        height_m: 0.18,
        depth_m: 0.065,
        tip_fraction: 0.4,
        shoulder: 0.7,
        profile: TrimProfile::Cove,
        surface: Surface::Paint,
        finish_slot: 0,
    };
    let profile = [Vec2::new(0., 0.065), Vec2::new(0.18, 0.035)];
    let mut run = Run::new(env, scene.room_size, 0, 0.005, false);
    let cut = run
        .cut_rect(Vec2::new(-1.97, 0.), Vec2::new(-1.92, 1.), &section)
        .unwrap();
    run.fit_ends(section.depth_m, 0.001, &[cut]);
    assert_eq!(run.ends[0], run.inward);
    let mut a = Geometry::default();
    sweep(&mut a, &run, &profile, &[cut], None);
    closed_stock(&a);
    // A validated simple polygon can have a cusp too sharp for this gauge.
    env.footprint = vec![
        Vec2::new(0., 0.),
        Vec2::new(8., 0.),
        Vec2::new(0.09, 0.02),
        Vec2::new(0., 4.),
        Vec2::new(-4., 4.),
        Vec2::new(-4., 0.),
    ];
    polygon::validate(&env.footprint).unwrap();
    for edge in 0..env.footprint.len() {
        let mut run = Run::new(env, scene.room_size, edge, 0.005, false);
        run.fit_ends(section.depth_m, 0.001, &[]);
        if run.span.y > run.span.x + CUT_TOLERANCE {
            let mut g = Geometry::default();
            sweep(&mut g, &run, &profile, &[], None);
            closed_stock(&g);
        }
    }
    // The opposite turn can also be too sharp: a full mitre would project
    // several metres into the room from a narrow recess in the outer wall.
    env.footprint = vec![
        Vec2::new(-4., -3.),
        Vec2::new(4., -3.),
        Vec2::new(4., 3.),
        Vec2::new(0.006, 3.),
        Vec2::new(0., -1.),
        Vec2::new(-0.006, 3.),
        Vec2::new(-4., 3.),
    ];
    polygon::validate(&env.footprint).unwrap();
    for (edge, end) in [(3, 1), (4, 0)] {
        let mut run = Run::new(env, scene.room_size, edge, 0.005, false);
        run.fit_ends(section.depth_m, 0.001, &[]);
        assert_eq!(run.ends[end], run.inward);
        let mut g = Geometry::default();
        sweep(&mut g, &run, &profile, &[], None);
        closed_stock(&g);
    }
}

#[test]
fn rounded_stock_wraps_one_metric_texture_and_shades_continuously_at_curve_knots() {
    let scene = room(50801);
    let env = scene.envelope.as_ref().unwrap();
    let mut section = TrimProgram::for_scene(&scene)
        .unwrap()
        .base
        .clone()
        .unwrap();
    for surface in [Surface::Paint, Surface::WoodEdge] {
        section.surface = surface;
        for shape in [TrimProfile::Round, TrimProfile::Cove, TrimProfile::Chamfer] {
            section.profile = shape;
            for crown in [false, true] {
                let mut run = Run::new(env, scene.room_size, 0, 0.005, crown);
                run.timber_origin = surface.is_cut_timber().then_some(Vec2::new(1.25, 0.77));
                run.gradient = Vec2::new(0.14, -0.11);
                let mut profile = section.knots();
                if crown {
                    profile.reverse();
                    for p in &mut profile {
                        p.x = 1. - p.x;
                    }
                }
                let mut facing = facing::Facing::new(&run, Some((&section, crown)));
                let mut g = Geometry::default();
                for pair in profile.windows(2) {
                    let [lo, hi] = [pair[0], pair[1]]
                        .map(|p| p * Vec2::new(section.height_m, section.depth_m));
                    let band = facing.band(lo.x, hi.x, hi.y - lo.y);
                    band.front(
                        &mut g,
                        [
                            run.point(-1., lo.x, lo.y),
                            run.point(1., lo.x, lo.y),
                            run.point(1., hi.x, hi.y),
                            run.point(-1., hi.x, hi.y),
                        ],
                    );
                }
                let mut smooth = 0;
                let mut creases = 0;
                for (i, quad) in g.uvs.chunks_exact(4).enumerate() {
                    let (longitudinal, transverse, sign) = if surface.is_cut_timber() {
                        (1, 0, -1.)
                    } else {
                        (0, 1, 1.)
                    };
                    assert!(quad[2][transverse] > quad[1][transverse]);
                    let vertex = 4 * i;
                    let length = Vec3::from(g.positions[vertex])
                        .distance(Vec3::from(g.positions[vertex + 1]));
                    assert!(
                        (sign * (quad[1][longitudinal] - quad[0][longitudinal]) - length).abs()
                            < 2e-6
                    );
                    if i == 0 {
                        continue;
                    }
                    assert_eq!(g.positions[vertex - 1], g.positions[vertex]);
                    assert_eq!(g.uvs[vertex - 1], g.uvs[vertex]);
                    if Vec3::from(g.normals[vertex - 1]).distance(Vec3::from(g.normals[vertex]))
                        < 1e-5
                    {
                        smooth += 1;
                    } else {
                        creases += 1;
                    }
                }
                match shape {
                    TrimProfile::Round => assert_eq!((smooth, creases), (5, 0)),
                    TrimProfile::Cove => assert_eq!((smooth, creases), (4, 1)),
                    TrimProfile::Chamfer => assert_eq!((smooth, creases), (0, 1)),
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn old_manifests_keep_square_ends_and_new_entries_fit_the_frame_stock() {
    use crate::scene::procedural_indoor::envelope::construction;
    let scene = room(50801);
    let env = scene.envelope.as_ref().unwrap();
    let edge = (0..env.footprint.len())
        .find(|&i| env.shared_edge(i, scene.room_size))
        .unwrap();
    let trim = TrimProgram::for_scene(&scene).unwrap();
    let s = trim.base.as_ref().unwrap();
    let mut parts = Assembly::default();
    build(&mut parts, &scene, edge, trim, s, &[], false);
    let mesh = &parts.parts[&(s.surface, format!("wall#finish{}", s.finish_slot))];
    closed_stock(mesh);
    let frame = env.wall_transform(edge);
    let half = (scene.entry_width()
        + crate::scene::procedural_indoor::architecture::doors::FRAME_WIDTH_M)
        * 0.5;
    for p in &mesh.positions {
        let x = (frame.rotation.inverse() * (Vec3::from(*p) - frame.translation)).x;
        assert!((x + scene.door_x).abs() >= half + trim.miter_joint_m.unwrap() - 1e-5);
    }
    let mut archived = serde_json::to_value(trim).unwrap();
    archived.as_object_mut().unwrap().remove("miter_joint_m");
    let legacy: TrimProgram = serde_json::from_value(archived).unwrap();
    assert!(legacy.miter_joint_m.is_none());
    legacy.validate().unwrap();
    let mut old = scene.clone();
    old.program
        .as_mut()
        .unwrap()
        .finishes
        .as_mut()
        .unwrap()
        .trim = Some(legacy);
    let current = construction::build(&scene);
    let replay = construction::build(&old);
    let key = (s.surface, format!("wall#finish{}", s.finish_slot));
    assert_ne!(current.parts[&key].positions, replay.parts[&key].positions);
    for invalid in [f32::NAN, -0.001, 0., 0.1] {
        let mut p = trim.clone();
        p.miter_joint_m = Some(invalid);
        assert!(p.validate().is_err());
    }
}
