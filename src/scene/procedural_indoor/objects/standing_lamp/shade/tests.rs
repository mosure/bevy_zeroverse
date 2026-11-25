use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version};

#[test]
fn generated_shades_keep_the_bulb_pose_envelope_and_semantic_owner() {
    let mut sections = [0; 4];
    let mut fittings = [0; 2];
    let mut lined = 0;
    let mut reduced = 0;
    let mut base_and_shade = [[0; 2]; 2];
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::FloorLamp, seed);
        let p = parameters(&o);
        let cut = ShadeCut::sample(&o, &p);
        if let Some(foot) = p.pedestal(o.seed).filter(|_| p.shade.shape < 3) {
            base_and_shade[usize::from(foot.sides < 32)]
                [usize::from(matches!(cut.section, ShadeSection::Panelled { .. }))] += 1;
        }
        let current = build_object_at_version(&o, None, CUT_SHADE_VERSION);
        let old = build_object_at_version(&o, None, CUT_SHADE_VERSION - 1);
        let check = audit::check(&o, &current).unwrap();
        assert!(check.max_envelope_overrun_m < 1e-5, "seed {seed}");
        assert!(current
            .parts
            .keys()
            .all(|(_, label)| part_label(label) == "lamp"));
        let (key, emissive) = current
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::Light)
            .unwrap();
        assert_eq!(emissive.positions, old.parts[key].positions);
        assert_eq!(emissive.indices, old.parts[key].indices);
        sections[match cut.section {
            ShadeSection::Round => 0,
            ShadeSection::Oval { .. } => 1,
            ShadeSection::SoftRectangle { .. } => 2,
            ShadeSection::Panelled { .. } => 3,
        }] += 1;
        fittings[usize::from(cut.fitter == Fitter::Harp)] += 1;
        if cut.lining == Some(Surface::Paper) {
            lined += 1;
            assert!(current.parts.keys().any(|(s, _)| *s == Surface::Paper));
        }
        if matches!(cut.section, ShadeSection::Panelled { .. }) {
            let old_tris: usize = old.parts.values().map(|g| g.indices.len() / 3).sum();
            assert!(check.triangles < old_tris, "faceted stock grew on {seed}");
            reduced += 1;
        }
    }
    assert!(sections.iter().all(|n| *n >= 12), "{sections:?}");
    assert!(fittings.iter().all(|n| *n >= 40));
    assert!(lined >= 20 && reduced >= 12);
    assert!(
        base_and_shade.iter().flatten().all(|count| *count > 0),
        "{base_and_shade:?}"
    );
}

#[test]
fn shade_hardware_clears_the_bulb_and_each_spoke_reaches_the_actual_hem() {
    use crate::scene::procedural_indoor::tests::point_triangle_distance_squared;
    for seed in 0..160 {
        let o = audit::fixture(ObjectKind::FloorLamp, seed);
        let mut p = parameters(&o);
        let cut = ShadeCut::sample(&o, &p);
        // Isolate the wire hardware from the shade parts for this mesh probe.
        p.shade.finish = Surface::FabricAlt;
        p.shade.trim = Surface::Chrome;
        let cut = ShadeCut {
            lining: Some(Surface::Paper),
            ..cut
        };
        let s = o.size;
        let bulb = Vec3::new(
            s.x * p.stem_offset,
            s.y - p.shade_height * 0.62,
            s.z * p.stem_depth,
        );
        let radii = luminaires::bulb_radii(&p.shade, p.bulb_radius);
        let root = bulb - Vec3::Y * (radii.y + p.bulb_radius * 0.42);
        let nominal = (s.x.min(s.z) * p.shade_radius_fraction)
            .min((s.x * 0.5 - bulb.x.abs()).min(s.z * 0.5 - bulb.z.abs()) / p.shade_taper.max(1.));
        let mut assembly = Assembly::default();
        cut.build(&mut assembly, &o, &p, nominal, bulb, root);
        for g in assembly.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = tri.map(|i| (Vec3::from(g.positions[i as usize]) - bulb) / radii);
                assert!(
                    point_triangle_distance_squared(Vec3::ZERO, a, b, c) > 1.,
                    "shade or fitter intersects lamp envelope on {seed}"
                );
            }
        }
        let hardware = &assembly.parts[&(Surface::Metal, "lamp".into())];
        let trim = &assembly.parts[&(Surface::Chrome, "lamp".into())];
        let lip_y = if cut.fitter == Fitter::Harp {
            s.y
        } else {
            s.y - p.shade_height
        };
        // Each uncapped cylindrical spoke ends in a six-sided ring with a UV seam.
        let ends = &hardware.positions[hardware.positions.len() - 14 * cut.spokes as usize..];
        for spoke in ends.as_chunks::<14>().0 {
            let tip = spoke[7..13].iter().copied().map(Vec3::from).sum::<Vec3>() / 6.;
            assert!((tip.y - lip_y).abs() < 0.0021);
            assert!(
                trim.indices.as_chunks::<3>().0.iter().any(|tri| {
                    let [a, b, c] = tri.map(|i| Vec3::from(trim.positions[i as usize]));
                    a.y == b.y
                        && b.y == c.y
                        && (a.y - lip_y).abs() < 1e-6
                        && point_triangle_distance_squared(tip.with_y(lip_y), a, b, c) < 1e-11
                }),
                "unseated {:?} wire on {seed}",
                cut.fitter
            );
        }
    }
}

#[test]
fn cut_rims_share_edges_and_panel_normals_follow_the_actual_face() {
    let cut = ShadeSection::Panelled {
        minor_ratio: 0.73,
        panels: 5,
    };
    let outline = cut.outline().unwrap();
    let rotation = Quat::from_rotation_y(0.73);
    let lower = 0.16;
    let height = 0.24;
    let mut p = luminaires::parameters(0);
    p.shape = 0;
    p.finish = Surface::FabricAlt;
    p.trim = Surface::Chrome;
    let mut assembly = Assembly::default();
    luminaires::ShadeStock::Cut {
        outline: &outline,
        rotation,
        lining: Surface::Paper,
    }
    .build(&mut assembly, "lamp", &p, Vec3::ZERO, lower, height, 0.62);
    let part = |s| &assembly.parts[&(s, "lamp".into())];
    let trim = part(p.trim);
    for (surface, sign) in [(p.finish, 1.), (Surface::Paper, -1.)] {
        let skin = part(surface);
        // Every seam really meets the rim in all three dimensions.
        for v in &skin.positions {
            assert!(trim
                .positions
                .iter()
                .any(|p| Vec3::from(*p).distance(Vec3::from(*v)) < 1e-6));
        }
        for tri in skin.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| Vec3::from(skin.positions[i as usize]));
            let n = (b - a).cross(c - a).normalize();
            assert!(n.dot((a + b + c).with_y(0.)) * sign > 0.);
            for &i in tri {
                assert!(n.dot(Vec3::from(skin.normals[i as usize])) > 0.99999);
            }
        }
    }
    // Pick an oblique point in the middle of an edge, where a circular
    // spoke tip would miss the pentagonal shoulder.
    let point = outline.perimeter_point(0.1);
    let [a, b] = [0., 0.2].map(|t| outline.perimeter_point(t));
    assert!(point.distance((a + b) * 0.5) < 1e-6);
    assert!(point.length() < 0.9);
    for i in 0..5 {
        let (_, _, u0) = outline.meridians().nth(i * 2).unwrap();
        let (_, _, u1) = outline.meridians().nth(i * 2 + 1).unwrap();
        let skin = part(p.finish);
        let chord =
            Vec3::from(skin.positions[i * 2]).distance(Vec3::from(skin.positions[i * 2 + 1]));
        assert!((chord - lower * (u1 - u0)).abs() < 1e-6);
    }
}
