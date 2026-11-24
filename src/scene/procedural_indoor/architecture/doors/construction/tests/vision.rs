use super::*;

const SHAPES: [VisionShape; 3] = [
    VisionShape::Porthole { diameter_m: 0.47 },
    VisionShape::Slot {
        width_m: 0.22,
        height_m: 0.81,
    },
    VisionShape::Rounded {
        width_m: 0.53,
        height_m: 0.66,
        radius_m: 0.093,
    },
];

fn stock(shape: VisionShape) -> VisionLight {
    VisionLight {
        shape,
        center_height_m: 1.54,
        latch_bias: 0.67,
        metal_bead: true,
    }
}

fn pane(d: &DoorInstallation, index: usize, local: bool) -> (Assembly, Leaf, Rect) {
    let j = Joinery::sample(&d.program);
    let mut leaf = d.leaves()[index];
    if local {
        leaf.transform = Transform::IDENTITY;
    }
    let fit = Fit::new(d, leaf, index, &j);
    let mut a = Assembly::default();
    panel::build(&mut a, &d.program, leaf, &fit, &j, index, &labels(&j));
    (a, leaf, fit.opening.unwrap())
}

#[test]
fn shaped_light_matches_analytic_aperture_and_opaque_shoulders_from_both_faces() {
    for (index, shape) in SHAPES.into_iter().enumerate() {
        let mut d = door(31, DoorMechanism::Paired);
        d.width = 1.12; // The requested porthole has to fit a narrower leaf.
        d.program.panel = DoorPanel::VisionLite;
        let v = stock(shape);
        d.program.vision_light = Some(v);
        let j = Joinery::sample(&d.program);
        let (a, leaf, hole) = pane(&d, 0, true);
        let radius = v.radius(hole);
        let half = hole.size() * 0.5;
        let bead = j.bead.min(radius * 0.55);
        let clear = bead + j.seal.min(radius * 0.20);
        if index == 0 {
            assert!((hole.width() - hole.height()).abs() < 1e-6 && hole.width() < 0.47);
        }
        for g in a.parts.values() {
            for idx in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = idx.map(|i| Vec3::from_array(g.positions[i as usize]));
                let n = (b - a).cross(c - a);
                assert!(n.length_squared() > 1e-16);
                for &i in idx {
                    let normal = Vec3::from_array(g.normals[i as usize]);
                    assert!(normal.is_normalized() && n.dot(normal) > 0.);
                    assert!(g.uvs[i as usize].iter().all(|u| u.is_finite()));
                }
            }
        }
        for side in [-1., 1.] {
            for x in 0..25 {
                for y in 0..31 {
                    let frac = Vec2::new((x as f32 + 0.37) / 25., (y as f32 + 0.69) / 31.);
                    let xy = hole.min - 0.004 + frac * (hole.size() + 0.008);
                    let q = (xy - hole.center()).abs() - half + radius;
                    let sd = q.max(Vec2::ZERO).length() + q.max_element().min(0.) - radius;
                    // The analytic outline bounds the inscribed polygon by 0.8 mm.
                    if [0., bead, clear]
                        .iter()
                        .any(|edge| (sd + edge).abs() < 0.001)
                    {
                        continue;
                    }
                    let visible = hit(&a, xy, side, |_, _| true).unwrap();
                    let opaque = hit(&a, xy, side, |s, _| s != Surface::GlassInterior);
                    if sd < -clear {
                        assert!(opaque.is_none());
                        assert_eq!(visible.1, Surface::GlassInterior);
                        assert!((visible.0 - j.inset - side * j.glass.metres() * 0.5).abs() < 2e-6);
                    } else {
                        let (z, surface, label) = opaque.unwrap();
                        assert_eq!(visible.1, surface);
                        assert!(label.starts_with("door#"));
                        if sd > 0. {
                            assert_eq!(surface, d.program.finish);
                            assert!((z * side - leaf.size.z * 0.5).abs() < 2e-6);
                        } else {
                            assert_eq!(
                                surface,
                                if sd > -bead {
                                    Surface::Metal
                                } else {
                                    Surface::Rubber
                                }
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fitted_light_and_material_charts_follow_every_moving_leaf() {
    for mechanism in [
        DoorMechanism::Hinged,
        DoorMechanism::Paired,
        DoorMechanism::Sliding,
        DoorMechanism::Telescoping,
    ] {
        for shape in SHAPES {
            let mut d = door(19, mechanism);
            d.program.panel = DoorPanel::VisionLite;
            d.program.vision_light = Some(stock(shape));
            for i in 0..d.leaves().len() {
                let (local, _, _) = pane(&d, i, true);
                for open in [0., 0.48, 1.] {
                    d.program.opening = open;
                    d.yaw = -0.88;
                    let (world, leaf, _) = pane(&d, i, false);
                    for (key, g) in &local.parts {
                        let target = &world.parts[key];
                        assert_eq!((&g.indices, &g.uvs), (&target.indices, &target.uvs));
                        for (p, q) in g.positions.iter().zip(&target.positions) {
                            assert!(leaf
                                .transform
                                .transform_point(Vec3::from_array(*p))
                                .abs_diff_eq(Vec3::from_array(*q), 2e-5));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn archived_door_uses_square_cut_and_shaped_stock_is_validated_and_replayable() {
    let mut counts = [0; 4];
    let mut trim = [0; 2];
    for seed in 0..2048 {
        let p = DoorProgram::sample(seed, DoorMechanism::Hinged);
        if p.panel != DoorPanel::VisionLite {
            assert!(p.vision_light.is_none());
            continue;
        }
        let encoded = serde_json::to_value(&p).unwrap();
        assert_eq!(p, serde_json::from_value(encoded.clone()).unwrap());
        assert!(p.validate());
        let class = match p.vision_light.map(|v| v.shape) {
            None => 0,
            Some(VisionShape::Porthole { .. }) => 1,
            Some(VisionShape::Slot { .. }) => 2,
            Some(VisionShape::Rounded { .. }) => 3,
        };
        counts[class] += 1;
        if let Some(v) = p.vision_light {
            trim[v.metal_bead as usize] += 1;
            let mut archived = encoded;
            archived.as_object_mut().unwrap().remove("vision_light");
            let legacy: DoorProgram = serde_json::from_value(archived.clone()).unwrap();
            assert!(legacy.vision_light.is_none());
            assert_eq!(serde_json::to_value(legacy).unwrap(), archived);
            let mut bad = p;
            bad.vision_light.as_mut().unwrap().latch_bias = f32::NAN;
            assert!(!bad.validate());
        }
    }
    assert!(counts.into_iter().all(|n| n > 25));
    assert!(trim.into_iter().all(|n| n > 35));
}
