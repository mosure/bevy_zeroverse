use super::{fit::temple_path, *};
use crate::scene::procedural_indoor::humans::{
    appearance::FaceProgram, body, face::projection::Topology, morphology, sample_person,
    HumanPoseKind, HumanSurface,
};
use burn_human::AnnyInput;

#[test]
fn sampled_rims_finishes_and_optics_are_replayable_and_physically_bounded() {
    let mut rims = [0; 3];
    let mut heights = [0; 4];
    for seed in 0..2048 {
        let metal = f32::from(seed % 2 == 0);
        let p = EyewearProgram::sample(seed, metal);
        assert_eq!(p, EyewearProgram::sample(seed, metal));
        assert_eq!(
            p,
            serde_json::from_value(serde_json::to_value(p).unwrap()).unwrap()
        );
        assert!(p.valid());
        if metal == 0. {
            assert_ne!(p.rim, Rim::DrillMounted);
        }
        rims[match p.rim {
            Rim::Full => 0,
            Rim::Upper => 1,
            Rim::DrillMounted => 2,
        }] += 1;
        heights[(((p.lens_height_m - 0.029) / 0.019 * 4.) as usize).min(3)] += 1;
        assert!(!EyewearProgram {
            lens_opacity: f32::NAN,
            ..p
        }
        .valid());
    }
    assert!(rims.into_iter().all(|n| n > 145), "{rims:?}");
    assert!(heights.into_iter().all(|n| n > 385), "{heights:?}");
    println!("2048 eyewear programs: full/upper/drill {rims:?}, height quartiles {heights:?}");
}

#[test]
fn fitted_optics_clear_actual_eyes_cheeks_nose_and_temples_under_rigid_transforms() {
    let body = &crate::procedural_human::reference::get().unwrap().body;
    let (names, _) = body.bone_hierarchy();
    let eye_indices = ["eye.R", "eye.L"].map(|name| names.iter().position(|n| n == name).unwrap());
    let mut min_lens = f32::INFINITY;
    let mut min_arm = f32::INFINITY;
    let mut max_ear = 0.0_f32;
    for seed in [8, 36, 51, 108, 251, 921] {
        let h = sample_person(
            seed,
            0,
            Vec3::ZERO,
            0.,
            HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        let shape = morphology::phenotype(&h);
        let params: Vec<_> = body
            .metadata()
            .metadata
            .phenotype_labels
            .iter()
            .map(|n| shape.value(n))
            .collect();
        let output = body
            .forward(AnnyInput {
                phenotype_inputs: Some(&params),
                ..Default::default()
            })
            .unwrap();
        let indoors = |x: f64, y: f64, z: f64| Vec3::new(-x as f32, z as f32, y as f32);
        let scale = if seed % 2 == 0 { 0.91 } else { 1.08 };
        let rot = Quat::from_euler(EulerRot::YXZ, 0.074 * seed as f32, -0.25, 0.2);
        let world = |p| Vec3::new(0.3, 0.5, -2.3) + rot * (p * scale);
        let posed: Vec<_> = output
            .rest_vertices
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[x, y, z]| world(indoors(x, y, z)))
            .collect();
        let eyes = eye_indices.map(|i| {
            let m = &output.rest_bone_poses.data[i * 16..];
            world(indoors(m[3], m[7], m[11]))
        });
        let origin = (eyes[0] + eyes[1]) * 0.5;
        let mut eyes = eyes.map(|p| rot.inverse() * (p - origin));
        eyes.sort_by(|a, b| a.x.total_cmp(&b.x));
        // The full reference is an independent face selection for a standing
        // rest pose; its hands and clothes are nowhere near the face.
        let head = Topology::new(&body.faces_quads().data, posed.len(), |_| true).project(
            &posed,
            origin,
            rot,
            scale,
            Rect::from_corners(
                Vec2::new(-0.10, -0.05) * scale,
                Vec2::new(0.10, 0.05) * scale,
            ),
            true,
        );
        for index in 0..16 {
            let p = EyewearProgram::sample(97 * seed + index, f32::from(index % 2 == 0));
            let fit = Fit::new(&p, eyes, scale, &head.front);
            for (l, eye) in fit.lenses.iter().zip(eyes) {
                let clearance = head.front.front(eye.truncate()).unwrap().0
                    - l.point(eye.truncate() - l.centre.truncate()).z;
                assert!(clearance / scale >= p.cornea_clearance_m - 1e-5);
                for i in 0..fit::SAMPLES {
                    let a = l.offsets[i];
                    let b = l.offsets[(i + 1) % fit::SAMPLES];
                    for f in [0.18, 0.58, 0.87, 1.] {
                        let xy = a.lerp(b, 0.5) * f;
                        let q = l.point(xy);
                        if let Some((z, _)) = head.front.front(q.truncate()) {
                            min_lens = min_lens.min((z - q.z) / scale);
                            assert!(
                                (z - q.z) / scale > 0.0001,
                                "lens hit skin, phenotype {seed}, frame {index}"
                            );
                        }
                        let n = l.normal(xy);
                        assert!(n.is_normalized() && n.z < -0.74);
                    }
                }
                let side = &head.temples.as_ref().unwrap()[usize::from(l.side > 0.)];
                let route = temple_path(l, &p, scale, &posed, origin, rot, side);
                assert_eq!(route[0], l.mount(2, p.rim));
                for pair in route[1..7].windows(2) {
                    for k in 0..=17 {
                        let v = pair[0].lerp(pair[1], k as f32 / 17.);
                        if let Some((depth, _)) = side.front(v.zy()) {
                            let gap = (l.side * v.x + depth) / scale;
                            min_arm = min_arm.min(gap);
                            assert!(
                                gap > p.rim_radius_m * 0.55,
                                "temple hit skin {gap}, phenotype {seed}, frame {index}"
                            );
                        }
                    }
                }
                let saddle = posed[if l.side > 0. { 5742 } else { 12323 }];
                max_ear = max_ear.max((origin + rot * route[7]).distance(saddle) / scale);
                assert!(max_ear < 0.004);
                assert!(route[9].y < route[7].y - 0.006 * scale && route[9].z > route[7].z);
            }
            for segment in fit.bridge(&head.front).windows(2) {
                for k in 0..8 {
                    let v = segment[0].lerp(segment[1], k as f32 / 8.);
                    if let Some((z, _)) = head.front.front(v.truncate()) {
                        assert!(
                            (z - v.z) / scale > p.rim_radius_m * 0.80,
                            "bridge hit nose, phenotype {seed}, frame {index}"
                        );
                    }
                }
            }
        }
    }
    println!("96 fits / 192 lenses: minimum face clearance {min_lens:.6} m, anterior arm clearance {min_arm:.6} m, maximum saddle offset {max_ear:.6} m (rest units)");
}

#[test]
fn wearable_mesh_is_optional_keeps_the_person_surface_and_has_smaller_frames() {
    let mut counts = [0usize; 2];
    for (i, pose) in [
        HumanPoseKind::StandingRelaxed,
        HumanPoseKind::StandingReading,
        HumanPoseKind::SeatedWorking,
    ]
    .into_iter()
    .enumerate()
    {
        for offset in 0..3 {
            let seed = 58330 + i as u64 * 17 + offset;
            let mut h = sample_person(seed, 0, Vec3::ZERO, 0.37 * i as f32, pose, None, false);
            let bare = {
                h.glasses = false;
                body::build(&h)
            };
            assert!(
                !bare.parts.contains_key(&HumanSurface::Lens)
                    && !bare.parts.contains_key(&HumanSurface::Eyewear)
            );
            h.glasses = true;
            let fit = body::build(&h);
            let face = &mut h.appearance.as_mut().unwrap().face;
            let mut legacy = serde_json::to_value(&*face).unwrap();
            legacy.as_object_mut().unwrap().remove("eyewear");
            *face = serde_json::from_value::<FaceProgram>(legacy).unwrap();
            assert!(face.eyewear.is_none());
            let old = body::build(&h);
            for (s, g) in &bare.parts {
                // The two face probe bounds can change floating point triangle
                // tie-breaks only at shared triangle edges.
                let clothed = &fit.parts[s];
                assert_eq!(g.indices, clothed.indices);
                assert_eq!(g.positions.len(), clothed.positions.len());
                assert!(g
                    .positions
                    .iter()
                    .zip(&clothed.positions)
                    .all(|(&a, &b)| Vec3::from(a).distance(Vec3::from(b)) < 1e-5));
            }
            let triangle_count = |m: &super::super::HumanAssembly| {
                [HumanSurface::Eyewear, HumanSurface::Lens]
                    .into_iter()
                    .map(|s| m.parts[&s].indices.len() / 3)
                    .sum::<usize>()
            };
            let (original, fitted) = (triangle_count(&old), triangle_count(&fit));
            assert!(fitted < original * 2 / 3, "{fitted} vs {original}");
            counts[0] += original;
            counts[1] += fitted;
            for s in [HumanSurface::Eyewear, HumanSurface::Lens] {
                let g = &fit.parts[&s];
                assert_eq!(g.positions.len(), g.normals.len());
                assert!(g.normals.iter().all(|&n| Vec3::from(n).is_normalized()));
                for t in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = t.map(|v| Vec3::from(g.positions[v as usize]));
                    let geometric = (b - a).cross(c - a);
                    assert!(geometric.is_finite() && geometric.length_squared() > 1e-23);
                }
                if s == HumanSurface::Lens {
                    // The first curved optical fan must face the same direction
                    // as its shading normals even when the lens is mirrored.
                    for t in g.indices.as_chunks::<3>().0.iter().take(fit::SAMPLES * 3) {
                        let [a, b, c] = t.map(|v| Vec3::from(g.positions[v as usize]));
                        assert!(
                            (b - a)
                                .cross(c - a)
                                .dot(Vec3::from(g.normals[t[0] as usize]))
                                > 0.
                        );
                    }
                }
            }
        }
    }
    println!(
        "9 matched people, frame and lens triangles: old {}, fitted {}",
        counts[0], counts[1]
    );
}
