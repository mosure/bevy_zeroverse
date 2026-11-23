use super::*;
use crate::scene::procedural_indoor::humans::{
    appearance::FaceProgram, body, sample_person, HumanOutfit, HumanSurface,
};

#[test]
fn rest_chart_faces_forward_and_preserves_every_geometric_attribute() {
    let chart = EyeChart {
        origin: Vec3::new(0.031, 1.6, -0.03),
        forward: Vec3::NEG_Z,
        u: Vec3::X * 50.,
        v: Vec3::NEG_Y * 50.,
    };
    for z in [-0.012, 0.012] {
        let offsets = [
            Vec2::new(-0.002, -0.002),
            Vec2::new(0.002, -0.002),
            Vec2::splat(0.002),
            Vec2::new(-0.002, 0.002),
        ];
        let before = offsets.map(|xy| GarmentVertex {
            rest: chart.origin + xy.extend(z),
            position: (xy * 2.).extend(z + 0.2),
            normal: Vec3::X,
            uv: Vec2::splat(0.1),
            skin_field: 0.75,
        });
        let mut after = before;
        chart.map_face(&mut after);
        for (i, vertex) in after.iter().enumerate() {
            assert_eq!(vertex.position, before[i].position);
            assert_eq!(vertex.rest, before[i].rest);
            assert_eq!(vertex.normal, before[i].normal);
            if z > 0. {
                assert_eq!(vertex.uv, Vec2::ZERO);
            } else {
                let expected = Vec2::splat(0.5) + offsets[i] * Vec2::new(50., -50.);
                assert!(vertex.uv.distance(expected) < 0.00001);
            }
        }
    }
}

#[test]
fn both_eyes_converge_in_metric_head_space_across_poses_and_body_scales() {
    let mut left = 0;
    let mut right = 0;
    for seed in 0..256 {
        let optics = EyeOptics::sample(seed);
        for pose in [
            HumanPoseKind::SeatedWorking,
            HumanPoseKind::StandingReading,
            HumanPoseKind::StandingRelaxed,
        ] {
            let offset = fixation(seed, pose);
            let near = pose != HumanPoseKind::StandingRelaxed;
            assert!(if near {
                (0.39..0.95).contains(&offset.length())
            } else {
                (1.09..5.51).contains(&offset.length())
            });
            assert!(offset.z < 0. && offset.y.abs() < 0.25 * offset.length());
            if !near {
                left += usize::from(offset.x < -0.03);
                right += usize::from(offset.x > 0.03);
            }
            for scale in [0.84, 1.0, 1.15] {
                let origins = [
                    Vec3::new(-0.030, 1.61, -0.04),
                    Vec3::new(0.031, 1.61, -0.04),
                ];
                let charts = EyeCharts::new(optics, seed, pose, [17, 19], origins, scale);
                assert!(charts.for_bone(18).is_none());
                let target = (origins[0] + origins[1]) * 0.5 + offset / scale;
                for chart in &charts.charts {
                    let ray = target - chart.origin;
                    assert!((ray - chart.forward * ray.dot(chart.forward)).length() < 2e-6);
                    assert!(chart.forward.dot(chart.u).abs() < 3e-5);
                }
                assert!(charts.charts[0].forward.x > charts.charts[1].forward.x);
            }
        }
    }
    assert!(left > 70 && right > 70);
}

#[test]
fn saved_faces_without_optics_keep_the_partitioned_mesh_and_current_eyes_keep_anny_topology() {
    fn area(g: &super::super::Geometry) -> f64 {
        g.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]).as_dvec3());
                (b - a).cross(c - a).length() * 0.5
            })
            .sum()
    }
    for seed in [28, 36, 51] {
        let mut h = sample_person(
            seed,
            0,
            Vec3::ZERO,
            0.,
            HumanPoseKind::StandingReading,
            None,
            false,
        );
        h.outfit = HumanOutfit::Tee;
        let current = body::build(&h);
        let eye = &current.parts[&HumanSurface::Eye];
        assert!(!current.parts.contains_key(&HumanSurface::Iris));
        assert!(!current.parts.contains_key(&HumanSurface::Detail));
        assert!(eye
            .uvs
            .iter()
            .any(|p| (Vec2::from_array(*p) - 0.5).length() < 0.20));
        assert!(eye.uvs.contains(&[0., 0.]));
        #[cfg(feature = "human_motion")]
        {
            let (rest, _) = body::build_rest(&h);
            // Splitting a shading normal can append duplicate vertices after
            // the original face corners. The pigment field follows those corners.
            let rest_eye = &rest.parts[&HumanSurface::Eye];
            let count = eye.indices.len();
            assert_eq!(count, rest_eye.indices.len());
            assert_eq!(&eye.uvs[..count], &rest_eye.uvs[..count]);
        }

        let face = &mut h.appearance.as_mut().unwrap().face;
        let mut old = serde_json::to_value(&*face).unwrap();
        old.as_object_mut().unwrap().remove("optics");
        *face = serde_json::from_value::<FaceProgram>(old).unwrap();
        assert!(face.optics.is_none());
        let archived = body::build(&h);
        let old_parts = [HumanSurface::Eye, HumanSurface::Iris, HumanSurface::Detail]
            .map(|s| &archived.parts[&s]);
        let old_area: f64 = old_parts.iter().map(|g| area(g)).sum();
        let old_triangles: usize = old_parts.iter().map(|g| g.indices.len() / 3).sum();
        assert!((area(eye) / old_area - 1.).abs() < 0.001);
        assert!(eye.indices.len() / 3 < old_triangles);
        assert_eq!(current.bounds(), archived.bounds());
        assert_eq!(current.local_joints, archived.local_joints);
    }
}
