use super::*;

#[test]
fn long_faces_follow_the_member_and_wrap_at_a_single_seam() {
    let size = Vec3::new(1.2, 0.04, 0.27);
    for axis in [GrainAxis::X, GrainAxis::Y, GrainAxis::Z] {
        let [along, a, b] = axis.frame();
        let mut g = Geometry::default();
        TimberCuts::new(77, 0).cuboid(&mut g, Surface::Wood, size, 0., Transform::IDENTITY, axis);
        let circumference = 2. * size.dot(a + b);
        let mut seams = 0;
        for i in 0..g.positions.len() {
            if Vec3::from(g.normals[i]).dot(along) != 0. {
                continue;
            }
            let p = Vec3::from(g.positions[i]);
            for j in i + 1..g.positions.len() {
                let q = Vec3::from(g.positions[j]);
                if g.normals[i] == g.normals[j] {
                    let run = (q - p).dot(along);
                    assert!((g.uvs[j][1] - g.uvs[i][1] + run).abs() < 1e-6);
                } else if p == q && Vec3::from(g.normals[j]).dot(along) == 0. {
                    assert_eq!(g.uvs[i][1], g.uvs[j][1]);
                    let difference = (g.uvs[i][0] - g.uvs[j][0]).abs();
                    if difference > 1e-6 {
                        assert!((difference - circumference).abs() < 1e-6);
                        seams += 1;
                    }
                }
            }
        }
        assert_eq!(seams, 2, "one longitudinal seam, at its two endpoints");
    }
}

#[test]
fn charts_preserve_shape_winding_and_valid_tangents_at_end_caps_and_bevels() {
    let tf = Transform::from_xyz(2., 0.3, -1.).with_rotation(Quat::from_euler(
        EulerRot::YXZ,
        1.2,
        0.3,
        -0.5,
    ));
    for size in [Vec3::new(1.4, 0.017, 0.32), Vec3::new(0.03, 2.1, 0.16)] {
        for radius in [0., 0.002] {
            let mut legacy = Geometry::default();
            legacy.cuboid(size, radius, tf);
            for surface in [
                Surface::Paint,
                Surface::Metal,
                Surface::Wood,
                Surface::WoodEdge,
            ] {
                for axis in [GrainAxis::X, GrainAxis::Y, GrainAxis::Z] {
                    let mut g = Geometry::default();
                    TimberCuts::new(71, 99).cuboid(&mut g, surface, size, radius, tf, axis);
                    assert_eq!(g.positions, legacy.positions);
                    assert_eq!(g.normals, legacy.normals);
                    assert_eq!(g.indices, legacy.indices);
                    if !surface.is_cut_timber() {
                        assert_eq!(g.uvs, legacy.uvs);
                    }
                    for tri in g.indices.as_chunks::<3>().0 {
                        let [u, v, w] = tri.map(|i| Vec2::from(g.uvs[i as usize]));
                        assert!(
                            (v - u).perp_dot(w - u) < -1e-9,
                            "collapsed or mirrored face chart"
                        );
                    }
                    let mesh = g.into_mesh();
                    let Some(VertexAttributeValues::Float32x4(tangents)) =
                        mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
                    else {
                        panic!("missing tangent frame")
                    };
                    assert!(tangents
                        .iter()
                        .all(|t| (Vec4::from(*t).truncate().length() - 1.).abs() < 1e-5
                            && t[3].abs() == 1.));
                }
            }
        }
    }
}

#[test]
fn cut_phases_replay_across_transforms_and_vary_between_adjacent_members() {
    let surfaces = [Surface::Wood, Surface::Wood, Surface::Wood];
    let make = |seed, tf| {
        let mut cuts = TimberCuts::new(seed, 2);
        surfaces.map(|s| {
            let mut g = Geometry::default();
            cuts.cuboid(
                &mut g,
                s,
                Vec3::new(1.1, 0.032, 0.08),
                0.002,
                tf,
                GrainAxis::X,
            );
            g.uvs
        })
    };
    let cuts = make(9, Transform::IDENTITY);
    assert_eq!(
        cuts,
        make(
            9,
            Transform::from_xyz(0.2, 1., 3.).with_rotation(Quat::from_rotation_y(0.9))
        )
    );
    assert_ne!(cuts, make(10, Transform::IDENTITY));
    for pair in cuts.windows(2) {
        assert_ne!(pair[0][0], pair[1][0]);
    }
}
