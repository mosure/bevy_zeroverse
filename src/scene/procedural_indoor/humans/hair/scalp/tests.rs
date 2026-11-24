use super::*;
use crate::scene::procedural_indoor::humans::{
    build_human, sample_person, HumanPoseKind, HumanSurface,
};

#[test]
fn clipping_opens_both_banks_inside_a_cell_and_closes_the_visible_rims() {
    let vertices = [
        (-0.015, -0.012),
        (-0.015, 0.012),
        (0.015, 0.012),
        (0.015, -0.012),
    ];
    let half_width = 0.0015;
    // The hairline crosses the same cell as the part. All four corners are
    // outside the gap, so alpha, a per-vertex threshold or a centroid mask fail.
    let quad = vertices.map(|(x, z)| Vertex {
        p: Vec3::new(x, 0.008, z),
        inner: Vec3::new(x, -0.00035, z),
        n: Vec3::Y,
        uv: Vec2::new(x, z),
        field: z,
        bank: Vec2::new(x, half_width),
    });
    let mut g = Geometry::default();
    for indices in [[0, 1, 2], [0, 2, 3]] {
        cut_triangle(indices.map(|i| quad[i]), true, &mut g);
    }
    let mut area = 0.;
    let mut lengths = [0.; 2];
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        let [ua, ub, uc] = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
        let normal = (b - a).cross(c - a);
        assert!(normal.length_squared() > 1e-18);
        assert!(
            (ub - ua).perp_dot(uc - ua).abs() > 1e-11,
            "singular tangent chart"
        );
        assert!([a, b, c]
            .iter()
            .all(|v| v.z >= -1e-8 && v.x.abs() >= half_width - 1e-8));
        if normal.y > 0. {
            area += normal.y * 0.5;
            assert!(
                [a, b, c].iter().all(|v| v.x * a.x > 0.),
                "hair bridges the part"
            );
        } else if normal.x.abs() > normal.z.abs() {
            assert!(normal.x * a.x < 0., "rim faces away from the exposed skin");
            assert!([a, b, c].iter().any(|v| v.y < 0.));
            lengths[usize::from(a.x > 0.)] += 0.5 * normal.length() / 0.00835;
        }
    }
    assert!((area - (0.030 - 2. * half_width) * 0.012).abs() < 1e-9);
    assert!(
        lengths.iter().all(|s| (s - 0.012).abs() < 1e-7),
        "{lengths:?}"
    );
}

#[test]
fn shared_edges_ignore_winding_and_skin_follows_the_actual_anny_diagonal() {
    let q = [9, 6, 13, 4];
    let next = [13, 6, 20, 16];
    for steps in [2, 3] {
        for t in 0..=steps {
            assert_eq!(
                shared_point(q, steps, t, steps),
                shared_point(next, steps - t, 0, steps)
            );
        }
        assert!(shared_point(q, 1, 1, steps).is_none());
    }
    let skin = [Vec3::ZERO, Vec3::X, Vec3::ONE, Vec3::Z];
    for (u, v) in [(0.75, 0.25), (0.25, 0.75), (0.4, 0.4)] {
        assert!(skin_point(&skin, [0, 1, 2, 3], u, v).abs_diff_eq(Vec3::new(u, u.min(v), v), 1e-7));
    }
}

// Orthographic rays through the complete posed meshes. Use the nearest
// front-facing triangle of each surface, as a depth-tested capture would.
fn crown_depths(mesh: &Geometry, lo: Vec2, span: Vec2, size: usize) -> Vec<f32> {
    let mut depth = vec![f32::NEG_INFINITY; size * size];
    for tri in mesh.indices.as_chunks::<3>().0 {
        let p = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
        let [a, b, c] = p.map(|v| (v.xz() - lo) / span * size as f32);
        let area = (b - a).perp_dot(c - a);
        if area >= -1e-8 {
            continue;
        }
        let min = a.min(b).min(c).max(Vec2::ZERO).floor().as_uvec2();
        let max = a
            .max(b)
            .max(c)
            .min(Vec2::splat(size as f32))
            .ceil()
            .as_uvec2();
        for z in min.y..max.y {
            for x in min.x..max.x {
                let offset = Vec2::new(x as f32 + 0.5, z as f32 + 0.5) - a;
                let v = offset.perp_dot(c - a) / area;
                let w = (b - a).perp_dot(offset) / area;
                if v >= 0. && w >= 0. && v + w <= 1. {
                    let y = p[0].y * (1. - v - w) + p[1].y * v + p[2].y * w;
                    let index = z as usize * size + x as usize;
                    depth[index] = depth[index].max(y);
                }
            }
        }
    }
    depth
}

#[test]
fn combed_part_reveals_the_actors_own_anny_scalp_in_a_posed_mesh() {
    let mut person = sample_person(
        50801,
        0,
        Vec3::ZERO,
        0.,
        HumanPoseKind::StandingRelaxed,
        None,
        false,
    );
    person.hairstyle = HairStyle::Swept as u8;
    let cut = person
        .appearance
        .as_mut()
        .unwrap()
        .hair_program
        .scalp_cut
        .as_mut()
        .unwrap();
    cut.part_width_m = 0.004;
    cut.part_definition = 0.;
    let unparted = build_human(&person);
    person
        .appearance
        .as_mut()
        .unwrap()
        .hair_program
        .scalp_cut
        .as_mut()
        .unwrap()
        .part_definition = 1.;
    let parted = build_human(&person);
    let skin = &parted.parts[&HumanSurface::Skin];
    assert_eq!(
        skin.positions,
        unparted.parts[&HumanSurface::Skin].positions
    );
    assert_eq!(parted.local_joints, unparted.local_joints);
    let hair = &unparted.parts[&HumanSurface::Hair];
    let bounds = hair
        .positions
        .iter()
        .map(|p| Vec3::from_array(*p).xz())
        .fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(p), hi.max(p)),
        );
    let span = bounds.1 - bounds.0;
    let size = 160;
    let skin_depth = crown_depths(skin, bounds.0, span, size);
    let closed_depth = crown_depths(hair, bounds.0, span, size);
    let parted_depth = crown_depths(&parted.parts[&HumanSurface::Hair], bounds.0, span, size);
    let mut exposed = 0usize;
    let mut covered = 0usize;
    for ((skin, closed), parted) in skin_depth.into_iter().zip(closed_depth).zip(parted_depth) {
        if skin.is_finite() && closed > skin + 0.00035 {
            covered += 1;
            if parted < skin {
                exposed += 1;
            }
        }
    }
    assert!(
        exposed >= 30 && exposed < covered / 8,
        "{exposed} out of {covered} crown rays exposed"
    );
}
