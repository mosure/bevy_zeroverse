//! Continuous swept cylinder for an enclosed, arbitrarily mounted impeller.
//! Sampling blade angles misses narrow obstacles. The corners of a swept box,
//! conversely, would collide with the outside of a correctly fitted grille.
use super::*;

pub(super) fn sweep(a: &Assembly, joint: &Joint, object: Transform) -> Option<Sweep> {
    let frame = Transform::from_translation(joint.pivot)
        .with_rotation(Quat::from_rotation_arc(Vec3::Y, joint.axis));
    let inverse = frame.rotation.inverse();
    let (mut lo, mut hi, mut radius_sq) = (f32::INFINITY, f32::NEG_INFINITY, 0.0_f32);
    for vertex in joint.parts.iter().flat_map(|key| &a.parts[key].positions) {
        let p = inverse * (Vec3::from_array(*vertex) - joint.pivot);
        lo = lo.min(p.y);
        hi = hi.max(p.y);
        radius_sq = radius_sq.max(p.xz().length_squared());
    }
    let pad = 0.0001;
    let r = radius_sq.sqrt() + pad;
    (lo <= hi).then(|| Sweep {
        local: (Vec3::new(-r, lo - pad, -r), Vec3::new(r, hi + pad, r)),
        frame: object.mul_transform(frame),
        shape: SweepShape::Rotor,
    })
}

pub(super) fn world_bounds((lo, hi): Bounds, frame: Transform) -> Bounds {
    let m = frame.compute_affine().matrix3;
    let x = Vec3::from(m.x_axis);
    let z = Vec3::from(m.z_axis);
    let radial = Vec3::new(x.x.hypot(z.x), x.y.hypot(z.y), x.z.hypot(z.z)) * hi.x;
    let axial = Vec3::from(m.y_axis).abs() * ((hi.y - lo.y) * 0.5);
    let center = frame.transform_point(Vec3::Y * ((hi.y + lo.y) * 0.5));
    (center - radial - axial, center + radial + axial)
}

/// Clip the static triangle at both end caps, then test the resulting convex
/// polygon against the disk. A triangle can cross the disk with all three
/// vertices outside. Two parallel plane clips produce at most five vertices.
pub(super) fn intersects(triangle: [Vec3; 3], (lo, hi): Bounds) -> bool {
    fn clip(input: &[Vec3], out: &mut [Vec3; 6], height: f32, sign: f32) -> usize {
        let mut count = 0;
        let mut previous = input[input.len() - 1];
        let mut a = sign * (previous.y - height);
        for &current in input {
            let b = sign * (current.y - height);
            if (a < 0. && b > 0.) || (a > 0. && b < 0.) {
                out[count] = previous.lerp(current, a / (a - b));
                count += 1;
            }
            if b >= 0. {
                out[count] = current;
                count += 1;
            }
            previous = current;
            a = b;
        }
        count
    }

    let mut lower = [Vec3::ZERO; 6];
    let mut slab = [Vec3::ZERO; 6];
    let n = clip(&triangle, &mut lower, lo.y, 1.);
    if n == 0 {
        return false;
    }
    let n = clip(&lower[..n], &mut slab, hi.y, -1.);
    if n == 0 {
        return false;
    }

    let radius_sq = hi.x * hi.x;
    let mut a = slab[n - 1].xz();
    let (mut left, mut right) = (false, false);
    for p in &slab[..n] {
        let b = p.xz();
        let edge = b - a;
        let length_sq = edge.length_squared();
        let t = if length_sq > 0. {
            (-a.dot(edge) / length_sq).clamp(0., 1.)
        } else {
            0.
        };
        if (a + t * edge).length_squared() <= radius_sq {
            return true;
        }
        let side = a.perp_dot(b);
        left |= side > 0.;
        right |= side < 0.;
        a = b;
    }
    // Consistent nonzero winding puts the axis inside. A collinear projection
    // that contains it was already detected by the segment distance test.
    left != right
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cylinder_hits_boundaries_and_excludes_guard_corners() {
        let cap = (Vec3::new(-1., -0.1, -1.), Vec3::new(1., 0.1, 1.));
        let tri = |p: [[f32; 3]; 3]| p.map(Vec3::from_array);
        for (triangle, hit) in [
            ([[-2., 0., -2.], [2., 0., -2.], [0., 0., 3.]], true),
            ([[1., -1., 0.], [1., 1., 0.], [2., 0., 0.]], true),
            ([[0., 0.1, 0.], [2., 1., 0.], [0., 1., 2.]], true),
            (
                [[0.84, 0., 0.84], [0.95, 0.1, 0.8], [0.8, -0.1, 0.95]],
                false,
            ),
            ([[0., 1., 0.], [2., -0.1, -0.3], [2., -0.1, 0.3]], false),
            ([[0., -0.2, 0.], [1., -0.3, 0.], [0., -0.4, 1.]], false),
            ([[2., 0., 0.], [3., 0., 0.], [4., 0., 0.]], false),
        ] {
            assert_eq!(intersects(tri(triangle), cap), hit);
            assert_eq!(
                intersects(tri([triangle[2], triangle[1], triangle[0]]), cap),
                hit
            );
        }
    }

    #[test]
    #[ignore = "builds reference and bounded BVHs over three complete furnished rooms"]
    fn furnished_rotor_admission_matches_full_scene_and_measures_preparation() {
        use crate::scene::procedural_indoor::layout::IndoorLayout;
        let policy = IndoorMotionConfig {
            desk_fan_fraction: 1.,
            chair_fraction: 0.,
            door_fraction: 0.,
            covering_fraction: 0.,
            appliance_fraction: 0.,
            prop_fraction: 0.,
            cart_fraction: 0.,
            cabinet_fraction: 0.,
            fan_fraction: 0.,
            ..default()
        };
        let mut samples = vec![];
        for (i, seed) in [78700, 78717, 78739].into_iter().enumerate() {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.72, 3, 0.25)
                    .unwrap();
            scene.indoor_motion = Some(IndoorMotionReport::new(policy.clone()));
            let (g, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
            let measure = |scoped| {
                let start = std::time::Instant::now();
                let tree = if scoped {
                    Tree::for_admission(&scene, &g)
                } else {
                    Tree::new(&scene, &g)
                };
                let build_ms = start.elapsed().as_secs_f64() * 1e3;
                let mut contacts = vec![];
                for (id, a) in g.objects.iter().enumerate() {
                    for (index, j) in a.joints.iter().enumerate() {
                        assert_eq!(j.kind, MotionKind::Fan);
                        for s in sweeps(a, j, scene.objects[id].transform()) {
                            contacts.push(tree.hits(
                                s,
                                Some(id),
                                &[index],
                                j.attachment.as_deref(),
                                None,
                            ));
                        }
                    }
                }
                (tree.triangles.len(), build_ms, contacts)
            };
            let (full, local) = if i % 2 == 0 {
                (measure(false), measure(true))
            } else {
                let local = measure(true);
                (measure(false), local)
            };
            assert!(!full.2.is_empty() && full.2 == local.2);
            assert!(
                local.0 < full.0 / 2,
                "seed {seed}: {} of {} triangles",
                local.0,
                full.0
            );
            samples.push(serde_json::json!({"seed":seed,"rotors":full.2.len(),"contacts":full.2,"full_triangles":full.0,"scoped_triangles":local.0,"full_build_ms":full.1,"scoped_build_ms":local.1}));
        }
        let dir = std::path::Path::new("out/indoor_rotating_objects");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("planning.json"),
            serde_json::to_vec_pretty(&samples).unwrap(),
        )
        .unwrap();
    }
}
