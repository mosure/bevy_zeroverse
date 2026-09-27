use super::*;
use std::collections::BTreeMap;

fn intersection_area(a: [Vec2; 3], b: [Vec2; 3]) -> f32 {
    let orient = (b[1] - b[0]).perp_dot(b[2] - b[0]).signum();
    let mut polygon = a.to_vec();
    for i in 0..3 {
        let start = b[i];
        let edge = b[(i + 1) % 3] - start;
        let side = |p: Vec2| orient * edge.perp_dot(p - start);
        let mut next = Vec::new();
        for j in 0..polygon.len() {
            let p = polygon[j];
            let q = polygon[(j + 1) % polygon.len()];
            let (dp, dq) = (side(p), side(q));
            if dp >= 0.0 {
                next.push(p);
            }
            if (dp > 0.0 && dq < 0.0) || (dp < 0.0 && dq > 0.0) {
                next.push(p.lerp(q, dp / (dp - dq)));
            }
        }
        polygon = next;
    }
    (0..polygon.len())
        .map(|i| polygon[i].perp_dot(polygon[(i + 1) % polygon.len()]))
        .sum::<f32>()
        .abs()
        * 0.5
}

#[test]
fn architectural_trim_has_no_competing_coplanar_faces() {
    for seed in 0..16 {
        let scene = IndoorManifest::generate_with_humans(
            seed,
            super::super::layout::IndoorLayout::Mixed,
            0.5,
            0,
            0.0,
        )
        .unwrap();
        let assembly = architecture(&scene);
        let mut planes = BTreeMap::<(usize, i32, i32), Vec<([Vec2; 3], String)>>::new();
        for ((surface, label), geometry) in assembly.parts {
            for triangle in geometry.indices.as_chunks::<3>().0 {
                let p: [Vec3; 3] = std::array::from_fn(|i| {
                    Vec3::from_array(geometry.positions[triangle[i] as usize])
                });
                let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                let axis = (0..3)
                    .max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
                    .unwrap();
                if n[axis].abs() < 0.99999 {
                    continue;
                }
                // Contacts hidden below the floor/above the ceiling cannot compete
                // in an interior view (wall and jamb bottom caps share the slab).
                if axis == 1 && (p[0].y < 0.002 || p[0].y > scene.room_size.y - 0.002) {
                    continue;
                }
                let key = (
                    axis,
                    n[axis].signum() as i32,
                    (p[0][axis] * 10000.0).round() as i32,
                );
                let projected = p.map(|p| Vec2::new(p[(axis + 1) % 3], p[(axis + 2) % 3]));
                planes
                    .entry(key)
                    .or_default()
                    .push((projected, format!("{label}/{surface:?}")));
            }
        }
        let mut failures = Vec::new();
        for (plane, triangles) in planes {
            for (i, (a, la)) in triangles.iter().enumerate() {
                for (b, lb) in &triangles[i + 1..] {
                    if !(la.starts_with("door/")
                        || la.starts_with("window/")
                        || lb.starts_with("door/")
                        || lb.starts_with("window/"))
                    {
                        continue;
                    }
                    let area = intersection_area(*a, *b);
                    if area > 0.00001 && failures.len() < 8 {
                        failures.push(format!(
                            "{plane:?}: {la} / {lb}, area={area}, triangles={a:?}/{b:?}"
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "seed {seed}: {failures:?}; sill={} partitions={:?}",
            scene.window_sill,
            scene.program.as_ref().map(|p| &p.partitions)
        );
    }
}
