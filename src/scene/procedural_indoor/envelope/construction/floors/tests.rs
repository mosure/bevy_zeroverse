use super::*;

#[test]
fn exterior_apron_leaves_the_sunken_landing_and_flight_open() {
    // Measured outline from workshop 75153. The former apron extruded a
    // diagonal between the far-right wall and the notch through this flight.
    let footprint = [
        [-3.4389687, -2.8461258],
        [3.3975334, -2.8461258],
        [3.5502717, 2.8461258],
        [-3.5502717, 2.8461258],
        [-3.5315988, 1.8911521],
        [-2.8979707, 1.6437134],
        [-2.851679, -0.72373456],
        [-3.4751499, -0.995754],
    ]
    .map(Vec2::from_array);
    let half = Vec2::new(3.5502717, 2.8461258) + Vec2::splat(0.25);
    let mut g = Geometry::default();
    exterior_apron(&mut g, &footprint, half);

    let mut areas = [0.; 3]; // top, underside, vertical returns
    for ids in g.indices.as_chunks::<3>().0 {
        let v = ids.map(|i| Vec3::from_array(g.positions[i as usize]));
        // Clip the projected triangle, including edge-on faces. An area-only
        // check misses a vertical sheet several metres long.
        assert!(
            polygon::clip_box(
                &v.map(|p| p.xz()),
                Vec2::new(-1.32, -1.87),
                Vec2::new(2.47, 0.24),
            )
            .is_empty(),
            "exterior face inside the floor cutout: {v:?}"
        );
        let cross = (v[1] - v[0]).cross(v[2] - v[0]);
        let side = if cross.y > 0. {
            0
        } else if cross.y < 0. {
            1
        } else {
            2
        };
        areas[side] += f64::from(cross.length()) * 0.5;
    }
    let cap = f64::from(4. * half.x * half.y - polygon::area(&footprint));
    let perimeter = polygon::edges(&footprint)
        .map(|(a, b)| a.distance(b))
        .sum::<f32>();
    let sides = 0.16 * f64::from(perimeter + 4. * half.element_sum());
    for (built, expected) in areas.into_iter().zip([cap, cap, sides]) {
        assert!((built - expected).abs() < 1e-5, "{built} != {expected}");
    }
}
