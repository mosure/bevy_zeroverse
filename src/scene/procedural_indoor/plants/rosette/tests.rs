use super::*;
use crate::scene::procedural_indoor::{layout::ObjectKind, objects::audit};

#[test]
fn fleshy_blade_is_closed_with_outward_normals_and_metric_cuticle_charts() {
    for seed in 0..24 {
        let growth = Growth::sample(seed);
        let habit = RosetteHabit::sample(seed, &growth);
        let mut mesh = Geometry::default();
        let blade = Blade {
            root: Vec3::new(0., 0.14, 0.),
            direction: radial(growth.shoot_phase),
            reach: 0.05,
            rise: 0.018,
            width: 0.012,
            bend: habit.tip_curl * 0.015,
            texture_origin: Vec2::new(0.15, 0.05),
        };
        blade.write(&mut mesh, &habit);
        let mut edges = std::collections::BTreeMap::new();
        for tri in mesh.indices.as_chunks::<3>().0 {
            let p = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
            let n = (p[1] - p[0]).cross(p[2] - p[0]);
            let face_normal: Vec3 = tri
                .iter()
                .map(|&i| Vec3::from_array(mesh.normals[i as usize]))
                .sum();
            assert!(n.dot(face_normal) > 0.);
            for k in 0..3 {
                let a = p[k].to_array().map(f32::to_bits);
                let b = p[(k + 1) % 3].to_array().map(f32::to_bits);
                let (edge, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let (count, balance) = edges.entry(edge).or_insert((0, 0));
                *count += 1;
                *balance += sign;
            }
        }
        assert!(edges.values().all(|&e| e == (2, 0)));
        for row in 0..STATIONS.len() {
            let first = 1 + row * (SIDES + 1);
            assert_eq!(mesh.normals[first], mesh.normals[first + SIDES]);
            for k in first..first + SIDES {
                let du = mesh.uvs[k + 1][0] - mesh.uvs[k][0];
                let chord = Vec3::from_array(mesh.positions[k + 1])
                    .distance(Vec3::from_array(mesh.positions[k]));
                assert!((du - chord).abs() < 1e-6 && du > 0.);
            }
        }
    }
}

#[test]
fn crowns_clear_the_pot_and_fleshy_blades_attach_in_both_spiral_directions() {
    const VERTICES: usize = 2 + STATIONS.len() * (SIDES + 1);
    let mut counts = std::collections::BTreeSet::new();
    let mut turns = [0; 2];
    let mut phases = [0; 8];
    let mut broadest = Vec::new();
    let mut pairs = 0;
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::Plant, seed);
        o.size = Vec3::new(0.36, 0.29, 0.36);
        let growth = Growth::sample(seed);
        let habit = RosetteHabit::sample(seed, &growth);
        let soil = Vec3::Y * 0.1;
        let rim = 0.122;
        let soil_radius = 0.11;
        let mut a = Assembly::default();
        build(&mut a, &o, &growth, soil, 0.13, soil_radius, rim);
        let mesh = a.part(Surface::Succulent, "other_prop");
        let (main, pup) = crowns(seed, &growth, &habit, 0.13, soil_radius);
        if let Some(pup) = pup {
            pairs += 1;
            assert!(main.spread >= 0.60 * 0.13);
            assert!(main.offset.distance(pup.offset) > 1.20 * (main.spread + pup.spread));
        }
        let mut index = 0;
        for crown in std::iter::once(main).chain(pup) {
            let p = |i| Vec3::from_array(mesh.positions[index + i]) - crown.offset;
            let low = p(0).y;
            let high = p(9).y;
            let r0 = p(0).xz().length();
            let r1 = p(9).xz().length();
            assert!(low < soil.y && high > rim && crown.offset.length() + r0 < soil_radius);
            // Two 9-vertex bark rings and two split cap fans.
            index += 2 * 9 + 2 * 10;
            let mut last_y = low;
            let mut pitches = Vec::new();
            for i in 0..crown.leaves {
                let leaf = &mesh.positions[index..index + VERTICES];
                index += VERTICES;
                let root = Vec3::from_array(leaf[0]);
                let tip = Vec3::from_array(*leaf.last().unwrap());
                let axis = tip - root;
                let chord = axis.xz();
                let t = (root.y - low) / (high - low);
                assert!(
                    root.y > last_y
                        && root.y < high
                        && (root - crown.offset).xz().length() < r0.lerp(r1, t) * (PI / 8.).cos()
                );
                assert!(leaf.iter().map(|p| Vec3::from_array(*p)).all(|p| p.y >= rim
                    && p.y <= o.size.y
                    && (p - crown.offset).xz().length() < crown.spread * 1.20));
                last_y = root.y;
                pitches.push(axis.y.atan2(chord.length()));
                let middle = 1 + 2 * (SIDES + 1);
                let width =
                    Vec3::from_array(leaf[middle + SIDES / 2]) - Vec3::from_array(leaf[middle]);
                assert!(width.xz().normalize().dot(chord.normalize()).abs() < 3e-5);
                if i == 0 {
                    let angle = chord.y.atan2(chord.x).rem_euclid(TAU);
                    phases[(angle * 8. / TAU) as usize % 8] += 1;
                    if crown.offset == main.offset {
                        broadest.push(width.length() / axis.length());
                    }
                }
            }
            assert!(pitches.last().unwrap() - pitches[0] > 0.50);
        }
        assert_eq!(index, mesh.positions.len());
        counts.insert(habit.leaf_count);
        turns[usize::from(habit.divergence_radians > 0.)] += 1;
    }
    broadest.sort_by(f32::total_cmp);
    assert!(counts.len() >= 12 && turns.into_iter().all(|n| n >= 24));
    assert!((24..72).contains(&pairs) && phases.into_iter().all(|n| n >= 3));
    assert!(broadest[86] > 1.55 * broadest[9]);
}
