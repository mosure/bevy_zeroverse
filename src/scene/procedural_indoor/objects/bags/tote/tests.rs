use super::*;
use crate::scene::procedural_indoor::envelope::ray_triangle;
use std::collections::{BTreeMap, BTreeSet};

fn hit(g: &Geometry, eye: Vec3, direction: Vec3) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            ray_triangle(
                eye,
                direction,
                t.map(|i| Vec3::from_array(g.positions[i as usize])),
            )
        })
        .min_by(f32::total_cmp)
}

#[test]
fn open_cut_has_a_watertight_lined_cavity_and_physical_texture_coordinates() {
    let mut count = 0;
    let mut handle_pairs = BTreeSet::new();
    let mut lining_types = BTreeSet::new();
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::Bag, seed);
        o.size = match seed % 3 {
            0 => Vec3::new(0.25, 0.47, 0.13),
            1 => Vec3::new(0.52, 0.25, 0.26),
            _ => o.size,
        };
        let p = parameters(&o);
        let Some(c) = ToteCut::sample(&o, &p) else {
            continue;
        };
        let shell = Shell::new(&c, o.size, &p);
        let mut skin = Geometry::default();
        shell.skin(&mut skin, false);
        shell.skin(&mut skin, true);
        shell.binding(&mut skin);
        let key = |pos: [f32; 3]| pos.map(|x| (x * 1e7).round() as i32);
        let mut edges = BTreeMap::<_, (u16, i16)>::new();
        for tri in skin.indices.as_chunks::<3>().0 {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let (a, b) = (
                    key(skin.positions[a as usize]),
                    key(skin.positions[b as usize]),
                );
                let edge = edges.entry((a.min(b), a.max(b))).or_default();
                edge.0 += 1;
                edge.1 += if a < b { 1 } else { -1 };
            }
        }
        assert!(edges.values().all(|&e| e == (2, 0)), "seed {seed}");
        let built = build_object(&o);
        assert!(audit::check(&o, &built).unwrap().triangles < 1200);
        assert!(built.parts.keys().all(|(_, l)| part_label(l) == "bag"));
        for g in built.parts.values() {
            for t in g.indices.as_chunks::<3>().0 {
                let uv = t.map(|i| Vec2::from_array(g.uvs[i as usize]));
                assert!(
                    (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-12,
                    "collapsed fabric chart {seed}"
                );
            }
        }
        // A model trained from above should see the lining floor through the bag mouth.
        let eye = Vec3::new(o.size.x * 0.031, o.size.y, -o.size.z * 0.027);
        let depth = built
            .parts
            .values()
            .filter_map(|g| hit(g, eye, Vec3::NEG_Y))
            .min_by(f32::total_cmp)
            .unwrap();
        assert!((eye.y - depth - c.base_m).abs() < 1e-5);
        let mut archived = Assembly::default();
        super::super::build(&mut archived, &o, OPEN_TOTE_VERSION - 1);
        let closed = archived
            .parts
            .values()
            .filter_map(|g| hit(g, eye, Vec3::NEG_Y))
            .min_by(f32::total_cmp)
            .unwrap();
        assert!(depth - closed > o.size.y * 0.4);
        let lip = shell.inside.last().unwrap();
        assert!((lip[4].z - lip[16].z - (c.mouth_depth_m - 2. * c.hem_m)).abs() < 1e-6);
        for face in 0..2 {
            let grip = shell.handle(&c, o.size, &p, face);
            let sign = if face == 0 { 1. } else { -1. };
            for end in [grip.points[0], grip.points[16]] {
                let dir = Vec3::NEG_Z * sign;
                let t = hit(&skin, end - dir * 0.015, dir).unwrap();
                assert!(
                    (t - 0.015).abs() < grip.thickness * 0.5,
                    "unsewn handle {seed}"
                );
            }
            let avg = (grip.points[0].y + grip.points[16].y) * 0.5;
            assert_eq!(grip.points[8].y < avg, c.draped_handles[face]);
        }
        handle_pairs.insert(c.draped_handles);
        lining_types.insert(c.lining);
        count += 1;
    }
    assert!(count > 55 && handle_pairs.len() == 4 && lining_types.len() >= 2);
}

#[test]
fn changing_the_lining_finish_keeps_the_cut_and_handle_poses() {
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::Bag, seed);
        let p = parameters(&o);
        let Some(first) = ToteCut::sample(&o, &p) else {
            continue;
        };
        o.finish_seed = Some(seed ^ 2345);
        let next = ToteCut::sample(&o, &parameters(&o)).unwrap();
        assert_eq!(
            Shell::new(&first, o.size, &p).outside,
            Shell::new(&next, o.size, &p).outside
        );
        assert_eq!(first.draped_handles, next.draped_handles);
        assert_eq!(first.handle_offset, next.handle_offset);
    }
}
