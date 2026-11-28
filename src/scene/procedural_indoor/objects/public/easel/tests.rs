use super::*;
use crate::scene::procedural_indoor::{
    materials::canvas_stock,
    objects::{audit, build_object, part_label},
};
use std::collections::BTreeSet;

#[test]
fn easel_frames_clamp_an_upright_canvas_inside_the_admitted_volume() {
    let mut combinations = BTreeSet::new();
    let mut shallow = 1.;
    let mut steep = 0.;
    // Audit consecutive products, then explicitly exercise any rarer stock/frame
    // combinations still missing without rebuilding already-covered meshes.
    for seed in 0..1536 {
        if seed >= 192 && combinations.len() == 16 {
            break;
        }
        let o = audit::fixture(ObjectKind::ArtistEasel, seed);
        let p = Easel::sample(&o);
        let new = combinations.insert((p.stand as u8, p.canvas as u8, p.frame_finish as u8));
        if seed >= 192 && !new {
            continue;
        }
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(check.triangles < 3000);
        let frame = p.rail_frame(&o);
        let mut ground = f32::INFINITY;
        for g in a.parts.values() {
            for v in &g.positions {
                ground = ground.min(v[1]);
            }
        }
        assert!(ground.abs() < 0.001);
        shallow = f32::min(shallow, p.lean_radians);
        steep = f32::max(steep, p.lean_radians);
        assert!(a
            .parts
            .keys()
            .all(|(_, l)| matches!(part_label(l), "easel" | "picture")));
        assert_eq!(
            a.parts.keys().any(|(_, l)| part_label(l) == "picture"),
            p.canvas != Canvas::Empty
        );
        if p.canvas == Canvas::Empty {
            continue;
        }
        let (surface, key) = match p.canvas {
            Canvas::Painted => (Surface::Art, "picture#uv_local"),
            Canvas::Linen => (Surface::Paper, canvas_stock::RAW),
            Canvas::Primed => (Surface::Paper, canvas_stock::PRIMED),
            Canvas::Empty => unreachable!(),
        };
        let (_, g) = a
            .parts
            .iter()
            .find(|((s, l), _)| *s == surface && l.starts_with(key))
            .unwrap();
        let n = frame.rotation * Vec3::Z;
        let indices = (0..g.positions.len())
            .filter(|i| Vec3::from_array(g.normals[*i]).dot(n) > 0.99999)
            .collect::<Vec<_>>();
        assert_eq!(indices.len(), 4);
        let local =
            |i| frame.rotation.inverse() * (Vec3::from_array(g.positions[i]) - frame.translation);
        let v = local(indices[0]);
        let opposite = local(indices[2]);
        assert!((v.z - p.canvas_back_z() - p.canvas_depth_m).abs() < 0.00001);
        assert!((v.y - p.tray_position_m).abs() < 0.00001);
        assert!((opposite.x - v.x - p.canvas_width_m).abs() < 0.00001);
        assert!((opposite.y - v.y - p.canvas_height_m).abs() < 0.00001);
        if p.canvas == Canvas::Painted {
            assert_eq!(g.uvs[indices[0]], [0., 1.]);
            assert_eq!(g.uvs[indices[2]], [1., 0.]);
        }
    }
    assert_eq!(combinations.len(), 16);
    assert!(shallow < 0.06 && steep > 0.23);
}
