use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle,
    objects::{audit, build_object, part_label},
};

fn hit(a: &Assembly, eye: Vec3, ray: Vec3) -> Option<(f32, &str)> {
    a.parts
        .iter()
        .filter_map(|((_, label), mesh)| {
            let distance = mesh
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter_map(|tri| {
                    ray_triangle(
                        eye,
                        ray,
                        tri.map(|i| Vec3::from_array(mesh.positions[i as usize])),
                    )
                })
                .min_by(f32::total_cmp)?;
            Some((distance, part_label(label)))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
}

#[test]
fn pressed_fountains_have_open_basins_drains_and_usable_water_outlets() {
    let mut forms = [0; 3];
    let mut ceramic = 0;
    let mut left = 0;
    let mut heights = [false; 6];
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::DrinkingFountain, seed);
        o.finish_seed = Some(seed);
        heights[((mount_height_m(seed) - 0.505) / 0.026) as usize] = true;
        let f = Fountain::for_object(&o);
        let a = build_object(&o);
        let audit = audit::check(&o, &a).unwrap();
        assert!(audit.triangles < 4500);
        let (lo, hi) = a.bounds();
        assert!(
            lo.y.abs() < 1e-5
                && (lo.z + o.size.z * 0.5).abs() < 1e-5
                && (hi.y - o.size.y).abs() < 1e-5
        );
        let center = Fountain::bowl_center(o.size);
        let sample = |xy: Vec3| {
            let eye = xy.with_y(f.deck_m + 0.001);
            let (distance, class) = hit(&a, eye, Vec3::NEG_Y).unwrap();
            assert_eq!(class, "sink");
            eye.y - distance
        };
        let low = f.deck_m - f.basin_depth_m;
        // A face across any part of the opening or a shroud sealing the waste fails these rays.
        let floor = sample(center + Vec3::X * (0.12 * o.size.x));
        assert!((low..low + 0.012).contains(&floor));
        assert!((sample(center) - (low - 0.015)).abs() < 0.0001);
        assert!(
            (sample(center + Vec3::X * f.throat_radius_m * 0.32) - (low - 0.0019999)).abs()
                < 0.0001
        );
        assert!((sample(center + Vec3::Z * o.size.z * 0.425) - f.deck_m).abs() < 0.0001);
        let face = f.face(o.size);
        let tip = face.transform_point(Vec3::Y * f.guard_length_m);
        let axis = face.rotation * Vec3::Y;
        let (bore, class) = hit(&a, tip + axis * 0.008, -axis).unwrap();
        assert_eq!(class, "faucet");
        assert!((bore - (0.008 + 0.33 * f.guard_length_m)).abs() < 0.0002);
        assert!(sample(tip.with_x(tip.x * 0.8)) < f.deck_m - 0.008);
        if f.form == Form::BottleFiller {
            let eye = Vec3::new(0., o.size.y - 0.102, -0.163 * o.size.z);
            let (down, class) = hit(&a, eye, Vec3::NEG_Y).unwrap();
            assert_eq!(class, "sink");
            assert!(down > 0.30 && eye.y - down < f.deck_m);
            let (up, class) = hit(&a, eye, Vec3::Y).unwrap();
            assert_eq!(class, "faucet");
            assert!((eye.y + up - (o.size.y - 0.073)).abs() < 0.0001);
        }
        ceramic += usize::from(f.bowl == Surface::Ceramic);
        left += usize::from(f.left_bubbler);
        forms[f.form as usize] += 1;
    }
    assert!(heights.into_iter().all(|used| used));
    assert!(ceramic < forms[Form::Bracket as usize]);
    assert!(forms.iter().all(|&n| n >= 18) && ceramic >= 6 && (36..94).contains(&left));
}
