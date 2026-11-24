use super::*;
use crate::scene::procedural_indoor::{
    items::sizing,
    objects::{audit, build_object},
};
#[test]
fn handsets_fit_replay_and_preserve_screen_and_semantic_contracts() {
    let mut layouts = std::collections::BTreeSet::new();
    let mut dimensions = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Phone, seed);
        o.size = sizing::loose(o.kind, &mut stream(seed, 9891)).unwrap();
        let p = parameters(&o);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-6);
        assert!(lo
            .cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - Vec3::splat(1e-5))
            .all());
        assert!(
            hi.cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + Vec3::splat(1e-5))
                .all(),
            "{seed}: {hi:?} / {:?}",
            o.size
        );
        for index in 0..32 {
            let theta = TAU * index as f32 / 32.;
            let shape = |v: f32| v.signum() * v.abs().powf(2. / p.island_roundness);
            let point = p.island_center
                + p.island_span * 0.5 * Vec2::new(shape(theta.cos()), shape(theta.sin()));
            let normalized = point.abs() / (p.span * 0.47);
            assert!(
                normalized.x.powf(p.roundness) + normalized.y.powf(p.roundness) < 1.,
                "island outside back: {seed} {point:?}"
            );
            for lens in &p.lens_centers[..p.lens_count] {
                let point = *lens + Vec2::new(theta.cos(), theta.sin()) * p.lens_radius;
                let normalized = (point - p.island_center).abs() / (p.island_span * 0.48);
                assert!(
                    normalized.x.powf(p.island_roundness) + normalized.y.powf(p.island_roundness)
                        < 1.,
                    "lens outside island: {seed} {point:?}"
                );
            }
        }
        assert!(a.parts.keys().all(|(_, label)| label.starts_with("phone")));
        let replay = build_object(&o);
        for (key, g) in &a.parts {
            assert_eq!(g.positions, replay.parts[key].positions);
            assert_eq!(g.indices, replay.parts[key].indices);
            if key.0 == Surface::PhoneScreen {
                assert!(g.normals.iter().all(|&n| n == Vec3::Y.to_array()));
                for axis in 0..2 {
                    let min = g.uvs.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min);
                    let max = g
                        .uvs
                        .iter()
                        .map(|v| v[axis])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!(min.abs() < 1e-6 && (max - 1.).abs() < 1e-6);
                }
            }
        }
        assert_eq!(
            a.parts.keys().any(|(s, _)| *s == Surface::PhoneScreen),
            p.face_up
        );
        layouts.insert((p.case, p.camera_layout, p.face_up));
        dimensions.insert(((p.span.x * 1000.) as u32, (p.body_height * 10000.) as u32));
        o.finish_seed = Some(1234);
        let q = parameters(&o);
        assert_eq!(p.span, q.span);
        assert_eq!(p.lens_centers, q.lens_centers);
        assert_eq!(p.face_up, q.face_up);
    }
    assert_eq!(layouts.len(), 24);
    assert!(dimensions.len() > 230);
}

#[test]
fn optical_inlays_cover_the_original_face_once_and_pass_through_rgb_screen_uvs() {
    use crate::scene::procedural_indoor::materials::handset;
    use std::collections::BTreeSet;
    let mut mounts = BTreeSet::new();
    let mut flash_positions = BTreeSet::new();
    let mut lateral = [false; 2];
    let mut glazed = BTreeSet::new();
    let mut matched_clusters = BTreeSet::new();
    let mut primary_indices = BTreeSet::new();
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::Phone, seed);
        o.size = sizing::loose(o.kind, &mut stream(seed, 37)).unwrap();
        let p = parameters(&o);
        let built = build_object(&o);
        let g = &built.parts[&(Surface::Ink, handset::LENS.into())];
        if let Some(camera) = p.front_camera {
            mounts.insert(camera.mount);
            if camera.mount == FrontMount::Punch && camera.center.x.abs() > p.span.x * 0.19 {
                lateral[usize::from(camera.center.x > 0.)] = true;
            }
            let (extent, z, y) = if camera.mount == FrontMount::Bezel {
                assert!(camera.center.y + camera.half_span.y < p.screen_z - p.screen_span.y * 0.5);
                assert!(camera.center.y - camera.half_span.y > -p.span.y * 0.49 + p.rim_width);
                (p.span * 0.49, 0., p.body_height)
            } else {
                (
                    p.screen_span * 0.5,
                    p.screen_z,
                    p.body_height + p.face_height,
                )
            };
            let mut cap = Geometry::default();
            Geometry::default().housing_with_top(
                &mut cap,
                &[(0., extent), (y, extent)],
                p.roundness,
                Transform::from_xyz(0., 0., z),
            );
            let tris = |g: &Geometry| -> Vec<[Vec2; 3]> {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.map(|i| Vec3::from_array(g.positions[i as usize]).xz()))
                    .collect()
            };
            let area = |t: &[Vec2; 3]| (t[2] - t[0]).perp_dot(t[1] - t[0]) * 0.5;
            let original: f32 = tris(&cap).iter().map(area).sum();
            let mut lens = Geometry::default();
            camera.inlay(&mut cap, &mut lens);
            assert_eq!(lens.positions, g.positions);
            assert!(g.positions.iter().all(|v| v[1] == y));
            let face = tris(&cap);
            let glass = tris(g);
            assert!(
                glass.iter().chain(&face).all(|t| area(t) > 0.),
                "inverted cap on {seed}"
            );
            let total: f32 = glass.iter().chain(&face).map(area).sum();
            assert!((total - original).abs() < original * 3e-6);
            let contains = |t: &[Vec2; 3], v: Vec2| {
                (0..3).all(|i| (t[(i + 1) % 3] - t[i]).perp_dot(v - t[i]) <= 1e-13)
            };
            for (offset, is_lens) in [
                (Vec2::new(0.16, -0.23), true),
                (Vec2::new(-0.72, 0.13), true),
                (Vec2::new(0.23, 1.2), false),
                (Vec2::new(-1.2, 0.14), false),
            ] {
                let ray = camera.center + camera.half_span * offset;
                assert_eq!(
                    glass.iter().filter(|t| contains(t, ray)).count(),
                    usize::from(is_lens)
                );
                assert_eq!(
                    face.iter().filter(|t| contains(t, ray)).count(),
                    usize::from(!is_lens)
                );
            }
            assert!(p.rear_flash.is_none());
        } else {
            assert!(p.lens_radii[p.lens_count..].iter().all(|&v| v == 0.));
            let active = &p.lens_radii[..p.lens_count];
            assert!(p.lens_radius <= 0.0125 && active.contains(&p.lens_radius));
            assert!(active.iter().all(|&v| v > 0. && v <= p.lens_radius));
            if p.lens_count > 1 {
                let matched = active.iter().all(|&v| v == p.lens_radius);
                matched_clusters.insert(matched);
                if !matched {
                    primary_indices
                        .insert(active.iter().position(|&v| v == p.lens_radius).unwrap());
                }
            }
            glazed.insert(p.glazed_island);
            flash_positions.insert(p.rear_flash.map(|f| f.on_island));
            assert_eq!(
                built
                    .parts
                    .contains_key(&(Surface::Ink, handset::DIFFUSER.into())),
                p.rear_flash.is_some()
            );
            if let Some(f) = p.rear_flash {
                for c in &p.lens_centers[..p.lens_count] {
                    assert!(f.center.distance(*c) > f.radius + p.lens_radius);
                }
                for i in 0..32 {
                    let (s, c) = (TAU * i as f32 / 32.).sin_cos();
                    let point = f.center + f.radius * Vec2::new(c, s);
                    let outline = point.abs() / (p.span * 0.47);
                    assert!(outline.x.powf(p.roundness) + outline.y.powf(p.roundness) < 1.);
                    let island = (point - p.island_center).abs()
                        / (p.island_span * if f.on_island { 0.48 } else { 0.5 });
                    assert_eq!(
                        island.x.powf(p.island_roundness) + island.y.powf(p.island_roundness) < 1.,
                        f.on_island
                    );
                }
            }
        }
    }
    assert_eq!(mounts.len(), 3);
    assert_eq!(flash_positions.len(), 3);
    assert_eq!(glazed.len(), 2);
    assert_eq!(matched_clusters.len(), 2);
    assert!(primary_indices.len() >= 3);
    assert_eq!(lateral, [true; 2]);
}
