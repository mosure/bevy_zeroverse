use super::*;

#[test]
fn displaced_driver_baffles_tile_the_rectangle_without_folds_or_holes() {
    for aspect in [0.08, 0.25, 1., 4., 12.] {
        for offset in [-0.8, -0.3, 0., 0.3, 0.8] {
            let half: f32 = 0.2;
            let height = half * 2. * aspect;
            let radius = half.min(height * 0.5) * 0.75;
            let mut p = parameters(&super::super::super::audit::fixture(ObjectKind::Speaker, 0));
            p.horizontal = false;
            p.body.x = half * 2. + p.wall * 2.;
            let d = Driver {
                low: 0.,
                high: height,
                center: Vec2::new((half - radius) * offset, height * 0.5),
                radius,
                depth: 0.01,
                dome: 0.3,
            };
            let mut g = Geometry::default();
            baffle(&mut g, &p, &d, Transform::IDENTITY);
            let mut area = 0.;
            for triangle in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                let twice_area = (b - a).cross(c - a).z;
                assert!(
                    twice_area > 0.,
                    "aspect={aspect}, offset={offset}: {a:?} {b:?} {c:?}"
                );
                area += twice_area * 0.5;
            }
            let expected = 2. * half * height - 16. * radius * radius * (TAU / 32.).sin();
            assert!((area - expected).abs() < expected * 2e-5);
        }
    }
}

#[test]
fn supported_and_wall_installations_share_product_design_and_fit_their_reserved_space() {
    use crate::scene::procedural_indoor::objects::{audit, build_object, part_label, wall};
    let mut counts = [0; 3];
    let mut pitched = 0;
    let mut tall_pitches = std::collections::BTreeSet::new();
    let mut faces = std::collections::BTreeSet::new();
    for seed in 0..300 {
        for size in [
            Vec3::new(0.09, 0.145, 0.067),
            Vec3::new(0.25, 0.62, 0.18),
            Vec3::new(1.08, 0.15, 0.11),
        ] {
            let mut o = audit::fixture(ObjectKind::Speaker, seed);
            o.finish_seed = Some(seed.rotate_right(9));
            o.size = size;
            o.solid = false;
            o.id = (2 * seed) as usize;
            o.support = if seed % 3 == 0 { Some(4) } else { None };
            o.position.y = if o.support.is_some() { 0.75 } else { 2.075 };
            let p = parameters(&o);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let (lo, hi) = a.bounds();
            assert!(
                lo.y >= -1e-6
                    && hi.y <= size.y + 1e-6
                    && lo.x >= -size.x * 0.5 - 1e-6
                    && hi.x <= size.x * 0.5 + 1e-6
                    && hi.z <= size.z * 0.5 + 1e-6
            );
            assert!(a
                .parts
                .keys()
                .all(|(_, label)| part_label(label) == "speaker"));
            let has = |s, label: &str| {
                a.parts
                    .keys()
                    .any(|(surface, l)| *surface == s && l == label)
            };
            assert_eq!(has(Surface::Rubber, acoustic::DAMPER_LABEL), !p.covered);
            assert!(has(
                if p.covered {
                    p.grille
                } else {
                    Surface::Plastic
                },
                p.acoustic_label()
            ));
            assert_eq!(
                has(Surface::Metal, acoustic::YOKE_LABEL),
                p.mount.kind == MountKind::WallYoke
            );
            faces.insert(p.face_name());
            if let Some(offset) = wall::mount_offset(&o) {
                assert!((lo.z + offset - 0.001).abs() < 1e-6);
            }
            match p.mount.kind {
                MountKind::Desktop => {
                    counts[0] += 1;
                    assert!(
                        o.support.is_some() && lo.y.abs() < 1e-6 && p.mount.pitch_radians == 0.
                    );
                }
                MountKind::FlushWall => {
                    counts[1] += 1;
                    assert!(o.support.is_none() && p.mount.pitch_radians == 0.);
                }
                MountKind::WallYoke => {
                    counts[2] += 1;
                    let dir = p.mount.cabinet(p.body).rotation * Vec3::Z;
                    assert!(dir.y < 0. && (-dir.y).asin() <= 28.01_f32.to_radians());
                    let cabinet_min = a
                        .parts
                        .iter()
                        .filter(|((_, label), _)| label != acoustic::YOKE_LABEL)
                        .flat_map(|(_, g)| g.positions.iter().map(|v| v[2]))
                        .fold(f32::INFINITY, f32::min);
                    assert!(cabinet_min > p.mount.rear_plane + p.mount.strap);
                    tall_pitches.insert((p.mount.pitch_radians.to_degrees() * 10.) as i32);
                    o.id += 1;
                    let other = parameters(&o);
                    assert_eq!(p.body, other.body);
                    assert_eq!(
                        (p.mount.kind, p.finish, p.covered, p.grille),
                        (other.mount.kind, other.finish, other.covered, other.grille)
                    );
                    pitched += usize::from(
                        (p.mount.pitch_radians - other.mount.pitch_radians).abs() > 0.002,
                    );
                }
            }
        }
    }
    assert!(
        counts.iter().all(|n| *n > 65)
            && pitched > 75
            && tall_pitches.len() > 55
            && faces.len() == 3,
        "{counts:?}, {pitched}, {tall_pitches:?}, {faces:?}"
    );
}
