use super::*;

// Intersect coplanar front/back triangles, including triangles within one batched
// material mesh. Shared edges are legitimate; overlapping positive area is not.
fn intersection_area(mut subject: Vec<Vec2>, mut clip: [Vec2; 3]) -> f32 {
    if (clip[1] - clip[0]).perp_dot(clip[2] - clip[0]) < 0. {
        clip.swap(1, 2);
    }
    for i in 0..3 {
        let a = clip[i];
        let b = clip[(i + 1) % 3];
        let mut out = Vec::new();
        for j in 0..subject.len() {
            let u = subject[j];
            let v = subject[(j + 1) % subject.len()];
            let du = (b - a).perp_dot(u - a);
            let dv = (b - a).perp_dot(v - a);
            if du >= 0. {
                out.push(u);
            }
            if (du < 0.) != (dv < 0.) {
                out.push(u.lerp(v, du / (du - dv)));
            }
        }
        subject = out;
        if subject.len() < 3 {
            return 0.;
        }
    }
    subject
        .iter()
        .enumerate()
        .map(|(i, a)| a.perp_dot(subject[(i + 1) % subject.len()]))
        .sum::<f32>()
        .abs()
        * 0.5
}

#[test]
fn mounted_alarm_faces_have_no_positive_area_coplanar_overlap() {
    use crate::scene::procedural_indoor::objects::{audit, build_object};
    for kind in [
        ObjectKind::FireAlarm,
        ObjectKind::FirePullStation,
        ObjectKind::FireDetector,
    ] {
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            // Include small and large installed devices, not just one envelope.
            o.size *= [0.65, 1., 1.6][seed as usize % 3];
            let assembly = build_object(&o);
            audit::check(&o, &assembly).unwrap();
            let mut faces = vec![];
            for ((surface, label), geometry) in &assembly.parts {
                for tri in geometry.indices.as_chunks::<3>().0 {
                    let p = tri.map(|i| Vec3::from_array(geometry.positions[i as usize]));
                    // Detector's mounting face is horizontal; wall alarms face +Z.
                    let p = if kind == ObjectKind::FireDetector {
                        p.map(|p| Vec3::new(p.x, p.z, p.y))
                    } else {
                        p
                    };
                    if (p[0].z - p[1].z).abs().max((p[0].z - p[2].z).abs()) < 1e-7 {
                        faces.push((p.map(|p| p.xy()), p[0].z, surface, label));
                    }
                }
            }
            for (i, a) in faces.iter().enumerate() {
                for b in &faces[i + 1..] {
                    if (a.1 - b.1).abs() < 1e-7 {
                        assert!(
                            intersection_area(a.0.to_vec(), b.0) < o.size.length_squared() * 1e-7,
                            "coplanar {kind:?}/{seed}: {:?}/{} and {:?}/{} at {}",
                            a.2,
                            a.3,
                            b.2,
                            b.3,
                            a.1
                        );
                    }
                }
            }
        }
    }
}

// First intersection with a rendered material batch, measured in object metres.
fn first_hit(g: &Geometry, origin: Vec3, direction: Vec3) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            let (e, f) = (b - a, c - a);
            let k = direction.cross(f);
            let det = e.dot(k);
            if det.abs() < 1e-12 {
                return None;
            }
            let off = origin - a;
            let (u, v) = (off.dot(k) / det, direction.dot(off.cross(e)) / det);
            let d = f.dot(off.cross(e)) / det;
            (u >= 0. && v >= 0. && u + v <= 1. && d >= 0.).then_some(d)
        })
        .min_by(f32::total_cmp)
}

#[test]
fn radial_rays_enter_the_detector_chamber_between_ribs_and_the_lens_is_seated() {
    use super::detector::{parameters, DetectorFace, CHAMBER};
    use crate::scene::procedural_indoor::{
        materials::registered_finish,
        objects::{audit, build_object},
    };
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::FireDetector, seed);
        o.size = [
            Vec3::new(0.09, 0.067, 0.09),
            Vec3::new(0.184, 0.025, 0.184),
            o.size,
        ][(seed / 4 % 3) as usize];
        let p = parameters(&o);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap();
        let get = |slot| &a.parts[&(Surface::Plastic, format!("fire_detector#finish{slot}"))];
        let (shell, chamber, lens) = (get(p.finish), get(CHAMBER), get(p.lens_finish));
        assert!(
            registered_finish(Surface::Plastic, CHAMBER)
                && registered_finish(Surface::Plastic, p.lens_finish)
        );
        assert!((a.bounds().1.y - o.size.y).abs() < 1e-6 && a.bounds().0.y > 0.);
        let (pitch, t) = (TAU / p.inlets as f32, 0.28);
        let y = p.inlet_bottom_m + t * (p.inlet_top_m - p.inlet_bottom_m);
        for bay in 0..p.inlets {
            let center = p.phase_rad + pitch * bay as f32 + t * p.rib_rake_rad;
            for (angle, solid) in [(center, true), (center + pitch * 0.5, false)] {
                let (s, c) = angle.sin_cos();
                let d = Vec3::new(c, 0., s);
                let eye = (d * p.radius_m * 1.1).with_y(y);
                let back = first_hit(chamber, eye, -d).unwrap();
                let radius = p.radius_m * 1.1 - back;
                assert!((p.chamber_radius_m * 0.98..=p.chamber_radius_m * 1.001).contains(&radius));
                assert_eq!(
                    first_hit(shell, eye, -d).is_some_and(|v| v < back),
                    solid,
                    "{seed}/{bay}"
                );
            }
        }
        let lens_origin = Vec3::new(p.lens_center_m.x, 0., p.lens_center_m.y);
        let cap = first_hit(lens, lens_origin, Vec3::Y).unwrap();
        assert!(cap < p.face_y_m);
        assert!((first_hit(shell, lens_origin, Vec3::Y).unwrap() - p.face_y_m).abs() < 1e-5);
        if p.face == DetectorFace::GuardedProbe {
            let metal = &a.parts[&(
                Surface::Metal,
                format!("fire_detector#finish{}", FINISH_BASE + 4),
            )];
            assert!(first_hit(metal, lens_origin, Vec3::Y).is_none());
            assert!(first_hit(metal, Vec3::ZERO, Vec3::Y).unwrap() < p.face_y_m);
        }
    }
}

#[test]
fn paired_detector_products_replay_independently_of_instance_id_and_yaw() {
    use crate::scene::procedural_indoor::objects::{audit, build_object};
    for seed in 0..16 {
        let a = audit::fixture(ObjectKind::FireDetector, seed);
        let mut b = a.clone();
        b.id = a.id + 1;
        b.yaw = a.yaw + 0.49;
        assert_eq!(audit::program(&a), audit::program(&b));
        assert_eq!(
            audit::descriptor(&a, &build_object(&a)),
            audit::descriptor(&b, &build_object(&b))
        );
    }
}
