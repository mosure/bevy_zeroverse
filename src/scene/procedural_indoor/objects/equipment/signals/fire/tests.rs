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
