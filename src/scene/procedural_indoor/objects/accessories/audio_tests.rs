use super::*;
use crate::scene::procedural_indoor::{
    clutter::microphone,
    items::sizing,
    objects::{audit, build_object},
};
use std::collections::BTreeSet;

#[test]
fn audio_and_stationery_extremes_replay_ground_and_fit() {
    use ObjectKind::*;
    let cases = [
        (
            Speaker,
            [
                [0.075, 0.095, 0.065],
                [0.19, 0.38, 0.23],
                [0.52, 0.065, 0.09],
            ],
        ),
        (
            Headphones,
            [[0.16, 0.055, 0.18], [0.24, 0.26, 0.14], [0.20, 0.09, 0.25]],
        ),
        (
            Microphone,
            [[0.075, 0.018, 0.08], [0.10, 0.38, 0.09], [0.17, 0.17, 0.20]],
        ),
        (
            Stapler,
            [
                [0.023, 0.024, 0.08],
                [0.065, 0.08, 0.19],
                [0.04, 0.045, 0.17],
            ],
        ),
        (
            TapeDispenser,
            [
                [0.031, 0.038, 0.09],
                [0.096, 0.135, 0.20],
                [0.07, 0.07, 0.19],
            ],
        ),
    ];
    let mut errors = Vec::new();
    for (kind, sizes) in cases {
        for seed in 0..96 {
            for size in sizes {
                let mut o = audit::fixture(kind, seed);
                o.size = Vec3::from_array(size);
                let a = build_object(&o);
                let b = build_object(&o);
                match audit::check(&o, &a) {
                    Err(e) => errors.push(format!("{kind:?}/{seed}/{size:?}: {e}")),
                    Ok(q) => {
                        if q.max_envelope_overrun_m >= 0.0005
                            || a.bounds().0.y.abs() >= 0.0001
                            || q.triangles > 12000
                        {
                            errors.push(format!(
                                "{kind:?}/{seed}/{size:?}: {q:?} bounds={:?}",
                                a.bounds()
                            ));
                        }
                    }
                }
                assert_eq!(a.parts.len(), b.parts.len());
                for (key, g) in &a.parts {
                    let h = &b.parts[key];
                    assert_eq!(g.positions, h.positions);
                    assert_eq!(g.normals, h.normals);
                    assert_eq!(g.uvs, h.uvs);
                    assert_eq!(g.indices, h.indices);
                }
            }
        }
    }
    assert!(
        errors.is_empty(),
        "{} failures: {:?}",
        errors.len(),
        errors.iter().take(12).collect::<Vec<_>>()
    );
}

fn front_at(a: &Assembly, q: Vec2) -> f32 {
    let mut z = f32::NEG_INFINITY;
    for g in a.parts.values() {
        for ids in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = ids.map(|i| Vec3::from_array(g.positions[i as usize]));
            let e = b.xy() - a.xy();
            let f = c.xy() - a.xy();
            let v = q - a.xy();
            let det = e.perp_dot(f);
            if det.abs() < 1e-10 {
                continue;
            }
            let u = v.perp_dot(f) / det;
            let w = e.perp_dot(v) / det;
            if u >= -1e-5 && w >= -1e-5 && u + w <= 1.00001 {
                z = z.max(a.z + u * (b.z - a.z) + w * (c.z - a.z));
            }
        }
    }
    z
}
#[test]
fn exposed_speaker_drivers_have_real_apertures_and_separated_cells() {
    let mut drivers = 0;
    let mut counts = BTreeSet::new();
    let mut covered = 0;
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Speaker, seed);
        o.size = sizing::loose(o.kind, &mut stream(seed, 9191)).unwrap();
        let p = speaker::parameters(&o);
        counts.insert(p.drivers.len());
        if p.covered {
            covered += 1;
            continue;
        }
        let a = build_object(&o);
        for d in &p.drivers {
            let q = if p.horizontal {
                Vec2::new(d.center.y, p.body.y * 0.5 - d.center.x)
            } else {
                Vec2::new(d.center.x, p.body.y * 0.5 + d.center.y)
            };
            let z = front_at(&a, q);
            assert!(
                z < p.body.z * 0.5 - d.depth * 0.20 && z > p.body.z * 0.5 - d.depth * 1.1,
                "{seed}: driver obscured/absent at {q:?}, z={z}"
            );
            assert!(d.center.y - d.radius > d.low && d.center.y + d.radius < d.high);
            drivers += 1;
        }
    }
    assert!(drivers > 80 && counts.len() >= 4 && covered > 25 && covered < 95);
    println!("speaker audit: drivers={drivers}, counts={counts:?}, covered={covered}/128");
}
#[test]
fn tape_webs_are_tangent_and_rolls_clear_the_support() {
    let mut with_web = 0;
    let mut open = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::TapeDispenser, seed);
        o.size = sizing::loose(o.kind, &mut stream(seed, 8181)).unwrap();
        let p = stationery::tape_parameters(&o);
        assert!(p.center.y - p.radius > p.base.y);
        assert!(p.radius * p.core + 0.0011 < p.radius);
        assert!(
            p.width * 0.52 < p.base.x * 0.34,
            "roll/core intersects cradle cheeks"
        );
        let radial = p.tangent - p.center;
        let web = p.blade + Vec3::Y * 0.0006 - p.tangent;
        assert!((radial.length() - p.radius).abs() < 1e-5);
        assert!(
            radial.normalize().dot(web.normalize()).abs() < 1e-5,
            "{seed}: web not tangent"
        );
        with_web += usize::from(p.web);
        open += usize::from(p.open_frame);
    }
    assert!(with_web > 100 && with_web < 245 && open > 60 && open < 210);
}
#[test]
fn audio_form_factors_and_mechanical_states_survive_metric_sampling() {
    let mut flat = 0;
    let mut microphone_types = [0; 3];
    let mut openings = BTreeSet::new();
    let mut poses = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for seed in 0..256 {
        let mut h = audit::fixture(ObjectKind::Headphones, seed);
        h.size = sizing::loose(h.kind, &mut stream(seed, 234)).unwrap();
        let p = headphones::parameters(&h);
        flat += usize::from(p.flat);
        poses.insert(serde_json::to_string(&p.resting_rotation).unwrap());
        let a = build_object(&h);
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-5);
        if p.flat {
            assert!(hi.y < (hi.z - lo.z) * 0.70);
        }
        let mut m = audit::fixture(ObjectKind::Microphone, seed);
        m.size = sizing::loose(m.kind, &mut stream(seed, 235)).unwrap();
        microphone_types[microphone::parameters(&m).construction as usize] += 1;
        let s = audit::fixture(ObjectKind::Stapler, seed);
        let p = stationery::stapler_parameters(&s);
        openings.insert((p.opening * 1000.) as u32);
        shapes.insert(serde_json::to_string(&p).unwrap());
    }
    assert!(
        flat > 80 && flat < 185 && poses.len() == 256 && microphone_types.iter().all(|v| *v > 35)
    );
    assert!(openings.len() > 100 && shapes.len() == 256);
    println!("form factor audit: flat_headphones={flat}/256, microphone_types={microphone_types:?}, distinct_rest_rotations={}, distinct_stapler_openings={}",poses.len(),openings.len());
}
#[test]
fn audio_and_stationery_finishes_are_independent_of_geometry() {
    for kind in [
        ObjectKind::Speaker,
        ObjectKind::Headphones,
        ObjectKind::Microphone,
        ObjectKind::Stapler,
        ObjectKind::TapeDispenser,
    ] {
        let mut changed = 0;
        for seed in 0..64 {
            let o = audit::fixture(kind, seed);
            let mut b = o.clone();
            b.finish_seed = Some(seed ^ 0xdeadbeef);
            let a = build_object(&o);
            let b = build_object(&b);
            let vertices = |a: &Assembly| {
                let mut v: Vec<_> = a
                    .parts
                    .values()
                    .flat_map(|g| g.positions.iter().map(|p| p.map(f32::to_bits)))
                    .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(vertices(&a), vertices(&b));
            changed += usize::from(
                a.parts.keys().collect::<Vec<_>>() != b.parts.keys().collect::<Vec<_>>(),
            );
        }
        assert!(changed > 20, "{kind:?}: finish stream collapsed");
    }
}

#[test]
fn headphone_poses_have_separated_contacts_and_flat_backs_have_area() {
    for seed in 0..128 {
        for flat in [false, true] {
            let mut o = audit::fixture(ObjectKind::Headphones, seed);
            o.size = if flat {
                Vec3::new(0.20, 0.08, 0.24)
            } else {
                Vec3::new(0.20, 0.23, 0.12)
            };
            let a = build_object(&o);
            let (lo, hi) = a.bounds();
            let points: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| &g.positions)
                .map(|v| Vec3::from_array(*v))
                .filter(|v| v.y < lo.y + 1e-5)
                .map(|v| v.xz())
                .collect();
            let min = points
                .iter()
                .copied()
                .min_by(|a, b| a.x.total_cmp(&b.x))
                .unwrap();
            let max = points
                .iter()
                .copied()
                .max_by(|a, b| a.x.total_cmp(&b.x))
                .unwrap();
            assert!(
                max.x - min.x > (hi.x - lo.x) * 0.45,
                "{seed}/{flat}: single-cup support"
            );
            if flat {
                let area = points
                    .iter()
                    .map(|v| (max - min).perp_dot(*v - min).abs())
                    .fold(0., f32::max);
                assert!(
                    area > (hi.x - lo.x) * (hi.z - lo.z) * 0.025,
                    "{seed}: flat pair lacks contact area"
                );
            }
        }
    }
}

#[test]
fn sampled_audio_forms_have_valid_meshes_and_local_gallery() {
    let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&path).unwrap();
    let mut image = image::RgbImage::new(8 * 192, 3 * 192);
    let mut samples = Vec::new();
    for (row, (kind, span, height)) in [
        (ObjectKind::Headphones, 0.43, 0.12),
        (ObjectKind::Speaker, 0.75, 0.19),
        (ObjectKind::Microphone, 0.62, 0.18),
    ]
    .into_iter()
    .enumerate()
    {
        for seed in 0..8 {
            let mut o = audit::fixture(kind, seed);
            o.size = sizing::loose(kind, &mut stream(seed, 234 + row as u64)).unwrap();
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let frame = crate::scene::procedural_indoor::geometry::preview::orthographic(
                &a,
                Vec3::Y * height,
                span,
                192,
                Vec3::new(1.1, 1.25, 1.6),
            );
            image::imageops::replace(&mut image, &frame, seed as i64 * 192, row as i64 * 192);
            samples.push(serde_json::json!({"kind":kind,"seed":seed,"size":o.size,"program":audit::program(&o)}));
        }
    }
    image.save(path.join("form-factors.png")).unwrap();
    std::fs::write(
        path.join("form-factors.json"),
        serde_json::to_vec_pretty(&samples).unwrap(),
    )
    .unwrap();
}
