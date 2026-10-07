//! Mesh qualification; one assembly is held at a time.
use super::*;
use crate::scene::procedural_indoor::objects::{build_object, Assembly};

pub fn validate_geometry(scene: &IndoorManifest) -> Result<GeometryStats, String> {
    let mut stats = GeometryStats::default();
    accumulate(
        scene,
        &mut stats,
        super::super::architecture::architecture(scene),
    )?;
    for object in &scene.objects {
        if !object.size.is_finite() || object.size.min_element() <= 0. {
            return Err(format!(
                "seed {}: invalid object geometry dimensions",
                scene.seed
            ));
        }
        let assembly = build_object(object);
        let (lo, hi) = assembly.bounds();
        let reserved_lo = Vec3::new(-object.size.x * 0.5, 0., -object.size.z * 0.5);
        let reserved_hi = Vec3::new(object.size.x * 0.5, object.size.y, object.size.z * 0.5);
        let overrun = (reserved_lo - lo)
            .max(hi - reserved_hi)
            .max_element()
            .max(0.);
        // Millimetre trim and upholstery piping are expected. Larger overruns
        // invalidate placement, camera collision and support assumptions.
        if !lo.is_finite() || !hi.is_finite() || overrun > 0.003 {
            return Err(format!(
                "seed {}: {:?} instance {} mesh escapes its reserved envelope by {overrun:.6} m: {lo:?}..{hi:?}, size={:?}",
                scene.seed, object.kind, object.id, object.size
            ));
        }
        if (object.solid || object.support.is_some()) && lo.y > 0.003 {
            return Err(format!(
                "seed {}: {:?} instance {} has no floor/support contact ({:.6} m gap)",
                scene.seed, object.kind, object.id, lo.y
            ));
        }
        stats.object_envelopes_checked += 1;
        *stats
            .object_envelopes_by_kind
            .entry(format!("{:?}", object.kind))
            .or_default() += 1;
        stats.object_envelope_max_overrun_metres =
            stats.object_envelope_max_overrun_metres.max(overrun);
        accumulate(scene, &mut stats, assembly)?;
    }
    for human in &scene.humans {
        let human_geometry = super::super::humans::build_human(human);
        qualify_human(scene.seed, human, &human_geometry, &mut stats)?;
        accumulate(
            scene,
            &mut stats,
            Assembly {
                parts: human_geometry
                    .parts
                    .into_iter()
                    .map(|(surface, geometry)| {
                        (
                            (
                                super::super::materials::Surface::Fabric,
                                format!("person#{surface:?}"),
                            ),
                            geometry,
                        )
                    })
                    .collect(),
            },
        )?;
    }
    Ok(stats)
}

fn qualify_human(
    seed: u64,
    human: &super::super::humans::IndoorHuman,
    mesh: &super::super::humans::HumanAssembly,
    stats: &mut GeometryStats,
) -> Result<(), String> {
    let (lo, hi) = mesh.bounds();
    if human
        .worktop_contact
        .as_ref()
        .is_some_and(|c| mesh.contacts.len() != c.palms.iter().filter(|&&v| v).count())
        || mesh.contacts.iter().any(|r| {
            !r.minimum_gap_metres.is_finite()
                || !r.palm_gap_metres.is_finite()
                || !(-1e-5..=0.003).contains(&r.palm_gap_metres)
                || !(-1e-5..=0.003).contains(&r.minimum_gap_metres)
                || !r.cloth_compression_metres.is_finite()
                || r.cloth_compression_metres > 0.025
                || !r.skin_compression_metres.is_finite()
                || r.skin_compression_metres > 0.006
                || r.contact_vertices == 0
        })
    {
        return Err(format!(
            "seed {seed}: person {} has invalid tabletop contact: {:?}",
            human.id, mesh.contacts
        ));
    }
    let overrun = (human.bounds_min - lo)
        .max(hi - human.bounds_max)
        .max_element()
        .max(0.);
    if !lo.is_finite() || !hi.is_finite() || overrun > 0.003 {
        return Err(format!(
            "seed {seed}: person {} mesh escapes its placement envelope by {overrun:.6} m: {lo:?}..{hi:?}",
            human.id
        ));
    }
    if lo.y.abs() > 0.003 {
        return Err(format!(
            "seed {seed}: person {} has no floor contact ({:.6} m)",
            human.id, lo.y
        ));
    }
    let m = mesh.body_measurements;
    if [
        m.rest_stature_metres,
        m.rest_shoulder_bone_span_metres,
        m.rest_torso_width_metres,
        m.rest_torso_depth_metres,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v <= 0.)
        || (m.rest_stature_metres - human.stature).abs() > 0.00001
        || !m.shoulder_attachment_offset_metres.is_finite()
        || !(0.0..=0.045).contains(&m.shoulder_attachment_offset_metres)
    {
        return Err(format!(
            "seed {seed}: person {} has invalid Anny body dimensions",
            human.id
        ));
    }
    stats.human_envelopes_checked += 1;
    stats.human_envelope_max_overrun_metres = stats.human_envelope_max_overrun_metres.max(overrun);
    stats.human_measurements.push(HumanGeometryReport {
        id: human.id,
        requested_stature_metres: human.stature,
        phenotype: super::super::humans::morphology::phenotype(human),
        body: m,
        posed_mesh_bounds: [lo.to_array(), hi.to_array()],
        placement_bounds: [human.bounds_min.to_array(), human.bounds_max.to_array()],
        envelope_overrun_metres: overrun,
        worktop_contacts: mesh.contacts.clone(),
    });
    Ok(())
}

fn accumulate(
    scene: &IndoorManifest,
    stats: &mut GeometryStats,
    a: Assembly,
) -> Result<(), String> {
    stats.assemblies += 1;
    for ((surface, label), g) in a.parts {
        stats.batches += 1;
        stats.vertices += g.positions.len();
        stats.triangles += g.indices.len() / 3;
        *stats
            .semantic_triangles
            .entry(super::super::objects::part_label(&label).to_owned())
            .or_default() += g.indices.len() / 3;
        if g.positions.len() != g.normals.len()
            || g.positions.len() != g.uvs.len()
            || g.indices.len() % 3 != 0
        {
            return Err("inconsistent mesh attribute lengths".into());
        }
        if g.positions
            .iter()
            .any(|p| !Vec3::from_array(*p).is_finite())
            || g.uvs.iter().any(|uv| !Vec2::from_array(*uv).is_finite())
        {
            return Err("non-finite mesh attribute".into());
        }
        if g.normals.iter().any(|n| {
            let n = Vec3::from_array(*n);
            !n.is_finite() || (n.length() - 1.0).abs() > 0.005
        }) {
            return Err("invalid mesh normal".into());
        }
        for tri in g.indices.as_chunks::<3>().0.iter() {
            if tri.iter().any(|&i| i as usize >= g.positions.len()) {
                return Err("mesh index out of bounds".into());
            }
            let p = |i: u32| Vec3::from_array(g.positions[i as usize]);
            let normal = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0]));
            if normal.length_squared() < 1e-20 {
                stats.degenerate_triangles += 1;
                *stats
                    .degenerate_by_semantic
                    .entry(super::super::objects::part_label(&label).to_owned())
                    .or_default() += 1;
            }
            let n = Vec3::from_array(g.normals[tri[0] as usize])
                + Vec3::from_array(g.normals[tri[1] as usize])
                + Vec3::from_array(g.normals[tri[2] as usize]);
            if normal.dot(n) < -1e-6 {
                return Err(format!(
                        "inverted triangle in seed {}: {surface:?}/{label}, positions={:?}, normal={n:?}",
                        scene.seed,
                        tri.iter()
                            .map(|i| g.positions[*i as usize])
                            .collect::<Vec<_>>()
                    ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Default, Serialize)]
pub struct GeometryStats {
    pub object_envelopes_checked: usize,
    pub object_envelopes_by_kind: BTreeMap<String, usize>,
    pub object_envelope_max_overrun_metres: f32,
    pub human_envelopes_checked: usize,
    pub human_envelope_max_overrun_metres: f32,
    pub human_measurements: Vec<HumanGeometryReport>,
    pub assemblies: usize,
    pub batches: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub degenerate_triangles: usize,
    pub degenerate_by_semantic: BTreeMap<String, usize>,
    pub semantic_triangles: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct HumanGeometryReport {
    pub id: usize,
    pub requested_stature_metres: f32,
    pub phenotype: super::super::humans::morphology::Phenotype,
    pub body: super::super::humans::morphology::BodyMeasurements,
    pub posed_mesh_bounds: [[f32; 3]; 2],
    pub placement_bounds: [[f32; 3]; 2],
    pub envelope_overrun_metres: f32,
    pub worktop_contacts: Vec<super::super::humans::contact::ContactReport>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{geometry::Geometry, materials::Surface};

    #[test]
    fn human_mesh_qualification_checks_bounds_contact_and_metric_stature() {
        let scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.5, 0, 1.).unwrap();
        let mut human = scene.humans[0].clone();
        let mut mesh = super::super::super::humans::build_human(&human);
        let mut stats = GeometryStats::default();
        qualify_human(scene.seed, &human, &mesh, &mut stats).unwrap();
        assert_eq!(stats.human_envelopes_checked, 1);
        assert_eq!(stats.human_measurements.len(), 1);
        assert!(
            (stats.human_measurements[0].body.rest_stature_metres - human.stature).abs() < 1e-5
        );
        let bounds = human.bounds_max;
        human.bounds_max.y = mesh.bounds().1.y - 0.01;
        assert!(qualify_human(scene.seed, &human, &mesh, &mut stats)
            .unwrap_err()
            .contains("placement envelope"));
        human.bounds_max = bounds;
        mesh.body_measurements.rest_stature_metres += 0.01;
        assert!(qualify_human(scene.seed, &human, &mesh, &mut stats)
            .unwrap_err()
            .contains("body dimensions"));
        mesh.body_measurements.rest_stature_metres = human.stature;
        for g in mesh.parts.values_mut() {
            for p in &mut g.positions {
                p[1] += 0.01;
            }
        }
        human.bounds_max.y += 0.01;
        assert!(qualify_human(scene.seed, &human, &mesh, &mut stats)
            .unwrap_err()
            .contains("floor contact"));
    }

    #[test]
    fn nonfinite_normals_cannot_bypass_the_unit_length_check() {
        let scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.5, 0, 0.).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut a = Assembly::default();
            a.parts.insert(
                (Surface::Paint, "wall".into()),
                Geometry {
                    positions: vec![[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]],
                    normals: vec![[value, 1., 0.]; 3],
                    uvs: vec![[0., 0.]; 3],
                    indices: vec![0, 1, 2],
                },
            );
            assert_eq!(
                accumulate(&scene, &mut GeometryStats::default(), a).unwrap_err(),
                "invalid mesh normal"
            );
        }
    }
}
