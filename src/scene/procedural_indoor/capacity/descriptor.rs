use super::*;
use crate::scene::procedural_indoor::{architecture, layout::IndoorManifest, objects, validation};
use bevy::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn quantize(value: f64, step: f64) -> Result<i64, String> {
    let scaled = (value / step).round();
    if !scaled.is_finite() || scaled.abs() >= i64::MAX as f64 {
        return Err("nonfinite or out-of-range descriptor coordinate".into());
    }
    Ok(scaled as i64)
}

/// Remove latent RNG/instance identities. Serialize accepted parameters instead
/// of counting arbitrary hashes or unexpanded seed-dependent recipes as novelty.
pub(super) fn canonical(value: Value, step: f64) -> Result<Value, String> {
    Ok(match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "id" | "seed" | "finish_seed" | "material_seed" | "lighting_seed"
                ) {
                    continue;
                }
                out.insert(key, canonical(value, step)?);
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|v| canonical(v, step))
                .collect::<Result<_, _>>()?,
        ),
        Value::Number(n) if n.is_f64() => {
            json!(quantize(n.as_f64().ok_or("invalid number")?, step)?)
        }
        v => v,
    })
}
fn hash(value: &Value) -> Result<String, String> {
    Ok(sha(&serde_json::to_vec(value).map_err(|e| e.to_string())?))
}

pub fn observe(
    scene: &IndoorManifest,
    plan: &CapacityPlan,
    draw: u32,
    mesh_checked: bool,
) -> Result<CapacityRecord, String> {
    validate_plan(plan)?;
    if scene.seed != seed_at(&plan.request.seed_policy, draw)
        || scene.generator_version != plan.population.generator.generator_version
    {
        return Err("scene is not a draw from this capacity plan".into());
    }
    if scene.cameras.len() != plan.population.capture.cameras as usize {
        return Err("scene has incomplete cameras".into());
    }
    if scene.indoor_motion.is_some()
        || plan.factors.iter().any(|f| {
            matches!(f.id.as_str(), "human_motion" | "object_motion")
                && f.state == CapturedFactorState::Unqualified
        })
    {
        return Err("manifest-only qualification cannot observe unresolved actor trajectories; use completed captures".into());
    }
    validation::validate_layout(scene)?;
    let topology = scene.room_topology();
    let mut diagnostics = BTreeMap::new();
    diagnostics.insert(format!("profile/{:?}", scene.layout), 1.);
    let region = scene.primary_region();
    let primary = |p: Vec3| region.contains_surface(p, 0.03);
    let functional = scene.functional_graph()?;
    let primary_objects: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| !o.neighbor && primary(o.position))
        .collect();
    let primary_people: Vec<_> = scene
        .humans
        .iter()
        .filter(|h| !h.neighbor && primary(h.position))
        .collect();
    for o in &primary_objects {
        *diagnostics
            .entry(format!("objects/{:?}", o.kind))
            .or_default() += 1.;
    }
    diagnostics.insert("people/count".into(), primary_people.len() as f64);
    diagnostics.insert("lighting/sun_lux".into(), f64::from(scene.daylight_lux));
    diagnostics.insert("lighting/electric_lux".into(), f64::from(scene.target_lux));
    diagnostics.insert("lighting/ev100".into(), f64::from(scene.ev100()));
    diagnostics.insert("architecture/regions".into(), topology.regions.len() as f64);
    diagnostics.insert("architecture/portals".into(), topology.portals.len() as f64);
    diagnostics.insert(
        "assembly/requested_seats".into(),
        scene
            .assembly_diagnostics
            .accepted
            .iter()
            .map(|a| a.requested_capacity as f64)
            .sum(),
    );
    diagnostics.insert(
        "assembly/accepted_seats".into(),
        scene
            .assembly_diagnostics
            .accepted
            .iter()
            .map(|a| a.seats.len() as f64)
            .sum(),
    );
    diagnostics.insert(
        "assembly/rejected_meetings".into(),
        scene.assembly_diagnostics.rejected_meetings as f64,
    );
    for c in &functional.coverage {
        *diagnostics
            .entry(format!("functional/{}/requested", c.program))
            .or_default() += c.requested_capacity as f64;
        *diagnostics
            .entry(format!("functional/{}/accepted", c.program))
            .or_default() += c.accepted_capacity as f64;
        *diagnostics
            .entry(format!("functional/{}/rejected", c.program))
            .or_default() += c.rejected_candidates as f64;
    }
    for e in &functional.edges {
        *diagnostics
            .entry(format!("relationships/{:?}", e.relation))
            .or_default() += 1.;
    }
    for h in &primary_people {
        *diagnostics
            .entry(format!("people/pose/{:?}", h.pose))
            .or_default() += 1.;
    }
    let mut descriptors = BTreeMap::new();
    // Construct each assembly once, then share accepted geometry across all
    // declared resolutions. Drop it after observation; no global audit cache.
    let architecture = architecture::architecture(scene);
    let assemblies: Vec<_> = scene.objects.iter().map(objects::build_object).collect();
    for definition in &plan.request.definitions {
        if definition.descriptor != DescriptorKind::AcceptedProgram {
            continue;
        }
        let mut roles = BTreeSet::new();
        let mut surface_area = BTreeMap::<String, f64>::new();
        let architecture_points = geometry(
            &architecture,
            definition,
            |p| !definition.primary_only || primary(p),
            &mut roles,
            &mut surface_area,
        )?;
        let mut furniture = Vec::new();
        let mut assignments = Vec::new();
        for (o, a) in scene.objects.iter().zip(&assemblies) {
            if definition.primary_only && (o.neighbor || !primary(o.position)) {
                continue;
            }
            let points = geometry(a, definition, |_| true, &mut roles, &mut surface_area)?;
            let support = o
                .support
                .and_then(|id| scene.objects.get(id))
                .map(|o| json!([o.kind, o.position]));
            let target = o
                .interaction_target
                .and_then(|id| scene.objects.get(id))
                .map(|o| json!([o.kind, o.position]));
            furniture.push(json!({"kind":o.kind,"position":canonical(json!(o.position),definition.metres)?,
                "yaw":quantize(o.yaw as f64,definition.radians)?,"size":canonical(json!(o.size),definition.metres)?,
                "geometry":points,"support":canonical(json!(support),definition.metres)?,"target":canonical(json!(target),definition.metres)?}));
            let assigned = assigned_materials(scene, a, definition)?;
            assignments.push(json!({"kind":o.kind,"position":canonical(json!(o.position),definition.metres)?,"assigned":assigned}));
        }
        let mut people: Vec<_> = scene
            .humans
            .iter()
            .filter(|h| !definition.primary_only || (!h.neighbor && primary(h.position)))
            .map(|h| {
                let mut value = json!(h);
                value.as_object_mut().unwrap().remove("chair");
                canonical(value, definition.metres)
            })
            .collect::<Result<_, _>>()?;
        people.sort_by_key(Value::to_string);
        let recipes: Vec<_> = scene
            .program
            .iter()
            .flat_map(|p| p.materials.iter())
            .filter(|m| roles.contains(&m.surface))
            .map(|m| canonical(json!(m), definition.appearance_step))
            .collect::<Result<_, _>>()?;
        let cameras:Vec<_>=scene.cameras.iter().map(|c| {
            let mut states:Vec<_>=plan.population.capture.progress.iter().map(|&t| {
                let tf=c.transform_at(t);
                Ok(json!({"translation":canonical(json!(tf.translation),definition.metres)?,
                    "rotation":canonical(json!(tf.rotation),definition.radians)?}))
            }).collect::<Result<_,String>>()?;
            states.dedup();
            Ok(json!({"states":states,"fovy":quantize(c.fov_degrees.to_radians() as f64,definition.radians)?,
                "principal_point":canonical(json!(c.principal_point_offset),definition.appearance_step)?}))
        }).collect::<Result<_,String>>()?;
        let mut blocks = BTreeMap::new();
        let regions: Vec<_> = topology
            .regions
            .iter()
            .filter(|r| !definition.primary_only || r.primary)
            .map(|r| json!({"triangles":r.triangles,"area_m2":r.area_m2}))
            .collect();
        blocks.insert("architecture".into(),hash(&json!({"mesh":architecture_points,"regions":canonical(json!(regions),definition.metres)?}))?);
        // Sort by content, not generation IDs. Reordering an identical accepted
        // set cannot create another class.
        furniture.sort_by_key(Value::to_string);
        blocks.insert("furniture".into(), hash(&json!(furniture))?);
        blocks.insert("people".into(), hash(&json!(people))?);
        assignments.sort_by_key(Value::to_string);
        blocks.insert(
            "materials".into(),
            hash(&json!({"base":recipes,"assignments":assignments}))?,
        );
        blocks.insert("lighting".into(),hash(&json!({"sun":quantize((f64::from(scene.daylight_lux)+1.).log2(),definition.log2_light_step)?,
            "electric":quantize((f64::from(scene.target_lux)+1.).log2(),definition.log2_light_step)?,"ev100":quantize(f64::from(scene.ev100()),definition.log2_light_step)?,
            "azimuth":quantize(f64::from(scene.sun_azimuth),definition.radians)?,"elevation":quantize(f64::from(scene.sun_elevation),definition.radians)?,
            "kelvin":quantize(f64::from(scene.light_kelvin),250.)?}))?);
        if plan.request.unit != SampleUnit::DressedScene {
            blocks.insert("cameras".into(), hash(&json!(cameras))?);
        }
        let class_sha256 = hash(&json!(blocks))?;
        descriptors.insert(
            definition.id.clone(),
            Descriptor {
                class_sha256,
                blocks,
            },
        );
        for (role, area) in surface_area {
            diagnostics.insert(format!("constructed_m2/{role}"), area);
        }
    }
    if descriptors.is_empty() {
        return Err("manifest observation needs an accepted_program definition".into());
    }
    let family_sha256 = super::family_identity(scene, plan);
    let record = CapacityRecord {
        draw,
        seed: scene.seed,
        mesh_checked,
        outcome: SampleOutcome::Accepted {
            family_sha256,
            descriptors,
            diagnostics,
            delivered_views: 0,
            delivered_progress: Vec::new(),
            capture_sha256: None,
        },
    };
    validate_record(plan, &record)?;
    Ok(record)
}

fn assigned_materials(
    scene: &IndoorManifest,
    a: &objects::Assembly,
    d: &DiversityDefinition,
) -> Result<Value, String> {
    use super::super::materials::{self, Surface};
    let Some(program) = &scene.program else {
        return Ok(Value::Null);
    };
    let mut recipes = Vec::new();
    for (role, label) in a.parts.keys() {
        let slot = label
            .split('#')
            .find_map(|s| s.strip_prefix("finish")?.parse::<usize>().ok());
        let base = &program.materials[*role as usize];
        let recipe = if let Some((slot, _)) = materials::rugs::key(label) {
            materials::rugs::recipe(base, slot)
        } else {
            slot.map_or_else(|| base.clone(), |i| base.variant(i))
        };
        let content = slot
            .map(|i| materials::variants::screen_seed(scene.material_seed(), i))
            .map(|seed| match role {
                Surface::Screen => json!(materials::screens::parameters_for(
                    seed,
                    materials::screens::DisplayKind::Desktop
                )),
                Surface::PhoneScreen => json!(materials::screens::parameters_for(
                    seed,
                    materials::screens::DisplayKind::Phone
                )),
                Surface::Television => json!(materials::screens::parameters_for(
                    seed,
                    materials::screens::DisplayKind::Television
                )),
                Surface::Whiteboard => json!(materials::boards::parameters(seed)),
                Surface::Art => json!(materials::art::parameters(seed)),
                _ => Value::Null,
            });
        recipes.push(json!([
            role,
            canonical(json!(recipe), d.appearance_step)?,
            canonical(json!(content), d.appearance_step)?
        ]));
    }
    recipes.sort_by_key(Value::to_string);
    Ok(json!(recipes))
}

fn geometry(
    a: &objects::Assembly,
    d: &DiversityDefinition,
    keep: impl Fn(Vec3) -> bool,
    roles: &mut BTreeSet<super::super::materials::Surface>,
    areas: &mut BTreeMap<String, f64>,
) -> Result<Value, String> {
    let mut parts = Vec::new();
    for ((role, label), g) in &a.parts {
        let mut cells = BTreeSet::new();
        let mut area = 0.;
        for t in g.indices.as_chunks::<3>().0 {
            let p = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            if !keep((p[0] + p[1] + p[2]) / 3.) {
                continue;
            }
            area += f64::from((p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5);
            for v in p {
                cells.insert([
                    quantize(v.x as f64, d.metres)?,
                    quantize(v.y as f64, d.metres)?,
                    quantize(v.z as f64, d.metres)?,
                ]);
            }
        }
        if cells.is_empty() {
            continue;
        }
        roles.insert(*role);
        *areas.entry(format!("{role:?}")).or_default() += area;
        // Semantic ownership retained, material suffix/random instance IDs are not.
        parts.push(json!([label.split('#').next().unwrap_or(label), cells]));
    }
    parts.sort_by_key(Value::to_string);
    Ok(json!(parts))
}
