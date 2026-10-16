use super::Category;
use anyhow::{ensure, Result};
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest, ObjectKind},
    materials::{art, boards, glass::GlassRecipe, program, rugs, screens, variants, Surface},
    objects,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

// Role-specific food bindings from materials/kitchen.rs intentionally use a
// separate namespace from the twelve generic object-finish slots.
fn food_finish(surface: Surface, label: &str, slot: usize) -> bool {
    label.starts_with("food#finish")
        && matches!(
            (surface, slot),
            (Surface::Ceramic, 100..=102) | (Surface::Terracotta, 103) | (Surface::Paper, 104)
        )
}

pub fn recipe(category: &Category, seed: u64) -> program::MaterialRecipe {
    let recipes = program::sample_with_floor(seed, category.floor).unwrap();
    if category.name == "Rug" {
        return rugs::recipe(&recipes[category.surface as usize], rugs::slot(seed));
    }
    recipes[category.surface as usize].variant(variants::slot(seed))
}
pub fn record(category: &Category, seed: u64) -> Value {
    let r = recipe(category, seed);
    let content_seed = variants::screen_seed(seed, variants::slot(seed));
    let content = match category.surface {
        Surface::Screen => json!(screens::parameters_for(
            content_seed,
            screens::DisplayKind::Desktop
        )),
        Surface::PhoneScreen => json!(screens::parameters_for(
            content_seed,
            screens::DisplayKind::Phone
        )),
        Surface::Television => json!(screens::parameters_for(
            content_seed,
            screens::DisplayKind::Television
        )),
        Surface::Whiteboard => json!(boards::parameters(content_seed)),
        Surface::Art => json!(art::parameters(content_seed)),
        _ => Value::Null,
    };
    let glass = matches!(category.surface, Surface::Glass | Surface::GlassInterior)
        .then(|| GlassRecipe::sample(seed, category.surface));
    json!({"seed":seed,"finish_slot":if category.name == "Rug" { Some(rugs::slot(seed)) } else { variants::supports(category.surface).then(|| variants::slot(seed)) },
        "structure_group":if category.name == "Rug" { rugs::slot(seed) } else { variants::slot(seed) % variants::structure_count(category.surface) },
        "recipe":r,"glass":glass,"content":content})
}

/// Count actual assembled parts, not the surfaces an object might use.
pub fn run(output: &Path, categories: &[Category], samples: usize, rooms: usize) -> Result<()> {
    let mut usage: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    let mut object_count = 0;
    let mut category_counts = BTreeMap::<String, usize>::new();
    let mut part_count = 0;
    let mut room_rows = Vec::new();
    for seed in 0..rooms as u64 {
        let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
            .map_err(anyhow::Error::msg)?;
        let mut room_objects = 0;
        for o in scene.objects.iter().filter(|o| !o.neighbor) {
            object_count += 1;
            *category_counts.entry(format!("{:?}", o.kind)).or_default() += 1;
            room_objects += 1;
            let a = objects::build_object(o);
            ensure!(!a.parts.is_empty(), "empty assembly {:?} / {seed}", o.kind);
            let mut object_surfaces = BTreeSet::new();
            for ((surface, label), geometry) in a.parts {
                ensure!(!geometry.positions.is_empty(), "empty material part");
                part_count += 1;
                let slot = label
                    .rsplit_once("#finish")
                    .map(|(_, s)| s.parse::<usize>())
                    .transpose()?;
                let rug_slot = rugs::key(&label).map(|(slot, _)| slot);
                let food_slot = slot.filter(|&slot| food_finish(surface, &label, slot));
                if let Some(slot) = rug_slot {
                    ensure!(
                        o.kind == ObjectKind::Rug
                            && surface == Surface::FabricAlt
                            && slot == rugs::slot(o.finish_seed.unwrap_or(o.seed)),
                        "invalid rug program binding"
                    );
                }
                ensure!(
                    slot.is_none_or(|i| i < variants::COUNT) || food_slot.is_some(),
                    "invalid finish slot {label}"
                );
                ensure!(
                    !variants::supports(surface) || slot.is_some() || rug_slot.is_some(),
                    "missing object finish {surface:?}/{label}"
                );
                // Books on storage furniture intentionally select independent cover slots.
                if variants::supports(surface)
                    && !label.starts_with("books#")
                    && rug_slot.is_none()
                    && food_slot.is_none()
                {
                    ensure!(
                        slot == Some(variants::object_slot(surface, o.seed, o.finish_seed)),
                        "object finish seed not honored"
                    );
                }
                let name = if rug_slot.is_some() {
                    "Rug".into()
                } else if surface == Surface::Floor {
                    ["FloorTimber", "FloorCarpet", "FloorStone"][scene.floor_style as usize].into()
                } else {
                    format!("{surface:?}")
                };
                let row = usage.entry(format!("{:?}", o.kind)).or_default().entry(name.clone())
                    .or_insert_with(|| json!({"parts":0,"objects":0,"slots":vec![0u64;variants::COUNT],"special_slots":{},"base_parts":0}));
                row["parts"] = json!(row["parts"].as_u64().unwrap() + 1);
                if object_surfaces.insert(name) {
                    row["objects"] = json!(row["objects"].as_u64().unwrap() + 1);
                }
                if let Some(slot) = food_slot {
                    let key = slot.to_string();
                    let count = row["special_slots"][&key].as_u64().unwrap_or(0);
                    row["special_slots"][&key] = json!(count + 1);
                } else if let Some(slot) = rug_slot.or(slot) {
                    row["slots"][slot] = json!(row["slots"][slot].as_u64().unwrap() + 1);
                } else {
                    row["base_parts"] = json!(row["base_parts"].as_u64().unwrap() + 1);
                }
            }
        }
        room_rows.push(
            json!({"seed":seed,"primary_objects":room_objects,"floor_style":scene.floor_style}),
        );
    }
    let mut rows = Vec::new();
    for c in categories {
        let records: Vec<_> = (0..samples as u64).map(|seed| record(c, seed)).collect();
        for seed in 0..samples as u64 {
            let r = recipe(c, seed);
            r.validate().map_err(anyhow::Error::msg)?;
            ensure!(r == recipe(c, seed), "recipe replay failed");
        }
        rows.push(json!({"name":c.name,"surface":c.surface,"floor_style":c.floor,
            "structure_count":if c.name == "Rug" { rugs::COUNT } else { variants::structure_count(c.surface) },"finish_slots":if c.name == "Rug" { rugs::COUNT } else if variants::supports(c.surface){variants::COUNT}else{1},
            "samples":records}));
    }
    fs::write(
        output.join("audit.json"),
        serde_json::to_vec(&json!({
        "schema_version":1,"build_provenance":bevy_zeroverse::provenance::capture_provenance(),"tool_sources":super::tool_sources(),
            "sample_count":samples,"room_count":rooms,"room_density":0.65,"human_density":0.,
        "object_count":object_count,"object_counts_by_category":category_counts,"part_count":part_count,"usage":usage,"rooms":room_rows,
            "unobserved_object_categories":ObjectKind::ALL.iter().filter(|k| !usage.contains_key(&format!("{k:?}"))).collect::<Vec<_>>(),
            "categories":rows,"failures":[]
        }))?,
    )?;
    println!("Material assignment audit: {rooms} rooms, {object_count} primary objects, {part_count} parts");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserved_food_finishes_do_not_alias_generic_slots() {
        for (surface, slot) in [
            (Surface::Ceramic, 100),
            (Surface::Ceramic, 101),
            (Surface::Ceramic, 102),
            (Surface::Terracotta, 103),
            (Surface::Paper, 104),
        ] {
            assert!(food_finish(surface, &format!("food#finish{slot}"), slot));
            assert!(!food_finish(
                surface,
                &format!("furniture#finish{slot}"),
                slot
            ));
        }
        assert!(!food_finish(Surface::Metal, "food#finish104", 104));
        assert!(!food_finish(Surface::Paper, "food#finish105", 105));
        let output = tempfile::tempdir().unwrap();
        run(output.path(), &[], 0, 32).unwrap();
        let audit: Value =
            serde_json::from_slice(&fs::read(output.path().join("audit.json")).unwrap()).unwrap();
        assert!(audit["usage"]
            .as_object()
            .unwrap()
            .values()
            .any(|surfaces| {
                surfaces.as_object().unwrap().values().any(|row| {
                    row["special_slots"]
                        .as_object()
                        .is_some_and(|slots| !slots.is_empty())
                })
            }));
    }
    #[test]
    fn inventory_and_seeded_structure_contract() {
        let c = crate::categories();
        assert_eq!(c.len(), 39);
        assert_eq!(c.iter().map(|c| &c.name).collect::<BTreeSet<_>>().len(), 39);
        for c in c {
            let r = recipe(&c, 7);
            assert_eq!(r, recipe(&c, 7));
            assert_eq!(r.surface, c.surface);
            r.validate().unwrap();
            let groups = super::super::SEEDS
                .into_iter()
                .map(|seed| {
                    if c.name == "Rug" {
                        rugs::slot(seed)
                    } else {
                        variants::slot(seed) % variants::structure_count(c.surface)
                    }
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(
                groups.len(),
                if c.name == "Rug" {
                    rugs::COUNT
                } else {
                    variants::structure_count(c.surface)
                },
                "unillustrated structure for {}",
                c.name
            );
        }
    }
}
