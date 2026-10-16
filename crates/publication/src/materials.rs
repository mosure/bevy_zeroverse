//! Reproducible, current-generator material atlas. No GPU dependency in the publisher.
mod paper;
use crate::{io, pipeline};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::GeneratorIdentity;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::Command,
};

pub const DIRECTORY: &str = "www/project/static/media/materials";
pub const TOOLS: [&str; 3] = [
    "examples/qualify_materials.rs",
    "examples/material_study/audit.rs",
    "examples/material_study/studio.rs",
];
const SEEDS: [u64; 6] = [0, 1, 4, 19, 53, 101];
const MODES: [&str; 5] = ["color", "normal", "albedo", "normal_map", "roughness"];
const SAMPLES: usize = 256;
const ROOMS: usize = 256;
fn special_parts(surface: &str, row: &Value) -> Result<u64> {
    let Some(slots) = row.get("special_slots") else {
        return Ok(0);
    };
    let mut total = 0u64;
    for (slot, count) in slots
        .as_object()
        .context("invalid special finish histogram")?
    {
        ensure!(
            matches!(
                (surface, slot.as_str()),
                ("Ceramic", "100" | "101" | "102") | ("Terracotta", "103") | ("Paper", "104")
            ),
            "unregistered special finish {surface}/{slot}"
        );
        let count = count
            .as_u64()
            .filter(|&n| n > 0)
            .context("invalid finish count")?;
        total = total.checked_add(count).context("finish count overflow")?;
    }
    Ok(total)
}
fn sources(root: &Path) -> Result<BTreeMap<String, String>> {
    TOOLS
        .iter()
        .map(|p| Ok((p.to_string(), io::hash(&root.join(p))?)))
        .collect()
}
fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                if k != "seed" && k != "surface" {
                    flatten(&format!("{prefix}.{k}"), v, out)
                }
            }
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                flatten(&format!("{prefix}.{i}"), v, out)
            }
        }
        Value::Null => (),
        _ => {
            out.insert(prefix.into(), v.clone());
        }
    }
}
fn parameters(samples: &[Value]) -> Value {
    let mut columns: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for s in samples {
        let mut values = BTreeMap::new();
        flatten("recipe", &s["recipe"], &mut values);
        flatten("glass", &s["glass"], &mut values);
        for (k, v) in values {
            columns.entry(k).or_default().push(v);
        }
    }
    Value::Object(columns.into_iter().map(|(k,values)|{
        let v=if values.iter().all(Value::is_number) {
            let mut xs:Vec<_>=values.iter().map(|v|v.as_f64().unwrap()).collect();
            xs.sort_by(f64::total_cmp);
            let lo=xs[0];let hi=xs[xs.len()-1];let mut bins=[0usize;8];
            for x in &xs {let index=if hi>lo {(((x-lo)/(hi-lo)*8.) as usize).min(7)}else{0};bins[index]+=1;}
            json!({"type":"numeric","present":xs.len(),"min":lo,"max":hi,"median":xs[xs.len()/2],"bins":bins})
        }else {
            let mut counts=BTreeMap::<String,usize>::new();for v in values {*counts.entry(v.to_string()).or_default()+=1;}
            json!({"type":"categorical","counts":counts})
        };(k,v)
    }).collect())
}
fn validate_samples(c: &Value) -> Result<()> {
    let samples = c["samples"]
        .as_array()
        .context("missing material samples")?;
    ensure!(samples.len() == SAMPLES, "material sample denominator");
    let structures = c["structure_count"].as_u64().context("structure count")?;
    ensure!((1..=4).contains(&structures), "invalid structure count");
    for (i, s) in samples.iter().enumerate() {
        ensure!(
            s["seed"] == i && s["recipe"]["surface"] == c["surface"],
            "seed/material mismatch"
        );
        ensure!(
            s["structure_group"]
                .as_u64()
                .is_some_and(|v| v < structures),
            "structure out of bounds"
        );
        let slot = &s["finish_slot"];
        ensure!(
            slot.is_null() || slot.as_u64().is_some_and(|v| v < 12),
            "finish out of bounds"
        );
        ensure!(
            s["recipe"]["period_m"]
                .as_f64()
                .is_some_and(|v| v > 0. && v.is_finite()),
            "invalid period"
        );
    }
    Ok(())
}
pub fn prepare(root: &Path, input: &Path, output: &Path) -> Result<()> {
    let identity = pipeline::identity(root)?;
    let audit: Value = io::read(&input.join("audit.json"))?;
    let render: Value = io::read(&input.join("renders.json"))?;
    for v in [&audit, &render] {
        ensure!(
            v["tool_sources"] == json!(sources(root)?),
            "stale material study tool binary"
        );
        ensure!(
            v["build_provenance"]["source_sha256"] == identity.source_sha256,
            "stale material study"
        );
    }
    ensure!(
        audit["sample_count"] == SAMPLES && audit["room_count"] == ROOMS,
        "material study protocol mismatch"
    );
    ensure!(
        audit["failures"].as_array().is_some_and(Vec::is_empty),
        "material assignment failures"
    );
    ensure!(
        render["selected_seeds"] == json!(SEEDS) && render["image_size"] == json!([384, 384]),
        "render protocol mismatch"
    );
    fs::create_dir_all(output)?;
    let mut categories = Vec::new();
    for category in audit["categories"]
        .as_array()
        .context("missing categories")?
    {
        validate_samples(category)?;
        let name = category["name"].as_str().context("missing category name")?;
        ensure!(
            name.chars().all(|c| c.is_ascii_alphanumeric()),
            "unsafe material name"
        );
        let mut c = category.clone();
        let samples = c["samples"].as_array().unwrap().clone();
        c["parameters"] = parameters(&samples);
        c["samples_file"] = json!(format!("{name}-samples.json"));
        c["selected"] = json!(render["records"]
            .as_array()
            .context("missing records")?
            .iter()
            .filter(|r| r["name"] == name)
            .collect::<Vec<_>>());
        for (i, seed) in SEEDS.into_iter().enumerate() {
            let row = &c["selected"][i];
            ensure!(
                row["seed"] == seed && row["column"] == i,
                "mismatched rendered seed"
            );
            for key in ["recipe", "glass", "finish_slot", "structure_group"] {
                ensure!(
                    row[key] == samples[seed as usize][key],
                    "render/audit mismatch {name}/{key}"
                );
            }
            for mode in MODES {
                let image = image::open(input.join(format!("{name}-{i}-{mode}.png")))?;
                image.save(output.join(format!("{name}-{i}-{mode}.webp")))?;
            }
        }
        io::write(&output.join(format!("{name}-samples.json")), &samples)?;
        c.as_object_mut().unwrap().remove("samples");
        categories.push(c);
    }
    let mut usage = audit.clone();
    usage.as_object_mut().unwrap().remove("categories");
    io::write(&output.join("usage.json"), &usage)?;
    let mut hashes = io::files(output, output)?;
    hashes.remove("atlas.json");
    let atlas = json!({"schema_version":2,"generator_identity":identity,"tool_sources":sources(root)?,
        "sample_count":SAMPLES,"room_count":ROOMS,"selected_seeds":SEEDS,"image_size":[384,384],
        "build_provenance":render["build_provenance"],"adapter":render["adapter"],
        "map_baker":render["map_baker"],"rig":render["rig"],"object_count":audit["object_count"],"part_count":audit["part_count"],
        "categories":categories,"files":hashes});
    io::write(&output.join("atlas.json"), &atlas)?;
    verify(output, &identity)?;
    Ok(())
}
pub fn verify(directory: &Path, identity: &GeneratorIdentity) -> Result<Value> {
    let a: Value = io::read(&directory.join("atlas.json"))?;
    ensure!(
        matches!(a["schema_version"].as_u64(), Some(1 | 2))
            && a["generator_identity"] == json!(identity)
            && a["sample_count"] == SAMPLES
            && a["room_count"] == ROOMS,
        "stale material atlas"
    );
    let mut files = io::files(directory, directory)?;
    files.remove("atlas.json");
    ensure!(
        a["files"] == json!(files),
        "material artifact hash mismatch"
    );
    ensure!(
        a["selected_seeds"] == json!(SEEDS),
        "altered material sample selection"
    );
    let categories = a["categories"].as_array().context("material categories")?;
    let mut inventory:BTreeSet<_>="Paint Accent Wood WoodEdge FloorTimber FloorCarpet FloorStone Ceiling Metal Chrome Plastic Fabric FabricAlt Glass Ceramic Soil Leaf LeafLight Paper Screen Ink Light Concrete Art Rubber LeafVariegated Terracotta Bark GlassInterior ContainerGlass Liquid Drink PhoneScreen PrintedPaper Leather Whiteboard Television Succulent".split_whitespace().collect();
    // A retained checksum-bound v1 snapshot remains readable. Fresh production
    // always emits v2, whose mandatory rug row cannot silently disappear.
    if a["schema_version"] == 2 {
        inventory.insert("Rug");
    }
    ensure!(
        categories.len() == inventory.len(),
        "incomplete material inventory"
    );
    ensure!(
        categories
            .iter()
            .filter_map(|c| c["name"].as_str())
            .collect::<BTreeSet<_>>()
            == inventory,
        "material roles missing or substituted"
    );
    let mut names = BTreeSet::new();
    let mut expected = BTreeSet::from(["usage.json".to_string()]);
    for c in categories {
        let name = c["name"].as_str().context("missing name")?;
        ensure!(
            names.insert(name) && name.chars().all(|c| c.is_ascii_alphanumeric()),
            "material name"
        );
        let sample_file = format!("{name}-samples.json");
        ensure!(
            c["samples_file"] == sample_file,
            "unexpected material data path"
        );
        let samples: Value = io::read(&directory.join(&sample_file))?;
        expected.insert(sample_file);
        let mut full = c.clone();
        full["samples"] = samples.clone();
        validate_samples(&full)?;
        ensure!(
            c["parameters"] == parameters(samples.as_array().unwrap()),
            "material parameter summary mismatch"
        );
        ensure!(
            c["selected"].as_array().is_some_and(|v| v.len() == 6),
            "missing material columns"
        );
        for (i, seed) in SEEDS.into_iter().enumerate() {
            let row = &c["selected"][i];
            ensure!(
                row["seed"] == seed && row["name"] == name && row["column"] == i,
                "incorrect sample link"
            );
            for key in ["recipe", "glass", "finish_slot", "structure_group"] {
                ensure!(
                    row[key] == samples[seed as usize][key],
                    "material sample mismatch"
                );
            }
            for mode in MODES {
                let file = format!("{name}-{i}-{mode}.webp");
                expected.insert(file.clone());
                let (w, h) = image::image_dimensions(directory.join(file))?;
                if mode == "color" || mode == "normal" {
                    ensure!((w, h) == (384, 384), "render dimensions");
                } else {
                    let meta = &row["actual_material"]["maps"][mode];
                    ensure!(
                        w == h && meta["width"] == w && w.is_power_of_two(),
                        "map dimensions"
                    );
                    ensure!(
                        meta["mip_levels"]
                            .as_u64()
                            .is_some_and(|n| n > 0 && n <= u64::from(w.ilog2() + 1)),
                        "map mip count"
                    );
                }
            }
        }
    }
    ensure!(
        files.keys().cloned().collect::<BTreeSet<_>>() == expected,
        "unbound or missing material files"
    );
    let usage: Value = io::read(&directory.join("usage.json"))?;
    ensure!(
        usage["room_count"] == ROOMS
            && usage["object_count"] == a["object_count"]
            && usage["part_count"] == a["part_count"],
        "usage counts"
    );
    let rooms = usage["rooms"].as_array().context("missing usage cohort")?;
    ensure!(
        rooms.len() == ROOMS && rooms.iter().enumerate().all(|(i, r)| r["seed"] == i),
        "usage seed coverage"
    );
    ensure!(
        rooms
            .iter()
            .map(|r| r["primary_objects"].as_u64().unwrap_or(0))
            .sum::<u64>()
            == usage["object_count"].as_u64().unwrap_or(0),
        "usage object denominator"
    );
    let denominators = usage["object_counts_by_category"]
        .as_object()
        .context("missing object-category denominators")?;
    ensure!(
        denominators
            .values()
            .map(|v| v.as_u64().unwrap_or(0))
            .sum::<u64>()
            == usage["object_count"].as_u64().unwrap_or(0),
        "object-category denominator mismatch"
    );
    let mut parts = 0;
    for (kind, category) in usage["usage"]
        .as_object()
        .context("missing assignment table")?
    {
        let denominator = denominators
            .get(kind)
            .and_then(Value::as_u64)
            .context("missing category denominator")?;
        for (surface, row) in category.as_object().context("missing assignment row")? {
            let slots = row["slots"]
                .as_array()
                .context("missing finish histogram")?;
            ensure!(slots.len() == 12, "finish histogram size");
            let total = slots.iter().map(|n| n.as_u64().unwrap_or(0)).sum::<u64>()
                + special_parts(surface, row)?
                + row["base_parts"].as_u64().unwrap_or(0);
            ensure!(row["parts"] == total, "part/slot count mismatch");
            ensure!(
                row["objects"]
                    .as_u64()
                    .is_some_and(|n| n > 0 && n <= denominator && n <= total),
                "material usage object count"
            );
            parts += total;
        }
    }
    ensure!(
        usage["part_count"] == parts && usage["failures"].as_array().is_some_and(Vec::is_empty),
        "failed material assignment audit"
    );
    Ok(a)
}
pub fn refresh_cache(root: &Path, cache: &Path, identity: &GeneratorIdentity) -> Result<()> {
    if let Ok(a) = verify(cache, identity) {
        if a["tool_sources"] == json!(sources(root)?) {
            return Ok(());
        }
    }
    let temp = tempfile::Builder::new()
        .prefix("materials-")
        .tempdir_in(cache.parent().context("cache parent")?)?
        .keep();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    ensure!(
        Command::new(&cargo)
            .current_dir(root)
            .args([
                "run",
                "--locked",
                "--no-default-features",
                "--features",
                "multi_threaded",
                "--example",
                "qualify_materials",
                "--",
                "--output"
            ])
            .arg(temp.join("raw"))
            .status()?
            .success(),
        "material study failed; published output unchanged"
    );
    prepare(root, &temp.join("raw"), &temp.join("atlas"))?;
    if cache.exists() {
        fs::remove_dir_all(cache)?;
    }
    fs::rename(temp.join("atlas"), cache)?;
    Ok(())
}
pub fn build(cache: &Path, stage: &Path, identity: &GeneratorIdentity) -> Result<()> {
    verify(cache, identity)?;
    let dest = stage.join(DIRECTORY);
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    io::copy_tree(cache, &dest)?;
    write_paper(stage)
}
pub fn write_paper(stage: &Path) -> Result<()> {
    paper::write(stage)
}
pub fn page() -> &'static str {
    include_str!("../templates/materials.html")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn specialized_food_binding_receipts_are_typed_and_counted() {
        assert_eq!(special_parts("Paper", &json!({})).unwrap(), 0);
        assert_eq!(
            special_parts("Paper", &json!({"special_slots":{"104":3}})).unwrap(),
            3
        );
        assert_eq!(
            special_parts("Ceramic", &json!({"special_slots":{"100":2,"102":4}})).unwrap(),
            6
        );
        for (surface, slots) in [
            ("Metal", json!({"104":1})),
            ("Paper", json!({"100":1})),
            ("Paper", json!({"105":1})),
            ("Paper", json!({"104":0})),
            ("Paper", json!({"104":1.5})),
            ("Paper", json!([])),
        ] {
            assert!(special_parts(surface, &json!({"special_slots":slots})).is_err());
        }
    }
    #[test]
    fn parameter_marginals_keep_conditionals_and_exclude_seed_ids() {
        let samples = vec![
            json!({"recipe":{"seed":u64::MAX,"surface":"Fabric","textile":{"knit":true,"pitch":2.}},"glass":null}),
            json!({"recipe":{"seed":1,"surface":"Fabric","textile":null},"glass":null}),
        ];
        let p = parameters(&samples);
        assert_eq!(p["recipe.textile.pitch"]["present"], 1);
        assert_eq!(
            p["recipe.textile.pitch"]["bins"],
            json!([1, 0, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(p["recipe.textile.knit"]["counts"]["true"], 1);
        assert!(p.get("recipe.seed").is_none());
    }
    #[test]
    fn parameter_bins_include_both_endpoints() {
        let p = parameters(&[
            json!({"recipe":{"roughness":0.1}}),
            json!({"recipe":{"roughness":0.9}}),
        ]);
        assert_eq!(
            p["recipe.roughness"]["bins"],
            json!([1, 0, 0, 0, 0, 0, 0, 1])
        );
    }
    #[test]
    fn sample_contract_rejects_missing_seeds_and_invalid_bindings() {
        let rows:Vec<_>=(0..SAMPLES).map(|i|json!({"seed":i,"finish_slot":i%12,"structure_group":i%3,"recipe":{"surface":"Fabric","period_m":0.1}})).collect();
        let c = json!({"surface":"Fabric","structure_count":3,"samples":rows});
        validate_samples(&c).unwrap();
        for (field, value) in [
            ("seed", json!(8)),
            ("finish_slot", json!(12)),
            ("structure_group", json!(3)),
        ] {
            let mut bad = c.clone();
            bad["samples"][0][field] = value;
            assert!(validate_samples(&bad).is_err(), "accepted {field}");
        }
        let mut bad = c.clone();
        bad["samples"].as_array_mut().unwrap().pop();
        assert!(validate_samples(&bad).is_err());
    }
}
