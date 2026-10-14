//! Fixed-envelope construction diagnostics, separate from room diversity.
use crate::io;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub(super) fn prepare(
    input: &Path,
    output: &Path,
    identity: &Value,
    assets: &mut BTreeMap<String, String>,
) -> Result<Value> {
    let audit: Value = io::read(&input.join("audit.json"))?;
    let renders: Value = io::read(&input.join("renders.json"))?;
    for source in [&audit, &renders] {
        let p = &source["build_provenance"];
        ensure!(
            p["source_sha256"] == identity["source_sha256"]
                && p["generator_version"] == identity["generator_version"],
            "stale object study"
        );
    }
    let count = audit["sample_count_per_category"]
        .as_u64()
        .context("missing sample count")? as usize;
    ensure!(
        (8..=1024).contains(&count) && renders["image_size"] == json!([320, 320]),
        "invalid studio protocol"
    );
    let categories = audit["categories"]
        .as_array()
        .context("missing categories")?;
    let rendered = renders["categories"]
        .as_array()
        .context("missing studio categories")?;
    ensure!(
        !categories.is_empty() && categories.len() == rendered.len(),
        "incomplete studio inventory"
    );
    let mut names = BTreeSet::new();
    let mut summary = Vec::new();
    fs::create_dir_all(output.join("objects"))?;
    let mut html = String::from(HEADER);
    for category in categories {
        let kind = category["kind"].as_str().context("missing object kind")?;
        ensure!(
            names.insert(kind) && kind.chars().all(|c| c.is_ascii_alphanumeric()),
            "unsafe or repeated category"
        );
        ensure!(
            category["failures"].as_array().is_some_and(Vec::is_empty),
            "failed object geometry"
        );
        let samples = category["samples"].as_array().context("missing samples")?;
        ensure!(samples.len() == count, "incomplete object sample count");
        let base = &samples[0]["object"]["size"];
        for (seed, sample) in samples.iter().enumerate() {
            ensure!(
                sample["object"]["seed"] == seed
                    && sample["object"]["kind"] == kind
                    && &sample["object"]["size"] == base,
                "not a fixed-envelope cohort"
            );
            ensure!(
                sample["mesh"]["triangles"].as_u64().is_some_and(|n| n > 0)
                    && sample["mesh"]["max_envelope_overrun_m"]
                        .as_f64()
                        .is_some_and(|v| (0.0..=0.003).contains(&v)),
                "invalid audited mesh"
            );
        }
        let row = rendered
            .iter()
            .find(|row| row["kind"] == kind)
            .context("missing rendered category")?;
        let selected = row["selected"]
            .as_array()
            .context("missing selected objects")?;
        ensure!(
            selected.len() == 6 && selected.iter().all(|s| samples.contains(s)),
            "studio examples differ from audit"
        );
        ensure!(
            selected
                .iter()
                .filter_map(|s| s["object"]["seed"].as_u64())
                .collect::<BTreeSet<_>>()
                .len()
                == 6,
            "repeated studio examples"
        );
        let coverage: Vec<_> = category["coverage"].as_array().context("missing coverage")?.iter().map(|v|json!({"threshold":v["threshold"],"representatives":v["representatives"].as_array().map(Vec::len)})).collect();
        let coverage_text = coverage
            .iter()
            .map(|c| {
                let threshold = c["threshold"].as_f64().context("invalid shape threshold")?;
                Ok(format!(
                    "{} at distance {threshold:.3}",
                    c["representatives"]
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .join("; ");
        html.push_str(&format!("<section><h2>{kind}</h2><p>{count} checked constructions. Depth-map representatives: {coverage_text}</p><div class='grid'>"));
        for (i, sample) in selected.iter().enumerate() {
            for mode in ["color", "normal"] {
                let name = format!("{kind}-{i}-{mode}.png");
                let source = input.join(&name);
                ensure!(
                    image::image_dimensions(&source)? == (320, 320),
                    "invalid studio image"
                );
                fs::copy(&source, output.join("objects").join(&name))?;
                assets.insert(format!("objects/{name}"), io::hash(&source)?);
            }
            html.push_str(&format!("<figure><img loading='lazy' data-base='{kind}-{i}' src='{kind}-{i}-color.png' alt='{kind} construction {i}'><figcaption>Seed {}</figcaption></figure>",sample["object"]["seed"]));
        }
        html.push_str("</div></section>");
        summary.push(json!({"kind":kind,"sample_count":count,"coverage":coverage,"selected_seeds":selected.iter().map(|s|s["object"]["seed"].clone()).collect::<Vec<_>>()}));
    }
    html.push_str(FOOTER);
    fs::write(output.join("objects/index.html"), html)?;
    assets.insert(
        "objects/index.html".into(),
        io::hash(&output.join("objects/index.html"))?,
    );
    for name in ["audit.json", "renders.json"] {
        fs::copy(input.join(name), output.join("objects").join(name))?;
        assets.insert(format!("objects/{name}"), io::hash(&input.join(name))?);
    }
    Ok(json!({"categories":summary,"gallery":"objects/index.html","audit":"objects/audit.json"}))
}

const HEADER: &str = r#"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Functional object construction</title><style>body{font:16px system-ui;background:#10171d;color:#e9edf2;margin:30px auto;max-width:1400px;padding:0 24px}p{line-height:1.65;color:#adc0cd}a{color:#81cbbd}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}img{width:100%}figure{margin:0}figcaption{font-size:12px}section{margin:36px 0}select{font:inherit;padding:8px;background:#22313c;color:white}@media(max-width:850px){.grid{grid-template-columns:repeat(3,1fr)}}</style><a href='../index.html'>← Room study</a><h1>Functional object construction</h1><p>Six representatives per category under fixed camera, metric envelope, finish seed and studio lighting. Representatives maximize depth-map spread; they are illustrative, not a frequency estimate.</p><label>Channel <select id='mode'><option>color</option><option>normal</option></select></label>"#;
const FOOTER: &str = r#"<p>Coverage uses six 24 × 24 first-surface depth maps. Fine controls may be under-resolved. A count of one indicates limited shape variation at this resolution; varying dimensions, finish or sub-pixel details does not establish distinct geometry. Geometry checks do not certify every hidden contact or photographic realism.</p><a href='audit.json'>All construction parameters and mesh measurements</a><script>document.querySelector('#mode').onchange=e=>document.querySelectorAll('img[data-base]').forEach(i=>i.src=i.dataset.base+'-'+e.target.value+'.png')</script>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_receipt_rejects_stale_source_repeated_examples_and_scale_drift() {
        let input = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let identity = json!({"source_sha256":"test", "generator_version":44});
        let samples: Vec<_> = (0..8).map(|seed|json!({"object":{"kind":"Lectern","seed":seed,"size":[1,1,1]},"mesh":{"triangles":12,"max_envelope_overrun_m":0.0}})).collect();
        let mut audit = json!({"build_provenance":identity,"sample_count_per_category":8,"categories":[{"kind":"Lectern","samples":samples,"failures":[],"coverage":[]}]});
        let mut renders = json!({"build_provenance":identity,"image_size":[320,320],"categories":[{"kind":"Lectern","selected":samples[..6]}]});
        for i in 0..6 {
            for mode in ["color", "normal"] {
                image::RgbImage::new(320, 320)
                    .save(input.path().join(format!("Lectern-{i}-{mode}.png")))
                    .unwrap();
            }
        }
        let check = |a: &Value, r: &Value| {
            io::write(&input.path().join("audit.json"), a).unwrap();
            io::write(&input.path().join("renders.json"), r).unwrap();
            prepare(input.path(), output.path(), &identity, &mut BTreeMap::new())
        };
        check(&audit, &renders).unwrap();
        renders["build_provenance"]["source_sha256"] = "stale".into();
        assert!(check(&audit, &renders).is_err());
        renders["build_provenance"] = identity.clone();
        renders["categories"][0]["selected"][1] = samples[0].clone();
        assert!(check(&audit, &renders).is_err());
        renders["categories"][0]["selected"] = json!(samples[..6]);
        audit["categories"][0]["samples"][7]["object"]["size"] = json!([2, 1, 1]);
        assert!(check(&audit, &renders).is_err());
    }
}
