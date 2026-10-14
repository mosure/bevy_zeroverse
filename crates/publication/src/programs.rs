//! Local, current-state functional-program gallery. Inputs are complete native
//! cohorts; every selected scene and annotation is verified before any report.
use crate::{cohorts, io, pipeline};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
mod figures;
mod objects;

/// A gallery is a qualification receipt, so failed or truncated audits may not
/// be silently attached to it as if they supported acceptance.
fn validate_audit(audit: &Value) -> Result<()> {
    if let Some(profiles) = audit["profiles"].as_array() {
        ensure!(!profiles.is_empty(), "empty structural audit");
        for profile in profiles {
            ensure!(
                profile["requested"].as_u64().is_some_and(|n| n > 0),
                "missing audit population"
            );
            ensure!(
                profile["failures"].as_array().is_some_and(Vec::is_empty),
                "structural audit contains failures"
            );
        }
    } else if let Some(rows) = audit["rows"].as_array() {
        ensure!(
            audit["complete"] == true && !rows.is_empty(),
            "incomplete camera audit"
        );
        ensure!(
            rows.iter()
                .all(|row| row.get("error").is_some_and(Value::is_null)),
            "camera audit contains failures or missing results"
        );
    } else {
        anyhow::bail!("unsupported qualification audit");
    }
    Ok(())
}

pub fn prepare(
    root: &Path,
    inputs: &[PathBuf],
    audits: &[PathBuf],
    object_inputs: Option<&Path>,
    output: &Path,
) -> Result<()> {
    let final_destination = crate::study::local_destination(root, output)?;
    ensure!(
        !final_destination.exists(),
        "program-study destination already exists"
    );
    let mut qualification = cohorts::collect(root, inputs)?;
    ensure!(
        qualification["identity"] == json!(pipeline::identity(root)?),
        "functional captures do not match the current source"
    );
    let parent = final_destination
        .parent()
        .context("missing output parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".program-study-")
        .tempdir_in(parent)?;
    let destination = stage.path().to_owned();
    let mut receipts = Vec::new();
    let mut camera_examples = Vec::new();
    let mut assets = std::collections::BTreeMap::new();
    for (i, input) in audits.iter().enumerate() {
        let source = root.join(input);
        let audit: Value = io::read(&source)?;
        validate_audit(&audit).with_context(|| format!("qualifying {}", input.display()))?;
        ensure!(
            audit["provenance"]["source_sha256"] == qualification["identity"]["source_sha256"],
            "audit source does not match native captures: {}",
            input.display()
        );
        let name = format!("audit_{i:02}.json");
        fs::copy(&source, destination.join(&name))?;
        assets.insert(name.clone(), io::hash(&destination.join(&name))?);
        for row in audit["rows"].as_array().into_iter().flatten() {
            let Some(name) = row["schematic"].as_str() else {
                continue;
            };
            ensure!(
                Path::new(name).file_name().and_then(|p| p.to_str()) == Some(name),
                "invalid schematic filename"
            );
            let path = source.parent().context("missing audit parent")?.join(name);
            fs::copy(&path, destination.join(name))?;
            assets.insert(name.into(), io::hash(&path)?);
            camera_examples.push(json!({"profile":row["profile"],"seed":row["seed"],"baseline":row["baseline"],"cameras":row["cameras"],"file":name,"geometry":row["geometry"]}));
        }
        receipts.push(json!({"file":name,"sha256":io::hash(&source)?,"policy":audit["policy"],"source":input}));
    }
    qualification["audits"] = json!(receipts);
    qualification["camera_examples"] = json!(camera_examples);
    let mut scenes = Vec::new();
    for input in inputs {
        let directory = root.join(input);
        let selection: Value = io::read(&directory.join("render_selection.json"))?;
        for seed in selection["selected_seeds"]
            .as_array()
            .context("missing seeds")?
        {
            let seed = seed.as_u64().context("invalid seed")?;
            let source = directory.join(format!("seed_{seed:06}"));
            let manifest: Value = io::read(&source.join("manifest.json"))?;
            let capture: Value = io::read(&source.join("capture.json"))?;
            let profile = manifest["layout"]
                .as_str()
                .context("missing room activity")?;
            ensure!(
                profile.chars().all(|c| c.is_ascii_alphanumeric()),
                "invalid activity slug"
            );
            let relative = format!("media/{profile}/{seed}");
            let target = destination.join(&relative);
            fs::create_dir_all(&target)?;
            for file in fs::read_dir(&source)? {
                let file = file?;
                let name = file.file_name();
                let name = name.to_str().context("non-UTF8 capture name")?;
                if name.ends_with(".png")
                    || name == "schematic.svg"
                    || name == "manifest.json"
                    || name == "capture.json"
                {
                    ensure!(
                        file.file_type()?.is_file(),
                        "capture assets must be regular files"
                    );
                    fs::copy(file.path(), target.join(name))?;
                    assets.insert(format!("{relative}/{name}"), io::hash(&target.join(name))?);
                }
            }
            let views = capture["views"]
                .as_array()
                .context("missing recorded views")?;
            let mut counts = std::collections::BTreeMap::<String, usize>::new();
            for object in manifest["objects"].as_array().context("missing objects")? {
                if object["neighbor"] != true {
                    *counts
                        .entry(object["kind"].as_str().context("missing kind")?.into())
                        .or_default() += 1;
                }
            }
            let max_error = views
                .iter()
                .filter_map(|v| v["annotation_alignment"]["reprojection_max_pixels"].as_f64())
                .fold(0.0_f64, f64::max);
            let min_classes = views
                .iter()
                .filter_map(|v| v["semantic_colors"].as_u64())
                .min();
            scenes.push(json!({"profile":profile,"seed":seed,"path":relative,"views":views.len(),"objects":counts,"people":manifest["humans"].as_array().map_or(0,Vec::len),"seconds":capture["elapsed_seconds"],"maximum_reprojection_error_pixels":max_error,"minimum_semantic_classes":min_classes,"co_visibility":capture["co_visibility_metadata"],"programs":manifest["assembly_diagnostics"]["programs"],"kitchens":manifest["assembly_diagnostics"]["kitchens"],"image_size":capture["image_size"]}));
        }
    }
    qualification["scenes"] = json!(scenes);
    let qualification_notes = root.join("docs/procedural_public_spaces.md");
    fs::copy(&qualification_notes, destination.join("qualification.md"))?;
    assets.insert("qualification.md".into(), io::hash(&qualification_notes)?);
    if let Some(input) = object_inputs {
        qualification["object_study"] = objects::prepare(
            &root.join(input),
            &destination,
            &qualification["identity"],
            &mut assets,
        )?;
    }
    qualification["assets_sha256"] = json!(assets);
    qualification["scope"]=json!("Instantaneous native captures; all requested seeds retained. Geometric and annotation qualification is distinct from photographic realism and downstream learning utility. Concurrent GPU workloads may affect elapsed times. Local review only.");
    figures::write(&destination, &qualification)?;
    for name in [
        "program_plate.svg",
        "program_plate.png",
        "program_metrics.svg",
        "program_metrics.png",
        "programs.tex",
        "figure_sources.json",
    ] {
        qualification["assets_sha256"][name] = io::hash(&destination.join(name))?.into();
    }
    io::write(&destination.join("report.json"), &qualification)?;
    let data = serde_json::to_string(&qualification)?.replace('<', "\\u003c");
    fs::write(
        destination.join("index.html"),
        HTML.replace("__DATA__", &data),
    )?;
    fs::rename(&destination, &final_destination)?;
    Ok(())
}

const HTML: &str = r##"<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zeroverse · Functional spaces</title>
<style>:root{color-scheme:dark;font:15px system-ui;color:#e9edf2;background:#10171d}body{margin:0}header{padding:30px 4vw 20px;border-bottom:1px solid #33414c;background:#16212b}h1{font-size:32px;font-weight:620;letter-spacing:-.8px;margin:8px 0}p{color:#adc0cd;max-width:950px;line-height:1.6}small{color:#81cbbd;text-transform:uppercase;letter-spacing:2px}main{padding:22px 4vw}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}label{display:grid;gap:6px;color:#aec0cf;font-size:12px}select,button{padding:10px 14px;background:#22313c;color:white;border:1px solid #4b6272;border-radius:7px;font:15px system-ui;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:#1b2833;border-radius:10px;overflow:hidden}img{width:100%;display:block}figcaption{padding:10px 14px;color:#a8bdce}.detail{display:grid;grid-template-columns:360px 1fr;gap:24px;margin-top:24px}pre{white-space:pre-wrap;font:12px/1.6 ui-monospace;max-height:520px;overflow:auto;color:#b6cddd}.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;margin:25px 0}.strip button{padding:0;overflow:hidden;font-size:11px}.strip img{aspect-ratio:1;object-fit:cover}.stat{color:#81cbbd;font-size:18px}a{color:#81cbbd}@media(max-width:850px){.grid{grid-template-columns:1fr}.detail{grid-template-columns:1fr}}</style>
<header><small>Bevy Zeroverse · Local qualification</small><h1>Spaces with a purpose</h1><p>Transit, preparation, service, dining, teaching and exhibition. Each room below is a consecutive generated sample with calibrated views and aligned geometry annotations.</p><div class="stat" id="stats"></div></header><main><div class="controls"><label>Room program<select id="family"></select></label><label>Room seed<select id="seed"></select></label><label>Annotation<select id="mode"><option value="color">PBR color</option><option value="depth">Depth</option><option value="normal">Surface normal</option><option value="position">World position</option><option value="semantic">Semantic classes</option><option value="co_visibility">Co-visibility</option></select></label><button id="prev">Previous</button><button id="next">Next</button></div><p id="legend" hidden></p><div class="grid" id="views"></div><div class="detail"><figure><img id="plan"><figcaption>Metric floor plan, geometry and capture trajectories</figcaption></figure><section><h2 id="title"></h2><p id="facts"></p><details><summary>Sampled program and object counts</summary><pre id="parameters"></pre></details><p>Co-visibility uses a shared camera-index bit palette across these views. Depth and position remain metric in the capture API; these images are previews.</p><a href="report.json">Qualification receipt and complete metrics</a></section></div><p id="object-link" hidden><a href="objects/index.html">Object construction: fixed-envelope PBR and normal grids</a></p><p><a href="qualification.md">Construction contracts, qualification scope and remaining limits</a></p><h2>Measured population</h2><figure><img src="program_metrics.svg" alt="Per-program content and capture latency"><figcaption>Every captured room contributes. Timings include annotation validation and image export.</figcaption></figure><p><a href="program_plate.png">Program contact sheet</a> · <a href="programs.tex">Generated paper fragment</a> · <a href="report.json">Source-bound qualification receipts</a></p><section id="camera-study" hidden><h2>Camera count and baseline</h2><p>The same geometry at 1, 3, 6 and 12 cameras, with baseline 0, 0.5 and 1. These diagrams show actual accepted trajectories; overlap in the audit receipts is the placement oracle.</p><label>Camera configuration<select id="camera-example"></select></label><figure style="max-width:640px;margin:18px 0"><img id="camera-plan" alt="Accepted camera trajectories"><figcaption id="camera-facts"></figcaption></figure></section><h2>Every room in this cohort</h2><div class="strip" id="rooms"></div><p id="scope"></p></main><script>
const data=__DATA__, family=document.querySelector('#family'), seed=document.querySelector('#seed'),mode=document.querySelector('#mode');let subset=[];
const option=(text,value)=>{const o=document.createElement('option');o.textContent=text;o.value=value;return o};
for(const p of [...new Set(data.scenes.map(s=>s.profile))])family.append(option(p,p));
document.querySelector('#stats').textContent=`${data.rooms} rooms · ${data.views} calibrated views · generator ${data.identity.generator_version}`;document.querySelector('#scope').textContent=data.scope;document.querySelector('#object-link').hidden=!data.object_study;
function show(){const s=subset[seed.selectedIndex];if(!s)return;const legend=document.querySelector('#legend');legend.hidden=mode.value!=='co_visibility';legend.replaceChildren();for(const c of s.co_visibility?.legend||[]){const item=document.createElement('span');item.style.cssText='display:inline-flex;align-items:center;gap:7px;margin-right:20px';const swatch=document.createElement('span');swatch.style.cssText=`width:14px;height:14px;border-radius:3px;background:rgb(${c.rgb8.join(',')})`;item.append(swatch,document.createTextNode(`Camera ${c.camera_index}`));legend.append(item)}legend.append(document.createTextNode('Colors add for shared visibility; the source-camera bit is excluded.'));const v=document.querySelector('#views');v.replaceChildren();for(let i=0;i<s.views;i++){const fig=document.createElement('figure'),img=document.createElement('img'),c=document.createElement('figcaption');img.src=`${s.path}/view_${String(i).padStart(2,'0')}_${mode.value}.png`;img.alt=`${s.profile} seed ${s.seed} camera ${i} ${mode.value}`;c.textContent=`Camera ${i} · ${mode.selectedOptions[0].textContent}`;fig.append(img,c);v.append(fig)}document.querySelector('#plan').src=`${s.path}/schematic.svg`;document.querySelector('#title').textContent=`${s.profile} · seed ${s.seed}`;document.querySelector('#facts').textContent=`${Object.values(s.objects).reduce((a,b)=>a+b,0)} primary/context object instances · ${s.people} people · ${s.image_size.join(' × ')} pixels`;document.querySelector('#parameters').textContent=JSON.stringify({programs:s.programs,kitchens:s.kitchens,objects:s.objects},null,2);history.replaceState(null,'',`#${s.profile}/${s.seed}/${mode.value}`)}
function choose(){subset=data.scenes.filter(s=>s.profile===family.value);seed.replaceChildren(...subset.map(s=>option(s.seed,s.seed)));const strip=document.querySelector('#rooms');strip.replaceChildren();for(let i=0;i<subset.length;i++){const s=subset[i],b=document.createElement('button'),im=document.createElement('img');im.loading='lazy';im.src=`${s.path}/view_00_color.png`;im.alt=`Seed ${s.seed}`;b.append(im,document.createTextNode(s.seed));b.onclick=()=>{seed.selectedIndex=i;show()};strip.append(b)}show()}
family.onchange=choose;seed.onchange=show;mode.onchange=show;document.querySelector('#prev').onclick=()=>{seed.selectedIndex=(seed.selectedIndex+subset.length-1)%subset.length;show()};document.querySelector('#next').onclick=()=>{seed.selectedIndex=(seed.selectedIndex+1)%subset.length;show()};const configs=data.camera_examples||[];if(configs.length){document.querySelector('#camera-study').hidden=false;const sel=document.querySelector('#camera-example');configs.forEach((c,i)=>sel.append(option(`${c.profile} · ${c.cameras} cameras · baseline ${c.baseline}`,i)));const draw=()=>{const c=configs[+sel.value];document.querySelector('#camera-plan').src=c.file;document.querySelector('#camera-facts').textContent=`Seed ${c.seed} · minimum pairwise baseline ${Number(c.geometry?.min_pairwise_baseline_m??0).toFixed(2)} m`};sel.onchange=draw;draw()}const hash=location.hash.slice(1).split('/');if([...family.options].some(o=>o.value===hash[0]))family.value=hash[0];choose();if([...seed.options].some(o=>o.value===hash[1]))seed.value=hash[1];if([...mode.options].some(o=>o.value===hash[2]))mode.value=hash[2];show();</script>"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_rejects_missing_failed_and_incomplete_audits() {
        for audit in [
            json!({}),
            json!({"profiles": []}),
            json!({"profiles": [{"requested": 4}]}),
            json!({"profiles": [{"requested": 4,"failures": ["invalid support"]}]}),
            json!({"complete": false,"rows": [{"error": null}]}),
            json!({"complete": true,"rows": [{}]}),
            json!({"complete": true,"rows": [{"error": "invalid trajectory"}]}),
        ] {
            assert!(validate_audit(&audit).is_err(), "{audit}");
        }
        validate_audit(&json!({"profiles": [{"requested": 4,"failures": []}]})).unwrap();
        validate_audit(&json!({"complete": true,"rows": [{"error": null}]})).unwrap();
    }
}
