//! Offline image-space companion to fixed-partition capacity evidence. Model
//! inference remains the optional `indoor_embed` binary, with verified cached
//! shards. This module never loads a model or launches a renderer.
pub mod kernel;
use super::*;
use crate::{graphics, io};
use kernel::{KernelConfig, KernelSummary};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt::Write, path::PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ImageSample {
    path: PathBuf,
    sha256: String,
    cohort: String,
    seed: u64,
    camera: usize,
    time: f32,
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Room {
    draw: u32,
    seed: u64,
    family_sha256: String,
    image_start: usize,
    image_count: usize,
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Evidence {
    scope: String,
    receipt_sha256: String,
    records_sha256: String,
    generator: bevy_zeroverse_capture::GeneratorIdentity,
    config_sha256: String,
    failed_draws: u32,
    loaded_weight_sha256: String,
    kernel: KernelConfig,
    rooms: Vec<Room>,
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Index {
    schema_version: u32,
    samples: Vec<ImageSample>,
    cohorts: Evidence,
}

fn read<T: serde::de::DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
    ensure!(
        fs::metadata(path)?.len() <= limit,
        "oversized evidence file {}",
        path.display()
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut tmp, value)?;
    tmp.persist_noclobber(path)?;
    Ok(())
}

/// Freeze images, model identity and kernel before inference. Every accepted draw
/// enters with its complete view/time set; failed draws remain in the contract.
pub fn index(
    receipt_path: &Path,
    records_path: &Path,
    captures: &Path,
    weights_sha256: &str,
    config: KernelConfig,
    output: &Path,
) -> Result<()> {
    config.validate()?;
    ensure!(
        weights_sha256.len() == 64 && weights_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected loaded-weight SHA256 required"
    );
    let receipt: CapacityReceipt = read(receipt_path, 32 << 20)?;
    validate_receipt(&receipt.plan, &receipt).map_err(anyhow::Error::msg)?;
    ensure!(
        receipt.estimate.accepted_rooms > 0
            && receipt.estimate.accepted_rooms as usize <= config.max_rooms,
        "cohort does not fit the declared exact kernel budget"
    );
    ensure!(
        receipt.estimate.rendered_rooms == receipt.estimate.accepted_rooms,
        "kernel requires completed renders"
    );
    ensure!(
        fs::metadata(records_path)?.len() <= 64 << 20,
        "oversized records"
    );
    let bytes = fs::read(records_path)?;
    ensure!(sha(&bytes) == receipt.records_sha256, "records changed");
    let lines: Vec<_> = std::str::from_utf8(&bytes)?.lines().collect();
    let records: Vec<CapacityRecord> = lines
        .iter()
        .map(|line| serde_json::from_str(line))
        .collect::<Result<_, _>>()?;
    let checked = qualify(&receipt.plan, &records, sha(&bytes), 0.)?;
    ensure!(
        equivalent_statistics(
            &serde_json::to_value(&checked.estimate)?,
            &serde_json::to_value(&receipt.estimate)?
        ),
        "receipt does not describe these observations"
    );
    let mut samples = Vec::new();
    let mut rooms = Vec::new();
    let capture = &receipt.plan.population.capture;
    for (record, line) in records.into_iter().zip(lines) {
        let SampleOutcome::Accepted { family_sha256, .. } = &record.outcome else {
            continue;
        };
        let folder = captures.join(format!("draw_{:04}", record.draw));
        let preview: Value = read(&folder.join("previews.json"), 1 << 20)?;
        // Bind the producer's original record bytes. Reserializing parsed f64
        // diagnostics can change the decimal spelling without changing meaning.
        ensure!(
            preview["schema_version"] == 1 && preview["record_sha256"] == sha(line.as_bytes()),
            "preview binding differs from completed observation"
        );
        let images = preview["images"]
            .as_array()
            .context("missing preview images")?;
        ensure!(
            images.len() == capture.cameras as usize * capture.progress.len(),
            "preview drops views or times"
        );
        rooms.push(Room {
            draw: record.draw,
            seed: record.seed,
            family_sha256: family_sha256.clone(),
            image_start: samples.len(),
            image_count: images.len(),
        });
        for (i, image) in images.iter().enumerate() {
            let filename = format!("view_{i:02}.png");
            ensure!(image["path"] == filename, "unexpected preview filename");
            let path = folder.join(filename).canonicalize()?;
            let hash = io::hash(&path)?;
            let camera = i % capture.cameras as usize;
            let time = capture.progress[i / capture.cameras as usize];
            ensure!(
                image["sha256"] == hash
                    && image["camera"] == camera
                    && image["time"]
                        .as_f64()
                        .is_some_and(|t| (t - time as f64).abs() < 1e-7),
                "preview image or calibration changed"
            );
            ensure!(
                image::image_dimensions(&path)? == (capture.image_size[0], capture.image_size[1]),
                "preview dimensions differ"
            );
            samples.push(ImageSample {
                path,
                sha256: hash,
                cohort: "current".into(),
                seed: record.seed,
                camera,
                time,
            });
        }
    }
    let population = &receipt.plan.population;
    atomic_json(output, &Index { schema_version:1, samples, cohorts:Evidence {
        scope:"whole-image sRGB; equal view/time average per room, L2-normalized; sample kernel only".into(),
        receipt_sha256:io::hash(receipt_path)?, records_sha256:receipt.records_sha256,
        generator:population.generator.clone(), config_sha256:population.config_sha256.clone(),
        failed_draws:receipt.estimate.failed_rooms, loaded_weight_sha256:weights_sha256.into(), kernel:config, rooms,
    }})
}

/// Verify exact index, captures, weights and tensor bytes; never infer IID
/// confidence or extrapolate an embedding-kernel sample score.
pub fn report(index_path: &Path, embeddings: &Path, output: &Path) -> Result<()> {
    let index: Index = read(index_path, 32 << 20)?;
    let metadata: Value = read(&embeddings.join("embeddings.json"), 32 << 20)?;
    ensure!(!output.exists(), "kernel report already exists");
    index.cohorts.kernel.validate()?;
    for hash in [
        &index.cohorts.generator.source_sha256,
        &index.cohorts.config_sha256,
        &index.cohorts.loaded_weight_sha256,
    ] {
        ensure!(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid index identity"
        );
    }
    let (vectors, dims) = load_vectors(
        &index,
        &metadata,
        &io::hash(index_path)?,
        &embeddings.join("embeddings.f32"),
    )?;
    for sample in &index.samples {
        ensure!(
            io::hash(&sample.path)? == sample.sha256,
            "embedding image changed"
        );
    }
    let centroids = room_centroids(&index, &vectors, dims)?;
    let summary = kernel::analyze(&centroids, &index.cohorts.kernel)?;
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix("kernel-report-")
        .tempdir_in(parent)?;
    let stage = staging.path();
    atomic_json(
        &stage.join("kernel.json"),
        &json!({
            "schema_version":1,"kind":"rbf-sample-vendi-v1","population_capacity_estimate":null,
            "scope":index.cohorts.scope,"binding":index.cohorts,"result":summary,
            "index_sha256":io::hash(index_path)?,"tensor_sha256":metadata["tensor_sha256"],
            "loaded_weights":metadata["loaded_weights"],"preprocessing":metadata["preprocessing"],
            "limits":["Bandwidth is prespecified, not a calibrated perceptual equivalence threshold.","Whole-image embeddings include exterior context; they do not measure only the primary reconstruction region.","Vendi and nearest neighbors describe this observed sample, not expected distinct classes in ten million draws or learning benefit."]
        }),
    )?;
    write_gallery(&index, &summary, stage)?;
    fs::rename(stage, output)?;
    Ok(())
}

fn load_vectors(
    index: &Index,
    metadata: &Value,
    index_sha: &str,
    path: &Path,
) -> Result<(Vec<f64>, usize)> {
    ensure!(
        index.schema_version == 1
            && metadata["schema_version"] == 1
            && metadata["encoder"] == "SigLIP2",
        "unsupported embedding contract"
    );
    ensure!(
        equivalent_statistics(&metadata["samples"], &serde_json::to_value(&index.samples)?)
            && equivalent_statistics(&metadata["cohorts"], &serde_json::to_value(&index.cohorts)?)
            && metadata["index_sha256"] == index_sha,
        "embedding run does not bind this frozen index"
    );
    let w = &metadata["loaded_weights"];
    ensure!(
        w["part_count"].as_u64().is_some_and(|n| n > 0)
            && w["part_count"] == w["sha256_verified"]
            && w["loaded_weight_sha256"] == index.cohorts.loaded_weight_sha256,
        "model differs or shards were not verified"
    );
    ensure!(
        metadata["normalization"] == "L2"
            && metadata["dtype"] == "little-endian float32"
            && metadata["preprocessing"]
                == "RGB; direct bilinear resize to model square; /255; mean=std=0.5",
        "unknown embedding preprocessing"
    );
    let rows = metadata["shape"][0].as_u64().context("missing rows")? as usize;
    let dims = metadata["shape"][1]
        .as_u64()
        .context("missing dimensions")? as usize;
    ensure!(
        rows == index.samples.len() && (1..=32768).contains(&rows) && (1..=4096).contains(&dims),
        "unbounded or inconsistent tensor shape"
    );
    ensure!(
        fs::metadata(path)?.len() == (rows * dims * 4) as u64,
        "embedding byte count differs"
    );
    let bytes = fs::read(path)?;
    ensure!(
        sha(&bytes) == metadata["tensor_sha256"],
        "embedding tensor changed"
    );
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    for row in values.chunks_exact(dims) {
        ensure!(
            row.iter().all(|x| x.is_finite())
                && (row.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-4,
            "invalid normalized embedding"
        );
    }
    Ok((values, dims))
}

// Aggregate f64 values may move one ULP through serde_json's decimal parser.
// Counts, strings, ordering and input hashes remain exact; permit only numerical
// roundoff in recomputed statistics, never changed descriptors or input bytes.
fn equivalent_statistics(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equivalent_statistics(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| equivalent_statistics(a, b)))
        }
        _ => a == b,
    }
}

fn room_centroids(index: &Index, values: &[f64], dims: usize) -> Result<Vec<Vec<f64>>> {
    ensure!(
        !index.cohorts.rooms.is_empty()
            && index.cohorts.rooms.len() <= index.cohorts.kernel.max_rooms,
        "unbounded rooms"
    );
    let mut cursor = 0;
    let mut centroids = Vec::new();
    for room in &index.cohorts.rooms {
        ensure!(
            room.image_start == cursor
                && room.image_count > 0
                && room.image_count <= index.samples.len().saturating_sub(cursor),
            "overlapping, empty or omitted room images"
        );
        let mut centroid = vec![0.; dims];
        for i in cursor..cursor + room.image_count {
            ensure!(
                index.samples[i].seed == room.seed,
                "room/image identity mismatch"
            );
            for (d, value) in centroid.iter_mut().enumerate() {
                *value += values[i * dims + d];
            }
        }
        cursor += room.image_count;
        let norm = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(norm > 1e-12 && norm.is_finite(), "zero room embedding");
        for x in &mut centroid {
            *x /= norm;
        }
        centroids.push(centroid);
    }
    ensure!(cursor == index.samples.len(), "unassigned image embeddings");
    Ok(centroids)
}

fn write_gallery(index: &Index, summary: &KernelSummary, output: &Path) -> Result<()> {
    let mut pairs = BTreeMap::new();
    for (i, j) in summary.nearest_room.iter().enumerate() {
        if let Some(j) = j {
            pairs.insert(
                (i.min(*j), i.max(*j)),
                summary.nearest_cosine_distance[i].context("missing nearest distance")?,
            );
        }
    }
    let mut pairs: Vec<_> = pairs.into_iter().collect();
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut gallery = String::new();
    for ((i, j), distance) in pairs.iter().take(12) {
        write!(
            gallery,
            "<article><h3>Rooms {i} / {j} · cosine distance {distance:.4}</h3><div class=pair>"
        )?;
        for &room in [i, j] {
            let record = &index.cohorts.rooms[room];
            write!(
                gallery,
                "<div><p>Draw {} · seed {}</p><div class=views>",
                record.draw, record.seed
            )?;
            for k in 0..record.image_count.min(4) {
                let filename = format!("room_{room:04}_{k:02}.png");
                fs::copy(
                    &index.samples[record.image_start + k].path,
                    output.join(&filename),
                )?;
                write!(
                    gallery,
                    "<img src=\"{filename}\" alt=\"Room {room}, observation {k}\" loading=lazy>"
                )?;
            }
            gallery.push_str("</div></div>");
        }
        gallery.push_str("</div></article>");
    }
    let mut svg = graphics::begin(900, 380);
    graphics::text(&mut svg, 30., 32., 22, "Observed image-space diversity");
    graphics::text(
        &mut svg,
        30.,
        60.,
        13,
        &format!(
            "{} rooms; RBF bandwidth {}; equal room weights; no population extrapolation",
            summary.rooms, index.cohorts.kernel.bandwidth
        ),
    );
    let maximum = summary.eigenvalues[0].max(1e-12);
    for (i, &v) in summary.eigenvalues.iter().take(32).enumerate() {
        let x = 62. + i as f64 * 24.;
        let h = 200. * v / maximum;
        write!(
            svg,
            "<rect x=\"{x}\" y=\"{}\" width=\"18\" height=\"{h}\" fill=\"#147d70\"/>",
            310. - h
        )?;
    }
    graphics::text(
        &mut svg,
        30.,
        85.,
        12,
        &format!(
            "Top eigenvalue {:.4}; remaining bars use the same linear scale",
            maximum
        ),
    );
    for (i, label) in [(0, "1"), (7, "8"), (15, "16"), (23, "24"), (31, "32")] {
        graphics::text(&mut svg, 62. + 24. * i as f64, 335., 12, label);
    }
    graphics::text(&mut svg, 845., 335., 12, "rank");
    graphics::text(
        &mut svg,
        62.,
        363.,
        14,
        &format!(
            "Vendi {:.2} / {} observed rooms · participation {:.2}",
            summary.vendi, summary.rooms, summary.participation
        ),
    );
    svg.push_str("</g></svg>");
    graphics::save(&svg, &output.join("kernel.svg"))?;
    fs::write(
        output.join("index.html"),
        format!(
            r#"<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zeroverse · image-space diversity</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:3rem auto;padding:1rem;color:#20382e;background:#f7f8f3}}article{{background:white;border:1px solid #d8e2d9;border-radius:14px;padding:1rem;margin:1rem 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}.views{{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}}img{{width:100%;height:auto}}@media(max-width:650px){{.pair{{grid-template-columns:1fr}}}}</style><h1>Current cohort · nearest rooms</h1><p>{} completed rooms; {} failed draws retained. These are the closest room-centroid pairs in one population, with all views kept together. First four observations per room are shown.</p><img src="kernel.svg" alt="Uncentered RBF kernel spectrum"><p>The bandwidth was frozen before inference. This whole-image diagnostic includes exterior context. Vendi is sample diversity, not unseen support, a perceptual distinct-count estimate or a prediction of training benefit.</p><p><a href="kernel.json">Bound evidence and full eigenvalues</a></p>{gallery}"#,
            summary.rooms, index.cohorts.failed_draws
        ),
    )?;
    fs::write(
        output.join("kernel.tex"),
        format!(
            r"% Generated from frozen image index; sample statistic only.
\begin{{table}}[htbp]\centering
\begin{{tabular}}{{rrrr}}\toprule Rooms & RBF $\sigma$ & Vendi & Participation\\\midrule
{} & {} & {:.2} & {:.2}\\\bottomrule\end{{tabular}}
\caption{{Whole-image SigLIP2 sample diversity, using equal-weight normalized room centroids. This is not a generator support estimate or a learning scaling law. Source \texttt{{{}}}; configuration \texttt{{{}}}.}}\label{{tab:capacity-kernel}}
\end{{table}}
",
            summary.rooms,
            index.cohorts.kernel.bandwidth,
            summary.vendi,
            summary.participation,
            &index.cohorts.generator.source_sha256[..12],
            &index.cohorts.config_sha256[..12]
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernel_gallery_renders_a_complete_current_cohort() {
        let tmp = tempfile::tempdir().unwrap();
        let mut index = fixture();
        for (i, sample) in index.samples.iter_mut().enumerate() {
            sample.path = tmp.path().join(format!("input-{i}.png"));
            image::RgbImage::new(16, 16).save(&sample.path).unwrap();
        }
        let output = tmp.path().join("report");
        fs::create_dir(&output).unwrap();
        let summary =
            kernel::analyze(&[vec![1., 0.], vec![0., 1.]], &index.cohorts.kernel).unwrap();
        write_gallery(&index, &summary, &output).unwrap();
        assert!(output.join("kernel.png").exists());
        assert!(fs::read_to_string(output.join("index.html"))
            .unwrap()
            .contains("Rooms 0 / 1"));
        assert!(output.join("room_0000_01.png").exists());
    }
    #[test]
    fn completed_preview_binding_uses_original_record_bytes_and_rejects_image_changes() {
        let (mut plan, mut records) = super::super::tests::fixture();
        records.truncate(4);
        plan.request.budget.manifests = 4;
        plan.request.budget.rendered_rooms = 4;
        let tmp = tempfile::tempdir().unwrap();
        let mut lines = String::new();
        for r in &mut records {
            if let SampleOutcome::Accepted {
                capture_sha256,
                delivered_views,
                delivered_progress,
                diagnostics,
                ..
            } = &mut r.outcome
            {
                *capture_sha256 = Some(sha(b"raw capture fixture"));
                *delivered_views = 4;
                *delivered_progress = vec![0.];
                diagnostics.insert("fraction".into(), 0.012345678987654321);
            }
            let raw = serde_json::to_string(r).unwrap();
            let folder = tmp.path().join(format!("draw_{:04}", r.draw));
            fs::create_dir(&folder).unwrap();
            let mut images = Vec::new();
            for i in 0..4 {
                let name = format!("view_{i:02}.png");
                let path = folder.join(&name);
                image::RgbImage::new(16, 16).save(&path).unwrap();
                images.push(
                    json!({"path":name,"camera":i,"time":0.,"sha256":io::hash(&path).unwrap()}),
                );
            }
            fs::write(folder.join("previews.json"),serde_json::to_vec(&json!({"schema_version":1,"record_sha256":sha(raw.as_bytes()),"images":images})).unwrap()).unwrap();
            lines.push_str(&raw);
            lines.push('\n');
        }
        let receipt = qualify(&plan, &records, sha(lines.as_bytes()), 0.).unwrap();
        let receipt_path = tmp.path().join("receipt.json");
        let rows = tmp.path().join("records.jsonl");
        fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        fs::write(&rows, lines).unwrap();
        let config = KernelConfig {
            bandwidth: 0.35,
            max_rooms: 512,
        };
        index(
            &receipt_path,
            &rows,
            tmp.path(),
            &sha(b"weights"),
            config.clone(),
            &tmp.path().join("index.json"),
        )
        .unwrap();
        fs::write(tmp.path().join("draw_0000/view_00.png"), b"changed").unwrap();
        assert!(index(
            &receipt_path,
            &rows,
            tmp.path(),
            &sha(b"weights"),
            config,
            &tmp.path().join("changed.json")
        )
        .is_err());
    }
    #[test]
    fn roundtrip_tolerance_does_not_admit_changed_counts_or_material_statistics() {
        assert!(equivalent_statistics(
            &json!({"n":128,"value":1.0000000000000002}),
            &json!({"n":128,"value":1.0})
        ));
        assert!(!equivalent_statistics(&json!({"n":128}), &json!({"n":129})));
        assert!(!equivalent_statistics(
            &json!({"value":1.0}),
            &json!({"value":1.00001})
        ));
    }
    fn fixture() -> Index {
        Index {
            schema_version: 1,
            samples: (0..3)
                .map(|i| ImageSample {
                    path: format!("{i}.png").into(),
                    sha256: sha(b"image"),
                    cohort: "current".into(),
                    seed: if i < 2 { 10 } else { 11 },
                    camera: i % 2,
                    time: 0.,
                })
                .collect(),
            cohorts: Evidence {
                scope: "whole image".into(),
                receipt_sha256: sha(b"receipt"),
                records_sha256: sha(b"rows"),
                generator: bevy_zeroverse_capture::GeneratorIdentity {
                    schema_version: 1,
                    crate_version: "test".into(),
                    generator_version: 50,
                    source_sha256: sha(b"source"),
                },
                config_sha256: sha(b"config"),
                failed_draws: 1,
                loaded_weight_sha256: sha(b"weights"),
                kernel: KernelConfig {
                    bandwidth: 0.35,
                    max_rooms: 512,
                },
                rooms: vec![
                    Room {
                        draw: 0,
                        seed: 10,
                        family_sha256: sha(b"a"),
                        image_start: 0,
                        image_count: 2,
                    },
                    Room {
                        draw: 2,
                        seed: 11,
                        family_sha256: sha(b"b"),
                        image_start: 2,
                        image_count: 1,
                    },
                ],
            },
        }
    }
    #[test]
    fn room_aggregation_keeps_views_together_and_rejects_omissions() {
        let mut index = fixture();
        let values = [1., 0., 1., 0., 0., 1.];
        let rooms = room_centroids(&index, &values, 2).unwrap();
        assert_eq!(rooms, vec![vec![1., 0.], vec![0., 1.]]);
        index.cohorts.rooms[1].image_start = 1;
        assert!(room_centroids(&index, &values, 2).is_err());
        index.cohorts.rooms.pop();
        assert!(room_centroids(&index, &values, 2).is_err());
    }
    #[test]
    fn tensor_checks_bind_model_preprocessing_index_and_every_byte() {
        let index = fixture();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tensor.f32");
        let bytes: Vec<_> = [1_f32, 0., 1., 0., 0., 1.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        fs::write(&path, &bytes).unwrap();
        let mut metadata = json!({"schema_version":1,"encoder":"SigLIP2","samples":index.samples,"cohorts":index.cohorts,"index_sha256":"index","normalization":"L2","dtype":"little-endian float32","preprocessing":"RGB; direct bilinear resize to model square; /255; mean=std=0.5","shape":[3,2],"tensor_sha256":sha(&bytes),"loaded_weights":{"part_count":2,"sha256_verified":2,"loaded_weight_sha256":index.cohorts.loaded_weight_sha256}});
        assert!(load_vectors(&index, &metadata, "index", &path).is_ok());
        metadata["loaded_weights"]["sha256_verified"] = json!(1);
        assert!(load_vectors(&index, &metadata, "index", &path).is_err());
        metadata["loaded_weights"]["sha256_verified"] = json!(2);
        metadata["loaded_weights"]["loaded_weight_sha256"] = json!(sha(b"other"));
        assert!(load_vectors(&index, &metadata, "index", &path).is_err());
        metadata["loaded_weights"]["loaded_weight_sha256"] =
            json!(index.cohorts.loaded_weight_sha256);
        fs::write(&path, [0; 24]).unwrap();
        assert!(load_vectors(&index, &metadata, "index", &path).is_err());
    }
}
