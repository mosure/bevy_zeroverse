//! Optional, batched SigLIP2 audit of completed capture cohorts. No renderer/model coupling.
use anyhow::{bail, ensure, Context, Result};
use burn_siglip2::{
    load_backend_on_device, preprocess_dynamic_images,
    resolve_or_bootstrap_siglip2_weights_with_config, DefaultWgpuBackend, LoadRequest,
    Siglip2BootstrapConfig, Siglip2ModelVariant, WgpuDevice,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// Index produced by scripts/indoor_embedding_report.py index.
    #[arg(long)]
    index: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "base")]
    variant: String,
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Offline override; shards and their checksums are still verified.
    #[arg(long)]
    parts: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
}

#[derive(Deserialize, Serialize)]
struct Index {
    schema_version: u32,
    samples: Vec<Sample>,
    cohorts: serde_json::Value,
}

#[derive(Deserialize, Serialize)]
struct Sample {
    path: PathBuf,
    sha256: String,
    cohort: String,
    seed: u64,
    camera: usize,
    time: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=64).contains(&args.batch_size),
        "batch size must be 1..64"
    );
    ensure!(
        fs::metadata(&args.index)?.len() <= 32 * 1024 * 1024,
        "audit index exceeds 32 MiB; use sampled cohorts"
    );
    let bytes = fs::read(&args.index)?;
    let index: Index = serde_json::from_slice(&bytes)?;
    ensure!(
        index.schema_version == 1 && !index.samples.is_empty() && index.samples.len() <= 50000,
        "unsupported or unbounded audit index"
    );
    ensure!(
        !args.output.join("embeddings.json").exists()
            && !args.output.join("embeddings.f32").exists()
            && !args.output.join("embeddings.f32.partial").exists(),
        "output already contains an embedding run; choose a new directory"
    );
    let variant = Siglip2ModelVariant::from_model_size(&args.variant)
        .context("unsupported SigLIP2 variant")?;
    let started = Instant::now();
    let parts = if let Some(path) = args.parts {
        path
    } else {
        let mut config = Siglip2BootstrapConfig::for_variant(variant);
        config.cache_root = args.cache;
        config.download_tokenizer_assets = false;
        let artifacts = resolve_or_bootstrap_siglip2_weights_with_config(&config)?;
        ensure!(
            artifacts.using_parts,
            "embedding audit requires verified shards"
        );
        artifacts.parts_manifest_path
    };
    let runtime = load_backend_on_device::<DefaultWgpuBackend>(
        LoadRequest::from_parts_manifest(&parts, true),
        WgpuDevice::default(),
    )
    .map_err(anyhow::Error::msg)?;
    ensure!(
        runtime.model.config.supported_variant() == Some(variant),
        "loaded model variant differs from requested variant"
    );
    ensure!(
        runtime.load_stats.part_count > 0
            && runtime.load_stats.sha256_verified == runtime.load_stats.part_count,
        "unverified model shards"
    );
    let load_seconds = started.elapsed().as_secs_f64();
    fs::create_dir_all(&args.output)?;
    // Completion metadata is written only after the entire tensor file succeeds.
    // Refuse overwrite so an old completion marker cannot certify a partial run.
    let temporary = args.output.join("embeddings.f32.partial");
    let file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    ensure!(
        !args.output.join("embeddings.json").exists(),
        "output already contains a completed embedding run"
    );
    let mut writer = BufWriter::new(file);
    let mut digest = Sha256::new();
    let mut dimension = None;
    let mut evidence = None;
    let inference_start = Instant::now();
    for (batch, samples) in index.samples.chunks(args.batch_size).enumerate() {
        let images: Vec<_> = samples
            .iter()
            .map(|sample| {
                let bytes = fs::read(&sample.path)?;
                ensure!(
                    format!("{:x}", Sha256::digest(&bytes)) == sample.sha256,
                    "capture image changed: {}",
                    sample.path.display()
                );
                Ok(image::load_from_memory(&bytes)?)
            })
            .collect::<Result<_>>()?;
        let input = preprocess_dynamic_images::<DefaultWgpuBackend>(
            &images,
            &runtime.model.config,
            &runtime.device,
        )
        .map_err(anyhow::Error::msg)?;
        let mut response = runtime
            .encode_image(input, false)
            .map_err(anyhow::Error::msg)?;
        let [rows, dim] = response.embedding.dims();
        ensure!(
            dim > 0 && rows == samples.len() && dimension.is_none_or(|d| d == dim),
            "embedding shape mismatch"
        );
        dimension = Some(dim);
        let values = response
            .embedding
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("embedding readback: {e:?}"))?;
        response.evidence.host_readbacks += 1;
        for row in values.chunks_exact(dim) {
            let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            if !norm.is_finite() || norm <= 1e-12 {
                bail!("nonfinite/zero embedding in batch {batch}");
            }
            for &v in row {
                let bytes = ((v as f64 / norm) as f32).to_le_bytes();
                writer.write_all(&bytes)?;
                digest.update(bytes);
            }
        }
        evidence = Some(response.evidence);
        eprintln!(
            "embedded {}/{} images",
            ((batch + 1) * args.batch_size).min(index.samples.len()),
            index.samples.len()
        );
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, args.output.join("embeddings.f32"))?;
    fs::write(
        args.output.join("embeddings.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version":1,"encoder":"SigLIP2","variant":args.variant,"burn":"0.21.0",
            "shape":[index.samples.len(),dimension.unwrap()],"dtype":"little-endian float32","normalization":"L2",
            "index_sha256":format!("{:x}",Sha256::digest(&bytes)),"tensor_sha256":format!("{:x}",digest.finalize()),
            "model_parts":parts,"loaded_weights":runtime.load_stats,"last_batch_execution":evidence,
            "embedding_host_readbacks":index.samples.len().div_ceil(args.batch_size),
            "load_seconds":load_seconds,"inference_and_io_seconds":inference_start.elapsed().as_secs_f64(),"batch_size":args.batch_size,
            "preprocessing":"RGB; direct bilinear resize to model square; /255; mean=std=0.5",
            "samples":index.samples,"cohorts":index.cohorts,
        }))?,
    )?;
    Ok(())
}
