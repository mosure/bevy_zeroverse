use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use burn_siglip2::{LoadRequest, SIGLIP2_MAX_ENCODED_IMAGE_BYTES, load_backend};

fn usage() -> String {
    "usage: siglip2_short_bench <model.bpk.parts.json> <image> [iterations]".to_string()
}

fn read_image_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open image '{}': {err}", path.display()))?;
    let declared = file
        .metadata()
        .map_err(|err| format!("failed to inspect image '{}': {err}", path.display()))?
        .len();
    if declared > SIGLIP2_MAX_ENCODED_IMAGE_BYTES as u64 {
        return Err(format!(
            "encoded image '{}' is {declared} bytes; maximum is {SIGLIP2_MAX_ENCODED_IMAGE_BYTES}",
            path.display()
        ));
    }

    let mut bytes = Vec::with_capacity(declared as usize);
    file.take(SIGLIP2_MAX_ENCODED_IMAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read image '{}': {err}", path.display()))?;
    if bytes.len() > SIGLIP2_MAX_ENCODED_IMAGE_BYTES {
        return Err(format!(
            "encoded image '{}' grew beyond the {}-byte maximum while reading",
            path.display(),
            SIGLIP2_MAX_ENCODED_IMAGE_BYTES
        ));
    }
    Ok(bytes)
}

fn parse_args() -> Result<(PathBuf, PathBuf, usize), String> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let image = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let iterations = match args.next() {
        Some(value) => value
            .to_str()
            .ok_or_else(|| "iterations must be valid UTF-8".to_string())?
            .parse::<usize>()
            .map_err(|err| format!("invalid iterations value: {err}"))?,
        None => 3,
    };
    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }
    if args.next().is_some() {
        return Err(usage());
    }
    Ok((manifest, image, iterations))
}

fn main() -> Result<(), String> {
    let (manifest, image_path, iterations) = parse_args()?;
    let encoded_image = read_image_bounded(&image_path)?;

    let load_start = Instant::now();
    let runtime = load_backend(LoadRequest::from_parts_manifest(&manifest, true))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1_000.0;

    let warmup = runtime.encode_image_bytes(&encoded_image, false)?;
    let warmup_values = warmup
        .embedding
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read warmup output: {err:?}"))?;
    let warmup_norm = warmup_values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !warmup_norm.is_finite() || warmup_norm <= 0.0 {
        return Err(format!(
            "warmup embedding has invalid L2 norm {warmup_norm}"
        ));
    }

    let start = Instant::now();
    for _ in 0..iterations {
        runtime.encode_image_bytes(&encoded_image, false)?;
    }
    let total_ms = start.elapsed().as_secs_f64() * 1_000.0;

    println!("siglip2 production image bench");
    println!("manifest={}", manifest.display());
    println!(
        "model: image_size={} projection_dim={} layers={}",
        runtime.model.config.image_size,
        runtime.model.config.projection_dim,
        runtime.model.config.num_layers
    );
    println!(
        "load: milliseconds={load_ms:.3} parts={} bytes={} sha256_verified={} tensors={}",
        runtime.load_stats.part_count,
        runtime.load_stats.loaded_bytes,
        runtime.load_stats.sha256_verified,
        runtime.load_stats.tensors_loaded
    );
    println!("warmup_embedding_l2={warmup_norm:.6}");
    println!(
        "timing: iterations={iterations} total_ms={total_ms:.3} avg_ms={:.3}",
        total_ms / iterations as f64
    );
    Ok(())
}
