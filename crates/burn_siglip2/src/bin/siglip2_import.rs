#![recursion_limit = "256"]

use std::path::PathBuf;

use clap::{ArgAction, Parser};

use burn_siglip2::{
    bpk::Siglip2ArtifactMetadata,
    import::{
        Siglip2ImportOptions, Siglip2OutputPrecision, expected_upstream_model_id, import_hf_dir,
    },
};

#[derive(Debug, Parser)]
#[command(about = "Import a Hugging Face SigLIP checkpoint into burn_siglip2 BPK artifacts")]
struct ImportArgs {
    #[arg(long)]
    hf_dir: PathBuf,

    #[arg(long)]
    output: PathBuf,

    /// Exact supported checkpoint variant written into immutable artifact metadata.
    #[arg(
        long,
        value_parser = [
            "base-patch16-224",
            "large-patch16-256",
            "so400m-patch14-224"
        ]
    )]
    model_variant: String,

    /// Full 40-character Hugging Face git commit for the downloaded checkpoint.
    #[arg(long)]
    upstream_revision: String,

    /// Override the canonical google/siglip2-* model id (must still match the variant).
    #[arg(long)]
    upstream_model_id: Option<String>,

    /// Floating-point storage dtype. CDN bundles default to f16.
    #[arg(long, default_value_t = Siglip2OutputPrecision::F16)]
    precision: Siglip2OutputPrecision,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    parts: bool,

    #[arg(long, default_value_t = burn_siglip2::DEFAULT_PART_SIZE_MIB)]
    parts_max_mib: u64,

    #[arg(long, default_value_t = false)]
    parts_overwrite: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    copy_tokenizer_assets: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    reject_unused_source_tensors: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ImportArgs::parse();
    let upstream_model_id = args.upstream_model_id.unwrap_or_else(|| {
        expected_upstream_model_id(&args.model_variant)
            .expect("clap restricts model variants")
            .to_string()
    });
    let artifact_metadata = Siglip2ArtifactMetadata {
        model_variant: args.model_variant,
        upstream_model_id,
        upstream_revision: args.upstream_revision,
        storage_dtype: args.precision.storage_dtype().to_string(),
    };
    let options = Siglip2ImportOptions {
        write_parts: args.parts,
        parts_max_mib: args.parts_max_mib,
        parts_overwrite: args.parts_overwrite,
        copy_tokenizer_assets: args.copy_tokenizer_assets,
        output_precision: args.precision,
        artifact_metadata: Some(artifact_metadata),
        reject_unused_source_tensors: args.reject_unused_source_tensors,
    };
    let report = import_hf_dir(&args.hf_dir, &args.output, &options)?;

    println!("Imported SigLIP2 checkpoint");
    println!("  config: {:?}", report.config);
    println!("  bpk: {}", report.bpk_path.display());
    println!("  storage dtype: {}", report.output_precision);
    if let Some(artifact) = &report.artifact_metadata {
        println!(
            "  upstream: {}@{} ({})",
            artifact.upstream_model_id, artifact.upstream_revision, artifact.model_variant
        );
    }
    if let Some(parts) = report.parts {
        println!(
            "  parts manifest: {} ({} parts, {:.1} MiB source)",
            parts.manifest_path.display(),
            parts.part_paths.len(),
            parts.total_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  requested/actual largest part: {:.1}/{:.1} MiB",
            parts.requested_max_part_bytes as f64 / (1024.0 * 1024.0),
            parts.actual_max_part_bytes as f64 / (1024.0 * 1024.0)
        );
        if parts.parts_exceeding_requested_limit > 0 {
            eprintln!(
                "  warning: {} serialized part(s) exceed the requested target; {} remaining atomic oversized tensor(s) are listed in the manifest",
                parts.parts_exceeding_requested_limit,
                parts.oversized_tensors.len()
            );
            for tensor in parts.oversized_tensors.iter().take(8) {
                eprintln!(
                    "    {}: {:.1} MiB ({})",
                    tensor.name,
                    tensor.tensor_bytes as f64 / (1024.0 * 1024.0),
                    tensor.part_path
                );
            }
        }
    }
    if !report.copied_assets.is_empty() {
        println!("  copied sidecars:");
        for path in &report.copied_assets {
            println!("    {}", path.display());
        }
    }
    if !report.unused_source_tensors.is_empty() {
        println!(
            "  unused source tensors: {}",
            report.unused_source_tensors.len()
        );
    }
    println!("  remapped tensors: {}", report.remapped.len());
    Ok(())
}
