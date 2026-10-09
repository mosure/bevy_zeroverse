#![cfg(all(feature = "import", feature = "flex", feature = "tokenizer"))]

use std::path::PathBuf;

use burn_siglip2::{LoadRequest, Siglip2Tokenizer, import_hf_dir, load_backend};
use tempfile::tempdir;

#[test]
fn gated_real_hf_import_and_text_encoding_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!("skipping real HF import smoke test; set BURN_SIGLIP2_REAL_MODEL_DIR");
        return Ok(());
    };

    let temp = tempdir()?;
    let output_base = temp.path().join("siglip2-base-patch16-256");
    let report = import_hf_dir(&hf_dir, &output_base, &Default::default())?;

    let runtime = load_backend(LoadRequest::from_bpk(report.bpk_path.clone()))?;
    let tokenizer_path = temp.path().join("siglip2-base-patch16-256.tokenizer.json");
    let tokenizer = Siglip2Tokenizer::from_file(&tokenizer_path, &runtime.model.config)?;
    let response = runtime.encode_text_strings(&tokenizer, &["a photo of a cat"], false)?;
    assert_eq!(
        response.embedding.shape().dims::<2>()[1],
        runtime.model.config.projection_dim
    );

    if let Some(parts) = report.parts {
        let runtime_from_parts =
            load_backend(LoadRequest::from_parts_manifest(parts.manifest_path, true))?;
        assert!(runtime_from_parts.load_stats.part_count >= 1);
    }

    Ok(())
}

fn real_model_dir() -> Option<PathBuf> {
    std::env::var_os("BURN_SIGLIP2_REAL_MODEL_DIR").map(PathBuf::from)
}
