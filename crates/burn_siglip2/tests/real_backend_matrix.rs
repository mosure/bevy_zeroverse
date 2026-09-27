#![cfg(all(
    feature = "ndarray",
    feature = "flex",
    feature = "wgpu",
    feature = "pipeline"
))]

use std::{fs, path::Path, path::PathBuf};

use burn::tensor::{Tensor, backend::Backend};
use burn_siglip2::{
    DefaultFlexBackend, DefaultNdArrayBackend, DefaultWgpuBackend, Siglip2Runtime,
    Siglip2Tokenizer, load_model_from_parts_manifest_path,
};

const ROOT_ENV: &str = "BURN_SIGLIP2_BACKEND_MATRIX_ROOT";
const IMAGE_ENV: &str = "BURN_SIGLIP2_BACKEND_MATRIX_IMAGE";
const VARIANTS_ENV: &str = "BURN_SIGLIP2_BACKEND_MATRIX_VARIANTS";
const TEXTS: [&str; 3] = [
    "this is a photo of a cat.",
    "this is a photo of a dog.",
    "this is a photo of an airplane.",
];

#[derive(Debug)]
struct Outputs {
    image_shape: [usize; 2],
    text_shape: [usize; 2],
    logits_shape: [usize; 2],
    raw_image: Vec<f32>,
    raw_text: Vec<f32>,
    normalized_image: Vec<f32>,
    normalized_text: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct MatrixTolerance {
    raw_embedding: f32,
    normalized_embedding: f32,
    logits: f32,
    probabilities: f32,
}

const FLEX_TOLERANCE: MatrixTolerance = MatrixTolerance {
    raw_embedding: 1.0e-3,
    normalized_embedding: 1.0e-4,
    logits: 5.0e-3,
    probabilities: 1.0e-6,
};
const WGPU_TOLERANCE: MatrixTolerance = MatrixTolerance {
    raw_embedding: 1.0e-3,
    normalized_embedding: 1.0e-4,
    logits: 5.0e-3,
    probabilities: 1.0e-6,
};

#[test]
fn opt_in_all_model_sizes_match_across_ndarray_flex_and_wgpu() -> Result<(), String> {
    let Some(root) = std::env::var_os(ROOT_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping real backend matrix; set {ROOT_ENV} and {IMAGE_ENV} to run NdArray/Flex/WGPU parity"
        );
        return Ok(());
    };
    let image_path = std::env::var_os(IMAGE_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{IMAGE_ENV} is required when {ROOT_ENV} is set"))?;
    let image = fs::read(&image_path)
        .map_err(|err| format!("failed to read image '{}': {err}", image_path.display()))?;
    let variants = selected_variants()?;

    for variant in variants {
        let variant_dir = root.join(variant);
        let manifest = variant_dir.join(format!("siglip2-{variant}.bpk.parts.json"));
        let tokenizer = variant_dir.join(format!("siglip2-{variant}.tokenizer.json"));
        require_file(&manifest)?;
        require_file(&tokenizer)?;

        eprintln!("backend matrix: loading {variant} with NdArray");
        let ndarray = run_backend::<DefaultNdArrayBackend>(
            Default::default(),
            &manifest,
            &tokenizer,
            &image,
        )?;
        eprintln!(
            "backend matrix NdArray oracle: variant={variant}, logits={:?}, probabilities={:?}",
            ndarray.logits, ndarray.probabilities
        );

        eprintln!("backend matrix: loading {variant} with Flex CPU");
        let flex =
            run_backend::<DefaultFlexBackend>(Default::default(), &manifest, &tokenizer, &image)?;
        compare_outputs(variant, "Flex", &flex, &ndarray, FLEX_TOLERANCE)?;

        eprintln!("backend matrix: loading {variant} with native WGPU");
        let wgpu =
            run_backend::<DefaultWgpuBackend>(Default::default(), &manifest, &tokenizer, &image)?;
        compare_outputs(variant, "WGPU", &wgpu, &ndarray, WGPU_TOLERANCE)?;
    }
    Ok(())
}

fn selected_variants() -> Result<Vec<&'static str>, String> {
    let requested = std::env::var(VARIANTS_ENV)
        .unwrap_or_else(|_| "base-patch16-224,large-patch16-256,so400m-patch14-224".to_string());
    requested
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| match value.trim() {
            "base" | "base-patch16-224" => Ok("base-patch16-224"),
            "large" | "large-patch16-256" => Ok("large-patch16-256"),
            "so400m" | "so400m-patch14-224" => Ok("so400m-patch14-224"),
            other => Err(format!(
                "unsupported {VARIANTS_ENV} entry '{other}'; expected base, large, or so400m"
            )),
        })
        .collect()
}

fn require_file(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!(
            "required backend-matrix file is missing: {}",
            path.display()
        ));
    }
    Ok(())
}

fn run_backend<B: Backend>(
    device: B::Device,
    manifest: &Path,
    tokenizer_path: &Path,
    image: &[u8],
) -> Result<Outputs, String> {
    let (model, load_stats) = load_model_from_parts_manifest_path::<B>(&device, manifest, true)?;
    let tokenizer = Siglip2Tokenizer::from_file(tokenizer_path, &model.config)?;
    let runtime = Siglip2Runtime {
        model,
        device,
        load_stats,
    };
    let response = runtime.encode_image_bytes_and_text_strings(image, &tokenizer, &TEXTS, false)?;
    let image_shape = response.image_embedding.shape().dims::<2>();
    let text_shape = response.text_embedding.shape().dims::<2>();
    let logits_shape = response.logits_per_image.shape().dims::<2>();
    let outputs = Outputs {
        image_shape,
        text_shape,
        logits_shape,
        raw_image: tensor_values(response.image_embedding)?,
        raw_text: tensor_values(response.text_embedding)?,
        normalized_image: tensor_values(response.normalized_image_embedding)?,
        normalized_text: tensor_values(response.normalized_text_embedding)?,
        logits: tensor_values(response.logits_per_image)?,
        probabilities: tensor_values(response.probabilities_per_image)?,
    };
    validate_outputs(&outputs)?;
    Ok(outputs)
}

fn tensor_values<B: Backend>(tensor: Tensor<B, 2>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read backend tensor: {err:?}"))
}

fn validate_outputs(output: &Outputs) -> Result<(), String> {
    let [images, projection] = output.image_shape;
    let [texts, text_projection] = output.text_shape;
    if images != 1 || texts != TEXTS.len() || projection == 0 || projection != text_projection {
        return Err(format!(
            "unexpected dual-tower shapes: image={:?}, text={:?}",
            output.image_shape, output.text_shape
        ));
    }
    if output.logits_shape != [images, texts]
        || output.raw_image.len() != images * projection
        || output.raw_text.len() != texts * projection
        || output.normalized_image.len() != images * projection
        || output.normalized_text.len() != texts * projection
        || output.logits.len() != images * texts
        || output.probabilities.len() != images * texts
    {
        return Err("backend output shape/value counts are inconsistent".to_string());
    }
    if output
        .raw_image
        .iter()
        .chain(&output.raw_text)
        .chain(&output.normalized_image)
        .chain(&output.normalized_text)
        .chain(&output.logits)
        .chain(&output.probabilities)
        .any(|value| !value.is_finite())
    {
        return Err("backend produced a non-finite output".to_string());
    }
    for (index, (&logit, &probability)) in
        output.logits.iter().zip(&output.probabilities).enumerate()
    {
        let expected = 1.0 / (1.0 + (-logit).exp());
        if !(0.0..=1.0).contains(&probability) || (probability - expected).abs() > 1.0e-6 {
            return Err(format!(
                "invalid probability {index}: probability={probability}, sigmoid(logit)={expected}"
            ));
        }
    }
    Ok(())
}

fn compare_outputs(
    variant: &str,
    backend: &str,
    actual: &Outputs,
    expected: &Outputs,
    tolerance: MatrixTolerance,
) -> Result<(), String> {
    if actual.image_shape != expected.image_shape
        || actual.text_shape != expected.text_shape
        || actual.logits_shape != expected.logits_shape
    {
        return Err(format!(
            "{variant} {backend}/NdArray shape mismatch: actual image={:?} text={:?} logits={:?}; expected image={:?} text={:?} logits={:?}",
            actual.image_shape,
            actual.text_shape,
            actual.logits_shape,
            expected.image_shape,
            expected.text_shape,
            expected.logits_shape
        ));
    }
    let raw_image = max_abs(&actual.raw_image, &expected.raw_image)?;
    let raw_text = max_abs(&actual.raw_text, &expected.raw_text)?;
    let normalized_image = max_abs(&actual.normalized_image, &expected.normalized_image)?;
    let normalized_text = max_abs(&actual.normalized_text, &expected.normalized_text)?;
    let logits = max_abs(&actual.logits, &expected.logits)?;
    let probabilities = max_abs(&actual.probabilities, &expected.probabilities)?;
    if raw_image > tolerance.raw_embedding
        || raw_text > tolerance.raw_embedding
        || normalized_image > tolerance.normalized_embedding
        || normalized_text > tolerance.normalized_embedding
        || logits > tolerance.logits
        || probabilities > tolerance.probabilities
    {
        return Err(format!(
            "{variant} {backend}/NdArray mismatch: raw_image={raw_image:.6e}, raw_text={raw_text:.6e}, normalized_image={normalized_image:.6e}, normalized_text={normalized_text:.6e}, logits={logits:.6e}, probabilities={probabilities:.6e}; tolerance={tolerance:?}"
        ));
    }
    eprintln!(
        "backend matrix passed: variant={variant}, backend={backend}, raw_image={raw_image:.6e}, raw_text={raw_text:.6e}, normalized_image={normalized_image:.6e}, normalized_text={normalized_text:.6e}, logits={logits:.6e}, probabilities={probabilities:.6e}"
    );
    Ok(())
}

fn max_abs(actual: &[f32], expected: &[f32]) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "backend matrix length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max))
}
