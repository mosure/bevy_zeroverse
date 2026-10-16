//! Optional current-source capacity evidence in the canonical page/paper build.
//! No old receipt is relabeled, and absence removes any previously staged data.
use super::*;
use crate::io;
use bevy_zeroverse_capture::GeneratorIdentity;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, path::PathBuf};

pub const DIRECTORY: &str = "www/project/static/media/capacity";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema_version: u32,
    generator: GeneratorIdentity,
    receipt_sha256: Vec<String>,
}

pub fn read_current(path: &Path, current: &GeneratorIdentity) -> Result<CapacityReceipt> {
    ensure!(
        fs::metadata(path)?.len() <= 32 << 20,
        "oversized capacity receipt"
    );
    let receipt: CapacityReceipt = io::read(path)?;
    validate_receipt(&receipt.plan, &receipt).map_err(anyhow::Error::msg)?;
    ensure!(
        &receipt.plan.population.generator == current,
        "stale capacity source; requalify instead of relabeling"
    );
    Ok(receipt)
}

pub fn build(
    root: &Path,
    stage: &Path,
    inputs: &[PathBuf],
    current: &GeneratorIdentity,
) -> Result<()> {
    ensure!(
        inputs.len() <= 4,
        "at most four explicitly labeled capacity populations per publication"
    );
    // Check every source before changing any staged artifact.
    let receipts = inputs
        .iter()
        .map(|path| read_current(&root.join(path), current))
        .collect::<Result<Vec<_>>>()?;
    let output = stage.join(DIRECTORY);
    if output.exists() {
        fs::remove_dir_all(&output)?;
    }
    let generated = stage.join("tex/generated");
    fs::create_dir_all(&generated)?;
    for i in 0..4 {
        let old = generated.join(format!("capacity_observed_{i}.png"));
        if old.exists() {
            fs::remove_file(old)?;
        }
    }
    let mut tex = String::new();
    if !receipts.is_empty() {
        tex.push_str("\\clearpage\n\\subsection{Configuration-specific capacity evidence}\nThese are separate source-bound populations, not pooled estimates. All failed draws are retained. Fixed-bin distinct counts measure the stated representation; they are not a calibrated perceptual population size. No unseen-tail forecast or learning law is qualified.\n\n");
        fs::create_dir_all(&output)?;
    }
    let mut hashes = Vec::new();
    for (i, (path, receipt)) in inputs.iter().zip(&receipts).enumerate() {
        let directory = output.join(i.to_string());
        report::write(&root.join(path), &directory)?;
        hashes.push(io::hash(&directory.join("receipt.json"))?);
        let c = &receipt.plan.population.capture;
        let policy = match receipt.plan.request.seed_policy {
            SeedPolicy::UniformWithReplacement { audit_seed } => {
                format!("uniform with replacement, audit seed {audit_seed}")
            }
            SeedPolicy::Consecutive { first } => {
                format!("consecutive, first seed {first}; observed statistics only")
            }
            SeedPolicy::Fixed { seed } => format!("fixed seed {seed}; observed statistics only"),
        };
        writeln!(tex,"\\paragraph{{Population {}}} Unit {:?}; capture plan: {} cameras at ${}\\times{}$, {} time states, modes \\texttt{{{}}}. Seed policy: {policy}.\n",i+1,receipt.plan.request.unit,c.cameras,c.image_size[0],c.image_size[1],c.progress.len(),c.modes.join(", ").replace('_',"\\_"))?;
        let table = fs::read_to_string(directory.join("capacity.tex"))?.replace(
            "\\label{tab:capacity-observed}",
            &format!("\\label{{tab:capacity-observed-{i}}}"),
        );
        tex.push_str(&table);
        fs::copy(
            directory.join("capacity.png"),
            generated.join(format!("capacity_observed_{i}.png")),
        )?;
        writeln!(tex,"\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\linewidth]{{generated/capacity_observed_{i}.png}}\\caption{{Observed-cohort rarefaction and coverage for population {}. This curve does not extrapolate beyond its observed rooms.}}\\end{{figure}}",i+1)?;
        tex.push_str("\\clearpage\n");
    }
    fs::write(generated.join("capacity_evidence.tex"), tex)?;
    if !receipts.is_empty() {
        fs::write(
            output.join("index.json"),
            serde_json::to_vec_pretty(&Bundle {
                schema_version: 1,
                generator: current.clone(),
                receipt_sha256: hashes,
            })?,
        )?;
    }
    Ok(())
}

pub fn verify(root: &Path, expected_count: usize, current: &GeneratorIdentity) -> Result<()> {
    let path = root.join(DIRECTORY);
    if expected_count == 0 {
        ensure!(!path.exists(), "unconfigured capacity data in publication");
        return Ok(());
    }
    let bundle: Bundle = io::read(&path.join("index.json"))?;
    ensure!(
        bundle.schema_version == 1
            && bundle.generator == *current
            && bundle.receipt_sha256.len() == expected_count,
        "capacity publication identity/count differs"
    );
    for (i, hash) in bundle.receipt_sha256.iter().enumerate() {
        let receipt = path.join(i.to_string()).join("receipt.json");
        ensure!(io::hash(&receipt)? == *hash, "capacity receipt changed");
        read_current(&receipt, current)?;
    }
    Ok(())
}

pub fn page(root: &Path) -> Result<String> {
    let path = root.join(DIRECTORY);
    if !path.exists() {
        return Ok(String::new());
    }
    let bundle: Bundle = io::read(&path.join("index.json"))?;
    verify(root, bundle.receipt_sha256.len(), &bundle.generator)?;
    let mut html=String::from("<section id=\"capacity\" class=\"section\"><p class=\"eyebrow\">CONFIGURATION-SPECIFIC EVIDENCE</p><h2>Observed diversity</h2><p>Each population keeps its own configuration, sample unit and representation. Observed classes are not a prediction of ten-million-sample learning utility.</p><div class=\"card-grid\">");
    for i in 0..bundle.receipt_sha256.len() {
        let receipt = read_current(
            &path.join(i.to_string()).join("receipt.json"),
            &bundle.generator,
        )?;
        let e = receipt.estimate;
        write!(html,"<article class=\"card\"><h3>Population {}</h3><p>{} accepted / {} draws; {} rendered rooms; {} retained failures.</p><a href=\"static/media/capacity/{i}/index.html\">Inspect definitions, coverage and bounds</a></article>",i+1,e.accepted_rooms,e.requested_rooms,e.rendered_rooms,e.failed_rooms)?;
    }
    html.push_str("</div></section>");
    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_source_publication_rejects_stale_inputs_and_clears_removed_evidence() {
        let (plan, records) = super::super::tests::fixture();
        let receipt = qualify(&plan, &records, sha(b"rows"), 0.).unwrap();
        let root = tempfile::tempdir().unwrap();
        let stage = tempfile::tempdir().unwrap();
        let input = PathBuf::from("receipt.json");
        fs::write(
            root.path().join(&input),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        let current = &plan.population.generator;
        build(
            root.path(),
            stage.path(),
            std::slice::from_ref(&input),
            current,
        )
        .unwrap();
        verify(stage.path(), 1, current).unwrap();
        assert!(page(stage.path()).unwrap().contains("Observed diversity"));
        let mut other = current.clone();
        other.source_sha256 = sha(b"new source");
        assert!(build(root.path(), stage.path(), &[input], &other).is_err());
        verify(stage.path(), 1, current).unwrap();
        build(root.path(), stage.path(), &[], current).unwrap();
        verify(stage.path(), 0, current).unwrap();
        assert!(!stage
            .path()
            .join("tex/generated/capacity_observed_0.png")
            .exists());
        assert!(
            fs::read_to_string(stage.path().join("tex/generated/capacity_evidence.tex"))
                .unwrap()
                .is_empty()
        );
    }
}
