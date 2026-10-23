//! An explicit, single-release policy for retaining captured evidence unchanged.
//!
//! This never certifies an earlier measurement as a measurement of new source.
//! Scientific inputs/outputs stay immutable; page presentation may be rebuilt.
use crate::{config::Config, pipeline::Attestation};
use anyhow::{ensure, Result};
use bevy_zeroverse_capture::{provenance::source_digest, GeneratorIdentity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedEvidence {
    pub release_version: String,
    pub release_source_sha256: String,
    pub capture_evidence_sha256: String,
    /// Optional immutable pin for the separately attested lens inputs AND shipped
    /// gallery/papers. Without this pin, lens evidence must match current source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens_evidence_sha256: Option<String>,
    /// Explicit pins for updated capture tools. Retained measurements keep the
    /// original tool hashes in their attestation; these tools are not rerun.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub current_tool_sha256: BTreeMap<String, String>,
    pub reason: String,
}

impl RetainedEvidence {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.release_version.split('.').count() == 3
                && self
                    .release_version
                    .split('.')
                    .all(|s| s.parse::<u64>().is_ok())
                && !self.reason.trim().is_empty(),
            "retained evidence requires an explicit release version and reason"
        );
        for hash in [&self.release_source_sha256, &self.capture_evidence_sha256]
            .into_iter()
            .chain(self.lens_evidence_sha256.iter())
            .chain(self.current_tool_sha256.values())
        {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "retained evidence requires lowercase SHA-256 pins"
            );
        }
        for path in self.current_tool_sha256.keys() {
            ensure!(
                crate::objects::TOOLS.contains(&path.as_str())
                    || crate::materials::TOOLS.contains(&path.as_str())
                    || path == crate::doors::TOOL,
                "retained tool pin must identify a registered capture tool"
            );
        }
        Ok(())
    }

    pub(crate) fn tool_hash(
        &self,
        path: &str,
        current: String,
        original: Option<&String>,
    ) -> Result<String> {
        let Some(pin) = self.current_tool_sha256.get(path) else {
            return Ok(current);
        };
        self.validate()?;
        ensure!(&current == pin, "retained capture tool changed: {path}");
        original
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing original capture tool: {path}"))
    }

    pub(crate) fn verify_lens_pin(
        &self,
        root: &std::path::Path,
        current: &GeneratorIdentity,
    ) -> Result<bool> {
        let Some(pin) = &self.lens_evidence_sha256 else {
            return Ok(false);
        };
        self.validate()?;
        ensure!(
            current.crate_version == self.release_version
                && current.source_sha256 == self.release_source_sha256,
            "retained lens release/source pin changed"
        );
        ensure!(
            &lens_evidence_digest(root)? == pin,
            "retained lens evidence changed"
        );
        Ok(true)
    }

    pub(crate) fn verify(
        &self,
        current: &GeneratorIdentity,
        attestation: &Attestation,
        inputs: &BTreeMap<String, String>,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            current.crate_version == self.release_version
                && current.source_sha256 == self.release_source_sha256,
            "retained evidence release/source pin changed; explicitly renew the policy or refresh captures"
        );
        ensure!(
            capture_evidence_digest(attestation)? == self.capture_evidence_sha256,
            "retained scientific evidence changed"
        );
        ensure!(
            source_digest(&attestation.renderer_inputs)
                == attestation.generator_identity.source_sha256,
            "retained capture identity does not match its original renderer inputs"
        );
        ensure!(
            frozen_inputs(&attestation.publication_inputs) == frozen_inputs(inputs),
            "retained scientific inputs changed; study tools, paper and evidence must stay identical"
        );
        Ok(())
    }
}

/// Bind the complete original lens evidence and shipped gallery, including all
/// PDFs and receipts. Any changed, added or removed file invalidates retention.
pub fn lens_evidence_digest(root: &std::path::Path) -> Result<String> {
    let mut files = BTreeMap::new();
    for directory in [crate::fisheye::EVIDENCE, crate::fisheye::DIRECTORY] {
        ensure!(
            root.join(directory).join("receipt.json").is_file(),
            "missing retained lens receipt"
        );
        files.extend(crate::io::files(root, &root.join(directory))?);
    }
    Ok(bevy_zeroverse_capture::provenance::sha(
        &serde_json::to_vec(&("bevy_zeroverse.retained_lens_evidence.v1", files))?,
    ))
}

pub(crate) fn retains_lens(root: &std::path::Path, current: &GeneratorIdentity) -> Result<bool> {
    if !root.join("publication.toml").is_file() {
        return Ok(false);
    }
    Config::load(root)?
        .retained_evidence
        .as_ref()
        .map_or(Ok(false), |policy| policy.verify_lens_pin(root, current))
}

fn release_input(path: &str) -> bool {
    matches!(
        path,
        "publication.toml"
            | "crates/publication/Cargo.toml"
            | "crates/publication/build.rs"
            | "crates/publication/README.md"
            | "crates/publication/templates/project.html"
            | "crates/publication/templates/objects.html"
            | "www/project/static/js/objects.js"
            | "www/project/static/css/objects.css"
    ) || path == "crates/publication/templates/fisheye.html"
        || path.starts_with("docs/evidence/fisheye/")
        || path.starts_with("crates/publication/src/")
        || path.starts_with("crates/publication/tests/")
}

fn frozen_inputs(inputs: &BTreeMap<String, String>) -> BTreeMap<&String, &String> {
    inputs
        .iter()
        .filter(|(path, _)| !release_input(path))
        .collect()
}

/// Domain-separated pin for the original scientific evidence. Presentation
/// outputs and the independently attested lens chapter are excluded; original
/// images, measurements, source archives and paper remain pinned. The top-level
/// attestation binds the added chapter normally. Struct order/maps are canonical.
pub fn capture_evidence_digest(attestation: &Attestation) -> Result<String> {
    #[derive(Serialize)]
    struct Evidence<'a> {
        domain: &'static str,
        schema_version: u32,
        generator_identity: &'a GeneratorIdentity,
        protocol: &'a crate::config::Protocol,
        run_id: &'a str,
        capture_engine: &'a str,
        renderer_inputs: &'a BTreeMap<String, String>,
        publication_inputs: BTreeMap<&'a String, &'a String>,
        capture_sources: &'a BTreeMap<String, String>,
        artifacts: BTreeMap<&'a String, &'a String>,
        reference_studies: &'a [serde_json::Value],
    }
    let bytes = serde_json::to_vec(&Evidence {
        domain: "bevy_zeroverse.retained_capture_evidence.v1",
        schema_version: attestation.schema_version,
        generator_identity: &attestation.generator_identity,
        protocol: &attestation.protocol,
        run_id: &attestation.run_id,
        capture_engine: &attestation.capture_engine,
        renderer_inputs: &attestation.renderer_inputs,
        publication_inputs: frozen_inputs(&attestation.publication_inputs),
        capture_sources: &attestation.capture_sources,
        artifacts: attestation
            .artifacts
            .iter()
            .filter(|(name, _)| {
                !matches!(
                    name.as_str(),
                    "www/project/index.html" | "www/project/reference.html"
                ) && !name.starts_with("www/project/fisheye/")
            })
            .collect(),
        reference_studies: &attestation.reference_studies,
    })?;
    Ok(bevy_zeroverse_capture::provenance::sha(&bytes))
}

pub(crate) fn require_refresh_allowed(config: &Config) -> Result<()> {
    ensure!(
        config.retained_evidence.is_none(),
        "refresh is disabled by retained_evidence; explicitly remove the policy before capturing new evidence"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Protocol;

    fn fixture() -> (RetainedEvidence, GeneratorIdentity, Attestation) {
        let renderer_inputs = BTreeMap::from([("src/lib.rs".into(), "c".repeat(64))]);
        let identity = GeneratorIdentity {
            schema_version: 1,
            crate_version: "1.1.0".into(),
            generator_version: 1,
            source_sha256: "a".repeat(64),
        };
        let mut policy = RetainedEvidence {
            release_version: identity.crate_version.clone(),
            release_source_sha256: identity.source_sha256.clone(),
            capture_evidence_sha256: "b".repeat(64),
            lens_evidence_sha256: None,
            current_tool_sha256: BTreeMap::new(),
            reason: "No GPU work in this release".into(),
        };
        let attestation = Attestation {
            schema_version: 1,
            generator_identity: GeneratorIdentity {
                crate_version: "1.0.0".into(),
                source_sha256: source_digest(&renderer_inputs),
                ..identity.clone()
            },
            protocol: Protocol {
                seed: 0,
                audit_rooms: 4,
                rendered_rooms: 4,
                cameras: 4,
                width: 64,
                height: 64,
                playback_steps: 2,
                density: 0.5,
                human_density: 0.25,
                gi_rays: 1024,
                ssaa_factor: 0,
            },
            run_id: "original".into(),
            capture_engine: "original".into(),
            renderer_inputs,
            publication_inputs: BTreeMap::from([
                ("publication.toml".into(), "old-policy".into()),
                (
                    "crates/publication/src/pipeline.rs".into(),
                    "old-code".into(),
                ),
                (
                    "crates/publication/templates/page.html".into(),
                    "original".into(),
                ),
                ("docs/evidence/cohort.json".into(), "original".into()),
            ]),
            capture_sources: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            reference_studies: vec![],
        };
        policy.capture_evidence_sha256 = capture_evidence_digest(&attestation).unwrap();
        (policy, identity, attestation)
    }

    #[test]
    fn updated_tools_require_explicit_pins_and_preserve_original_evidence() {
        let (mut policy, identity, mut attestation) = fixture();
        let path = "examples/review_objects.rs";
        let original = "1".repeat(64);
        let current = "2".repeat(64);
        attestation
            .publication_inputs
            .insert(path.into(), original.clone());
        policy.capture_evidence_sha256 = capture_evidence_digest(&attestation).unwrap();
        policy
            .current_tool_sha256
            .insert(path.into(), current.clone());
        let mut inputs = attestation.publication_inputs.clone();
        inputs.insert(
            path.into(),
            policy
                .tool_hash(path, current.clone(), Some(&original))
                .unwrap(),
        );
        policy.verify(&identity, &attestation, &inputs).unwrap();
        assert_eq!(inputs[path], original);
        assert!(policy
            .tool_hash(path, "3".repeat(64), Some(&original))
            .is_err());
        assert!(policy.tool_hash(path, current.clone(), None).is_err());
        inputs.insert(path.into(), current);
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
        policy
            .current_tool_sha256
            .insert("tex/bevy_zeroverse.tex".into(), "4".repeat(64));
        assert!(policy.validate().is_err());
    }

    #[test]
    fn lens_retention_requires_explicit_source_pin_and_identical_inputs_and_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let (mut policy, identity, _) = fixture();
        assert!(!policy.verify_lens_pin(root, &identity).unwrap());
        for directory in [crate::fisheye::EVIDENCE, crate::fisheye::DIRECTORY] {
            std::fs::create_dir_all(root.join(directory)).unwrap();
            std::fs::write(
                root.join(directory).join("receipt.json"),
                b"original identity",
            )
            .unwrap();
            std::fs::write(root.join(directory).join("image.png"), b"original pixels").unwrap();
        }
        policy.lens_evidence_sha256 = Some(lens_evidence_digest(root).unwrap());
        assert!(policy.verify_lens_pin(root, &identity).unwrap());
        let mut wrong = identity.clone();
        wrong.crate_version = "2.0.0".into();
        assert!(policy.verify_lens_pin(root, &wrong).is_err());
        wrong = identity.clone();
        wrong.source_sha256 = "e".repeat(64);
        assert!(policy.verify_lens_pin(root, &wrong).is_err());
        for directory in [crate::fisheye::EVIDENCE, crate::fisheye::DIRECTORY] {
            let path = root.join(directory).join("image.png");
            std::fs::write(&path, b"different pixels").unwrap();
            assert!(policy.verify_lens_pin(root, &identity).is_err());
            std::fs::remove_file(&path).unwrap();
            assert!(policy.verify_lens_pin(root, &identity).is_err());
            std::fs::write(&path, b"original pixels").unwrap();
            let extra = root.join(directory).join("unexpected.png");
            std::fs::write(&extra, b"extra").unwrap();
            assert!(policy.verify_lens_pin(root, &identity).is_err());
            std::fs::remove_file(extra).unwrap();
        }
        assert!(policy.verify_lens_pin(root, &identity).unwrap());
    }

    #[test]
    fn permits_release_code_changes_without_relabeling_evidence() {
        let (policy, identity, attestation) = fixture();
        let mut inputs = attestation.publication_inputs.clone();
        inputs.insert("publication.toml".into(), "pinned-policy".into());
        inputs.insert(
            "crates/publication/src/retained.rs".into(),
            "new-code".into(),
        );
        inputs.remove("crates/publication/src/pipeline.rs");
        policy.verify(&identity, &attestation, &inputs).unwrap();
        assert_eq!(attestation.generator_identity.crate_version, "1.0.0");
    }

    #[test]
    fn lens_chapter_preserves_the_original_evidence_pin() {
        let (policy, identity, mut attestation) = fixture();
        attestation.artifacts.insert(
            "www/project/fisheye/whitepaper.pdf".into(),
            "separately-qualified-paper".into(),
        );
        attestation.publication_inputs.insert(
            "docs/evidence/fisheye/receipt.json".into(),
            "current-source-receipt".into(),
        );
        policy
            .verify(&identity, &attestation, &attestation.publication_inputs)
            .unwrap();
        attestation.artifacts.insert(
            "www/project/static/papers/bevy_zeroverse.pdf".into(),
            "changed-original-paper".into(),
        );
        assert!(policy
            .verify(&identity, &attestation, &attestation.publication_inputs)
            .is_err());
    }

    #[test]
    fn rejects_different_release_source_or_attestation() {
        let (policy, mut identity, mut attestation) = fixture();
        let inputs = attestation.publication_inputs.clone();
        let mut corrupt = policy.clone();
        corrupt.capture_evidence_sha256 = "d".repeat(64);
        assert!(corrupt.verify(&identity, &attestation, &inputs).is_err());
        identity.crate_version = "1.1.1".into();
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
        identity.crate_version = policy.release_version.clone();
        identity.source_sha256 = "e".repeat(64);
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
        identity.source_sha256 = policy.release_source_sha256.clone();
        attestation.renderer_inputs.clear();
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
    }

    #[test]
    fn rejects_modified_removed_or_added_scientific_inputs() {
        let (policy, identity, attestation) = fixture();
        for path in [
            "crates/publication/templates/page.html",
            "docs/evidence/cohort.json",
            "crates/publication/fonts/new.ttf",
            "tex/references.bib",
            "examples/review_objects.rs",
        ] {
            let mut inputs = attestation.publication_inputs.clone();
            inputs.insert(path.into(), "changed".into());
            assert!(
                policy.verify(&identity, &attestation, &inputs).is_err(),
                "{path}"
            );
        }
        let mut inputs = attestation.publication_inputs.clone();
        inputs.remove("docs/evidence/cohort.json");
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
    }

    #[test]
    fn page_rebuild_preserves_capture_and_paper_pins() {
        let (policy, identity, mut attestation) = fixture();
        attestation.publication_inputs.insert(
            "crates/publication/templates/objects.html".into(),
            "new-page-template".into(),
        );
        attestation.publication_inputs.insert(
            "www/project/static/js/objects.js".into(),
            "new-page-controls".into(),
        );
        attestation
            .artifacts
            .insert("www/project/index.html".into(), "new-page".into());
        let inputs = attestation.publication_inputs.clone();
        policy.verify(&identity, &attestation, &inputs).unwrap();
        for protected in [
            "www/project/static/media/doors/atlas.json",
            "www/project/static/papers/bevy_zeroverse.pdf",
            "tex/bevy_zeroverse.tex",
        ] {
            attestation
                .artifacts
                .insert(protected.into(), "altered".into());
            assert!(
                policy.verify(&identity, &attestation, &inputs).is_err(),
                "{protected}"
            );
            attestation.artifacts.remove(protected);
        }
        attestation.generator_identity.crate_version = identity.crate_version.clone();
        assert!(policy.verify(&identity, &attestation, &inputs).is_err());
    }

    #[test]
    fn refresh_fails_before_work_or_external_commands() {
        let (policy, _, attestation) = fixture();
        let root = tempfile::tempdir().unwrap();
        let config = Config {
            schema_version: 1,
            captures: "out/publication/captures".into(),
            capture: attestation.protocol,
            paper: crate::config::Paper {
                source: "tex/bevy_zeroverse.tex".into(),
            },
            references: vec![],
            camera_study: None,
            room_qualification: None,
            capacity_receipts: vec![],
            retained_evidence: Some(policy),
        };
        std::fs::write(
            root.path().join("publication.toml"),
            toml::to_string(&config).unwrap(),
        )
        .unwrap();
        for recapture in [false, true] {
            let error = crate::pipeline::refresh(root.path(), recapture).unwrap_err();
            assert!(error.to_string().contains("refresh is disabled"));
            assert!(!root.path().join("out").exists());
        }
    }
}
