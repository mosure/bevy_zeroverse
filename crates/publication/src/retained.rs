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
        for hash in [&self.release_source_sha256, &self.capture_evidence_sha256] {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "retained evidence requires lowercase SHA-256 pins"
            );
        }
        Ok(())
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
    ) || path.starts_with("crates/publication/src/")
        || path.starts_with("crates/publication/tests/")
}

fn frozen_inputs(inputs: &BTreeMap<String, String>) -> BTreeMap<&String, &String> {
    inputs
        .iter()
        .filter(|(path, _)| !release_input(path))
        .collect()
}

/// Domain-separated pin for the original scientific evidence. Presentation
/// outputs are deliberately excluded; all images, measurements, source archives
/// and the paper remain pinned. Struct field order and BTreeMaps are canonical.
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
                )
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
