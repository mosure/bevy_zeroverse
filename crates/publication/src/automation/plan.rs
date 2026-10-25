use crate::{config::Config, io, pipeline, qualification};
use anyhow::{ensure, Result};
use bevy_zeroverse_capture::provenance::{publisher_inputs, sha, source_digest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

/// Costs are explicit and bounded. Rendering quality is owned by the capture
/// contracts; automation cannot disable shadows, GI, people or annotations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Recipe {
    pub camera_audit_rooms: usize,
    pub camera_render_rooms: usize,
    pub preparation_rooms: usize,
    pub timing_rooms: usize,
    pub cpu_workers: usize,
}
impl Default for Recipe {
    fn default() -> Self {
        Self {
            camera_audit_rooms: 64,
            camera_render_rooms: 2,
            preparation_rooms: 32,
            timing_rooms: 8,
            cpu_workers: 2,
        }
    }
}
impl Recipe {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=512).contains(&self.camera_audit_rooms),
            "camera audit needs 2..512 rooms"
        );
        ensure!(
            (2..=self.camera_audit_rooms).contains(&self.camera_render_rooms),
            "rendered camera families must be a prespecified audit prefix"
        );
        ensure!(
            (2..=512).contains(&self.preparation_rooms) && (2..=128).contains(&self.timing_rooms),
            "invalid timing population"
        );
        ensure!(
            (1..=32).contains(&self.cpu_workers),
            "invalid publication CPU budget"
        );
        Ok(())
    }
}

pub fn require_current(config: &Config) -> Result<()> {
    ensure!(config.retained_evidence.is_none(), "automatic/current publication cannot retain earlier evidence; remove [retained_evidence] and run sync");
    ensure!(
        config.camera_study.is_some() && config.room_qualification.is_some(),
        "automatic publication requires camera_study and room_qualification"
    );
    let paths = [
        config.camera_study.as_deref().unwrap(),
        config.room_qualification.as_deref().unwrap(),
        Path::new(crate::fisheye::EVIDENCE),
        Path::new(super::RECEIPT),
    ];
    for (i, a) in paths.iter().enumerate() {
        ensure!(
            *a != Path::new("docs/evidence"),
            "evidence must be a child directory"
        );
        for b in &paths[i + 1..] {
            ensure!(
                !a.starts_with(b) && !b.starts_with(a),
                "overlapping automatic evidence destinations"
            );
        }
    }
    ensure!(
        config
            .capture
            .seed
            .checked_add(config.automation.camera_audit_rooms as u64)
            .is_some(),
        "camera audit seed overflow"
    );
    // The native camera study uses this exact aspect, population and density.
    ensure!(
        (config.capture.density - 0.65).abs() < 1e-6
            && (config.capture.human_density - 0.25).abs() < 1e-6,
        "camera qualification protocol requires density 0.65 and human density 0.25"
    );
    Ok(())
}

/// This key excludes generated evidence and includes *all* capture tools and
/// recipes. A new version, shader, tool, dependency or publisher invalidates it.
pub fn plan(root: &Path) -> Result<Value> {
    let config = Config::load(root)?;
    require_current(&config)?;
    let identity = pipeline::identity(root)?;
    let mut inputs = BTreeMap::new();
    for name in ["examples", "tests/support"] {
        inputs.extend(io::files(root, &root.join(name))?);
    }
    for name in ["publication.toml", "tex/arxiv.sty", "tex/references.bib"] {
        inputs.insert(name.into(), io::hash(&root.join(name))?);
    }
    for name in [
        "www/index.html",
        "assets/burn_human/fullbody_default.meta.json",
        "assets/burn_human/fullbody_default.safetensors",
    ] {
        if root.join(name).is_file() {
            inputs.insert(name.into(), io::hash(&root.join(name))?);
        }
    }
    for dir in ["www/project/static", "tex/generated"] {
        inputs.extend(io::files_where(root, &root.join(dir), &|name| {
            !pipeline::is_generated(name) && !crate::snapshot::retired(name)
        })?);
    }
    for path in &config.capacity_receipts {
        inputs.insert(
            path.to_string_lossy().into_owned(),
            io::hash(&root.join(path))?,
        );
    }
    let publisher = source_digest(&publisher_inputs(&root.join("crates/publication"))?);
    let mut result = json!({"schema_version":1, "generator_identity":identity,
        "publisher_sha256":publisher, "tool_inputs":inputs, "recipe":config.automation,
        "capture":config.capture, "stages":[
            "build_tools", "gallery_capture", "camera_audit", "camera_render", "camera_report",
            "room_qualification", "fisheye_capture_and_oracles", "pipeline_timing",
            "factor_object_door_material_atlases", "page_and_paper", "verify_current"],
        "camera_baselines":[0.,0.5,1.], "camera_counts":(1..=12).collect::<Vec<_>>(),
        "rendered_camera_counts":[2,4,8,12], "camera_image_size":[320,200],
        "camera_rendered_views":config.automation.camera_render_rooms * 3 * (2+4+8+12) * 2,
        "gallery_rendered_views":config.capture.rendered_rooms * config.capture.cameras * config.capture.playback_steps,
        "fisheye_rendered_views":36,
        "cache_policy":"exact source, tool, recipe and completed artifact hashes; never retained evidence",
        "scope":"current-only snapshot; no private before/after study inputs"});
    result["cache_key"] = json!(sha(&serde_json::to_vec(&result)?));
    Ok(result)
}

pub fn camera_recipe(config: &Config) -> qualification::Recipe {
    qualification::Recipe {
        schema_version: 1,
        seed: config.capture.seed,
        rooms_per_case: config.automation.camera_render_rooms,
        playback_steps: 2,
        human_density: 0.25,
        cases: [0., 0.5, 1.]
            .into_iter()
            .enumerate()
            .flat_map(|(b, baseline)| {
                [2, 4, 8, 12]
                    .into_iter()
                    .map(move |cameras| qualification::Case {
                        name: format!("baseline_{b}_cameras_{cameras}"),
                        cameras,
                        image_size: [320, 200],
                        camera: json!({"baseline":baseline}),
                        appearance: json!({}),
                    })
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recipe_is_bounded_and_rejects_unknown_controls() {
        Recipe::default().validate().unwrap();
        for field in [
            "camera_audit_rooms",
            "camera_render_rooms",
            "preparation_rooms",
            "timing_rooms",
            "cpu_workers",
        ] {
            let mut v = serde_json::to_value(Recipe::default()).unwrap();
            v[field] = json!(0);
            assert!(serde_json::from_value::<Recipe>(v)
                .unwrap()
                .validate()
                .is_err());
        }
        assert!(serde_json::from_value::<Recipe>(json!({"disable_shadows":true})).is_err());
    }

    fn checkout() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for dir in ["tex", "src", "examples", "crates/publication/src"] {
            std::fs::create_dir_all(root.path().join(dir)).unwrap();
        }
        for (name, content) in [
            ("Cargo.toml", "[package]\nname='example'\nversion='1.0.0'\n"),
            ("tex/arxiv.sty", "style"),
            ("tex/references.bib", "references"),
            ("examples/audit_cameras.rs", "audit"),
            ("src/lib.rs", "renderer"),
            (
                "crates/publication/Cargo.toml",
                "[package]\nname='publisher'\nversion='1.0.0'\n",
            ),
            ("crates/publication/src/lib.rs", "publisher"),
            ("crates/publication/build.rs", "build"),
            (
                "publication.toml",
                r#"schema_version=1
captures='out/publication/captures'
camera_study='docs/evidence/camera_counts'
room_qualification='docs/evidence/room_qualification'
[capture]
seed=0
audit_rooms=512
rendered_rooms=32
cameras=4
width=640
height=400
playback_steps=2
density=0.65
human_density=0.25
gi_rays=1024
[paper]
source='tex/bevy_zeroverse.tex'
"#,
            ),
        ] {
            std::fs::write(root.path().join(name), content).unwrap();
        }
        root
    }

    #[test]
    fn plan_binds_version_tools_recipe_and_renderer_but_not_generated_outputs() {
        let root = checkout();
        let original = plan(root.path()).unwrap();
        std::fs::create_dir_all(root.path().join("www/project")).unwrap();
        std::fs::write(root.path().join("www/project/index.html"), "generated").unwrap();
        assert_eq!(plan(root.path()).unwrap(), original);
        for name in [
            "src/lib.rs",
            "examples/audit_cameras.rs",
            "crates/publication/src/lib.rs",
            "tex/references.bib",
        ] {
            let path = root.path().join(name);
            let old = std::fs::read(&path).unwrap();
            std::fs::write(&path, "changed").unwrap();
            assert_ne!(
                plan(root.path()).unwrap()["cache_key"],
                original["cache_key"],
                "{name}"
            );
            std::fs::write(path, old).unwrap();
        }
        let mut config = Config::load(root.path()).unwrap();
        let recipe = camera_recipe(&config);
        assert_eq!(recipe.cases.len(), 12);
        assert!(recipe
            .cases
            .iter()
            .all(|c| c.appearance == json!({}) && c.image_size == [320, 200]));
        config.automation.camera_audit_rooms += 1;
        std::fs::write(
            root.path().join("publication.toml"),
            toml::to_string(&config).unwrap(),
        )
        .unwrap();
        assert_ne!(
            plan(root.path()).unwrap()["cache_key"],
            original["cache_key"]
        );
        std::fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname='example'\nversion='1.0.1'\n",
        )
        .unwrap();
        assert_eq!(
            plan(root.path()).unwrap()["generator_identity"]["crate_version"],
            "1.0.1"
        );
        config.room_qualification = config.camera_study.clone();
        assert!(require_current(&config).is_err());
        config.room_qualification = Some("docs/evidence/room_qualification".into());
        config.retained_evidence = Some(crate::retained::RetainedEvidence {
            release_version: "1.0.1".into(),
            release_source_sha256: "0".repeat(64),
            capture_evidence_sha256: "0".repeat(64),
            lens_evidence_sha256: None,
            current_tool_sha256: Default::default(),
            reason: "archived".into(),
        });
        assert!(require_current(&config).is_err());
    }
}
