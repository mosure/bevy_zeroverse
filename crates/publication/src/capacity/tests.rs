use super::*;

pub(super) fn fixture() -> (CapacityPlan, Vec<CapacityRecord>) {
    let request = CapacityRequest {
        budget: AuditBudget {
            manifests: 16,
            rendered_rooms: 0,
            mesh_checks: 0,
        },
        ..Default::default()
    };
    let plan = CapacityPlan {
        schema_version: SCHEMA_VERSION,
        population: ResolvedPopulation {
            generator: bevy_zeroverse_capture::GeneratorIdentity {
                schema_version: 1,
                crate_version: "test".into(),
                generator_version: 50,
                source_sha256: sha(b"test source"),
            },
            capture_engine: "test-engine".into(),
            platform: "test".into(),
            config_sha256: sha(b"{}"),
            resolved_config_json: "{}".into(),
            capture: CapturePlan {
                image_size: [16, 16],
                cameras: 4,
                progress: vec![0.],
                seconds: None,
                modes: vec!["color".into()],
                annotation_glass: "surface".into(),
            },
            failure_policy: "retain-error-no-resampling-v1".into(),
        },
        request,
        factors: vec![],
    };
    let records = (0..16)
        .map(|draw| CapacityRecord {
            draw,
            seed: seed_at(&plan.request.seed_policy, draw),
            mesh_checked: false,
            outcome: SampleOutcome::Accepted {
                family_sha256: sha(format!("family {}", draw / 2).as_bytes()),
                descriptors: BTreeMap::from([(
                    plan.request.definitions[0].id.clone(),
                    Descriptor {
                        class_sha256: sha(format!("class {}", draw / 4).as_bytes()),
                        blocks: BTreeMap::from([(
                            "geometry".into(),
                            sha(format!("class {}", draw / 4).as_bytes()),
                        )]),
                    },
                )]),
                diagnostics: if draw < 4 {
                    BTreeMap::from([("objects/Chair".into(), 2.)])
                } else {
                    BTreeMap::new()
                },
                delivered_views: 0,
                delivered_progress: vec![],
                capture_sha256: None,
            },
        })
        .collect();
    (plan, records)
}
#[test]
fn counts_failures_families_and_absent_categories_without_multiplying_views() {
    let (plan, mut records) = fixture();
    records[15].outcome = SampleOutcome::Failed {
        stage: "layout".into(),
        reason: "required function rejected".into(),
    };
    let receipt = qualify(&plan, &records, sha(b"rows"), 0.2).unwrap();
    let e = &receipt.estimate;
    assert_eq!(
        (e.accepted_rooms, e.failed_rooms, e.unique_families),
        (15, 1, 8)
    );
    assert_eq!(
        (e.definitions[0].rooms, e.definitions[0].observed_classes),
        (15, 4)
    );
    assert_eq!(e.diagnostics["objects/Chair"].rooms, 15);
    assert_eq!(e.diagnostics["objects/Chair"].min, 0.);
    assert_eq!(e.definitions[0].block_distinct_counts["geometry"], 4);
    let wire = serde_json::to_vec(&receipt).unwrap();
    let decoded: CapacityReceipt = serde_json::from_slice(&wire).unwrap();
    assert_eq!(estimate(&plan, &decoded).unwrap(), *e);
    let mut changed = plan.clone();
    changed.population.capture.image_size = [32, 32];
    assert!(estimate(&changed, &receipt).is_err());
}
#[test]
fn rejects_incomplete_duplicate_reordered_stale_and_nonfinite_evidence() {
    let (plan, records) = fixture();
    let q = |r: &[CapacityRecord]| qualify(&plan, r, sha(b"rows"), 0.);
    assert!(q(&records[..15]).is_err());
    let mut r = records.clone();
    r.swap(0, 1);
    assert!(q(&r).is_err());
    let mut r = records.clone();
    r[1] = r[0].clone();
    assert!(q(&r).is_err());
    let mut r = records.clone();
    r[0].seed ^= 1;
    assert!(q(&r).is_err());
    let mut r = records.clone();
    if let SampleOutcome::Accepted { diagnostics, .. } = &mut r[0].outcome {
        diagnostics.insert("invalid".into(), f64::NAN);
    }
    assert!(q(&r).is_err());
    let mut stale = plan.clone();
    stale.population.config_sha256 = sha(b"not config");
    assert!(qualify(&stale, &records, sha(b"rows"), 0.).is_err());
    let mut unknown = serde_json::to_value(&plan).unwrap();
    unknown["unknown_contract"] = true.into();
    assert!(serde_json::from_value::<CapacityPlan>(unknown).is_err());
}
#[test]
fn render_and_mesh_budgets_are_requirements_not_aspirational_labels() {
    let (mut plan, records) = fixture();
    plan.request.budget.mesh_checks = 4;
    assert!(qualify(&plan, &records, sha(b"rows"), 0.).is_err());
    plan.request.budget.mesh_checks = 0;
    plan.request.budget.rendered_rooms = 4;
    assert!(qualify(&plan, &records, sha(b"rows"), 0.).is_err());
}
#[test]
fn fixed_and_consecutive_runs_have_no_iid_forecast_bounds() {
    for policy in [
        SeedPolicy::Fixed { seed: 1 },
        SeedPolicy::Consecutive { first: 10 },
    ] {
        let (mut plan, mut records) = fixture();
        plan.request.seed_policy = policy;
        for r in &mut records {
            r.seed = seed_at(&plan.request.seed_policy, r.draw);
        }
        let receipt = qualify(&plan, &records, sha(b"rows"), 0.).unwrap();
        let d = &receipt.estimate.definitions[0];
        assert_eq!(d.status, EstimateStatus::ObservedOnly);
        assert!(d.collision_probability.confidence.is_none());
        assert!(d.requested_occupancy.distinct.lower.is_none());
    }
}
#[test]
fn report_is_bound_to_receipt_and_escapes_user_labels() {
    let (plan, records) = fixture();
    let mut receipt = qualify(&plan, &records, sha(b"rows"), 0.).unwrap();
    receipt
        .estimate
        .limitations
        .push("<script>bad()</script>".into());
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    fs::write(&input, serde_json::to_vec(&receipt).unwrap()).unwrap();
    let output = dir.path().join("report");
    report::write(&input, &output).unwrap();
    let html = fs::read_to_string(output.join("index.html")).unwrap();
    assert!(!html.contains("<script>"));
    assert!(html.contains(&receipt.records_sha256));
    assert!(output.join("capacity.png").is_file());
    assert!(report::write(&input, &output).is_err());
}
