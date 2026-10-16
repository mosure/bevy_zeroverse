use super::*;
use std::collections::BTreeSet;

pub(crate) fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
pub fn validate_plan(plan: &CapacityPlan) -> Result<(), String> {
    let r = &plan.request;
    let c = &plan.population.capture;
    if plan.schema_version != SCHEMA_VERSION || r.schema_version != SCHEMA_VERSION {
        return Err("unsupported capacity schema version".into());
    }
    if !(0.5..1.).contains(&r.confidence)
        || r.requested_draws == 0
        || r.requested_draws > 1_u64 << 53
        || r.budget.manifests == 0
        || r.budget.mesh_checks > r.budget.manifests
        || r.budget.rendered_rooms > r.budget.manifests
    {
        return Err("invalid confidence, horizon or audit budget".into());
    }
    if r.definitions.is_empty() || r.definitions.len() > 32 {
        return Err("capacity needs 1..32 frozen definitions".into());
    }
    let mut ids = BTreeSet::new();
    for d in &r.definitions {
        if d.id.is_empty()
            || !ids.insert(&d.id)
            || d.version != 1
            || ![d.metres, d.radians, d.appearance_step, d.log2_light_step]
                .iter()
                .all(|x| x.is_finite() && *x > 0.)
            || d.grid_side == 0
            || d.grid_side > 128
        {
            return Err("invalid or duplicate diversity definition".into());
        }
        if d.descriptor == DescriptorKind::RenderedGrid
            && (r.unit == SampleUnit::DressedScene
                || ["color", "normal", "position", "semantic"]
                    .iter()
                    .any(|mode| !c.modes.iter().any(|m| m == mode)))
        {
            return Err("rendered-grid requires an observation unit and RGB/normal/position/semantic capture modes".into());
        }
    }
    if c.image_size.contains(&0)
        || !(1..=16).contains(&c.cameras)
        || c.progress.is_empty()
        || c.progress.len() > 1024
        || !c.progress.iter().all(|t| (0.0..=1.).contains(t))
    {
        return Err("invalid capture dimensions, cameras or schedule".into());
    }
    // Equality rather than float ordering also catches repeated +0/-0 frames.
    if c.progress
        .iter()
        .enumerate()
        .any(|(i, t)| c.progress[..i].contains(t))
    {
        return Err("repeated capture progress does not create temporal diversity".into());
    }
    if let Some(times) = &c.seconds {
        if times.len() != c.progress.len()
            || !times.iter().all(|t| t.is_finite() && *t >= 0.)
            || times.windows(2).any(|t| t[1] <= t[0])
        {
            return Err("physical capture times must be finite and strictly increasing".into());
        }
    }
    if r.unit == SampleUnit::TemporalClip && c.progress.len() < 2 {
        return Err("a temporal clip requires at least two distinct captured states".into());
    }
    if c.modes.is_empty()
        || c.modes.iter().enumerate().any(|(i, m)| {
            c.modes[..i].contains(m)
                || !matches!(
                    m.as_str(),
                    "color"
                        | "depth"
                        | "normal"
                        | "position"
                        | "semantic"
                        | "optical_flow"
                        | "motion_vectors"
                        | "co_visibility"
                )
        })
        || !matches!(c.annotation_glass.as_str(), "surface" | "through")
        || !digest(&plan.population.config_sha256)
        || !digest(&plan.population.generator.source_sha256)
        || plan.population.capture_engine.is_empty()
        || plan.population.platform.is_empty()
        || plan.population.failure_policy != "retain-error-no-resampling-v1"
    {
        return Err("missing capture/provenance contract".into());
    }
    #[cfg(feature = "provenance")]
    if crate::provenance::sha(plan.population.resolved_config_json.as_bytes())
        != plan.population.config_sha256
    {
        return Err("capacity config hash mismatch".into());
    }
    let mut factors = BTreeSet::new();
    if plan
        .factors
        .iter()
        .any(|f| f.id.is_empty() || !factors.insert(&f.id))
        || plan
            .factors
            .iter()
            .any(|f| f.parents.iter().any(|p| p == &f.id || !factors.contains(p)))
    {
        return Err("invalid factor registry".into());
    }
    Ok(())
}

pub fn validate_record(plan: &CapacityPlan, record: &CapacityRecord) -> Result<(), String> {
    if record.draw >= plan.request.budget.manifests {
        return Err("sample exceeds the declared audit budget".into());
    }
    if record.mesh_checked && record.draw >= plan.request.budget.mesh_checks {
        return Err("mesh check was not in the prespecified cohort".into());
    }
    match &record.outcome {
        SampleOutcome::Failed { stage, reason } if stage.is_empty() || reason.is_empty() => {
            Err("failure must retain its stage and reason".into())
        }
        SampleOutcome::Failed { .. } => Ok(()),
        SampleOutcome::Accepted {
            family_sha256,
            descriptors,
            diagnostics,
            delivered_views,
            delivered_progress,
            capture_sha256,
        } => {
            if !digest(family_sha256) || !diagnostics.values().all(|v| v.is_finite()) {
                return Err("invalid family identity or nonfinite diagnostic".into());
            }
            if let Some(hash) = capture_sha256 {
                if !digest(hash)
                    || record.draw >= plan.request.budget.rendered_rooms
                    || *delivered_views != plan.population.capture.cameras
                    || delivered_progress != &plan.population.capture.progress
                {
                    return Err("incomplete rendered camera/time delivery".into());
                }
            } else if *delivered_views != 0 || !delivered_progress.is_empty() {
                return Err("manifest observations must not claim rendered delivery".into());
            }
            if record.draw < plan.request.budget.rendered_rooms && capture_sha256.is_none() {
                return Err("accepted draw is missing its required render".into());
            }
            if record.draw < plan.request.budget.mesh_checks && !record.mesh_checked {
                return Err("accepted draw is missing its required mesh check".into());
            }
            for (id, descriptor) in descriptors {
                let d = plan
                    .request
                    .definitions
                    .iter()
                    .find(|d| &d.id == id)
                    .ok_or("unrequested descriptor")?;
                if !digest(&descriptor.class_sha256)
                    || descriptor.blocks.is_empty()
                    || descriptor.blocks.values().any(|h| !digest(h))
                {
                    return Err("invalid descriptor identity".into());
                }
                if d.descriptor == DescriptorKind::RenderedGrid && capture_sha256.is_none() {
                    return Err("rendered descriptor without completed capture".into());
                }
            }
            for d in &plan.request.definitions {
                let required =
                    d.descriptor == DescriptorKind::AcceptedProgram || capture_sha256.is_some();
                if required && !descriptors.contains_key(&d.id) {
                    return Err(format!("missing requested descriptor {}", d.id));
                }
            }
            Ok(())
        }
    }
}

pub fn validate_receipt(plan: &CapacityPlan, receipt: &CapacityReceipt) -> Result<(), String> {
    validate_plan(plan)?;
    if receipt.schema_version != SCHEMA_VERSION || &receipt.plan != plan {
        return Err("capacity receipt configuration/source/definition mismatch".into());
    }
    let e = &receipt.estimate;
    if !receipt.complete
        || !digest(&receipt.records_sha256)
        || !receipt.qualification_seconds.is_finite()
        || receipt.qualification_seconds < 0.
        || e.estimator != ESTIMATOR_VERSION
        || e.requested_rooms != plan.request.budget.manifests
        || u64::from(e.accepted_rooms) + u64::from(e.failed_rooms) != u64::from(e.requested_rooms)
        || e.mesh_checked_rooms > plan.request.budget.mesh_checks
        || e.rendered_rooms > plan.request.budget.rendered_rooms
        || e.unique_families > e.accepted_rooms
        || e.definitions.len() != plan.request.definitions.len()
    {
        return Err("incomplete or inconsistent capacity receipt".into());
    }
    for summary in e.diagnostics.values() {
        if summary.rooms != e.accepted_rooms
            || summary.nonzero_rooms > summary.rooms
            || ![summary.sum, summary.min, summary.max]
                .iter()
                .all(|v| v.is_finite())
            || summary.min > summary.max
        {
            return Err("inconsistent diagnostic summary".into());
        }
    }
    for (d, expected) in e.definitions.iter().zip(&plan.request.definitions) {
        if &d.definition != expected
            || d.rooms > e.accepted_rooms
            || d.observed_classes > d.rooms
            || d.singletons > d.observed_classes
            || d.doubletons > d.observed_classes
            || d.disjoint_pairs != d.rooms / 2
            || d.matching_disjoint_pairs > d.disjoint_pairs
            || d.requested_occupancy.draws != plan.request.requested_draws
            || d.max_qualified_forecast_draws > u64::from(d.rooms)
            || !d.empirical_entropy_diversity.is_finite()
            || d.block_distinct_counts.values().any(|&v| v > d.rooms)
            || [
                d.collision_probability.point,
                d.collision_probability.lower,
                d.collision_probability.upper,
            ]
            .into_iter()
            .flatten()
            .any(|v| !(0.0..=1.).contains(&v))
            || d.nested_validation.iter().any(|v| {
                ![v.predicted_unseen_fraction, v.observed_unseen_fraction]
                    .iter()
                    .all(|p| (0.0..=1.).contains(p))
            })
        {
            return Err("inconsistent capacity estimate".into());
        }
        for s in std::iter::once(&d.collision_probability)
            .chain(std::iter::once(&d.collision_diversity))
            .chain(std::iter::once(&d.requested_occupancy.distinct))
            .chain(d.rarefaction.iter().map(|p| &p.distinct))
        {
            if [s.point, s.lower, s.upper]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite() || v < 0.)
                || s.lower.zip(s.upper).is_some_and(|(l, u)| l > u)
                || s.confidence.is_some_and(|v| !(0.5..1.).contains(&v))
            {
                return Err("nonfinite or reversed estimate interval".into());
            }
        }
    }
    Ok(())
}
