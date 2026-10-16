//! Explicit, bounded offline qualification. No renderer or model is initialized.
pub mod embeddings;
pub mod publication;
pub mod report;
pub mod statistics;
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::{capacity::*, provenance::sha};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::Path,
};

pub const MAX_IN_MEMORY_ROOMS: u32 = 65_536;
#[cfg(test)]
mod tests;

/// Analyze ordered root draws, including failed attempts. Views and frames never
/// enter the uncertainty calculation as independent rooms.
pub fn qualify(
    plan: &CapacityPlan,
    records: &[CapacityRecord],
    records_sha256: String,
    seconds: f64,
) -> Result<CapacityReceipt> {
    validate_plan(plan).map_err(anyhow::Error::msg)?;
    ensure!(
        records.len() == plan.request.budget.manifests as usize,
        "incomplete audit: retain every requested seed, including failures"
    );
    ensure!(records.len() <= MAX_IN_MEMORY_ROOMS as usize,"cohort exceeds the bounded qualification implementation; shard observation capture instead of allocating unbounded memory");
    let mut labels: BTreeMap<String, Vec<String>> = plan
        .request
        .definitions
        .iter()
        .map(|d| (d.id.clone(), Vec::new()))
        .collect();
    let mut diagnostics = BTreeMap::<String, DiagnosticSummary>::new();
    let mut families = BTreeSet::new();
    let (mut accepted, mut rendered, mut mesh) = (0, 0, 0);
    for (i, record) in records.iter().enumerate() {
        ensure!(
            record.draw as usize == i,
            "audit order has a duplicate or missing draw"
        );
        ensure!(
            record.seed == seed_at(&plan.request.seed_policy, record.draw),
            "seed does not match the declared sampling law"
        );
        validate_record(plan, record).map_err(anyhow::Error::msg)?;
        mesh += u32::from(record.mesh_checked);
        if let SampleOutcome::Accepted {
            family_sha256,
            descriptors,
            diagnostics: values,
            capture_sha256,
            ..
        } = &record.outcome
        {
            accepted += 1;
            rendered += u32::from(capture_sha256.is_some());
            families.insert(family_sha256);
            for (id, d) in descriptors {
                labels.get_mut(id).unwrap().push(d.class_sha256.clone());
            }
            for (name, &value) in values {
                let d = diagnostics
                    .entry(name.clone())
                    .or_insert(DiagnosticSummary {
                        rooms: 0,
                        nonzero_rooms: 0,
                        sum: 0.,
                        min: f64::INFINITY,
                        max: f64::NEG_INFINITY,
                    });
                d.rooms += 1;
                d.nonzero_rooms += u32::from(value != 0.);
                d.sum += value;
                d.min = d.min.min(value);
                d.max = d.max.max(value);
            }
        }
    }
    // Missing category observations mean zero, not a denominator restricted to
    // rooms that happen to contain the category.
    for d in diagnostics.values_mut() {
        if d.rooms < accepted {
            d.min = d.min.min(0.);
            d.max = d.max.max(0.);
        }
        d.rooms = accepted;
    }
    let definitions = plan
        .request
        .definitions
        .iter()
        .map(|d| {
            let mut estimate = statistics::estimate(d, &labels[&d.id], &plan.request)?;
            let mut blocks = BTreeMap::<String, BTreeSet<&str>>::new();
            for record in records {
                if let SampleOutcome::Accepted { descriptors, .. } = &record.outcome {
                    if let Some(descriptor) = descriptors.get(&d.id) {
                        for (name, hash) in &descriptor.blocks {
                            blocks.entry(name.clone()).or_default().insert(hash);
                        }
                    }
                }
            }
            estimate.block_distinct_counts = blocks
                .into_iter()
                .map(|(name, keys)| (name, keys.len() as u32))
                .collect();
            Ok(estimate)
        })
        .collect::<Result<_>>()?;
    let receipt = CapacityReceipt {
        schema_version:SCHEMA_VERSION,plan:plan.clone(),records_sha256,complete:true,qualification_seconds:seconds,
        estimate:CapacityEstimate {
            estimator:ESTIMATOR_VERSION.into(),requested_rooms:records.len() as u32,accepted_rooms:accepted,
            failed_rooms:records.len() as u32-accepted,mesh_checked_rooms:mesh,rendered_rooms:rendered,
            unique_families:families.len() as u32,definitions,diagnostics,
            limitations:vec![
                "Accepted-program bins measure joint construction, not rendered perceptual capacity.".into(),
                "Confidence bounds condition on successful draws; all failed attempts remain in the receipt.".into(),
                "No qualified unseen-tail forecast or downstream learning law; sample spread is not global capacity.".into(),
                "A metric, bin size or configuration selected after inspection requires an independent confirmation cohort.".into(),
            ],
        },
    };
    validate_receipt(plan, &receipt).map_err(anyhow::Error::msg)?;
    Ok(receipt)
}

/// SHA-256 counter stream, versioned by its domain string. No mutable worker RNG
/// state, scheduling dependence, consecutive-seed assumption or retry resampling.
pub use bevy_zeroverse_capture::capacity::seed_at;

pub fn qualify_files(plan_path: &Path, records_path: &Path, output: &Path) -> Result<()> {
    let plan: CapacityPlan = serde_json::from_slice(&fs::read(plan_path)?)?;
    ensure!(
        plan.request.budget.manifests <= MAX_IN_MEMORY_ROOMS,
        "qualification budget too large"
    );
    ensure!(
        !output.exists(),
        "refusing to overwrite immutable qualification receipt"
    );
    let started = std::time::Instant::now();
    let mut records = Vec::new();
    for line in BufReader::new(fs::File::open(records_path)?).lines() {
        ensure!(
            records.len() < plan.request.budget.manifests as usize,
            "too many audit rows"
        );
        records.push(serde_json::from_str(&line?).context("malformed audit record")?);
    }
    let receipt = qualify(
        &plan,
        &records,
        sha(&fs::read(records_path)?),
        started.elapsed().as_secs_f64(),
    )?;
    // Same-directory persist gives an atomic result; interrupted qualification
    // never produces a file that can be mistaken for a complete receipt.
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut tmp, &receipt)?;
    tmp.persist_noclobber(output)?;
    Ok(())
}

pub fn estimate(plan: &CapacityPlan, receipt: &CapacityReceipt) -> Result<CapacityEstimate> {
    validate_receipt(plan, receipt).map_err(anyhow::Error::msg)?;
    Ok(receipt.estimate.clone())
}
