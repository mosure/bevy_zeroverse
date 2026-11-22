//! Explicit capacity planning / CPU observations. Statistical qualification is
//! the publication crate's capacity-qualify command; neither operation publishes.
#[cfg(not(target_arch = "wasm32"))]
#[path = "indoor_capacity/capture.rs"]
mod capture;
#[path = "indoor_capacity/execution.rs"]
mod execution;
use anyhow::{ensure, Result};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    scene::{
        procedural_indoor::{capacity::*, validation::validate_geometry},
        ZeroverseSceneType,
    },
};
use clap::{Parser, Subcommand};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Completed native captures with frozen descriptors, all failures retained.
    #[cfg(not(target_arch = "wasm32"))]
    Capture {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Resolve a population without generating a room or loading any model.
    Plan {
        #[arg(long)]
        request: Option<PathBuf>,
        /// Full BevyZeroverseConfig JSON; default is the conditional indoor CLI preset.
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        capture: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Bounded CPU construction; includes every failed draw and optional real meshes.
    Audit {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        workers: usize,
    },
    /// Read a matching immutable estimate. Never launches capture or embeddings.
    Estimate {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
}
fn read<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn main() -> Result<()> {
    match Args::parse().command {
        #[cfg(not(target_arch = "wasm32"))]
        Action::Capture { plan, output } => capture::run(&read(&plan)?, &output)?,
        Action::Plan {
            request,
            config,
            capture,
            output,
        } => {
            let request: CapacityRequest =
                request.as_ref().map(read).transpose()?.unwrap_or_default();
            let capture = capture
                .as_ref()
                .map(read)
                .transpose()?
                .unwrap_or_else(default_capture);
            let config = config
                .as_ref()
                .map(read)
                .transpose()?
                .unwrap_or(BevyZeroverseConfig {
                    scene_type: ZeroverseSceneType::ProceduralIndoor,
                    headless: true,
                    width: 256.,
                    height: 256.,
                    num_cameras: 4,
                    playback_steps: 1,
                    ..Default::default()
                });
            let plan = config
                .capacity_plan(&request, capture)
                .map_err(anyhow::Error::msg)?;
            ensure!(!output.exists(), "plan already exists");
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(output, serde_json::to_vec_pretty(&plan)?)?;
        }
        Action::Audit {
            plan,
            output,
            workers,
        } => {
            let plan: CapacityPlan = read(&plan)?;
            validate_plan(&plan).map_err(anyhow::Error::msg)?;
            ensure!((1..=4).contains(&workers), "workers must be 1..4");
            ensure!(
                plan.request.budget.manifests <= 65_536,
                "bounded audit supports at most 65536 rooms"
            );
            ensure!(
                plan.request.budget.rendered_rooms == 0,
                "CPU audit cannot fulfill a render budget"
            );
            ensure!(
                !output.exists(),
                "audit output exists; receipts are immutable"
            );
            fs::create_dir_all(&output)?;
            std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
            fs::write(output.join("plan.json"), serde_json::to_vec_pretty(&plan)?)?;
            let started = Instant::now();
            let mut file = BufWriter::new(fs::File::create(output.join("records.jsonl.partial"))?);
            let mut failures = 0;
            let mut written = 0;
            execution::ordered(
                plan.request.budget.manifests,
                workers,
                |draw| {
                    let mesh_checked = draw < plan.request.budget.mesh_checks;
                    let result = std::panic::catch_unwind(|| -> Result<CapacityRecord, String> {
                        let scene = generate(&plan, draw)?;
                        if mesh_checked {
                            validate_geometry(&scene)?;
                        }
                        observe(&scene, &plan, draw, mesh_checked)
                    })
                    .unwrap_or_else(|_| Err("construction panic".into()));
                    result.unwrap_or_else(|reason| CapacityRecord {
                        draw,
                        seed: seed_at(&plan.request.seed_policy, draw),
                        mesh_checked: false,
                        outcome: SampleOutcome::Failed {
                            stage: "CPU construction/qualification".into(),
                            reason,
                        },
                    })
                },
                |record| {
                    failures += usize::from(matches!(record.outcome, SampleOutcome::Failed { .. }));
                    serde_json::to_writer(&mut file, &record)?;
                    writeln!(file)?;
                    written += 1;
                    if written % 64 == 0 {
                        eprintln!(
                            "{written}/{} draws; {failures} failures retained",
                            plan.request.budget.manifests
                        );
                    }
                    Ok(())
                },
            )?;
            file.flush()?;
            file.get_ref().sync_all()?;
            fs::rename(
                output.join("records.jsonl.partial"),
                output.join("records.jsonl"),
            )?;
            fs::write(
                output.join("execution.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"seconds":started.elapsed().as_secs_f64(),"workers":workers,"failures":failures,"scope":"CPU construction and descriptors, not completed rendered rooms/s"}),
                )?,
            )?;
            ensure!(failures==0,"audit retained {failures} failing rooms; qualify the complete record stream to inspect yield");
        }
        Action::Estimate { plan, receipt } => {
            let plan: CapacityPlan = read(&plan)?;
            let receipt: CapacityReceipt = read(&receipt)?;
            validate_receipt(&plan, &receipt).map_err(anyhow::Error::msg)?;
            println!("{}", serde_json::to_string_pretty(&receipt.estimate)?);
        }
    }
    Ok(())
}
