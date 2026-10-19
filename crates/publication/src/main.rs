use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Formal project-page and paper publication for bevy_zeroverse")]
struct Args {
    #[arg(long, default_value = ".", global = true)]
    root: PathBuf,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Print the canonical immutable capture/paper digest for a retained-evidence policy.
    EvidenceDigest,
    /// Build a local current-source paper section and page fragment; does not publish.
    CapacityPaper {
        #[arg(long, required=true, num_args=1..)]
        receipt: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Freeze a whole-image SigLIP2 cohort and kernel before optional inference.
    CapacityEmbeddingIndex {
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        captures: PathBuf,
        #[arg(long)]
        weights_sha256: String,
        #[arg(long, default_value_t = 0.35)]
        bandwidth: f64,
        #[arg(long, default_value_t = 512)]
        max_rooms: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify cached SigLIP2 inference and report room-weighted sample diversity.
    CapacityEmbeddingReport {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate local figures, HTML and a paper table from one qualified receipt.
    CapacityReport {
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Qualify a complete local capacity observation stream. Does not publish.
    CapacityQualify {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate analytical occupancy illustrations; never estimates room capacity.
    CapacityTheory {
        #[arg(long)]
        output: PathBuf,
    },
    /// Bind architectural door/cabinet construction, renders and semantics.
    DoorStudy {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Build a local current-state functional-room gallery from verified native cohorts.
    ProgramStudy {
        #[arg(long,required=true,num_args=1..)]
        input: Vec<PathBuf>,
        /// Source-bound structural, boundary or camera audit JSON receipts.
        #[arg(long, num_args=1..)]
        audit: Vec<PathBuf>,
        /// Optional fixed-envelope object audit and studio capture directory.
        #[arg(long)]
        objects: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify current rendered populations and prepare a snapshot receipt for page/paper use.
    CohortStudy {
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate a paper-only timing receipt, phase table and engine diagram.
    TimingStudy {
        #[arg(long, required = true, num_args = 1..)]
        captures: Vec<PathBuf>,
        #[arg(long, required = true, num_args = 1..)]
        preparation: Vec<PathBuf>,
        #[arg(long)]
        environment: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Bind production-material sampling and native PBR swatches into an atlas.
    MaterialStudy {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Bind the current object audit and studio captures into a local atlas.
    ObjectStudy {
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        renders: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify completed cohorts and report them separately, with bound inputs.
    AuditCohorts {
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Summarize a checksum-bound camera-count audit and optional native renders.
    CameraStudy {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        rendered: Option<PathBuf>,
    },
    /// Verify all bytes in an offline paired study, including every annotation.
    VerifyStudy {
        #[arg(long)]
        output: PathBuf,
    },
    /// Qualify and compare every room in matched cohorts; build a complete gallery.
    CompareIndoor {
        /// Explicit seed-matched architecture experiment; default requires identical geometry.
        #[arg(long)]
        allow_architecture_changes: bool,
        #[arg(long)]
        before: PathBuf,
        #[arg(long)]
        after: PathBuf,
        #[arg(long, default_value = "out/indoor_study")]
        output: PathBuf,
        /// Protocol TOML; defaults to 64 rooms, 3 views, 512 square, GI 1024.
        #[arg(long)]
        recipe: Option<PathBuf>,
        #[arg(long)]
        embeddings: Option<PathBuf>,
        #[arg(long)]
        before_population: Option<PathBuf>,
        #[arg(long)]
        after_population: Option<PathBuf>,
    },
    /// Capture a matched camera/aspect/appearance sweep, receipt and annotation gallery.
    Qualify {
        #[arg(long, default_value = "target/debug/indoor_validate")]
        validator: PathBuf,
        #[arg(long, default_value = "out/qualification")]
        output: PathBuf,
        #[arg(long)]
        recipe: Option<PathBuf>,
    },
    /// Verify every byte bound by a qualification receipt, including failed cases.
    VerifyQualification {
        #[arg(long, default_value = "out/qualification")]
        output: PathBuf,
    },
    /// Prepare/verify the latest page and paper before invoking cargo publish.
    Publish {
        #[arg(long, default_value = "bevy_zeroverse")]
        package: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Regenerate the page/paper from verified shipped captures (no GPU or out/ data).
    Rebuild {
        #[arg(long)]
        release_version: Option<String>,
    },
    /// One transaction: current captures → validation → media/page → paper → attestation.
    Refresh {
        /// Force a fresh render instead of reusing an identical completed capture run.
        #[arg(long)]
        recapture: bool,
    },
    /// Offline release/deployment gate. Needs neither Python, a GPU nor out/ captures.
    Verify {
        /// Optionally require the Cargo package version to match a release tag.
        #[arg(long)]
        release_version: Option<String>,
    },
}
fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Action::EvidenceDigest => {
            let attestation = bevy_zeroverse_publication::io::read(
                &args.root.join("www/project/publication.json"),
            )?;
            println!(
                "{}",
                bevy_zeroverse_publication::retained::capture_evidence_digest(&attestation)?
            );
        }
        Action::CapacityPaper { receipt, output } => {
            anyhow::ensure!(!output.exists(), "capacity paper output already exists");
            let identity = bevy_zeroverse_publication::pipeline::identity(&args.root)?;
            bevy_zeroverse_publication::capacity::publication::build(
                &args.root, &output, &receipt, &identity,
            )?;
        }
        Action::CapacityEmbeddingIndex {
            receipt,
            records,
            captures,
            weights_sha256,
            bandwidth,
            max_rooms,
            output,
        } => {
            bevy_zeroverse_publication::capacity::embeddings::index(
                &receipt,
                &records,
                &captures,
                &weights_sha256,
                bevy_zeroverse_publication::capacity::embeddings::kernel::KernelConfig {
                    bandwidth,
                    max_rooms,
                },
                &output,
            )?;
        }
        Action::CapacityEmbeddingReport {
            index,
            embeddings,
            output,
        } => {
            bevy_zeroverse_publication::capacity::embeddings::report(&index, &embeddings, &output)?;
        }
        Action::CapacityReport { receipt, output } => {
            bevy_zeroverse_publication::capacity::report::write(&receipt, &output)?
        }
        Action::CapacityQualify {
            plan,
            records,
            output,
        } => {
            bevy_zeroverse_publication::capacity::qualify_files(&plan, &records, &output)?;
        }
        Action::CapacityTheory { output } => {
            bevy_zeroverse_publication::capacity_theory::write(&output)?
        }
        Action::DoorStudy { input, output } => {
            bevy_zeroverse_publication::doors::prepare(&args.root, &input, &output)?
        }
        Action::ProgramStudy {
            input,
            audit,
            objects,
            output,
        } => bevy_zeroverse_publication::programs::prepare(
            &args.root,
            &input,
            &audit,
            objects.as_deref(),
            &output,
        )?,
        Action::CohortStudy { input, output } => {
            bevy_zeroverse_publication::cohorts::publication::prepare(&args.root, &input, &output)?;
        }
        Action::TimingStudy {
            captures,
            preparation,
            environment,
            output,
        } => {
            bevy_zeroverse_publication::timing::prepare(
                &args.root,
                &captures,
                &preparation,
                &environment,
                &output,
            )?;
        }
        Action::MaterialStudy { input, output } => {
            bevy_zeroverse_publication::materials::prepare(&args.root, &input, &output)?
        }
        Action::ObjectStudy {
            audit,
            renders,
            output,
        } => bevy_zeroverse_publication::objects::prepare(&args.root, &audit, &renders, &output)?,
        Action::AuditCohorts { input, output } => {
            bevy_zeroverse_publication::cohorts::audit(&args.root, &input, &output)?;
        }
        Action::CameraStudy {
            input,
            output,
            rendered,
        } => {
            bevy_zeroverse_publication::camera_study::prepare(&input, &output, rendered.as_deref())?
        }
        Action::VerifyStudy { output } => {
            bevy_zeroverse_publication::study::verify(&args.root.join(output))?
        }
        Action::CompareIndoor {
            allow_architecture_changes,
            before,
            after,
            output,
            recipe,
            embeddings,
            before_population,
            after_population,
        } => bevy_zeroverse_publication::study::compare_with_matching(
            &args.root,
            &before,
            &after,
            &output,
            recipe.as_deref(),
            embeddings.as_deref(),
            before_population.as_deref(),
            after_population.as_deref(),
            allow_architecture_changes,
        )?,
        Action::Qualify {
            validator,
            output,
            recipe,
        } => bevy_zeroverse_publication::qualification::run(
            &args.root,
            &validator,
            &output,
            recipe.as_deref(),
        )?,
        Action::VerifyQualification { output } => {
            bevy_zeroverse_publication::qualification::verify(&output)?
        }
        Action::Publish { package, dry_run } => {
            bevy_zeroverse_publication::pipeline::publish(&args.root, &package, dry_run)?
        }
        Action::Rebuild { release_version } => {
            bevy_zeroverse_publication::pipeline::rebuild(&args.root, release_version.as_deref())?
        }
        Action::Refresh { recapture } => {
            bevy_zeroverse_publication::pipeline::refresh(&args.root, recapture)?
        }
        Action::Verify { release_version } => {
            let report = bevy_zeroverse_publication::pipeline::verify(
                &args.root,
                release_version.as_deref(),
            )?;
            println!("Verified captured snapshot: crate {}, generator {}, {} rendered rooms, {} bound artifacts; checkout crate {}",report.generator_identity.crate_version,report.generator_identity.generator_version,report.protocol.rendered_rooms,report.artifacts.len(), bevy_zeroverse_publication::pipeline::identity(&args.root)?.crate_version);
        }
    }
    Ok(())
}
