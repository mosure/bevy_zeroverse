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
            println!("Verified current snapshot: crate {}, generator {}, {} rendered rooms, {} bound artifacts",report.generator_identity.crate_version,report.generator_identity.generator_version,report.protocol.rendered_rooms,report.artifacts.len());
        }
    }
    Ok(())
}
