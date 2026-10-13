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
            before,
            after,
            output,
            recipe,
            embeddings,
            before_population,
            after_population,
        } => bevy_zeroverse_publication::study::compare(
            &args.root,
            &before,
            &after,
            &output,
            recipe.as_deref(),
            embeddings.as_deref(),
            before_population.as_deref(),
            after_population.as_deref(),
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
