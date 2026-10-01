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
            println!("Verified crate {}, generator {}, {} rendered rooms, {} bound artifacts; references retain separate identities",report.generator_identity.crate_version,report.generator_identity.generator_version,report.protocol.rendered_rooms,report.artifacts.len());
        }
    }
    Ok(())
}
