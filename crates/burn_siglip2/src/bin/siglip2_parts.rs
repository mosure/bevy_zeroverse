#![recursion_limit = "256"]

use std::path::PathBuf;

use clap::Parser;

use burn_siglip2::{DEFAULT_PART_SIZE_MIB, write_bpk_parts};

#[derive(Debug, Parser)]
#[command(about = "Create bounded, checksummed SigLIP2 BPK shards from an imported BPK")]
struct Args {
    /// Imported monolithic BPK used as the immutable source artifact.
    #[arg(long)]
    bpk: PathBuf,

    /// Requested maximum shard size in MiB.
    #[arg(long, default_value_t = DEFAULT_PART_SIZE_MIB)]
    max_mib: u64,

    /// Replace an existing parts manifest and its listed shards.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let report = write_bpk_parts(&args.bpk, args.max_mib, args.overwrite)?
        .ok_or_else(|| "parts generation unexpectedly returned no report".to_string())?;
    println!("parts manifest: {}", report.manifest_path.display());
    println!("parts: {}", report.part_paths.len());
    println!("source bytes: {}", report.total_bytes);
    println!(
        "requested/actual max bytes: {}/{}",
        report.requested_max_part_bytes, report.actual_max_part_bytes
    );
    if report.parts_exceeding_requested_limit > 0 {
        return Err(format!(
            "{} shard(s) exceed the requested maximum; regenerate with a larger maximum or update the portable tensor chunking policy",
            report.parts_exceeding_requested_limit
        ));
    }
    Ok(())
}
