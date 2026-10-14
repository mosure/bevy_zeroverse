//! Production-material qualification; diagnostics never run in the generator.
#[path = "material_study/audit.rs"]
mod audit;
#[path = "material_study/studio.rs"]
mod studio;
use anyhow::{ensure, Result};
use bevy_zeroverse::scene::procedural_indoor::materials::{program, Surface};
use clap::Parser;
use std::{fs, path::PathBuf};

// Common illustration seeds. Seed four includes frosted interior glass;
// frequencies use the full consecutive cohort, never this selected grid.
pub const SEEDS: [u64; 6] = [0, 1, 4, 19, 53, 101];
pub const RESOLUTION: u32 = 384;
pub fn tool_sources() -> std::collections::BTreeMap<&'static str, String> {
    use sha2::{Digest, Sha256};
    [
        (
            "examples/qualify_materials.rs",
            include_str!("qualify_materials.rs"),
        ),
        (
            "examples/material_study/audit.rs",
            include_str!("material_study/audit.rs"),
        ),
        (
            "examples/material_study/studio.rs",
            include_str!("material_study/studio.rs"),
        ),
    ]
    .map(|(name, source)| (name, format!("{:x}", Sha256::digest(source.as_bytes()))))
    .into_iter()
    .collect()
}
#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    samples: usize,
    #[arg(long, default_value_t = 256)]
    rooms: usize,
    /// Diagnostic smoke test only; publication requires the full inventory.
    #[arg(long)]
    category: Option<String>,
    #[arg(long)]
    skip_renders: bool,
}
#[derive(Clone)]
pub struct Category {
    pub name: String,
    pub surface: Surface,
    pub floor: u32,
}
pub fn categories() -> Vec<Category> {
    program::SURFACES
        .into_iter()
        .flat_map(|surface| {
            let styles = if surface == Surface::Floor { 3 } else { 1 };
            (0..styles).map(move |floor| Category {
                name: if surface == Surface::Floor {
                    ["FloorTimber", "FloorCarpet", "FloorStone"][floor as usize].into()
                } else {
                    format!("{surface:?}")
                },
                surface,
                floor,
            })
        })
        .chain(std::iter::once(Category {
            name: "Rug".into(),
            surface: Surface::FabricAlt,
            floor: 0,
        }))
        .collect()
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (128..=1024).contains(&args.samples) && (1..=1024).contains(&args.rooms),
        "invalid study size"
    );
    fs::create_dir_all(&args.output)?;
    let categories: Vec<_> = categories()
        .into_iter()
        .filter(|c| args.category.as_ref().is_none_or(|s| s == &c.name))
        .collect();
    ensure!(!categories.is_empty(), "unknown material category");
    audit::run(&args.output, &categories, args.samples, args.rooms)?;
    if !args.skip_renders {
        studio::run(&args.output, &categories)?;
    }
    Ok(())
}
