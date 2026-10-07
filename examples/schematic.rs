//! CPU-only schematic and prediction overlay example.
//! cargo run --example schematic --no-default-features --features multi_threaded
use bevy::prelude::*;
use bevy_zeroverse::{
    annotation::schematic::{Overlay, RenderOptions, Schematic},
    scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest},
};
fn main() -> anyhow::Result<()> {
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.25)
            .map_err(anyhow::Error::msg)?;
        let plan = Schematic::from_manifest(&scene, 0.4).map_err(anyhow::Error::msg)?;
        plan.write(
            format!("out/schematic/{seed}"),
            RenderOptions::default(),
            Overlay::default(),
        )?;
        let mut predicted = plan.cameras[0].clone();
        predicted.label = "Predicted C0".into();
        let matrix = Mat4::from_cols_array_2d(&predicted.world_from_view);
        predicted.world_from_view = (Mat4::from_translation(Vec3::new(0.5, 0., 0.35))
            * matrix
            * Mat4::from_rotation_y(0.12))
        .to_cols_array_2d();
        predicted.path.clear();
        plan.write(
            format!("out/schematic/{seed}-prediction"),
            RenderOptions::default(),
            Overlay {
                cameras: vec![predicted],
                ..default()
            },
        )?;
    }
    Ok(())
}
