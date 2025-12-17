use std::{fs, mem, path::PathBuf, time::Duration};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless,
    io::channels,
    ovoxel::OvoxelVolume,
    ovoxel_mesh::{ovoxel_to_mesh, ovoxel_to_semantic_mesh, write_mesh_as_glb},
    render::RenderMode,
    sample::Sample,
    scene::ZeroverseSceneType,
};

#[derive(Debug, Parser)]
struct Args {
    /// Grid resolution used for voxelization.
    #[arg(long, default_value = "128")]
    resolution: u32,

    /// Output directory for GLB exports.
    #[arg(long, default_value = "data/ovoxel_autoencode")]
    output: PathBuf,

    /// When set, colorize the baked O-Voxel mesh by semantic label instead of base color.
    #[arg(long, default_value_t = false)]
    semantic_colors: bool,
}

fn request_sample(config: &BevyZeroverseConfig) -> Result<Sample> {
    headless::setup_globals(None);
    headless::setup_and_run_app(true, Some(config.clone()));

    channels::app_frame_sender()
        .send(())
        .context("signal app for next frame")?;

    let receiver = channels::sample_receiver().context("sample receiver missing")?;
    let lock = receiver
        .lock()
        .map_err(|_| anyhow!("sample receiver lock poisoned"))?;
    lock.recv_timeout(Duration::from_secs(120))
        .context("timed out waiting for sample")
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config = BevyZeroverseConfig {
        headless: true,
        image_copiers: true,
        editor: false,
        press_esc_close: false,
        keybinds: false,
        render_modes: vec![RenderMode::Color],
        num_cameras: 1,
        scene_type: ZeroverseSceneType::SemanticRoom,
        ovoxel_resolution: args.resolution,
        ..Default::default()
    };

    let mut sample = request_sample(&config)?;
    let ov = sample
        .ovoxel
        .take()
        .ok_or_else(|| anyhow!("no ovoxel volume in sample"))?;
    let volume = OvoxelVolume {
        coords: ov.coords,
        dual_vertices: ov.dual_vertices,
        intersected: ov.intersected,
        base_color: ov.base_color,
        semantics: ov.semantics,
        semantic_labels: ov.semantic_labels,
        resolution: ov.resolution,
        aabb: ov.aabb,
    };

    let auto_mesh = if args.semantic_colors {
        ovoxel_to_semantic_mesh(&volume)
    } else {
        ovoxel_to_mesh(&volume)
    };

    fs::create_dir_all(&args.output).context("create output dir")?;
    let auto_path = args.output.join("autoencoded.glb");
    write_mesh_as_glb(&auto_mesh, &auto_path)?;

    // Save the first view color as PNG for reference.
    if let Some(view) = sample.views.first() {
        let width = config.width.round() as u32;
        let height = config.height.round() as u32;
        let bytes_per_pixel = mem::size_of::<f32>() * 4;
        let row_stride = (width as usize * bytes_per_pixel)
            .div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT as usize)
            * COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        let expected_len = row_stride * height as usize;
        if view.color.len() >= expected_len {
            let mut rgba: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height as usize {
                let start = row * row_stride;
                let row_bytes = &view.color[start..start + width as usize * bytes_per_pixel];
                for chunk in row_bytes.chunks_exact(bytes_per_pixel) {
                    let r = f32::from_ne_bytes(chunk[0..4].try_into().unwrap());
                    let g = f32::from_ne_bytes(chunk[4..8].try_into().unwrap());
                    let b = f32::from_ne_bytes(chunk[8..12].try_into().unwrap());
                    let a = f32::from_ne_bytes(chunk[12..16].try_into().unwrap());
                    rgba.extend_from_slice(&[
                        (r.clamp(0.0, 1.0) * 255.0) as u8,
                        (g.clamp(0.0, 1.0) * 255.0) as u8,
                        (b.clamp(0.0, 1.0) * 255.0) as u8,
                        (a.clamp(0.0, 1.0) * 255.0) as u8,
                    ]);
                }
            }
            if let Some(img) = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba) {
                let color_path = args.output.join("color.png");
                img.save(&color_path)
                    .with_context(|| format!("write png {:?}", color_path))?;
                println!("saved color PNG to {:?}", color_path);
            }
        }
    }

    println!("saved autoencoded GLB to {:?}", auto_path);

    Ok(())
}
