use std::time::{Duration, Instant};

fn wait(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
}

fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024
}

fn main() {
    let hint = std::env::args().nth(1).unwrap_or_else(|| "memory".into());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("zeroverse dependency audit"),
        memory_hints: if hint == "memory" {
            wgpu::MemoryHints::MemoryUsage
        } else {
            wgpu::MemoryHints::Performance
        },
        ..Default::default()
    }))
    .unwrap();
    let side = 1024;
    let extent = wgpu::Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("4 MiB upload target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut pixels = vec![0_u8; (side * side * 4) as usize];
    let mut uploads = Vec::new();
    for batch in 0..4_u8 {
        pixels.fill(batch + 1);
        let start = Instant::now();
        for _ in 0..8 {
            queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: Some(side),
                },
                extent,
            );
        }
        let write_seconds = start.elapsed().as_secs_f64();
        queue.submit([]);
        wait(&device);
        uploads.push(serde_json::json!({"batch": batch, "warmup": batch == 0,
            "bytes": pixels.len() * 8, "write_seconds": write_seconds,
            "completion_seconds": start.elapsed().as_secs_f64()}));
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("upload verification"),
        size: pixels.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 4),
                rows_per_image: Some(side),
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    wait(&device);
    rx.recv().unwrap().unwrap();
    assert_eq!(&*readback.slice(..).get_mapped_range(), &pixels);
    readback.unmap();
    let commands_target = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("command retention target"),
        size: 4096,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut waves = Vec::new();
    for peak in [128, 512, 1024] {
        let mut pending = Vec::with_capacity(peak);
        let start = Instant::now();
        for _ in 0..peak {
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..128 {
                encoder.clear_buffer(&commands_target, 0, None);
            }
            pending.push(encoder.finish());
        }
        queue.submit(pending);
        wait(&device);
        waves.push(serde_json::json!({"peak": peak,
            "seconds": start.elapsed().as_secs_f64(), "rss_bytes": rss_bytes(),
            "idle_command_encoders": device.get_internal_counters().hal.command_encoders.read()}));
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "adapter": format!("{:?}", adapter.get_info()), "memory_hint": hint,
        "upload_readback_equal": true, "uploads": uploads, "command_waves": waves,
        "scope": "Bounded Vulkan upload and command-cache diagnostic; not a scene or unlimited-process benchmark."
    })).unwrap());
}
