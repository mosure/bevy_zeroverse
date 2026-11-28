use super::*;

/// Independent workspaces start empty. The control uses one global pass for
/// each dependent level, including 4x4, 2x2 and 1x1.
#[test]
#[ignore = "requires Vulkan or another native compute adapter"]
fn fused_material_mips_match_separate_dispatches_byte_for_byte() {
    let (device, queue) = device();
    let (layout, pipelines) = bake_pipelines(&device);
    let candidate = kernel::Workspace::new(&device);
    let control = kernel::Workspace::new(&device);
    let recipes = program::sample_with_floor(202, 1).unwrap();
    for (surface, size) in [
        (Surface::Fabric, 256),
        (Surface::Floor, 512),
        (Surface::Chrome, 256),
    ] {
        let recipe = recipe::Recipe::new(&recipes[surface as usize], 1).unwrap();
        assert_eq!(recipe.meta[1], size);
        let base = &pipelines[recipe.meta[0] as usize];
        let levels = kernel::levels(size);
        let plane = kernel::plane_bytes(size);
        let bytes = 3 * plane;
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material dispatch parity"),
            size: 2 * bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let binding = candidate.binding(&device, &layout, &recipe);
        candidate.encode(
            &mut encoder,
            [
                base,
                &pipelines[kernel::NORMALS],
                &pipelines[kernel::MIPS],
                &pipelines[kernel::MIP_TAIL],
            ],
            &binding,
            size,
            [None; 3],
        );
        encoder.copy_buffer_to_buffer(&candidate.pixels, 0, &read, 0, bytes);

        let control_binding = control.binding(&device, &layout, &recipe);
        let stride = (std::mem::size_of::<kernel::Dispatch>() as u32)
            .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment);
        let stages = [(0, base), (0, &pipelines[kernel::NORMALS])]
            .into_iter()
            .chain((1..levels.len()).map(|level| (level, &pipelines[kernel::MIPS])));
        for (level, pipeline) in stages {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &control_binding, &[level as u32 * stride]);
            let groups = levels[level].size.div_ceil(8);
            pass.dispatch_workgroups(groups, groups, 1);
        }
        encoder.copy_buffer_to_buffer(&control.pixels, 0, &read, bytes, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = read.slice(..).get_mapped_range().unwrap();
        let (actual, expected) = mapped.split_at(bytes as usize);
        for map in 0..3 {
            for mip in &levels {
                for y in 0..mip.size {
                    let start = (map * plane + mip.offset + u64::from(y * mip.stride)) as usize;
                    let row = start..start + (4 * mip.size) as usize;
                    assert!(
                        actual[row.clone()] == expected[row],
                        "{surface:?} map {map}, {}x{}, row {y}",
                        mip.size,
                        mip.size
                    );
                }
            }
        }
        eprintln!(
            "{surface:?} {size}x{size}: three maps, {} levels, exact",
            levels.len()
        );
    }
}
