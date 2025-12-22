use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_zeroverse::sample::{OvoxelSample, Sample, View};
use bevy_zeroverse_burn::chunk::save_chunk;
use bevy_zeroverse_burn::compression::Compression;
use bevy_zeroverse_burn::fs::save_sample_to_fs;
use tempfile::TempDir;

struct CountingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            DEALLOCATED.fetch_add(old_size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        let size = layout.size();
        DEALLOCATED.fetch_add(size, Ordering::Relaxed);
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn snapshot() -> (usize, usize) {
    let bytes = ALLOCATED
        .load(Ordering::Relaxed)
        .saturating_sub(DEALLOCATED.load(Ordering::Relaxed));
    let allocs = ALLOC_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(DEALLOC_COUNT.load(Ordering::Relaxed));
    (bytes, allocs)
}

fn make_rgba_bytes(width: u32, height: u32, seed: f32) -> Vec<u8> {
    let pixels = (width * height) as usize;
    let mut rgba: Vec<f32> = Vec::with_capacity(pixels * 4);
    for i in 0..pixels {
        let base = (i as f32 + seed) / pixels.max(1) as f32;
        rgba.extend_from_slice(&[base, base * 0.5, base * 0.25, 1.0]);
    }
    bytemuck::cast_slice(&rgba).to_vec()
}

fn make_sample(width: u32, height: u32) -> Sample {
    let view = View {
        color: make_rgba_bytes(width, height, 0.1),
        depth: make_rgba_bytes(width, height, 0.2),
        normal: make_rgba_bytes(width, height, 0.3),
        optical_flow: make_rgba_bytes(width, height, 0.4),
        position: make_rgba_bytes(width, height, 0.5),
        world_from_view: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        fovy: 1.0,
        near: 0.1,
        far: 10.0,
        time: 0.0,
    };

    let ov = OvoxelSample {
        coords: vec![[0, 0, 0], [1, 0, 0]],
        dual_vertices: vec![[1, 2, 3], [4, 5, 6]],
        intersected: vec![1, 0],
        base_color: vec![[255, 0, 0, 255], [0, 255, 0, 255]],
        semantics: vec![0, 1],
        semantic_labels: vec!["a".to_string(), "b".to_string()],
        resolution: 2,
        aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    };

    Sample {
        views: vec![view],
        view_dim: 1,
        aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        object_obbs: Vec::new(),
        ovoxel: Some(ov),
    }
}

fn run_roundtrip(sample: &Sample, width: u32, height: u32) {
    let dir = TempDir::new().expect("tempdir should be creatable");
    let samples = vec![sample.clone()];
    save_chunk(
        &samples,
        dir.path(),
        0,
        Compression::None,
        width,
        height,
        true,
    )
    .expect("chunk save should succeed");
    save_sample_to_fs(sample, dir.path(), 0, width, height, true)
        .expect("fs save should succeed");
}

#[test]
fn chunk_and_fs_roundtrip_do_not_leak_memory() {
    const WIDTH: u32 = 2;
    const HEIGHT: u32 = 2;
    const MAX_LEAK_BYTES: usize = 64 * 1024;
    const MAX_LEAK_ALLOCS: usize = 64;
    const WARMUP_ROUNDS: usize = 2;
    const CHECK_ROUNDS: usize = 6;

    let sample = make_sample(WIDTH, HEIGHT);

    for _ in 0..WARMUP_ROUNDS {
        run_roundtrip(&sample, WIDTH, HEIGHT);
    }

    let baseline = snapshot();
    let mut max_seen = baseline;

    for _ in 0..CHECK_ROUNDS {
        run_roundtrip(&sample, WIDTH, HEIGHT);
        let now = snapshot();
        if now.0 > max_seen.0 {
            max_seen.0 = now.0;
        }
        if now.1 > max_seen.1 {
            max_seen.1 = now.1;
        }
    }

    let bytes_growth = max_seen.0.saturating_sub(baseline.0);
    let allocs_growth = max_seen.1.saturating_sub(baseline.1);

    assert!(
        bytes_growth <= MAX_LEAK_BYTES,
        "allocated bytes grew by {bytes_growth} (baseline {}, max {})",
        baseline.0,
        max_seen.0
    );
    assert!(
        allocs_growth <= MAX_LEAK_ALLOCS,
        "allocation count grew by {allocs_growth} (baseline {}, max {})",
        baseline.1,
        max_seen.1
    );
}
