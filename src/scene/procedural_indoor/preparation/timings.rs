//! Wall-clock accounting, separate from deterministic scene metadata.
//!
//! `wall` partitions one preparation future's lifetime. Scoped parallel work is
//! counted once in `parallel_assets_seconds`; legacy branch timers and geometry
//! job totals are diagnostics and must never be added to that partition.
use bevy::prelude::Resource;
use serde::Serialize;

#[derive(Resource, Debug, Default, Clone, Serialize)]
pub struct PreparationTimings {
    pub layout_seconds: f64,
    /// Camera sampling only (layout validation has its own wall stage).
    pub cameras_seconds: f64,
    pub materials_seconds: f64,
    pub material_transport_seconds: f64,
    pub material_transport_mesh_seconds: f64,
    pub gi_setup_seconds: f64,
    pub transport_seconds: f64,
    /// Disjoint transport stages; service intervals can overlap mesh/material work.
    pub transport_work: TransportTimings,
    pub environment_seconds: f64,
    pub gpu_gi_setup_seconds: f64,
    pub geometry_seconds: f64,
    pub meshes_seconds: f64,
    /// Main-world asset insertion and ECS command recording, not deferred command
    /// execution, speculative upload, GPU completion or end-to-end preparation.
    pub asset_insertion_seconds: f64,
    pub wall: PreparationWall,
    pub geometry_jobs: GeometryJobs,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TransportTimings {
    pub emission_seconds: f64,
    pub probe_tree_seconds: f64,
    pub shadow_tree_seconds: f64,
    pub triangles: usize,
    pub probe_nodes: usize,
}

/// Non-overlapping monotonic wall intervals in execution order. Includes pool
/// waiting and (on Wasm) cooperative yields, not CPU cycles or summed core time.
/// GPU map/GI execution, asset installation and motion inference are downstream.
#[derive(Debug, Default, Clone, Serialize)]
pub struct PreparationWall {
    pub layout_seconds: f64,
    pub cameras_seconds: f64,
    pub validation_seconds: f64,
    pub motion_planning_seconds: f64,
    pub assembly_seconds: f64,
    pub material_selection_seconds: f64,
    pub parallel_assets_seconds: f64,
    pub transport_binding_seconds: f64,
    pub environment_seconds: f64,
    pub gi_request_seconds: f64,
    /// Wasm plans and realizes meshes after transport binding; native performs
    /// this work inside the parallel asset interval.
    pub serial_meshes_seconds: f64,
    pub group_binding_seconds: f64,
    pub other_seconds: f64,
    pub total_seconds: f64,
}
impl PreparationWall {
    pub fn measured_seconds(&self) -> f64 {
        self.layout_seconds
            + self.cameras_seconds
            + self.validation_seconds
            + self.motion_planning_seconds
            + self.assembly_seconds
            + self.material_selection_seconds
            + self.parallel_assets_seconds
            + self.transport_binding_seconds
            + self.environment_seconds
            + self.gi_request_seconds
            + self.serial_meshes_seconds
            + self.group_binding_seconds
    }

    pub(super) fn finish(&mut self, total_seconds: f64) {
        self.total_seconds = total_seconds;
        // Do not clamp: a negative residual reveals overlapping instrumentation
        // and must be rejected by qualification instead of concealed.
        self.other_seconds = total_seconds - self.measured_seconds();
    }
}

/// Per-builder elapsed service time, excluding time queued before it starts.
/// Calls run concurrently and can contain nested work; sum is not CPU core time.
#[derive(Debug, Default, Clone, Serialize)]
pub struct JobTiming {
    pub count: usize,
    pub sum_seconds: f64,
    pub max_seconds: f64,
}
impl JobTiming {
    pub(super) fn add(&mut self, seconds: f64) {
        self.count += 1;
        self.sum_seconds += seconds;
        self.max_seconds = self.max_seconds.max(seconds);
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct GeometryJobs {
    pub architecture: JobTiming,
    pub objects: JobTiming,
    pub humans: JobTiming,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_branches_are_not_added_to_the_wall_partition() {
        let mut timing = PreparationTimings {
            materials_seconds: 0.7,
            meshes_seconds: 0.5,
            transport_seconds: 0.2,
            wall: PreparationWall {
                layout_seconds: 0.1,
                parallel_assets_seconds: 0.75,
                ..Default::default()
            },
            ..Default::default()
        };
        timing.wall.finish(1.0);
        assert!((timing.wall.other_seconds - 0.15).abs() < 1e-12);
        timing.wall.finish(0.5);
        assert!(timing.wall.other_seconds < 0.0, "overlap must stay visible");
    }
}
