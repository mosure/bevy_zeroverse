//! Sample completion of speculative uploads with at most one outstanding signal.
//! The Studio uses it for pacing. Actual asset copies and draws are queue ordered.
use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        renderer::RenderQueue,
        Render, RenderApp, RenderSystems,
    },
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct UploadFence {
    issued: u64,
    completed: Arc<AtomicU64>,
}

impl UploadFence {
    pub(crate) fn ready(&self) -> bool {
        self.completed.load(Ordering::Acquire) == self.issued
    }
    pub(crate) fn issue(&mut self) {
        self.issued += 1;
    }
    fn complete(&self) {
        self.completed.fetch_max(self.issued, Ordering::Release);
    }
    #[cfg(test)]
    pub(crate) fn complete_for_test(&self) {
        self.complete();
    }
}

pub(crate) fn install(app: &mut App) {
    if app.get_sub_app(RenderApp).is_none()
        || app
            .world()
            .get_resource::<crate::app::BevyZeroverseConfig>()
            .is_some_and(|config| config.headless)
    {
        return;
    }
    app.init_resource::<UploadFence>()
        .add_plugins(ExtractResourcePlugin::<UploadFence>::default());
    app.sub_app_mut(RenderApp)
        .add_systems(Render, acknowledge.in_set(RenderSystems::Cleanup));
}

fn acknowledge(fence: Res<UploadFence>, queue: Res<RenderQueue>, mut submitted: Local<u64>) {
    let issued = fence.issued;
    if issued == *submitted {
        return;
    }
    *submitted = issued;
    let snapshot = fence.clone();
    // Extraction snapshots the serial with the assets. Register only after
    // this render frame has submitted its copies, bakes and presentation work.
    queue.on_submitted_work_done(move || snapshot.complete());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_old_studio_upload_completion_cannot_unlock_a_later_batch() {
        let mut fence = UploadFence::default();
        assert!(fence.ready());
        fence.issue();
        let extracted = fence.clone();
        assert!(!fence.ready());
        extracted.complete();
        assert!(fence.ready());
        fence.issue();
        extracted.complete();
        assert!(!fence.ready());
    }
}
