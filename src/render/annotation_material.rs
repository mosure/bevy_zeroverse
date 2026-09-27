//! Share equivalent annotation materials so mesh instancing survives mode changes.
//! The cap also bounds scene-dependent position extents in persistent viewers.
use bevy::prelude::*;
use std::collections::HashMap;

pub(crate) struct AnnotationMaterialCache<M: Asset>(HashMap<Vec<u32>, Handle<M>>);
impl<M: Asset> Default for AnnotationMaterialCache<M> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}
impl<M: Asset> AnnotationMaterialCache<M> {
    pub fn get(
        &mut self,
        materials: &mut Assets<M>,
        key: Vec<u32>,
        create: impl FnOnce() -> M,
    ) -> Handle<M> {
        if let Some(handle) = self.0.get(&key) {
            return handle.clone();
        }
        if self.0.len() >= 256 {
            self.0.clear();
        }
        let handle = materials.add(create());
        self.0.insert(key, handle.clone());
        handle
    }
}
