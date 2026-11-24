use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::CeilingGrid, objects::equipment::pendant::Canopy,
};

/// Installed height of a plumb product. Only the small fixing boss needs a
/// bearing surface at the apex. A soffit edge cannot carry half a canopy.
pub(crate) fn canopy_support_height(
    scene: &IndoorManifest,
    p: Vec2,
    canopy: Canopy,
) -> Option<f32> {
    if let Some(grid) = CeilingGrid::for_scene(scene) {
        return grid.fixture_support(scene, p, canopy.radius_m);
    }
    let Some(e) = &scene.envelope else {
        return Some(scene.room_size.y);
    };
    let (lo, hi) = (p - canopy.radius_m, p + canopy.radius_m);
    if e.soffits.iter().any(|patch| {
        patch.overlaps(lo, hi) && !(patch.min.cmple(lo).all() && patch.max.cmpge(hi).all())
    }) {
        return None;
    }
    // Minimum height of the planar roof over the circular bearing face.
    let roof = e.roof_height(scene.room_size, p)
        - (e.ceiling_drop / scene.room_size.xz()).length() * canopy.pad_radius_m;
    Some(scene.ceiling_height(p).min(roof))
}
