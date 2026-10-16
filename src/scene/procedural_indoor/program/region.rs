//! Borrowed primary reconstruction domain. Cache it once for pixel/point queries;
//! it shares the accepted envelope and support/roof functions with construction.
use super::super::{envelope::polygon, layout::IndoorManifest};
use bevy::prelude::*;

pub struct PrimaryRegion<'a> {
    scene: &'a IndoorManifest,
    pub min: Vec2,
    pub max: Vec2,
}
impl IndoorManifest {
    pub fn primary_region(&self) -> PrimaryRegion<'_> {
        let (min, max) = self.primary_room_bounds();
        PrimaryRegion {
            scene: self,
            min,
            max,
        }
    }
}
impl PrimaryRegion<'_> {
    /// X/Z ownership only. Does not invent floor in a concave notch.
    pub fn contains_projection(&self, p: Vec2, clearance: f32) -> bool {
        p.is_finite()
            && clearance.is_finite()
            && clearance >= 0.
            && p.cmpge(self.min + Vec2::splat(clearance)).all()
            && p.cmple(self.max - Vec2::splat(clearance)).all()
            && self
                .scene
                .envelope
                .as_ref()
                .is_none_or(|e| polygon::contains(&e.footprint, p, clearance))
    }
    /// Rendered surface ownership under sloped ceilings, soffits and stepped
    /// floors. Vertical tolerance handles the finite precision of surface hits;
    /// it never admits the sky proxy or neighboring exterior roof.
    pub fn contains_surface(&self, p: Vec3, vertical_tolerance: f32) -> bool {
        vertical_tolerance.is_finite()
            && vertical_tolerance >= 0.
            && p.is_finite()
            && self.contains_projection(p.xz(), 0.)
            && p.y >= self.scene.floor_height(p.xz()) - vertical_tolerance
            && p.y <= self.scene.ceiling_height(p.xz()) + vertical_tolerance
    }
    /// Enclosing architectural shell used by scene AABB / O-voxel crop. This is
    /// a conservative box; exact point membership is contains_surface.
    pub fn shell_bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(
                self.min.x - 0.20,
                self.scene
                    .envelope
                    .as_ref()
                    .map_or(0., |e| e.minimum_floor())
                    - 0.20,
                self.min.y - 0.20,
            ),
            Vec3::new(
                self.max.x + 0.20,
                self.scene.room_size.y + 0.14,
                self.max.y + 0.20,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    #[test]
    fn membership_uses_local_floor_roof_and_excludes_context() {
        for seed in 0..32 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let region = scene.primary_region();
            for r in scene.room_topology().regions.iter().filter(|r| r.primary) {
                for t in &r.triangles {
                    let p = (t[0] + t[1] + t[2]) / 3.;
                    let floor = scene.floor_height(p);
                    let roof = scene.ceiling_height(p);
                    assert!(region.contains_surface(Vec3::new(p.x, floor, p.y), 0.01));
                    assert!(region.contains_surface(Vec3::new(p.x, roof, p.y), 0.01));
                    assert!(!region.contains_surface(Vec3::new(p.x, roof + 0.1, p.y), 0.01));
                    assert!(!region.contains_surface(Vec3::new(p.x, floor - 0.1, p.y), 0.01));
                }
            }
            assert!(!region.contains_surface(Vec3::splat(f32::NAN), 0.01));
            assert!(!region.contains_surface(Vec3::new(0., 100., 0.), 0.01));
        }
    }
}
