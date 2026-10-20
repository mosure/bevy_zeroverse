//! Bounded ceiling installations shared by construction, placement and cameras.
//! Recipes describe real occupied volumes; meshes are batched by existing PBR roles.
mod construction;
mod legacy;
mod profile;
mod sampling;
pub(crate) use sampling::highest_floor;
#[cfg(test)]
mod tests;
use super::*;
pub use profile::{CeilingDetail, CeilingShape, TrussWeb};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CeilingSystem {
    AcousticCloud,
    Baffles,
    TimberSlats,
    RoundDuct,
    RectangularDuct,
    SprinklerMain,
    Truss,
    SuspendedTiles,
    OpenCell,
    Conduit,
    CableTray,
    Coffers,
    Ventilation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeilingModule {
    pub system: CeilingSystem,
    pub center: Vec3,
    /// Body reserve in room-local coordinates; `bounds` also includes suspension.
    pub size: Vec3,
    pub support_top_m: f32,
    pub pitch_m: f32,
    pub thickness_m: f32,
    pub seed: u64,
    /// Missing in older manifests: retain their original geometry exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<CeilingDetail>,
}
impl CeilingModule {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let (lo, mut hi) = self.body_bounds();
        hi.y = hi.y.max(self.support_top_m);
        (lo, hi)
    }
    pub(crate) fn headroom(&self, scene: &IndoorManifest) -> f32 {
        let (lo, hi) = self.body_bounds();
        lo.y - sampling::highest_floor(scene, lo.xz(), hi.xz())
    }
    fn body_bounds(&self) -> (Vec3, Vec3) {
        (self.center - self.size * 0.5, self.center + self.size * 0.5)
    }
    /// Production assembly, also useful for isolated geometry qualification.
    pub fn geometry(&self, scene: &IndoorManifest) -> Assembly {
        let mut a = Assembly::default();
        construction::build(&mut a, scene, self);
        a
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CeilingInstallation {
    pub exposed_slab: bool,
    pub modules: Vec<CeilingModule>,
}
impl CeilingInstallation {
    pub fn sample(scene: &IndoorManifest) -> Self {
        sampling::sample(scene)
    }
    pub fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        if self.modules.len() > 96 {
            return Err("unbounded ceiling installation".into());
        }
        let reservations = sampling::Reservations::new(scene);
        for (i, module) in self.modules.iter().enumerate() {
            if !module.center.is_finite()
                || !module.size.is_finite()
                || !module.support_top_m.is_finite()
                || module.size.xz().min_element() < 0.08
                || module.size.xz().max_element() < 0.40
                || !(0.10..=0.90).contains(&module.size.y)
                || module.size.max_element() > 40.
                || !(0.14..=3.2).contains(&module.pitch_m)
                || !(0.008..=0.12).contains(&module.thickness_m)
                || module
                    .detail
                    .as_ref()
                    .is_some_and(|p| !p.valid(module.system))
                || (module.detail.is_none()
                    && matches!(
                        module.system,
                        CeilingSystem::SuspendedTiles
                            | CeilingSystem::OpenCell
                            | CeilingSystem::Conduit
                            | CeilingSystem::CableTray
                            | CeilingSystem::Coffers
                            | CeilingSystem::Ventilation
                    ))
                || !sampling::admissible(scene, module, &reservations)
                || self.modules[..i]
                    .iter()
                    .any(|m| overlap(m.bounds(), module.bounds(), 0.012))
            {
                return Err(format!("invalid or intersecting ceiling module {i}"));
            }
        }
        Ok(())
    }
}

fn overlap((a, b): (Vec3, Vec3), (c, d): (Vec3, Vec3), margin: f32) -> bool {
    a.cmplt(d + Vec3::splat(margin)).all() && b.cmpgt(c - Vec3::splat(margin)).all()
}

pub(crate) fn build(a: &mut Assembly, scene: &IndoorManifest) {
    if let Some(installation) = &scene.ceiling_installation {
        for module in &installation.modules {
            construction::build(a, scene, module);
        }
    }
}
