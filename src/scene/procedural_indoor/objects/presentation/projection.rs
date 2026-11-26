//! Projection screen construction and the exact optical target surface.
use super::*;
mod construction;
mod deployment;
mod legacy;
mod product;
pub(crate) use deployment::selected as deployment_selected;
pub(crate) use product::dimensions;
pub use product::{Coating, Deployment, Installation, Mechanism, ScreenFormat};
pub(crate) const INSTALLED: u32 = 1 << 8;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScreenProgram {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installation: Option<Installation>,
    pub framed: bool,
    pub drop_fraction: f32,
    pub case_height_m: f32,
    pub case_depth_m: f32,
    pub cap_width_m: f32,
    pub sheet_width_m: f32,
    pub sheet_center_x_m: f32,
    pub sheet_bottom_m: f32,
    pub sheet_top_m: f32,
    pub sheet_front_m: f32,
    pub bar_radius_m: f32,
    pub bar_width_m: f32,
    pub surface_lo: Vec3,
    pub surface_hi: Vec3,
    pub housing: Surface,
    pub trim: Surface,
    pub mount_x_m: [f32; 2],
}
impl ScreenProgram {
    pub fn target_local(&self) -> Vec3 {
        (self.surface_lo + self.surface_hi) * 0.5
    }
    /// A projector can be paired only with a fully exposed image. Archived
    /// installations retain their recorded throw and target.
    pub fn projection_target_local(&self) -> Option<Vec3> {
        self.installation
            .as_ref()
            .is_none_or(|i| i.deployment == Deployment::Presented)
            .then(|| self.target_local())
    }
    pub(crate) fn viewing_target_local(&self) -> Vec3 {
        match &self.installation {
            Some(i) if i.deployment != Deployment::Presented => self
                .target_local()
                .with_y(self.sheet_top_m - 0.5 * i.full_image_height_m),
            _ => self.target_local(),
        }
    }
    pub fn image_width_m(&self) -> f32 {
        self.surface_hi.x - self.surface_lo.x
    }
}

pub fn parameters(o: &IndoorObject) -> ScreenProgram {
    if o.variant & INSTALLED == 0 {
        legacy::parameters(o)
    } else {
        product::parameters(o)
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters(o);
    if let Some(i) = &p.installation {
        construction::build(a, o, &p, i, version);
    } else {
        legacy::build(a, o, &p);
    }
}

#[cfg(test)]
mod tests;
