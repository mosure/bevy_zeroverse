//! Bodies and their thin supports are reserved separately. Empty plenum space
//! is not a solid obstacle; compatible service runs may cross at different levels.
use super::*;

impl CeilingModule {
    pub fn obstacles(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        let body = if self.detail.is_none()
            || self.suspension.is_none()
            || self.system == CeilingSystem::Ventilation
        {
            self.bounds()
        } else {
            self.body_bounds()
        };
        std::iter::once(body).chain(self.suspension.iter().flatten().map(|s| {
            (
                s.start.min(s.end) - Vec3::splat(0.012),
                s.start.max(s.end) + Vec3::splat(0.012),
            )
        }))
    }
    pub(super) fn validate_suspension(&self) -> Result<(), String> {
        let (lo, hi) = self.body_bounds();
        if let Some(supports) = &self.suspension {
            if supports.len() > 32
                || supports.iter().any(|s| {
                    !s.start.is_finite()
                        || !s.end.is_finite()
                        || !s.start.cmpge(lo - Vec3::splat(0.003)).all()
                        || !s.start.cmple(hi + Vec3::splat(0.003)).all()
                        || s.start.xz().distance_squared(s.end.xz()) > 1e-8
                        || s.end.y < s.start.y
                        || s.end.y > self.support_top_m + 0.001
                })
            {
                return Err("invalid ceiling suspension".into());
            }
        }
        Ok(())
    }
}
impl CeilingInstallation {
    pub fn obstacles(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        self.modules
            .iter()
            .flat_map(CeilingModule::obstacles)
            .chain(
                self.framing
                    .iter()
                    .flat_map(framing::CeilingFraming::bounds),
            )
    }
    pub fn attachment_height(&self, point: Vec3, roof: f32) -> f32 {
        self.framing
            .as_ref()
            .map_or(roof, |f| f.attachment_height(point, roof))
    }
}

pub(super) fn prepare(
    scene: &IndoorManifest,
    frame: Option<&framing::CeilingFraming>,
    m: &mut CeilingModule,
) {
    m.suspension = Some(
        construction::anchors(scene, m)
            .into_iter()
            .filter_map(|start| {
                let roof = scene.ceiling_height(start.xz());
                let top = frame.map_or(roof, |f| f.attachment_height(start, roof));
                (top - start.y > 0.003).then_some(Suspension {
                    start,
                    end: start.with_y(top - 0.003),
                })
            })
            .collect(),
    );
}
pub(super) fn conflicts(a: &CeilingModule, b: &CeilingModule, margin: f32) -> bool {
    a.obstacles()
        .any(|a| b.obstacles().any(|b| overlap(a, b, margin)))
}
pub(super) fn frame_clear(m: &CeilingModule, f: Option<&framing::CeilingFraming>) -> bool {
    let body = if m.detail.is_none() || m.system == CeilingSystem::Ventilation {
        m.bounds()
    } else {
        m.body_bounds()
    };
    f.is_none_or(|f| f.bounds().all(|b| !overlap(body, b, 0.008)))
}
