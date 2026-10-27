//! Roof-connected structure. Members follow the recorded roof plane; primary,
//! secondary and rib levels meet rather than occupying one suspended patch.
mod geometry;
#[cfg(test)]
mod qualification;
mod sampling;
#[cfg(test)]
mod tests;
use super::*;

pub const MAX_MEMBERS: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameRole {
    Rafter,
    Purlin,
    RoofRib,
    Bearing,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameSection {
    Solid,
    ISection,
    OpenWeb,
    Fluted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMember {
    /// Upper quad, in perimeter order; X/Z edges remain axis aligned.
    pub top: [Vec3; 4],
    pub depth: f32,
    pub role: FrameRole,
    pub section: FrameSection,
    pub material: Surface,
    pub finish: u8,
    pub web_pitch: f32,
}
impl FrameMember {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let lo = self
            .top
            .into_iter()
            .fold(Vec3::splat(f32::INFINITY), Vec3::min)
            - Vec3::Y * self.depth;
        let hi = self
            .top
            .into_iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
        let padding = if self.section == FrameSection::OpenWeb {
            self.top[0].distance(self.top[3]) * 0.13
        } else {
            0.
        };
        (
            lo - Vec3::new(padding, 0., padding),
            hi + Vec3::new(padding, 0., padding),
        )
    }
    pub fn point(&self, t: f32, u: f32, drop: f32) -> Vec3 {
        self.top[0]
            .lerp(self.top[1], t)
            .lerp(self.top[3].lerp(self.top[2], t), u)
            - Vec3::Y * drop
    }
    fn top_y_at(&self, p: Vec2) -> f32 {
        let a = self.top[0];
        let along = self.top[1].xz() - a.xz();
        let across = self.top[3].xz() - a.xz();
        let d = p - a.xz();
        self.point(
            d.dot(along) / along.length_squared(),
            d.dot(across) / across.length_squared(),
            0.,
        )
        .y
    }
    pub fn attachment_y(&self, p: Vec3) -> Option<f32> {
        if self.section == FrameSection::OpenWeb {
            // End posts and diagonal tubes can project past the upper quad.
            // Match the actual padded footprint before the triangle query.
            let (lo, hi) = self.bounds();
            return (p.xz().cmpge(lo.xz()).all() && p.xz().cmple(hi.xz()).all())
                .then(|| geometry::attachment(self, p))
                .flatten();
        }
        let a = self.top[0];
        let along = self.top[1].xz() - a.xz();
        let across = self.top[3].xz() - a.xz();
        let d = p.xz() - a.xz();
        let t = d.dot(along) / along.length_squared();
        let u = d.dot(across) / across.length_squared();
        if !(0.0..=1.).contains(&t) || !(0.0..=1.).contains(&u) {
            return None;
        }
        let drop = if self.section == FrameSection::Fluted {
            self.depth * (u / 0.22).min((1. - u) / 0.22).min(1.)
        } else {
            self.depth
        };
        let y = self.point(t, u, drop).y;
        (y > p.y + 0.003).then_some(y)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeilingFraming {
    pub roof_material: Surface,
    pub members: Vec<FrameMember>,
}
impl CeilingFraming {
    pub(super) fn sample(scene: &IndoorManifest, reserved: &super::sampling::Reservations) -> Self {
        sampling::sample(scene, reserved)
    }
    pub fn geometry(&self) -> Assembly {
        geometry::build(self)
    }
    pub fn bounds(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        self.members.iter().map(FrameMember::bounds)
    }
    pub fn attachment_height(&self, point: Vec3, roof: f32) -> f32 {
        self.members
            .iter()
            .filter_map(|m| m.attachment_y(point))
            .fold(roof, f32::min)
    }
    pub fn deepest_drop(&self, scene: &IndoorManifest) -> f32 {
        self.members
            .iter()
            .filter(|m| m.role == FrameRole::Rafter)
            .map(|m| scene.ceiling_height(m.top[0].xz()) - m.top[0].y + m.depth)
            .fold(0.1, f32::max)
    }
    /// Contact graph with a 5 mm joint allowance for fastening layers.
    pub(super) fn connected(&self) -> Vec<bool> {
        let bounds: Vec<_> = self.bounds().collect();
        let mut reached: Vec<_> = self
            .members
            .iter()
            .map(|m| matches!(m.role, FrameRole::RoofRib | FrameRole::Bearing))
            .collect();
        let mut queue: Vec<_> = reached
            .iter()
            .enumerate()
            .filter_map(|(i, &yes)| yes.then_some(i))
            .collect();
        while let Some(i) = queue.pop() {
            for j in 0..reached.len() {
                if !reached[j] && overlap(bounds[i], bounds[j], 0.005) {
                    // All layers follow the roof plane. Compare their local
                    // vertical intervals at the shared footprint, not global
                    // Y bounds: parallel sloped beams can have overlapping
                    // AABBs while remaining separated along their entire span.
                    let p = (bounds[i].0.xz().max(bounds[j].0.xz())
                        + bounds[i].1.xz().min(bounds[j].1.xz()))
                        * 0.5;
                    let a = self.members[i].top_y_at(p);
                    let b = self.members[j].top_y_at(p);
                    if a - self.members[i].depth <= b + 0.005
                        && b - self.members[j].depth <= a + 0.005
                    {
                        reached[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
        reached
    }
    pub(super) fn validate_integrity(&self) -> Result<(), String> {
        if self.members.len() > MAX_MEMBERS
            || !matches!(
                self.roof_material,
                Surface::Concrete | Surface::Metal | Surface::Wood
            )
        {
            return Err("invalid ceiling frame program".into());
        }
        for m in &self.members {
            let (lo, hi) = m.bounds();
            let length = m.top[0].xz().distance(m.top[1].xz());
            let width = m.top[0].xz().distance(m.top[3].xz());
            if !lo.is_finite()
                || !hi.is_finite()
                || !m.top.iter().all(|p| p.is_finite())
                || !(0.012..=1.6).contains(&m.depth)
                || !(0.04..=0.50).contains(&width)
                || !(0.12..=40.).contains(&length)
                || m.finish >= 12
                || !(0.25..=1.6).contains(&m.web_pitch)
                || !matches!(
                    m.material,
                    Surface::Wood | Surface::Metal | Surface::Paint | Surface::Concrete
                )
                || (m.top[0] + m.top[2] - m.top[1] - m.top[3]).length() > 0.001
                || (m.top[1].xz() - m.top[0].xz())
                    .dot(m.top[3].xz() - m.top[0].xz())
                    .abs()
                    > 1e-4
            {
                return Err("invalid ceiling frame member".into());
            }
        }
        Ok(())
    }
    pub(super) fn placement_valid(
        &self,
        scene: &IndoorManifest,
        reserved: &super::sampling::Reservations,
    ) -> bool {
        self.connected().iter().all(|&connected| connected)
            && self.members.iter().all(|m| {
                sampling::admissible(scene, m, reserved)
                    && (!matches!(m.role, FrameRole::RoofRib | FrameRole::Bearing)
                        || m.top
                            .iter()
                            .all(|p| (scene.ceiling_height(p.xz()) - p.y - 0.001).abs() < 2e-5))
            })
    }
}
