//! Select hand-operated lights within an aperture in wall-local metres. The same
//! installation drives meshes, daylight metering and exported window statistics.
use super::*;

const FIT_GAP_M: f32 = 0.004;
const LOW_HANDLE_M: f32 = 0.68;
const HIGH_HANDLE_M: f32 = 1.72;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SashStock {
    /// Zero disables manual lights. Otherwise one eligible light serves the
    /// opening and this probability admits additional eligible siblings.
    pub operating_share: f32,
    /// Preferred wall-local height of a casement latch, in metres.
    pub handle_height_m: f32,
    pub lever_length_m: f32,
    pub grip_projection_m: f32,
    /// Maximum tip travel from the closed glazing plane, in metres.
    pub swing_limit_m: f32,
    pub finish: Surface,
}

impl SashStock {
    pub(super) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"sashware"));
        Self {
            operating_share: r.random_range(0.22..0.78),
            handle_height_m: r.random_range(1.02..1.46),
            lever_length_m: r.random_range(0.064..0.111),
            grip_projection_m: r.random_range(0.022..0.044),
            swing_limit_m: r.random_range(0.065..0.14),
            finish: [Surface::Metal, Surface::Chrome, Surface::Plastic][r.random_range(0..3)],
        }
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.).contains(&self.operating_share)
            || !(0.95..=1.52).contains(&self.handle_height_m)
            || !(0.055..=0.12).contains(&self.lever_length_m)
            || !(0.018..=0.05).contains(&self.grip_projection_m)
            || !(0.04..=0.14).contains(&self.swing_limit_m)
            || !matches!(
                self.finish,
                Surface::Metal | Surface::Chrome | Surface::Plastic
            )
        {
            return Err("invalid window sash hardware".into());
        }
        Ok(())
    }
}

/// Hinge side as seen from inside the room. Casements and bottom-hung
/// hoppers open inward. A top-hung awning sheds water to the exterior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SashHinge {
    Fixed,
    LeftCasement,
    RightCasement,
    TopAwning,
    BottomHopper,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaneInstallation {
    /// Outside edges of the light, including the fitting clearance.
    pub min: Vec2,
    pub max: Vec2,
    pub hinge: SashHinge,
    /// Mounting-plate centre in the closed orientation, relative to the wall origin.
    pub handle: Option<Vec2>,
    /// Realized static opening, after treatment clearance and sibling selection.
    pub opening_fraction: f32,
    pub swing_radians: f32,
}

impl PaneInstallation {
    fn fixed(min: Vec2, max: Vec2) -> Self {
        Self {
            min,
            max,
            hinge: SashHinge::Fixed,
            handle: None,
            opening_fraction: 0.,
            swing_radians: 0.,
        }
    }

    pub(super) fn bounds(&self, fitted: bool) -> (Vec2, Vec2) {
        let gap = if fitted && self.hinge != SashHinge::Fixed {
            Vec2::splat(FIT_GAP_M)
        } else {
            Vec2::ZERO
        };
        (self.min + gap, self.max - gap)
    }

    pub(super) fn pose(&self, f: &Facade, face: f32, back: f32) -> Transform {
        let (lo, hi) = self.bounds(true);
        // Place the pin on the face facing the sweep. The 4 mm sash joint
        // accepts the opposite corner throughout the small permitted swing.
        let (pivot, rotation) = match self.hinge {
            SashHinge::Fixed => (Vec3::ZERO, Quat::IDENTITY),
            SashHinge::LeftCasement => (
                Vec3::new(lo.x, 0., face),
                Quat::from_rotation_y(-self.swing_radians),
            ),
            SashHinge::RightCasement => (
                Vec3::new(hi.x, 0., face),
                Quat::from_rotation_y(self.swing_radians),
            ),
            SashHinge::TopAwning => (
                Vec3::new(0., hi.y, back),
                Quat::from_rotation_x(self.swing_radians),
            ),
            SashHinge::BottomHopper => (
                Vec3::new(0., lo.y, face),
                Quat::from_rotation_x(self.swing_radians),
            ),
        };
        Transform::from_translation(pivot - rotation * pivot - Vec3::Z * f.recess)
            .with_rotation(rotation)
    }

    fn set_opening(&mut self, fraction: f32, limit: f32) {
        let span = self.max - self.min - Vec2::splat(2. * FIT_GAP_M);
        let travel = match self.hinge {
            SashHinge::LeftCasement | SashHinge::RightCasement => span.x,
            _ => span.y,
        };
        self.opening_fraction = fraction;
        self.swing_radians = fraction * (limit / travel).min(0.30).asin();
    }

    fn candidate(&self, stock: &SashStock, width: f32, r: &mut impl Rng) -> Option<Self> {
        let lo = self.min + FIT_GAP_M;
        let hi = self.max - FIT_GAP_M;
        let span = hi - lo;
        if span.min_element() < 0.32 || span.element_product() > 2.0 {
            return None;
        }
        let center = (lo + hi) * 0.5;
        let reach = LOW_HANDLE_M..=HIGH_HANDLE_M;
        let low = (lo.y + stock.lever_length_m + 0.035).max(LOW_HANDLE_M);
        let high = (hi.y - 0.055).min(HIGH_HANDLE_M);
        let side = span.x <= 1.28 && span.y <= 2.12 && low <= high;
        let half_rail = width * 0.5;
        let horizontal = span.x <= 1.85 && span.y <= 1.22;
        let awning = horizontal && span.y >= 0.45 && reach.contains(&(lo.y + half_rail));
        let hopper = horizontal && reach.contains(&(hi.y - half_rail));
        let hinge = if side && (!(awning || hopper) || r.random_bool(0.59)) {
            if r.random_bool(0.5) {
                SashHinge::LeftCasement
            } else {
                SashHinge::RightCasement
            }
        } else if awning && (!hopper || r.random_bool(0.60)) {
            SashHinge::TopAwning
        } else if hopper {
            SashHinge::BottomHopper
        } else {
            return None;
        };
        let handle = match hinge {
            SashHinge::LeftCasement | SashHinge::RightCasement => {
                let y = (stock.handle_height_m + r.random_range(-0.095..0.095)).clamp(low, high);
                Vec2::new(
                    if hinge == SashHinge::LeftCasement {
                        hi.x - half_rail
                    } else {
                        lo.x + half_rail
                    },
                    y,
                )
            }
            _ => Vec2::new(
                center.x + span.x * r.random_range(-0.15..0.15),
                if hinge == SashHinge::TopAwning {
                    lo.y + half_rail
                } else {
                    hi.y - half_rail
                },
            ),
        };
        Some(Self {
            hinge,
            handle: Some(handle),
            ..*self
        })
    }
}

impl Facade {
    /// Resolve the fitted lights without drawing from room, camera or furnishing RNG streams.
    pub fn pane_installations(&self, opening: &WindowOpening) -> Vec<PaneInstallation> {
        let (columns, levels) = self.divisions(opening);
        self.installations_at(opening, &columns, &levels)
    }

    pub(super) fn installations_at(
        &self,
        opening: &WindowOpening,
        columns: &[f32],
        levels: &[f32],
    ) -> Vec<PaneInstallation> {
        let mut panes = Vec::with_capacity((columns.len() - 1) * (levels.len() - 1));
        panes.extend(
            self.rectangles_at(columns, levels)
                .map(|(lo, hi)| PaneInstallation::fixed(lo, hi)),
        );
        let Some(glass) = opening.glazing.as_ref().filter(|_| opening.operable) else {
            return panes;
        };
        let opening_fraction = if self.allows_open_sash() {
            glass.opening
        } else {
            0.
        };
        let Some((j, stock)) = self
            .joinery
            .as_ref()
            .and_then(|j| j.sash_stock.as_ref().map(|stock| (j, stock)))
        else {
            for pane in &mut panes {
                let span = pane.max - pane.min;
                let width = self
                    .joinery
                    .as_ref()
                    .map_or(0.021, |j| j.sash_width_m)
                    .min(span.min_element() * 0.09);
                pane.hinge = SashHinge::BottomHopper;
                pane.handle = Some(Vec2::new(
                    pane.max.x - width * 0.5,
                    (pane.min.y + pane.max.y) * 0.5,
                ));
                pane.opening_fraction = opening_fraction;
                pane.swing_radians = opening_fraction * (0.14 / span.y).min(0.20).asin();
            }
            return panes;
        };
        if stock.operating_share == 0. {
            return panes;
        }
        let mut r = stream(glass.identity, u64::from_le_bytes(*b"sashlite"));
        let mut first: Option<(f32, usize, PaneInstallation)> = None;
        for (i, pane) in panes.iter_mut().enumerate() {
            let width = fitted_rail_width(j, pane.max - pane.min - 2. * FIT_GAP_M);
            if let Some(mut candidate) = pane.candidate(stock, width, &mut r) {
                let priority: f32 = r.random();
                if first.as_ref().is_none_or(|v| priority < v.0) {
                    first = Some((priority, i, candidate));
                }
                if priority < stock.operating_share {
                    // Independently latched siblings can stay shut or sit at a
                    // smaller angle while the principal operating light is open.
                    let fraction = if r.random_bool(0.3) {
                        0.
                    } else {
                        r.random_range(0.35..1.)
                    };
                    candidate.set_opening(opening_fraction * fraction, stock.swing_limit_m);
                    *pane = candidate;
                }
            }
        }
        if let Some((_, i, mut lead)) = first {
            lead.set_opening(opening_fraction, stock.swing_limit_m);
            panes[i] = lead;
        }
        panes
    }
}

pub(super) fn fitted_rail_width(j: &WindowJoinery, span: Vec2) -> f32 {
    (0.022 + j.sash_width_m).min(span.min_element() * 0.09)
}

#[cfg(test)]
mod tests;
