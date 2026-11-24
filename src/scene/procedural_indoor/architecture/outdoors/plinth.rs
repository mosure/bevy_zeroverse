//! The weather-resistant base of a neighbouring building follows its exposed
//! ground-floor perimeter. Corner profiles meet at a mitre. Jambs have square
//! returns to the wall, leaving the actual door threshold clear.
use super::*;
mod mesh;

pub(super) const FIRST_VERSION: u32 = 197;
const FOUNDATION: f32 = -0.075;
const JAMB_GAP: f32 = 0.012;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(u8)]
enum Stock {
    ContinuousWall,
    Render,
    Precast,
    FiredClay,
    Flashing,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct Course {
    stock: Stock,
    height: f32,
    projection: f32,
    slope: f32,
    sill_gap: f32,
    module: Option<f32>,
    joint: f32,
}

impl Course {
    pub(super) fn sample(seed: u64, cladding: Surface) -> Self {
        let mut r = stream(seed, 0x4241_5345_434f_5552);
        let draw: f32 = r.random();
        let stock = match cladding {
            _ if draw < 0.12 => Stock::ContinuousWall,
            Surface::WoodEdge | Surface::Metal if draw < 0.64 => Stock::Flashing,
            Surface::Paint | Surface::Concrete if draw < 0.53 => Stock::Render,
            Surface::Terracotta if draw < 0.62 => Stock::FiredClay,
            _ if (0.74..0.85).contains(&draw) => Stock::FiredClay,
            _ => Stock::Precast,
        };
        let (height, projection, module) = match stock {
            Stock::ContinuousWall => (0., 0., None),
            Stock::Render => (
                r.random_range(0.08..0.38),
                r.random_range(0.009..0.030),
                None,
            ),
            Stock::Precast => (
                r.random_range(0.15..0.56),
                r.random_range(0.035..0.115),
                r.random_bool(0.62).then(|| r.random_range(1.0..2.4)),
            ),
            Stock::FiredClay => (
                r.random_range(0.10..0.37),
                r.random_range(0.018..0.050),
                Some(r.random_range(0.36..0.88)),
            ),
            Stock::Flashing => (
                r.random_range(0.07..0.28),
                r.random_range(0.020..0.062),
                None,
            ),
        };
        Self {
            stock,
            height,
            projection,
            slope: r.random_range(0.085..0.24),
            module,
            joint: r.random_range(0.004..0.009),
            sill_gap: r.random_range(0.010..0.045),
        }
    }

    fn surface(&self) -> Option<Surface> {
        match self.stock {
            Stock::ContinuousWall => None,
            Stock::Render => Some(Surface::Paint),
            Stock::Precast => Some(Surface::Concrete),
            Stock::FiredClay => Some(Surface::Terracotta),
            Stock::Flashing => Some(Surface::Metal),
        }
    }

    /// Elevation relative to the doorway threshold/sidewalk. Keep the highest
    /// point of the weathering below the lowest window's outer frame.
    pub(super) fn top(&self, b: &Building) -> Option<f32> {
        self.surface().map(|_| {
            self.height
                .min(FOUNDATION + b.pitch * b.sill - b.frame_width - self.sill_gap)
        })
    }

    // At the re-entrant corner a mitre projects along the returning wall.
    // Leave room for the side wing's actual recess and the nearest doorway.
    fn fitted_projection(&self, b: &Building, size: Vec2) -> f32 {
        if b.split >= 1. {
            return self.projection;
        }
        let wing = size * Vec2::new(1. - b.split, b.wing_depth);
        let pier = b
            .window_installation
            .corner_pier(b.window_installation.recess(wing))
            - JAMB_GAP;
        self.projection
            .min(pier * 0.85)
            .min((size.y - wing.y) * 0.85)
    }

    pub(super) fn measure(&self, b: &Building, size: Vec2, out: &mut Vec<(&'static str, f32)>) {
        out.push(("exterior_plinth_stock", self.stock as u8 as f32));
        if let Some(top) = self.top(b) {
            out.push(("exterior_plinth_top_m", top));
            out.push((
                "exterior_plinth_projection_m",
                self.fitted_projection(b, size),
            ));
            if let Some(module) = self.module {
                out.push(("exterior_plinth_nominal_unit_length_m", module));
            }
        }
    }

    fn section(&self, top: f32, nose: f32) -> ([Vec2; 4], usize) {
        let shoulder = top - nose * self.slope;
        let (body, rise) = match self.stock {
            Stock::Flashing => (0.62 * nose, 0.40 * nose),
            Stock::Precast => (0.80 * nose, 0.48 * nose),
            _ => (nose, 0.),
        };
        let profile = [
            Vec2::new(body, FOUNDATION),
            Vec2::new(
                body,
                if rise > 0. {
                    shoulder - rise
                } else {
                    FOUNDATION
                },
            ),
            Vec2::new(nose, shoulder),
            Vec2::new(0., top),
        ];
        (profile, usize::from(rise == 0.))
    }
}

#[cfg(test)]
mod tests;
