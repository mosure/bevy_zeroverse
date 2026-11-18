//! A lift cord picks up blades from below. The suspended portion remains on its
//! ladder pitch and the picked-up blades nest on the moving bottom rail.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlatStock {
    pub depth_m: f32,
    pub thickness_m: f32,
    pub crown_m: f32,
    pub nesting_gap_m: f32,
}

impl SlatStock {
    pub(super) fn sample(r: &mut impl Rng, material: Surface) -> Self {
        let (depth_m, thickness_m, crown_ratio) = match material {
            Surface::Metal => {
                let depth = if r.random_bool(0.43) {
                    r.random_range(0.024..0.031)
                } else {
                    r.random_range(0.044..0.055)
                };
                (
                    depth,
                    r.random_range(0.00030..0.00065),
                    r.random_range(0.036..0.065),
                )
            }
            Surface::WoodEdge => (
                r.random_range(0.045..0.065),
                r.random_range(0.0026..0.0042),
                r.random_range(0.001..0.008),
            ),
            _ => (
                r.random_range(0.036..0.064),
                r.random_range(0.0015..0.0036),
                r.random_range(0.010..0.038),
            ),
        };
        Self {
            depth_m,
            thickness_m,
            crown_m: depth_m * crown_ratio,
            nesting_gap_m: r.random_range(0.00025..0.00075),
        }
    }

    pub(super) fn valid(self, pitch: f32) -> bool {
        (0.80..=1.06).contains(&(pitch / self.depth_m))
            && self.packed_pitch(1.2) < pitch
            && (0.023..=0.066).contains(&self.depth_m)
            && (0.00025..=0.0045).contains(&self.thickness_m)
            && (0.0..=self.depth_m * 0.066).contains(&self.crown_m)
            && (0.0002..=0.0008).contains(&self.nesting_gap_m)
    }

    fn packed_pitch(self, tilt: f32) -> f32 {
        let (sin, cos) = tilt.sin_cos();
        // Minimum clearance between two parallel curved shells, measured
        // vertically at a *common depth*, including the steepest blade edge.
        let slope = 4. * self.crown_m / self.depth_m;
        (self.thickness_m + self.nesting_gap_m) / (cos - slope * sin.abs()).max(0.1)
    }

    pub(super) fn layout(
        self,
        height: f32,
        ladder_pitch: f32,
        coverage: f32,
        tilt: f32,
    ) -> SlatLayout {
        let count = (height / ladder_pitch).floor().clamp(2., 80.) as usize;
        let pitch = height / count as f32;
        let (sin, cos) = tilt.sin_cos();
        let packed = self.packed_pitch(tilt);
        let mut top = f32::NEG_INFINITY;
        let mut bottom = f32::INFINITY;
        for i in 0..=6 {
            let v = i as f32 / 6. - 0.5;
            let y = v * self.depth_m * sin + self.crown_m * (1. - 4. * v * v) * cos;
            top = top.max(y + self.thickness_m * 0.5 * cos);
            bottom = bottom.min(y - self.thickness_m * 0.5 * cos);
        }
        SlatLayout {
            count,
            pitch,
            packed,
            suspended: (count - 1) as f32 * coverage,
            top,
            bottom,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SlatLayout {
    pub count: usize,
    pitch: f32,
    packed: f32,
    suspended: f32,
    pub top: f32,
    pub bottom: f32,
}

impl SlatLayout {
    pub fn drop_at(self, i: usize) -> f32 {
        let n = i as f32;
        n.min(self.suspended) * (self.pitch - self.packed) + n * self.packed
    }

    pub fn blocked_height(self) -> f32 {
        let silhouette = self.top - self.bottom;
        let full = self.suspended.floor();
        let fraction = self.suspended - full;
        let gaps = (self.count - 1) as f32;
        let transition = if full < gaps { 1. } else { 0. };
        silhouette
            + full * self.pitch.min(silhouette)
            + transition * (self.packed + fraction * (self.pitch - self.packed)).min(silhouette)
            + (gaps - full - transition) * self.packed.min(silhouette)
    }
}

pub(crate) fn clear_fraction(f: &Facade, span: Vec2) -> Option<f32> {
    let p = f.treatment.as_ref()?;
    let stock = p.slats?;
    if f.shade != Shade::Venetian {
        return None;
    }
    let coverage = f.shade_coverage.clamp(0., 1.);
    let tilt = f.shade_tilt.clamp(-1.2, 1.2) * coverage.sqrt();
    let layout = stock.layout(span.y - 0.028 - 0.095, p.slat_pitch, coverage, tilt);
    // Header and rail have disjoint silhouettes. The two thin lift ladders
    // have negligible area in this room-average frontal daylight estimate.
    let blocked = (span.x - 0.066) * (layout.blocked_height() + 0.014) + (span.x - 0.028) * 0.042;
    Some((1. - blocked / span.element_product()).clamp(0., 1.))
}
