//! Fabric vane stock and its traverse track. A linked tilt rod sets the angle
//! independently of the carrier spacing. All lengths are in wall-local metres.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerticalStock {
    pub width_m: f32,
    pub overlap_m: f32,
    pub crown_m: f32,
    pub thickness_m: f32,
    pub carrier_pitch_m: f32,
    pub hem_m: f32,
    pub sill_gap_m: f32,
    pub clip_drop_m: f32,
}

const JAMB: f32 = 0.034;
pub(super) const TRACK_Z: f32 = 0.115;
pub(super) const RAIL_TOP: f32 = 0.019;
pub(super) const RAIL_BOTTOM: f32 = 0.061;

impl VerticalStock {
    const MIN_TILT: f32 = 0.16;
    pub(crate) const MAX_TILT: f32 = 1.48;

    pub(crate) fn sample_tilt(seed: u64) -> f32 {
        let control = stream(seed, 0x56414e4554494c54).random_range(-1.0_f32..1.);
        control.signum() * Self::MIN_TILT.lerp(Self::MAX_TILT, control.abs())
    }

    pub(super) fn sample(r: &mut impl Rng) -> Self {
        let width_m = r.random_range(0.065..0.133);
        Self {
            width_m,
            overlap_m: width_m * r.random_range(0.08..0.21),
            crown_m: width_m * r.random_range(0.004..0.018),
            thickness_m: r.random_range(0.0007..0.0017),
            carrier_pitch_m: r.random_range(0.010..0.020),
            hem_m: r.random_range(0.020..0.047),
            sill_gap_m: r.random_range(0.014..0.055),
            clip_drop_m: r.random_range(0.013..0.032),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.065..=0.133).contains(&self.width_m)
            && (0.07..=0.22).contains(&(self.overlap_m / self.width_m))
            && (0.003..=0.019).contains(&(self.crown_m / self.width_m))
            && (0.0007..=0.0017).contains(&self.thickness_m)
            && (0.010..=0.020).contains(&self.carrier_pitch_m)
            && (0.020..=0.047).contains(&self.hem_m)
            && (0.014..=0.055).contains(&self.sill_gap_m)
            && (0.013..=0.032).contains(&self.clip_drop_m)
    }

    pub(crate) fn layout(
        self,
        p: &ShadeProgram,
        o: &WindowOpening,
        tilt: f32,
        draw: f32,
    ) -> VaneLayout {
        let usable = (o.max.x - o.min.x - 2. * JAMB).max(0.);
        let width = self.width_m.min(usable);
        let travel = (usable - width).max(0.);
        let nominal_pitch = self.width_m - self.overlap_m;
        let intervals = (travel / nominal_pitch).round().clamp(0., 159.) as usize;
        let count = intervals + 1;
        let pitch = if intervals == 0 {
            0.
        } else {
            travel / intervals as f32
        };
        let angle = tilt.signum() * tilt.abs().clamp(Self::MIN_TILT, Self::MAX_TILT);
        let (sin, cos) = angle.sin_cos();
        // Separation normal to the vane, not the projected width on the wall.
        // Even the two most bowed points keep a millimetre of air between them.
        let packed = self
            .carrier_pitch_m
            .max((self.crown_m + self.thickness_m + 0.001) / sin.abs())
            .min(pitch);
        let step = packed.lerp(pitch, draw.clamp(0., 1.));
        let end_stop = o.min.x + JAMB + width * 0.5;
        let single_draw = if count == 1 {
            travel * 0.5 * draw.clamp(0., 1.)
        } else {
            0.
        };
        let left = end_stop + single_draw;
        let right = end_stop + travel - single_draw;
        let left_count = match p.stack {
            Stack::Left => count,
            Stack::Right => 0,
            Stack::Pair if count > 2 => {
                ((count as f32 * p.split).round() as usize).clamp(1, count - 1)
            }
            // Two stationary end carriers cannot split-draw a two-vane bay.
            Stack::Pair if p.identity.is_multiple_of(2) => count,
            Stack::Pair => 0,
        };
        VaneLayout {
            count,
            width,
            angle,
            pitch,
            packed,
            step,
            left,
            right,
            left_count,
            // At the sampled crown and tilt, the transverse profile is monotone
            // in projected X. Its silhouette ends at the two selvages.
            silhouette: width * cos + self.thickness_m * sin.abs(),
            rail_y: o.max.y - RAIL_BOTTOM,
            top: o.max.y - RAIL_BOTTOM - self.clip_drop_m,
            bottom: o.min.y + self.sill_gap_m,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct VaneLayout {
    pub count: usize,
    pub width: f32,
    pub angle: f32,
    pub pitch: f32,
    pub packed: f32,
    step: f32,
    left: f32,
    right: f32,
    left_count: usize,
    silhouette: f32,
    pub rail_y: f32,
    pub top: f32,
    pub bottom: f32,
}

impl VaneLayout {
    pub(super) fn x(self, i: usize) -> f32 {
        if i < self.left_count {
            self.left + i as f32 * self.step
        } else {
            self.right - (self.count - 1 - i) as f32 * self.step
        }
    }

    /// Union of the projected vane intervals clipped to a pane or frit strip.
    /// Each stack is regularly spaced, so this costs the same for two or 150 vanes.
    fn blocked_width(self, lo: f32, hi: f32) -> f32 {
        let bank = |first: f32, n: usize, from: f32, to: f32| {
            if n == 0 || from >= to {
                return 0.;
            }
            let start = first - self.silhouette * 0.5;
            let extent = (n - 1) as f32 * self.step + self.silhouette;
            let prefix = |x: f32| {
                let t = (x - start).clamp(0., extent);
                if self.silhouette >= self.step {
                    return t;
                }
                let k = (t / self.step).floor().min((n - 1) as f32);
                k * self.silhouette + (t - k * self.step).min(self.silhouette)
            };
            prefix(to) - prefix(from)
        };
        let n = self.left_count;
        if n == self.count {
            return bank(self.left, n, lo, hi);
        }
        let right = self.x(n);
        if n == 0 {
            return bank(right, self.count, lo, hi);
        }
        let middle = (self.x(n - 1) + right) * 0.5;
        bank(self.left, n, lo, hi.min(middle)) + bank(right, self.count - n, lo.max(middle), hi)
    }
}

/// Frontal metering aperture, including actual pane edges, framing and frit.
/// Carrier stems and the control wand are below the scale of this exposure model.
pub(crate) fn visible_area(f: &Facade, o: &WindowOpening, draw: f32) -> Option<f32> {
    let p = f
        .treatment
        .as_ref()
        .filter(|_| f.shade == Shade::Vertical)?;
    let layout = p.vertical?.layout(p, o, f.shade_tilt, draw);
    Some(super::super::pane::visible_pane_area_with(
        f,
        o,
        |lo, hi| {
            let height = (hi.y.min(layout.top) - lo.y.max(layout.bottom)).max(0.);
            let rail_height = (hi.y.min(o.max.y - RAIL_TOP) - lo.y.max(layout.rail_y)).max(0.);
            let rail_width = (hi.x.min(o.max.x - 0.014) - lo.x.max(o.min.x + 0.014)).max(0.);
            height * layout.blocked_width(lo.x, hi.x) + rail_height * rail_width
        },
    ))
}
