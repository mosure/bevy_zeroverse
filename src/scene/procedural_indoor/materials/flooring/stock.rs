//! Metric cut-stock variation with constant-time, periodic interval lookup.
//! Adjacent pieces share boundaries; jitter never produces gaps or slivers.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloorStock {
    /// Fraction of nominal rectangular piece size. Each interval remains at
    /// least 20% wide. Parquet/hexagonal layouts retain their exact cut geometry.
    pub cut_jitter: [f32; 2],
    pub mottle_cells: [u32; 2],
    /// sRGB grout/joint reflectance, independent of the face material.
    pub joint_color: [f32; 3],
    pub joint_roughness: f32,
}

impl FloorStock {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .cut_jitter
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=0.40).contains(x))
            || self.mottle_cells.iter().any(|x| !(1..=16).contains(x))
            || self
                .joint_color
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=0.97).contains(x))
            || !(0.4..=1.).contains(&self.joint_roughness)
        {
            return Err("invalid floor cut stock, mottling or joint finish".into());
        }
        Ok(())
    }

    pub(super) fn cut(&self, f: &FloorFinish, r: &MaterialRecipe, uv: [f32; 2]) -> Cut {
        let (x, a, aw) = interval(
            uv[0],
            f.cells[0],
            self.cut_jitter[0],
            0,
            r.seed.wrapping_add(701),
        );
        let offset = if f.layout == FloorLayout::BrokenBond {
            hash(x, 7, r.seed.wrapping_add(211))
        } else {
            0.
        };
        // Straight installations retain continuous cross joints. Random-length
        // broken-bond boards may vary their end joints independently by lane.
        let lane = if f.layout == FloorLayout::BrokenBond {
            x
        } else {
            0
        };
        let (y, b, bw) = interval(
            uv[1] + offset / f.cells[1] as f32,
            f.cells[1],
            self.cut_jitter[1],
            lane,
            r.seed.wrapping_add(709),
        );
        Cut {
            id: [x, y],
            uv: [a, b],
            distance_m: r.period_m * (a.min(1. - a) * aw).min(b.min(1. - b) * bw),
        }
    }

    pub(in super::super) fn joint(&self, coverage: f32, t: &mut super::super::program::Texel) {
        if coverage <= 0. {
            return;
        }
        // Mix reflectance in linear light. Subpixel joints use area coverage,
        // rather than turning every filtered seam into a full-width dark line.
        t.color = super::super::field::mix(t.color, self.joint_color, coverage);
        t.roughness += (self.joint_roughness - t.roughness) * coverage;
    }
}

/// (wrapped piece ID, local coordinate, metric fraction of the atlas).
fn interval(u: f32, n: u32, jitter: f32, lane: u32, seed: u64) -> (u32, f32, f32) {
    let t = u.rem_euclid(1.) * n as f32;
    let boundary =
        |i: i32| i as f32 + (hash(i.rem_euclid(n as i32) as u32, lane, seed) * 2. - 1.) * jitter;
    let mut i = t.floor() as i32;
    if t < boundary(i) {
        i -= 1;
    } else if t >= boundary(i + 1) {
        i += 1;
    }
    let lo = boundary(i);
    let width = boundary(i + 1) - lo;
    (
        i.rem_euclid(n as i32) as u32,
        ((t - lo) / width).clamp(0., 1.),
        width / n as f32,
    )
}
