//! The heading fixes the carrier spacing and parked width. Each cut has its
//! own hang below the carriers. The same panel outlines drive the daylight meter.
use super::*;
use crate::scene::procedural_indoor::materials::window_textile::Lining;

pub(super) const TRACK_Z: f32 = 0.105;
pub(super) const TRACK_HEIGHT: f32 = 0.026;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurtainHeading {
    Wave,
    PinchPleat,
    Gathered,
}

impl CurtainHeading {
    pub(crate) fn parked_fraction(self, p: &ShadeProgram) -> f32 {
        let fullness = (p.fullness - 1.4).clamp(0., 1.);
        let lining = match p.textile.map(|stock| stock.lining) {
            Some(Lining::Blackout) => 1.,
            Some(Lining::Dimout) => 0.5,
            _ => 0.,
        };
        let bulk = 0.62 * fullness + 0.38 * lining;
        match self {
            Self::Wave => 0.17 + 0.08 * bulk,
            Self::PinchPleat => 0.24 + 0.11 * bulk,
            Self::Gathered => 0.19 + 0.13 * bulk,
        }
    }
}

pub(super) fn tied(p: &ShadeProgram, o: &WindowOpening, coverage: f32) -> bool {
    let Some(heading) = p.curtain_heading else {
        return false;
    };
    let size = o.max - o.min;
    let larger_leaf = if p.stack == Stack::Pair {
        p.split.max(1. - p.split)
    } else {
        1.
    };
    let parked = heading.parked_fraction(p);
    // A sewn-back bundle is stationary until someone releases the holdback.
    // Short transoms and wide cloth beyond arm's reach are left free.
    p.tieback > 0.
        && coverage < 0.78
        && size.y > 0.70
        && size.x * larger_leaf * parked.lerp(1., coverage) < size.y * 0.90
}

#[derive(Clone, Copy)]
pub(super) struct Panel {
    pub jamb: f32,
    pub left: bool,
    pub span: f32,
    pub top: f32,
    pub height: f32,
    pub folds: usize,
    pub tie_at: f32,
    pub held: bool,
    pub phase: f32,
    pub arc_bias: f32,
    pub flutter: f32,
    pub hem_lift: f32,
    pub fullness: f32,
    pub cut_origin: Vec2,
    pub fold_depth: f32,
    pub heading: CurtainHeading,
    width: f32,
    hem_flare: f32,
    bundle: f32,
}

pub(super) fn panels(
    p: &ShadeProgram,
    o: &WindowOpening,
    heading: CurtainHeading,
    coverage: f32,
    held: bool,
) -> impl Iterator<Item = Panel> {
    let left = o.min.x + 0.014;
    let right = o.max.x - 0.014;
    let width = right - left;
    let split = left + width * p.split;
    let bounds = match p.stack {
        Stack::Pair => [(left, split, true), (split, right, false)],
        Stack::Left => [(left, right, true), (right, right, false)],
        Stack::Right => [(left, left, true), (left, right, false)],
    };
    let parked = heading.parked_fraction(p);
    let deployed = parked.lerp(1., coverage.clamp(0., 1.));
    let seed = installation_seed(p, o);
    let fullness = p.fullness;
    let fold_depth = p.fold_depth;
    let hem_sag = p.hem_sag;
    let tension = p.tieback;
    let height = o.max.y - o.min.y - 0.123;
    let top = o.max.y - 0.072;
    // More cloth per metre needs more folds within the available reveal depth.
    let pitch = p.fold_pitch.min(0.32 / fullness);
    bounds
        .into_iter()
        .filter(|(lo, hi, _)| hi > lo)
        .map(move |(lo, hi, left)| {
            let span = hi - lo - 0.014;
            let mut rng = stream(seed, 0x4452415045435554 ^ u64::from(left));
            // Share the aperture's fold budget across both leaves. Preserve
            // the sewn edges and carrier geometry even on broad, tied curtains.
            let width_budget = (24. * span / width).round().max(2.);
            let folds = (span / pitch).round().clamp(2., width_budget) as usize;
            let width = span * deployed;
            Panel {
                jamb: if left { lo + 0.007 } else { hi - 0.007 },
                left,
                span,
                top,
                height,
                folds,
                heading,
                held,
                width,
                fullness,
                fold_depth,
                tie_at: rng.random_range(0.32..0.57),
                phase: rng.random_range(0.0..std::f32::consts::TAU),
                arc_bias: rng.random_range(-0.20..0.20),
                flutter: rng.random_range(0.35..0.95),
                hem_flare: rng.random_range(0.025..0.125),
                hem_lift: (hem_sag + if held { (0.045 * width).min(0.07) } else { 0. })
                    .min(height * 0.064),
                bundle: (span * rng.random_range(0.22..0.32) * (1. - 0.6 * tension))
                    .min(0.24 - 0.09 * tension)
                    .min(width * 0.85),
                cut_origin: Vec2::new(rng.random_range(0.0..2.), rng.random_range(0.0..5.)),
            }
        })
}

fn smooth(x: f32) -> f32 {
    x * x * (3. - 2. * x)
}

impl Panel {
    fn edge_bands(&self) -> (f32, f32) {
        let hem = (1.45 * self.hem_lift / self.height).clamp(0.02, 0.10);
        let head =
            ((self.span / self.folds as f32 * 0.09).min(0.030) / self.height).clamp(0.02, 0.10);
        (hem, head)
    }

    pub fn levels(&self) -> Vec<f32> {
        let (hem, head) = self.edge_bands();
        if self.held {
            let h = self.tie_at;
            vec![
                0.,
                hem,
                0.62 * h,
                h,
                h + 0.21 * (1. - h),
                h + 0.47 * (1. - h),
                h + 0.76 * (1. - h),
                1. - head,
                1.,
            ]
        } else {
            vec![0., hem, 0.28, 0.57, 0.83, 1. - head, 1.]
        }
    }

    // The sewn header and hem maintain their width across each binding.
    // Scalloping the edge inside those bands keeps even a deeply gathered
    // skirt from folding its projected triangles back across one another.
    pub fn profile_height(&self, v: f32) -> f32 {
        let (hem, head) = self.edge_bands();
        if v <= self.tie_at {
            self.tie_at * ((v - hem) / (self.tie_at - hem)).clamp(0., 1.)
        } else {
            self.tie_at
                + (1. - self.tie_at) * ((v - self.tie_at) / (1. - head - self.tie_at)).clamp(0., 1.)
        }
    }

    pub fn gather(&self, v: f32) -> f32 {
        if !self.held {
            return 0.;
        }
        let t = if v <= self.tie_at {
            v / self.tie_at
        } else {
            (1. - v) / (1. - self.tie_at)
        };
        smooth(t)
    }

    pub fn row_width(&self, v: f32) -> f32 {
        let skirt = (self.span - self.width) * self.hem_flare * (1. - v).powi(2);
        let mut reach = self.width + skirt;
        if self.held && v < self.tie_at {
            // Only the header is held out across the rail. Below the jamb tie
            // the weight falls under the bundle, with a flare limited by the
            // hanging length and the fullness of this cut.
            let splay = 0.14 + 2.4 * self.hem_flare + 0.11 * (self.fullness - 1.4);
            reach = reach.min(self.bundle + self.height * self.tie_at * splay);
        }
        reach.lerp(self.bundle, self.gather(v))
    }

    pub fn edge_x(&self, width: f32) -> f32 {
        if self.left {
            self.jamb + width
        } else {
            self.jamb - width
        }
    }
}

/// Frontal aperture for exposure metering. Clip the actual cut widths to each
/// glazed pane and frit band. Hem scallops and individual hooks are sub-band detail.
pub(crate) fn visible_area(f: &Facade, o: &WindowOpening, coverage: f32) -> Option<f32> {
    let p = f.treatment.as_ref().filter(|_| f.shade == Shade::Curtain)?;
    let heading = p.curtain_heading?;
    let mut bands = Vec::new();
    for panel in panels(p, o, heading, coverage, tied(p, o, coverage)) {
        let bottom = panel.top - panel.height + 0.5 * panel.hem_lift;
        let levels = panel.levels();
        for v in levels.windows(2) {
            let height_at = |v: f32| {
                if v == 0. {
                    bottom
                } else {
                    panel.top - panel.height * (1. - v)
                }
            };
            let y = Vec2::new(height_at(v[0]), height_at(v[1]));
            let edges = Vec2::new(
                panel.edge_x(panel.row_width(panel.profile_height(v[0]))),
                panel.edge_x(panel.row_width(panel.profile_height(v[1]))),
            );
            bands.push((y, panel.jamb, edges));
        }
    }
    let rail_y = o.max.y - 0.040;
    Some(super::super::pane::visible_pane_area_with(
        f,
        o,
        |lo, hi| {
            if hi.cmple(lo).any() {
                return 0.;
            }
            let mut area = ((rail_y + TRACK_HEIGHT * 0.5).min(hi.y)
                - (rail_y - TRACK_HEIGHT * 0.5).max(lo.y))
            .max(0.)
                * ((o.max.x - 0.014).min(hi.x) - (o.min.x + 0.014).max(lo.x)).max(0.);
            for &(y, jamb, edges) in &bands {
                let from = lo.y.max(y.x);
                let to = hi.y.min(y.y);
                if from >= to {
                    continue;
                }
                let width_at = |level| {
                    let u = (level - y.x) / (y.y - y.x);
                    (edges.x.lerp(edges.y, u).clamp(lo.x, hi.x) - jamb.clamp(lo.x, hi.x)).abs()
                };
                // Simpson integration is exact on uncut trapezoids; handles bands
                // crossing a mullion without constructing any meshes for metering.
                area += (to - from)
                    * (width_at(from) + 4. * width_at((from + to) * 0.5) + width_at(to))
                    / 6.;
            }
            area
        },
    ))
}
