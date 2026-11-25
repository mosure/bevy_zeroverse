//! A length of pressed cloth cut to the opening. Each facet rotates as the
//! bottom rail travels. A bonded second web encloses air cells across the width.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PleatConstruction {
    Single,
    Cellular,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PleatedStock {
    pub construction: PleatConstruction,
    pub packed_half_pitch_m: f32,
    pub cloth_thickness_m: f32,
    pub bond_width_m: f32,
    pub side_gap_m: f32,
    pub sill_gap_m: f32,
    pub rail_height_m: f32,
    pub cordless: bool,
}

pub(super) const TRACK_Z: f32 = 0.115;
const HEAD_TOP: f32 = 0.019;
const HEAD_BOTTOM: f32 = 0.061;
const INSERTION: f32 = 0.001;

impl PleatedStock {
    pub(super) fn sample(r: &mut impl Rng, crease: f32) -> Self {
        let cellular = r.random_bool(0.52);
        Self {
            construction: if cellular {
                PleatConstruction::Cellular
            } else {
                PleatConstruction::Single
            },
            packed_half_pitch_m: 0.00065 + crease * r.random_range(0.0030..0.0052),
            cloth_thickness_m: r.random_range(0.00030..0.00068),
            bond_width_m: if cellular {
                r.random_range(0.0045..0.014)
            } else {
                0.
            },
            side_gap_m: r.random_range(0.024..0.040),
            sill_gap_m: r.random_range(0.012..0.040),
            rail_height_m: r.random_range(0.013..0.029),
            cordless: r.random_bool(0.63),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.0010..=0.0021).contains(&self.packed_half_pitch_m)
            && (0.00030..=0.00068).contains(&self.cloth_thickness_m)
            && match self.construction {
                PleatConstruction::Single => self.bond_width_m == 0.,
                PleatConstruction::Cellular => (0.0045..=0.014).contains(&self.bond_width_m),
            }
            && (0.024..=0.040).contains(&self.side_gap_m)
            && (0.012..=0.040).contains(&self.sill_gap_m)
            && (0.013..=0.029).contains(&self.rail_height_m)
    }

    pub(crate) fn layout(self, p: &ShadeProgram, o: &WindowOpening, draw: f32) -> PleatLayout {
        let top = o.max.y - HEAD_BOTTOM + INSERTION;
        let full_drop = top - (o.min.y + self.sill_gap_m + self.rail_height_m - INSERTION);
        // Fit whole cells between the head and sill, distributing the length
        // remainder evenly across the installed pitch.
        let width_draw = ((p.fold_pitch - 0.10) / 0.15).clamp(0., 1.);
        let cellular = self.construction == PleatConstruction::Cellular;
        // A honeycomb has two facing webs. Its projected repeat is smaller
        // than a single web's at a comparable packed depth. Tall installations
        // use a coarser gauge so the entire drop fits on a single lift.
        let (fine_pitch, pitch_range, widest_pitch, max_cells) = if cellular {
            (0.019, 0.050, 0.084, 144.)
        } else {
            (0.029, 0.064, 0.100, 256.)
        };
        let pitch = fine_pitch + pitch_range * width_draw * width_draw;
        // Leave daylight below a raised shade. On a tall drop, fitting the
        // fine gauge can otherwise leave hundreds of millimetres of piled
        // cloth. Limit the stack while retaining a physically sized facet.
        let parked_drop = 0.040 + 2.6 * self.rail_height_m;
        let capacity = (parked_drop / (2. * self.packed_half_pitch_m)).floor();
        let limit = capacity
            .max((full_drop / widest_pitch).ceil())
            .clamp(1., max_cells);
        let count = (full_drop / pitch).round().clamp(1., limit) as usize;
        let full_rise = full_drop / (2 * count) as f32;
        let depth_draw = ((p.fold_depth - 0.012) / 0.022).clamp(0., 1.);
        let face_m = full_rise.hypot(full_rise * (0.40 + 0.48 * depth_draw));
        let packed_depth = if cellular {
            self.bond_width_m + 2. * face_m
        } else {
            self.cloth_thickness_m + face_m
        };
        let rail_z = if cellular {
            TRACK_Z
        } else {
            TRACK_Z + 0.5 * face_m
        };
        let rise = self
            .packed_half_pitch_m
            .min(full_rise)
            .lerp(full_rise, draw.clamp(0., 1.));
        PleatLayout {
            count,
            top,
            rise,
            face_m,
            rail_z,
            packed_depth,
            depth: ((face_m - rise) * (face_m + rise)).max(0.).sqrt(),
            width: o.max.x - o.min.x - 2. * self.side_gap_m,
            bottom: top - (2 * count) as f32 * rise,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PleatLayout {
    pub count: usize,
    pub top: f32,
    pub bottom: f32,
    pub width: f32,
    pub rise: f32,
    pub depth: f32,
    pub face_m: f32,
    pub rail_z: f32,
    pub packed_depth: f32,
}

impl PleatLayout {
    pub(super) fn rail_bottom(self, stock: PleatedStock) -> f32 {
        self.bottom + INSERTION - stock.rail_height_m
    }
}

/// Frontal clear area for the daylight meter. The three adjacent bands have
/// different widths. Each is clipped to the actual glazed panes and frit.
pub(crate) fn visible_area(f: &Facade, o: &WindowOpening, draw: f32) -> Option<f32> {
    let p = f.treatment.as_ref().filter(|_| f.shade == Shade::Pleated)?;
    let stock = p.pleated?;
    let l = stock.layout(p, o, draw);
    let center = (o.min.x + o.max.x) * 0.5;
    let rail_top = l.bottom + INSERTION;
    let header = o.max.y - HEAD_BOTTOM;
    Some(super::super::pane::visible_pane_area_with(
        f,
        o,
        |lo, hi| {
            let band = |width: f32, bottom: f32, top: f32| {
                (hi.x.min(center + width * 0.5) - lo.x.max(center - width * 0.5)).max(0.)
                    * (hi.y.min(top) - lo.y.max(bottom)).max(0.)
            };
            band(o.max.x - o.min.x - 0.028, header, o.max.y - HEAD_TOP)
                + band(l.width, rail_top, header)
                + band(l.width + 0.006, l.rail_bottom(stock), rail_top)
        },
    ))
}
