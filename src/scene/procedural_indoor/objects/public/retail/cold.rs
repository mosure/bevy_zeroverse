//! Open-front multidecks. Shelves and stock use the same interior section;
//! local +Z is the customer side reserved by the retail layout.
use super::*;
use crate::scene::procedural_indoor::materials::glass::SheetGauge;
mod enclosure;
use crate::scene::procedural_indoor::materials::{
    coated_stock, variants, worktop_finish::BrushedHardware,
};

pub(crate) const REFRIGERATED: u32 = 1 << 20;
const SHELF: f32 = 0.022;

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStock {
    CoatedSteel,
    MoldedPolymer,
    BrushedStainless,
}

impl CaseStock {
    fn binding(self, o: &IndoorObject) -> (Surface, String) {
        let substrate = match self {
            Self::BrushedStainless => {
                return (Surface::Metal, BrushedHardware::Stainless.label(FIXTURE))
            }
            Self::CoatedSteel => Surface::Metal,
            Self::MoldedPolymer => Surface::Plastic,
        };
        let finish = variants::object_slot(substrate, o.seed, o.finish_seed);
        let slot = coated_stock::slot(substrate, finish).expect("factory-coated case stock");
        (substrate, format!("{FIXTURE}#finish{slot}"))
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RefrigeratedCase {
    pub levels: usize,
    pub facings: usize,
    pub stock: f32,
    pub deck_height_m: f32,
    pub canopy_height_m: f32,
    pub upper_setback_m: f32,
    pub return_width_m: f32,
    pub end_thickness_m: f32,
    pub back_thickness_m: f32,
    pub shell: CaseStock,
    pub liner: CaseStock,
    /// The two ends in local -X, +X order. None denotes an insulated solid end.
    pub end_glass_m: [Option<SheetGauge>; 2],
    pub end_window_depth_fraction: f32,
    /// Height of the glazing sill above the lower merchandise deck.
    pub end_sill_m: f32,
}

pub(crate) fn is_refrigerated(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::GroceryDisplay && o.size.y >= 1.15 && o.variant & REFRIGERATED != 0
}

/// Recover the construction parameters from the actual archived fixture.
pub fn refrigerated_case(o: &IndoorObject) -> Option<RefrigeratedCase> {
    is_refrigerated(o).then(|| {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ColdDeck"));
        let s = o.size;
        let deck_height_m = s.y * r.random_range(0.165..0.245);
        let canopy_height_m = r.random_range(0.084..0.153);
        let mut ends = stream(o.seed, u64::from_le_bytes(*b"ColdEnds"));
        let gauge = [SheetGauge::Four, SheetGauge::Five, SheetGauge::Six][ends.random_range(0..3)];
        let end_glass_m = match ends.random_range(0..10) {
            0..=2 => [None, None],
            3..=6 => [Some(gauge), Some(gauge)],
            _ if ends.random() => [Some(gauge), None],
            _ => [None, Some(gauge)],
        };
        RefrigeratedCase {
            end_glass_m,
            end_window_depth_fraction: ends.random_range(0.57..0.83),
            end_sill_m: ends.random_range(0.070..0.165),
            levels: ((s.y - deck_height_m - canopy_height_m) / r.random_range(0.29..0.40))
                .round()
                .clamp(2., 5.) as usize,
            facings: (s.x / r.random_range(0.145..0.215)).floor().clamp(4., 12.) as usize,
            stock: r.random_range(0.48..0.97),
            deck_height_m,
            canopy_height_m,
            upper_setback_m: s.z * r.random_range(0.10..0.28),
            return_width_m: r.random_range(0.082..0.127),
            end_thickness_m: r.random_range(0.035..0.061),
            back_thickness_m: r.random_range(0.064..0.108),
            shell: [
                CaseStock::CoatedSteel,
                CaseStock::MoldedPolymer,
                CaseStock::BrushedStainless,
            ][r.random_range(0..3)],
            liner: if r.random_bool(0.58) {
                CaseStock::BrushedStainless
            } else {
                CaseStock::MoldedPolymer
            },
        }
    })
}

#[derive(Clone, Copy)]
pub(super) struct Shelf {
    pub half_width: f32,
    pub back: f32,
    pub front: f32,
    pub top: f32,
    pub clearance: f32,
}

impl RefrigeratedCase {
    pub(super) fn shelf(self, s: Vec3, level: usize) -> Shelf {
        let pitch = (s.y - self.canopy_height_m - self.deck_height_m) / self.levels as f32;
        Shelf {
            half_width: s.x * 0.5 - self.end_thickness_m,
            back: -s.z * 0.5 + self.back_thickness_m,
            front: s.z * 0.5
                - self.return_width_m
                - self.upper_setback_m * level as f32 / (self.levels - 1) as f32,
            top: self.deck_height_m + pitch * level as f32,
            clearance: pitch - SHELF,
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, case: RefrigeratedCase) {
    let s = o.size;
    let bottom = case.shelf(s, 0);
    let (shell, shell_label) = case.shell.binding(o);
    let (liner, liner_label) = case.liner.binding(o);
    let grille = BrushedHardware::Stainless.label(FIXTURE);
    let z = s.z * 0.5;
    let width = 2. * bottom.half_width;
    let toe = 0.075;
    let well = bottom.top - 0.035;
    let canopy = s.y - case.canopy_height_m;
    // The recessed toe supports the insulated base. The plenum stops below
    // the return grille so the gaps between its slats have real depth.
    board(
        a,
        Surface::Rubber,
        FIXTURE,
        Vec3::Y * (toe * 0.5),
        Vec3::new(s.x - 0.08, toe, s.z - 0.09),
    );
    enclosure::base(a, o, (shell, &shell_label), toe, well);
    enclosure::ends(a, o, case, (shell, &shell_label), well, canopy);
    board(
        a,
        liner,
        &liner_label,
        Vec3::new(
            0.,
            (well + canopy) * 0.5,
            bottom.back - case.back_thickness_m * 0.5,
        ),
        Vec3::new(width, canopy - well, case.back_thickness_m),
    );
    board(
        a,
        shell,
        &shell_label,
        Vec3::Y * (s.y - case.canopy_height_m * 0.5),
        Vec3::new(s.x, case.canopy_height_m, s.z),
    );
    // The canopy carries a flush-mounted, downward-facing diffuser.
    board(
        a,
        Surface::Light,
        FIXTURE,
        Vec3::new(0., canopy - 0.003, z - 0.07),
        Vec3::new(width - 0.065, 0.006, 0.036),
    );
    for level in 0..case.levels {
        let shelf = case.shelf(s, level);
        let t = if level == 0 { bottom.top - well } else { SHELF };
        board(
            a,
            liner,
            &liner_label,
            Vec3::new(0., shelf.top - t * 0.5, (shelf.back + shelf.front) * 0.5),
            Vec3::new(width, t, shelf.front - shelf.back),
        );
        if level > 0 {
            // The price rail caps the shelf edge, outside the merchandise footprint.
            board(
                a,
                shell,
                &shell_label,
                Vec3::new(0., shelf.top - 0.016, shelf.front + 0.004),
                Vec3::new(width, 0.032, 0.008),
            );
        }
    }
    let rim = 0.012;
    board(
        a,
        shell,
        &shell_label,
        Vec3::new(0., (bottom.top + well) * 0.5, z - rim * 0.5),
        Vec3::new(width, bottom.top - well, rim),
    );
    let spacing = (case.return_width_m - rim) / 4.;
    for i in 0..4 {
        board(
            a,
            Surface::Metal,
            &grille,
            Vec3::new(
                0.,
                bottom.top - 0.006,
                bottom.front + (i as f32 + 0.5) * spacing,
            ),
            Vec3::new(width, 0.012, spacing * 0.50),
        );
    }
    grocery::chilled::stock(a, o, case);
}
