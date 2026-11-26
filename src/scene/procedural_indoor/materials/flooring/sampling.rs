//! Independent product, pigment, cut-stock and finish domains. Continuous
//! variation stays inside buildable floor products, not arbitrary RGB noise.
use super::super::{carpet::dye::CarpetDye, Surface};
use super::*;

/// Granulated resilient flooring uses the existing chip-and-binder evaluator.
/// The rubber binder is matte, with through-coloured flecks and butt joints.
pub(in super::super) fn exercise_floor(r: &mut MaterialRecipe) {
    let mut stock = stream(r.seed, u64::from_le_bytes(*b"GymFloor"));
    let m = r.mineral.as_mut().expect("granulated floor substrate");
    let f = r.flooring.as_mut().expect("floor installation");
    let base = stock.random_range(0.15..0.41);
    let bias = stock.random_range(-0.045..0.045);
    r.color = [base - bias, base, base + bias];
    r.roughness = stock.random_range(0.76..0.93);
    r.relief_m = stock.random_range(0.00012..0.00040);
    r.joint_width = stock.random_range(0.0005..0.0015);
    m.casting = None;
    m.marble_mix = 0.;
    m.vein_strength = 0.;
    m.polish = 0.;
    m.porosity = stock.random_range(0.04..0.25);
    m.aggregate_radius = stock.random_range(0.12..0.28);
    m.aggregate_exposure = stock.random_range(0.26..0.72);
    m.aggregate_cells = [stock.random_range(55..=94); 2];
    m.aggregate_color = [0.28, 0.57].map(|light| {
        super::pigment::hsl(
            stock.random_range(0.0..360.0),
            stock.random_range(0.02..0.40),
            light,
        )
    });
    m.binder_variation = stock.random_range(0.02..0.15);
    f.layout = if stock.random_bool(0.37) {
        FloorLayout::BrokenBond
    } else {
        FloorLayout::Straight
    };
    let tiles = stock.random_range(2..=4);
    f.cells = [tiles; 2];
    r.period_m = tiles as f32 * stock.random_range(0.44..1.05);
    f.piece_tone = stock.random_range(0.015..0.07);
    f.piece_warmth = stock.random_range(0.003..0.016);
    f.accent_fraction = 0.;
    f.tile_definition = stock.random_range(0.24..0.64);
    if let Some(cut) = &mut f.stock {
        cut.cut_jitter = [0.; 2];
        cut.joint_color = r.color.map(|x| x * 0.8);
        cut.joint_roughness = r.roughness;
    }
}

/// Narrow strip hardwood with close joints and a sealed sports-floor finish.
pub(in super::super) fn sports_timber(r: &mut MaterialRecipe) {
    let mut rng = stream(r.seed, u64::from_le_bytes(*b"VBMaple "));
    let f = r.flooring.as_mut().expect("timber installation");
    let w = r.wood.as_mut().expect("timber floor");
    f.layout = FloorLayout::BrokenBond;
    let strips = rng.random_range(23..=35);
    f.cells = [strips, rng.random_range(2..=3)];
    r.period_m = strips as f32 * rng.random_range(0.057..0.086);
    r.color = super::pigment::hsl(
        rng.random_range(29.0..43.0),
        rng.random_range(0.33..0.57),
        rng.random_range(0.55..0.76),
    );
    r.roughness = rng.random_range(0.36..0.54);
    r.relief_m = rng.random_range(0.000018..0.000054);
    r.joint_width = rng.random_range(0.00045..0.0012);
    w.clearcoat = rng.random_range(0.42..0.78);
    w.coat_roughness = rng.random_range(0.26..0.40);
    w.pore_fill = rng.random_range(0.72..0.96);
    w.stain_strength = rng.random_range(0.01..0.16);
    w.bleach = rng.random_range(0.0..0.10);
    f.piece_tone = rng.random_range(0.040..0.14);
    f.piece_warmth = rng.random_range(0.008..0.035);
    f.piece_roughness = rng.random_range(0.015..0.045);
    f.accent_fraction = 0.;
    if let Some(stock) = &mut f.stock {
        stock.cut_jitter = [0., rng.random_range(0.14..0.32)];
        stock.joint_color = r.color.map(|x| x * 0.65);
        stock.joint_roughness = r.roughness.max(0.40);
    }
    if let Some(laying) = &mut f.laying {
        laying.turn_radians = if rng.random_bool(0.84) {
            0.
        } else {
            std::f32::consts::FRAC_PI_2
        };
    }
}

pub(in super::super) fn expand(r: &mut MaterialRecipe, style: u32) {
    debug_assert_eq!(r.surface, Surface::Floor);
    let mut product = stream(r.seed, 0x464c50524f445543);
    let mut pigment = stream(r.seed, 0x464c5049474d454e);
    let mut finish = stream(r.seed, 0x464c46494e495348);
    let seamless = r
        .flooring
        .as_ref()
        .is_some_and(|f| f.layout == FloorLayout::Seamless);
    r.period_m = super::scale::period(r.seed, seamless);
    match style {
        0 => {
            let hue = pigment.random_range(18.0..48.0);
            let sat = pigment.random_range(0.18..0.48);
            let value = pigment.random_range(0.17..0.78);
            r.color = super::pigment::hsl(hue, sat, value);
            let w = r.wood.as_mut().unwrap();
            w.clearcoat = finish.random_range(0.0_f32..1.).powi(2) * 0.9;
            w.coat_roughness = finish.random_range(0.09..0.44);
            w.stain_strength = pigment.random_range(0.0_f32..1.).powi(2) * 0.85;
            w.bleach = pigment.random_range(0.0_f32..1.).powi(3) * 0.65 * (1. - w.stain_strength);
            w.pore_fill = (finish.random_range(0.05..0.65) + w.clearcoat * 0.3).min(0.95);
            r.roughness = finish.random_range(0.30..0.82);
            r.relief_m = finish.random_range(0.00003..0.00020) * (1. - w.pore_fill * 0.7);
            r.grain_frequency = product.random_range(17.0..78.0);
        }
        1 => {
            let c = r.carpet.as_mut().unwrap();
            c.dye = Some(CarpetDye::sample(r.seed));
            c.contrast = pigment.random_range(0.08..0.72);
            c.line_width = product.random_range(0.035..0.34);
            c.palette = super::pigment::yarns(&mut pigment);
            r.color = c.palette[0];
            c.pile = finish.random_range(0.10..0.95);
            // Unresolved fibres broaden the highlight. The normal map captures
            // the relief of yarn bundles that survive the atlas footprint.
            r.roughness = 0.80 + 0.12 * c.pile + finish.random_range(0.0..0.065);
            r.relief_m = 0.00010 + c.pile * finish.random_range(0.00024..0.00061);
        }
        _ => {
            let m = r.mineral.as_mut().unwrap();
            r.color = super::pigment::mineral(&mut pigment, m.marble_mix);
            m.polish = finish.random_range(0.0..1.);
            r.roughness = 0.86 - m.polish * 0.66;
            r.relief_m = finish.random_range(0.000025..0.00016) * (1. - m.polish * 0.7);
            // Chip sizing and vein geometry vary independently of polish and
            // installation. Keep quiet porcelain and strongly figured stone.
            m.aggregate_radius = product.random_range(0.12..0.45);
            m.vein_width = product.random_range(0.003..0.028);
            m.vein_warp = product.random_range(0.03..0.42);
        }
    }
    r.relief_m *= r.detail_scale.unwrap_or(1.);
    let mut f = FloorFinish::sample(r.seed, style, r.period_m);
    let joint_color = if style == 0 {
        let tone = pigment.random_range(0.035..0.18);
        [tone, tone * 0.97, tone * 0.94]
    } else {
        super::pigment::joint(&mut pigment, r.color, style == 1)
    };
    f.stock = Some(stock::FloorStock {
        cut_jitter: [0.; 2],
        mottle_cells: [product.random_range(1..=11), product.random_range(1..=11)],
        joint_color,
        joint_roughness: if style == 1 {
            finish.random_range(0.84..0.98)
        } else {
            finish.random_range(0.65..0.97)
        },
    });
    r.joint_width = product.random_range(if style == 0 {
        0.0005..0.0025
    } else {
        0.001..0.006
    });
    r.flooring = Some(f);
    super::scale::resolve(r, style);
}

/// Weathered paving uses the existing mineral, joint, normal and roughness
/// channels. Discrete installation choices constrain the continuous stock sizes.
pub(in super::super) fn courtyard_paving(r: &mut MaterialRecipe) {
    let mut rng = stream(r.seed, u64::from_le_bytes(*b"CourtPav"));
    let f = r.flooring.as_mut().expect("mineral paving installation");
    let m = r.mineral.as_mut().expect("mineral paving substrate");
    let product = rng.random_range(0..4);
    let (pigment, course) = match product {
        0 => ([0.58, 0.33, 0.23], rng.random_range(0.095..0.17)),
        1 => ([0.60, 0.54, 0.43], rng.random_range(0.31..0.75)),
        2 => ([0.36, 0.40, 0.41], rng.random_range(0.24..0.60)),
        _ => ([0.54, 0.53, 0.50], rng.random_range(0.20..0.66)),
    };
    let exposed = product == 3;
    let tone = rng.random_range(0.76..1.18);
    r.color = pigment.map(|c| c * tone);
    r.roughness = rng.random_range(0.69..0.94);
    r.relief_m = rng.random_range(0.00022..0.00071) * r.detail_scale.unwrap_or(1.);
    r.joint_width = rng.random_range(0.0028..0.009);
    m.casting = None;
    m.marble_mix = 0.;
    m.vein_strength = if product == 1 || product == 2 {
        rng.random_range(0.03..0.19)
    } else {
        0.
    };
    m.polish = rng.random_range(0.01..0.19);
    m.porosity = rng.random_range(0.26..0.70);
    m.aggregate_exposure = if exposed {
        rng.random_range(0.24..0.46)
    } else {
        rng.random_range(0.02..0.14)
    };
    m.aggregate_color = [0.74, 1.15].map(|v| r.color.map(|c| (c * v).min(0.95)));
    m.binder_variation = rng.random_range(0.035..0.18);
    f.layout = match product {
        0 if rng.random_bool(0.50) => FloorLayout::Herringbone,
        3 if rng.random_bool(0.35) => FloorLayout::Hexagonal,
        _ if rng.random_bool(0.58) => FloorLayout::BrokenBond,
        _ => FloorLayout::Straight,
    };
    let n = rng.random_range(4..=9);
    match f.layout {
        FloorLayout::Herringbone => {
            let ratio = rng.random_range(2..=3);
            f.craft.as_mut().expect("mineral installation").plank_ratio = ratio;
            f.cells = [2 * ratio * rng.random_range(2..=3); 2];
        }
        FloorLayout::Hexagonal => {
            f.cells = [n * 2, (n as f32 * 2. / 3_f32.sqrt()).round() as u32];
        }
        _ => {
            f.cells = [n, (n as f32 * rng.random_range(1.0..2.5)).round() as u32];
        }
    }
    r.period_m = course * f.cells[1] as f32;
    f.piece_tone = rng.random_range(0.06..0.24);
    f.piece_warmth = rng.random_range(0.009..0.045);
    f.accent_fraction = 0.;
    f.tile_definition = rng.random_range(0.64..0.96);
    if let Some(cut) = &mut f.stock {
        cut.cut_jitter = [rng.random_range(0.0..0.08), rng.random_range(0.0..0.16)];
        let joint = rng.random_range(0.54..0.87);
        cut.joint_color = r.color.map(|c| c * joint);
        cut.joint_roughness = rng.random_range(0.81..0.97);
    }
}
