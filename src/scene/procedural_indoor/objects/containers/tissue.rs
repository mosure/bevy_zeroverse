//! Fitted tabletop dispensers. The case, cut edge, remaining pack and drawn
//! tissue have separate physical surfaces but share one tracked object.
use super::*;
mod case;
mod legacy;
mod sheet;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct TissueProgram {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dispenser: Option<Dispenser>,
    pub box_height: f32,
    pub opening: Vec2,
    pub lean: f32,
    pub fold: f32,
    pub phase: f32,
    pub finish: Surface,
}
/// Values up to three were used by archived plain boxes and external fixtures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(u32)]
pub enum Case {
    FoldingCarton = 4,
    MoldedCover,
    TimberSleeve,
    MetalCover,
}
impl Case {
    pub(crate) fn from_variant(variant: u32) -> Option<Self> {
        match variant {
            4 => Some(Self::FoldingCarton),
            5 => Some(Self::MoldedCover),
            6 => Some(Self::TimberSleeve),
            7 => Some(Self::MetalCover),
            _ => None,
        }
    }
}
pub(crate) fn installed_case(seed: u64) -> Case {
    match stream(seed, u64::from_le_bytes(*b"TissCase")).random_range(0..100) {
        0..=67 => Case::FoldingCarton,
        68..=82 => Case::MoldedCover,
        83..=92 => Case::TimberSleeve,
        _ => Case::MetalCover,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Draw {
    Sealed,
    LowTab,
    Fan,
    CornerPull,
}
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Dispenser {
    pub case: Case,
    pub draw: Draw,
    pub wall_m: f32,
    pub corner_m: f32,
    pub lip_m: f32,
    pub mouth_rounding: Vec2,
    pub mouth_offset_m: Vec2,
    pub fill_fraction: f32,
    pub pleats: u8,
    pub crest: f32,
    pub curl: f32,
    pub flare: f32,
}
fn draw_profile(use_state: &mut impl Rng, case: Case, footprint: Vec2) -> (Draw, f32) {
    let draw = match use_state.random_range(0..100) {
        0..=8 if case == Case::FoldingCarton => Draw::Sealed,
        0..=28 => Draw::LowTab,
        29..=66 => Draw::Fan,
        _ => Draw::CornerPull,
    };
    let compact = ((1.57 - footprint.x / footprint.y) / 0.60).clamp(0., 1.);
    let box_height = match draw {
        Draw::Sealed => 1.,
        Draw::LowTab => use_state.random_range(0.82..0.93),
        _ => use_state.random_range(0.48..0.68) + 0.11 * compact,
    };
    (draw, box_height)
}

/// The sizing sampler supplies the closed package dimensions. Add just the
/// exposed draw to the reservation before the support/clearance tests run.
pub(crate) fn fit_proposal(o: &mut IndoorObject) {
    let case = installed_case(o.seed);
    o.variant = case as u32;
    let mut rng = stream(o.seed, u64::from_le_bytes(*b"TissDraw"));
    let (_, case_fraction) = draw_profile(&mut rng, case, o.size.xz());
    o.size.y /= case_fraction;
}

pub fn parameters(o: &IndoorObject) -> TissueProgram {
    let Some(case) = Case::from_variant(o.variant) else {
        return legacy::parameters(o);
    };
    let mut r = stream(o.seed, u64::from_le_bytes(*b"TissForm"));
    let mut use_state = stream(o.seed, u64::from_le_bytes(*b"TissDraw"));
    let s = o.size;
    let (draw, box_height) = draw_profile(&mut use_state, case, s.xz());
    let (finish, wall, corner, lip): (Surface, f32, f32, f32) = match case {
        Case::FoldingCarton => (
            Surface::Paper,
            r.random_range(0.00038..0.00076),
            r.random_range(0.001..0.0035),
            r.random_range(0.0005..0.0014),
        ),
        Case::MoldedCover => (
            Surface::Plastic,
            r.random_range(0.002..0.004),
            r.random_range(0.008..0.022),
            r.random_range(0.002..0.005),
        ),
        Case::TimberSleeve => (
            Surface::Wood,
            r.random_range(0.0035..0.007),
            r.random_range(0.002..0.010),
            r.random_range(0.001..0.003),
        ),
        Case::MetalCover => (
            if r.random_bool(0.23) {
                Surface::Chrome
            } else {
                Surface::Metal
            },
            r.random_range(0.0008..0.0018),
            r.random_range(0.003..0.013),
            r.random_range(0.0012..0.0035),
        ),
    };
    let opening = Vec2::new(r.random_range(0.48..0.78), r.random_range(0.19..0.40));
    let oval = r.random_bool(0.52);
    let rounding = r.random_range(0.36..0.97);
    TissueProgram {
        box_height,
        opening,
        lean: use_state.random_range(-0.14..0.14),
        fold: use_state.random_range(0.055..0.145),
        phase: use_state.random_range(-PI..PI),
        finish,
        dispenser: Some(Dispenser {
            case,
            draw,
            wall_m: wall,
            corner_m: corner.min(s.x.min(s.z) * 0.18),
            lip_m: lip,
            mouth_rounding: if oval {
                Vec2::splat(rounding)
            } else {
                Vec2::new(rounding * (s.z * opening.y) / (s.x * opening.x), rounding)
                    .min(Vec2::splat(0.965))
            },
            mouth_offset_m: s.xz()
                * Vec2::new(r.random_range(-0.047..0.047), r.random_range(-0.062..0.062)),
            fill_fraction: use_state.random_range(0.30..0.87),
            pleats: use_state.random_range(2..=4),
            crest: use_state.random_range(-0.82..0.82),
            curl: use_state.random_range(-0.15..0.15),
            flare: use_state.random_range(0.60..0.95),
        }),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    if let Some(d) = p.dispenser {
        case::build(a, o, &p, d);
        if d.draw != Draw::Sealed {
            sheet::build(a, o, &p, d);
        }
    } else {
        legacy::build(a, o, &p);
    }
}
