//! Full-size sheet stock and its wall reservation are sampled together. Rolling
//! fabric onto the drum removes material from the top of the visible image.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum ScreenFormat {
    Square,
    FourThree,
    SixteenTen,
    SixteenNine,
    Cinema,
}
impl ScreenFormat {
    pub fn aspect(self) -> f32 {
        match self {
            Self::Square => 1.,
            Self::FourThree => 4. / 3.,
            Self::SixteenTen => 1.6,
            Self::SixteenNine => 16. / 9.,
            Self::Cinema => 2.35,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Mechanism {
    FixedFrame,
    SpringRoller,
    MotorRoller,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Deployment {
    Presented,
    PartlyLowered,
    Stowed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Coating {
    MatteWhite,
    NeutralGray,
    SatinWhite,
}
impl Coating {
    pub(crate) const ALL: [Self; 3] = [Self::MatteWhite, Self::NeutralGray, Self::SatinWhite];
    pub(crate) fn slot(self) -> usize {
        600 + self as usize
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Installation {
    pub format: ScreenFormat,
    pub mechanism: Mechanism,
    pub deployment: Deployment,
    pub coating: Coating,
    pub full_image_height_m: f32,
    pub sheet_border_m: f32,
    pub stock_length_m: f32,
    pub housing_width_m: f32,
    pub pull_length_m: f32,
    pub pull_x_m: f32,
    pub shell_bevel_m: f32,
}

struct Product {
    format: ScreenFormat,
    mechanism: Mechanism,
    deployment: Deployment,
    extension: f32,
    border: f32,
    clearance: f32,
    leader: f32,
    pull: f32,
    pull_x: f32,
    bevel: f32,
    case_height: f32,
    bar_radius: f32,
    lateral: f32,
    rail_extension: f32,
    bracket_span: f32,
    bracket_offset: f32,
}
impl Product {
    fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"AVscreen"));
        let format = match r.random_range(0..100) {
            0..=7 => ScreenFormat::Square,
            8..=28 => ScreenFormat::FourThree,
            29..=54 => ScreenFormat::SixteenTen,
            55..=96 => ScreenFormat::SixteenNine,
            _ => ScreenFormat::Cinema,
        };
        let mechanism = match r.random_range(0..100) {
            0..=29 => Mechanism::FixedFrame,
            30..=66 => Mechanism::SpringRoller,
            _ => Mechanism::MotorRoller,
        };
        let deployment = if mechanism == Mechanism::FixedFrame {
            Deployment::Presented
        } else {
            match r.random_range(0..100) {
                0..=18 => Deployment::Stowed,
                19..=36 => Deployment::PartlyLowered,
                _ => Deployment::Presented,
            }
        };
        Self {
            format,
            mechanism,
            deployment,
            extension: r.random_range(0.20..0.86),
            border: r.random_range(0.011..0.029),
            clearance: r.random_range(0.020..0.046),
            leader: r.random_range(0.048..0.16),
            pull: if mechanism == Mechanism::SpringRoller {
                r.random_range(0.013..0.032)
            } else {
                0.
            },
            pull_x: r.random_range(-0.11..0.11),
            bevel: r.random_range(0.08..0.32),
            case_height: r.random_range(0.55..0.84),
            bar_radius: r.random_range(0.075..0.14),
            lateral: r.random_range(-0.28..0.28),
            rail_extension: r.random_range(0.08..0.26),
            bracket_span: r.random_range(0.52..0.84),
            bracket_offset: r.random_range(-0.045..0.045),
        }
    }
    fn sheet_width(&self, w: f32) -> f32 {
        if self.mechanism == Mechanism::FixedFrame {
            w
        } else {
            w * (1. - 2. * self.clearance)
        }
    }
    fn image_height(&self, w: f32) -> f32 {
        (self.sheet_width(w) - 2. * w * self.border) / self.format.aspect()
    }
    fn height(&self, w: f32, d: f32) -> f32 {
        let h = self.image_height(w);
        if self.mechanism == Mechanism::FixedFrame {
            h + 2. * w * self.border
        } else {
            d * (self.case_height + 2. * self.bar_radius)
                + w * (self.border + self.leader + self.pull)
                + h
        }
    }
}
pub(crate) fn dimensions(seed: u64, width: f32, depth: f32) -> Vec3 {
    Vec3::new(width, Product::sample(seed).height(width, depth), depth)
}

pub(super) fn parameters(o: &IndoorObject) -> ScreenProgram {
    let product = Product::sample(o.seed);
    let s = o.size;
    // Review fixtures and imported instances can reserve an arbitrary aspect.
    let fit = (s.y / product.height(s.x, s.z)).min(1.);
    let (w, d) = (s.x * fit, s.z * fit);
    let framed = product.mechanism == Mechanism::FixedFrame;
    let h = product.image_height(w);
    let sheet_w = product.sheet_width(w);
    let border = w * product.border;
    let case_h = d * product.case_height;
    let bar = if framed { 0. } else { d * product.bar_radius };
    let rail = 2. * bar;
    let leader = w * product.leader;
    let top = if framed { s.y } else { s.y - case_h };
    let extension = if framed {
        h + 2. * border
    } else {
        match product.deployment {
            Deployment::Presented => rail + border + h + leader * product.extension,
            Deployment::PartlyLowered => rail + border + h * product.extension,
            Deployment::Stowed => rail + border * 0.18,
        }
    };
    let bottom = top - extension;
    let x = product.lateral * (w - sheet_w);
    let z = if framed {
        -s.z * 0.5 + d * 0.55
    } else {
        d * 0.08
    };
    let image_bottom = (bottom + rail + border).min(top);
    let image_top = (image_bottom + h).min(top);
    let half_image = if product.deployment == Deployment::Stowed {
        0.
    } else {
        sheet_w * 0.5 - border
    };
    let mut f = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"AVfinish"),
    );
    let coating = match f.random_range(0..10) {
        0..=6 => Coating::MatteWhite,
        7..=8 => Coating::NeutralGray,
        _ => Coating::SatinWhite,
    };
    ScreenProgram {
        installation: Some(Installation {
            format: product.format,
            mechanism: product.mechanism,
            deployment: product.deployment,
            coating,
            full_image_height_m: h,
            sheet_border_m: border,
            stock_length_m: if framed {
                h + 2. * border
            } else {
                border + h + leader
            },
            housing_width_m: w,
            pull_length_m: w * product.pull,
            pull_x_m: x + w * product.pull_x,
            shell_bevel_m: d * product.bevel,
        }),
        framed,
        drop_fraction: if framed {
            1.
        } else {
            extension / (rail + border + h + leader)
        },
        case_height_m: case_h,
        case_depth_m: d * 0.80,
        cap_width_m: w * product.clearance * 0.52,
        sheet_width_m: sheet_w,
        sheet_center_x_m: x,
        sheet_bottom_m: bottom,
        sheet_top_m: top,
        sheet_front_m: z,
        bar_radius_m: bar,
        bar_width_m: sheet_w + (w - sheet_w) * product.rail_extension,
        surface_lo: Vec3::new(x - half_image, image_bottom, z),
        surface_hi: Vec3::new(x + half_image, image_top, z),
        housing: [Surface::Metal, Surface::Plastic, Surface::WoodEdge][f.random_range(0..3)],
        trim: [Surface::Metal, Surface::Plastic, Surface::Chrome][f.random_range(0..3)],
        mount_x_m: [
            w * (product.bracket_offset - product.bracket_span * 0.5),
            w * (product.bracket_offset + product.bracket_span * 0.5),
        ],
    }
}
