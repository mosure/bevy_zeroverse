//! Local +Z is the customer side. Each sale unit and its peripherals share the
//! admitted envelope and instance identity; the checkout form is archived.
use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

mod fixtures;
pub(super) mod legacy;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 291;
const APPAREL_SELF_CHECKOUT: u32 = 0x514;
const COUNTER: &str = "counter";
const CONTROLS: &str = "control_panel";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CheckoutForm {
    ConveyorLane,
    BasketRegister,
    ApparelWrap,
    SelfCheckout,
}

impl CheckoutForm {
    pub(crate) fn tag(self) -> u32 {
        match self {
            Self::ConveyorLane => 0x510,
            Self::BasketRegister => 0x511,
            Self::ApparelWrap => 0x512,
            Self::SelfCheckout => 0x513,
        }
    }

    pub(crate) fn for_object(o: &IndoorObject) -> Option<Self> {
        if o.kind != ObjectKind::CheckoutCounter {
            return None;
        }
        match o.variant {
            0x510 => Some(Self::ConveyorLane),
            0x511 => Some(Self::BasketRegister),
            0x512 => Some(Self::ApparelWrap),
            0x513 | APPAREL_SELF_CHECKOUT => Some(Self::SelfCheckout),
            _ => None,
        }
    }

    pub(crate) fn self_service_tag(apparel: bool) -> u32 {
        if apparel {
            APPAREL_SELF_CHECKOUT
        } else {
            Self::SelfCheckout.tag()
        }
    }
    pub(crate) fn apparel_assortment(o: &IndoorObject) -> bool {
        o.variant == APPAREL_SELF_CHECKOUT || Self::for_object(o) == Some(Self::ApparelWrap)
    }

    pub(crate) fn staffed(activity: IndoorLayout, width: f32, seed: u64) -> Self {
        if activity == IndoorLayout::ClothingStore {
            Self::ApparelWrap
        } else if width >= 1.6 && stream(seed, u64::from_le_bytes(*b"ScanLane")).random_bool(0.76) {
            Self::ConveyorLane
        } else {
            Self::BasketRegister
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CheckoutStation {
    pub form: CheckoutForm,
    pub apparel_assortment: bool,
    pub cabinet: Surface,
    pub deck: Surface,
    pub worktop_height_m: f32,
    pub terminal_on_right: bool,
    /// Width of the terminal bay as a fraction of the installed station.
    pub terminal_fraction: f32,
    pub screen_width_m: f32,
    pub screen_height_m: f32,
    pub screen_tilt_radians: f32,
    pub bag_platform_height_m: f32,
    pub bag_rails: bool,
    pub wrapping_roll: bool,
}

pub(crate) fn self_checkout_size(r: &mut impl Rng) -> Vec3 {
    Vec3::new(
        r.random_range(0.90..1.46),
        r.random_range(1.34..1.61),
        r.random_range(0.57..0.76),
    )
}

pub fn checkout_station(o: &IndoorObject) -> Option<CheckoutStation> {
    let form = CheckoutForm::for_object(o)?;
    let installation = if form == CheckoutForm::SelfCheckout {
        o.finish_seed.unwrap_or(o.seed)
    } else {
        o.seed
    };
    let mut r = stream(installation, u64::from_le_bytes(*b"ShopDesk"));
    let mut stock = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"TillSkin") ^ u64::from(form.tag()),
    );
    let wrap = form == CheckoutForm::ApparelWrap;
    let kiosk = form == CheckoutForm::SelfCheckout;
    let cabinet = if wrap {
        [Surface::Wood, Surface::Paint, Surface::WoodEdge][stock.random_range(0..3)]
    } else {
        [Surface::Paint, Surface::Metal, Surface::Plastic][stock.random_range(0..3)]
    };
    let deck = if wrap {
        [Surface::Wood, Surface::WoodEdge, Surface::Plastic][stock.random_range(0..3)]
    } else {
        Surface::Metal
    };
    let top = r.random_range(0.83..0.94);
    let bay = if kiosk {
        r.random_range(0.42..0.59)
    } else {
        r.random_range(0.28..0.42)
    };
    let screen_width = r.random_range(0.265_f32..0.416).min(o.size.x * bay - 0.038);
    let screen_height = (screen_width
        * if kiosk {
            r.random_range(0.84..1.23)
        } else {
            r.random_range(0.58..0.78)
        })
    .min(o.size.y - top - 0.060);
    Some(CheckoutStation {
        form,
        apparel_assortment: CheckoutForm::apparel_assortment(o),
        cabinet,
        deck,
        worktop_height_m: top,
        terminal_fraction: bay,
        terminal_on_right: r.random(),
        screen_width_m: screen_width,
        screen_height_m: screen_height,
        screen_tilt_radians: r.random_range(0.09..0.34),
        bag_platform_height_m: if kiosk {
            r.random_range(0.46..0.68)
        } else {
            top
        },
        bag_rails: kiosk && r.random_bool(0.73),
        wrapping_roll: wrap && r.random_bool(0.76),
    })
}

impl CheckoutStation {
    fn bays(self, width: f32) -> [(f32, f32); 2] {
        let hand = if self.terminal_on_right { 1. } else { -1. };
        let core = width * self.terminal_fraction;
        let other = width - core;
        [(hand * 0.5 * other, core), (-hand * 0.5 * core, other)]
    }
    /// Opaque volumes for the camera planner. The raised display only blocks
    /// its own bay, and the bagging platform leaves open space above it.
    pub(crate) fn view_boxes(self, size: Vec3, mut emit: impl FnMut(Vec3, Vec3)) {
        let [(x, core), (work, bay)] = self.bays(size.x);
        let top = self.worktop_height_m;
        if self.form == CheckoutForm::SelfCheckout {
            emit(
                Vec3::new(x, top * 0.5, 0.),
                Vec3::new(core, top, size.z) * 0.5,
            );
            let h = self.bag_platform_height_m;
            emit(
                Vec3::new(work, (h - 0.043) * 0.5, 0.),
                Vec3::new(bay * 0.58, h - 0.043, size.z * 0.64) * 0.5,
            );
            emit(
                Vec3::new(work, h - 0.0215, 0.),
                Vec3::new(bay, 0.043, size.z * 0.86) * 0.5,
            );
        } else {
            emit(Vec3::Y * (top * 0.5), Vec3::new(size.x, top, size.z) * 0.5);
        }
        let display = self.display(size);
        let (s, c) = self.screen_tilt_radians.sin_cos();
        emit(
            display.translation,
            Vec3::new(
                self.screen_width_m,
                self.screen_height_m * c + fixtures::DISPLAY_DEPTH * s,
                self.screen_height_m * s + fixtures::DISPLAY_DEPTH * c,
            ) * 0.5,
        );
    }
    /// World-facing display orientation is derived by applying the instance transform.
    fn display(self, s: Vec3) -> Transform {
        let facing = if self.form == CheckoutForm::SelfCheckout {
            0.
        } else {
            PI
        };
        let tilt = self.screen_tilt_radians;
        let top_extent =
            (self.screen_height_m * tilt.cos() + fixtures::DISPLAY_DEPTH * tilt.sin()) * 0.5;
        Transform::from_xyz(self.bays(s.x)[0].0, s.y - top_extent, -s.z * 0.256)
            .with_rotation(Quat::from_rotation_y(facing) * Quat::from_rotation_x(-tilt))
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, d: CheckoutStation) {
    let s = o.size;
    let [(cx, core), (wx, work)] = d.bays(s.x);
    let top = d.worktop_height_m;
    let mut cuts = TimberCuts::new(o.seed, 0);
    if d.form == CheckoutForm::SelfCheckout {
        fixtures::base(a, cx, core, s.z, top - 0.035, d.cabinet);
        let h = d.bag_platform_height_m;
        // An independent lower pedestal supports the bagging scale.
        fixtures::base(a, wx, work * 0.58, s.z * 0.64, h - 0.043, d.cabinet);
        board(
            a,
            Surface::Metal,
            COUNTER,
            Vec3::new(wx, h - 0.0215, 0.),
            Vec3::new(work, 0.043, s.z * 0.86),
        );
        if d.bag_rails {
            fixtures::bag_holder(a, wx, work, h, s.z);
        }
    } else {
        fixtures::cabinet(a, o, d, &mut cuts);
        let deck = if d.form == CheckoutForm::ConveyorLane {
            Surface::Rubber
        } else {
            d.deck
        };
        let rail = if d.form == CheckoutForm::ConveyorLane {
            0.021
        } else {
            0.
        };
        a.board(
            &mut cuts,
            (
                deck,
                if rail > 0. {
                    crate::scene::procedural_indoor::materials::point_of_sale::stock::BELT
                } else {
                    COUNTER
                },
            ),
            Vec3::new(wx, top - 0.0175, 0.),
            Vec3::new(work, 0.035, s.z - 2. * rail),
            0.,
            GrainAxis::X,
        );
        if rail > 0. {
            for z in [-1., 1.] {
                board(
                    a,
                    Surface::Metal,
                    COUNTER,
                    Vec3::new(wx, top - 0.0175, z * (s.z - rail) * 0.5),
                    Vec3::new(work, 0.035, rail),
                );
            }
        }
    }
    if d.form == CheckoutForm::ApparelWrap {
        a.board(
            &mut cuts,
            (d.deck, COUNTER),
            Vec3::new(cx, top - 0.0175, 0.),
            Vec3::new(core, 0.035, s.z),
            0.,
            GrainAxis::X,
        );
    } else {
        fixtures::scanner(a, cx, core, top, s.z);
    }
    fixtures::terminal(a, o, d);
}
