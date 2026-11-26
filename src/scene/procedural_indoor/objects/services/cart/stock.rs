//! Resolve a cart's job after its top is dressed. Keep the inventory in the
//! cart's rest mesh so it receives the same rigid motion and optical flow.
use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorLayout,
    objects::kitchen::contents::{self, ShelfStock, StoredItem},
    program::IndoorProgram,
};

pub(crate) const INTRODUCED: u32 = 269;
// The low byte is the original furniture variant and is also used by nested
// utility objects. The two following bits persist the job in the manifest.
const JOB_SHIFT: u32 = 8;
pub(super) const JOB_MASK: u32 = 3 << JOB_SHIFT;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CartJob {
    Utility = 0,
    Beverage = 1,
    TableService = 2,
}

impl CartJob {
    pub(crate) fn read(o: &IndoorObject, version: u32) -> Self {
        if version >= INTRODUCED {
            match (o.variant & JOB_MASK) >> JOB_SHIFT {
                1 => return Self::Beverage,
                2 => return Self::TableService,
                _ => {}
            }
        }
        Self::Utility
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Utility => "utility",
            Self::Beverage => "beverage",
            Self::TableService => "table_service",
        }
    }

    pub(super) fn write(self, cart: &mut IndoorObject) {
        cart.variant = (cart.variant & !JOB_MASK) | ((self as u32) << JOB_SHIFT);
    }
}

pub(crate) fn dress_carts(
    objects: &mut [IndoorObject],
    layout: IndoorLayout,
    program: Option<&IndoorProgram>,
) {
    use ObjectKind::*;
    for index in 0..objects.len() {
        let cart = &objects[index];
        if cart.kind != MobileCart {
            continue;
        }
        let at = cart.position.xz();
        let activity = program
            .and_then(|p| {
                p.zones
                    .iter()
                    .find(|z| at.cmpge(z.min).all() && at.cmple(z.max).all())
            })
            .map_or(layout, |z| z.activity);
        let mut printer = false;
        let mut beverage = false;
        let mut dishes = false;
        for child in objects.iter().filter(|o| o.support == Some(cart.id)) {
            match child.kind {
                Printer | Projector => printer = true,
                CoffeeMaker | Kettle | DrinkDispenser => beverage = true,
                Mug | CoffeeCup | Bowl | Dishware | FruitBowl => dishes = true,
                _ => {}
            }
        }
        let draw: f32 = stream(cart.seed, 0x4341_5254_4a4f_4253).random();
        let (drinks, service) = if printer {
            (0., 0.)
        } else if beverage {
            (0.85, 0.10)
        } else if matches!(
            activity,
            IndoorLayout::Kitchenette
                | IndoorLayout::Kitchen
                | IndoorLayout::Cafe
                | IndoorLayout::Restaurant
                | IndoorLayout::Cafeteria
                | IndoorLayout::Breakroom
        ) {
            (0.24, 0.49)
        } else if dishes {
            (0.32, 0.24)
        } else {
            (0.055, 0.045)
        };
        let job = if draw < drinks {
            CartJob::Beverage
        } else if draw < drinks + service {
            CartJob::TableService
        } else {
            CartJob::Utility
        };
        job.write(&mut objects[index]);
    }
}

pub(crate) enum ShelfLoad {
    Utility(Vec<IndoorObject>),
    Service(Vec<StoredItem>),
}

impl ShelfLoad {
    pub(crate) fn sample(o: &IndoorObject, p: &CartProgram, level: usize, version: u32) -> Self {
        let stock = match CartJob::read(o, version) {
            CartJob::Utility => return Self::Utility(cargo(o, p, level)),
            CartJob::Beverage => ShelfStock::BeverageCart,
            CartJob::TableService => ShelfStock::ServingCart,
        };
        if !stream(o.seed, 0x4341_5254_4c4f_4144 + level as u64).random_bool(p.occupancy.into()) {
            return Self::Service(Vec::new());
        }
        let (lo, hi) = p.bay(o.size, level);
        Self::Service(stock.pack(
            o.seed,
            level as u32,
            (lo.xz(), hi.xz()),
            lo.y,
            hi.y - lo.y,
            version,
        ))
    }

    pub(super) fn build(self, a: &mut Assembly, version: u32) {
        match self {
            Self::Utility(items) => {
                for child in items {
                    public::append_at(
                        a,
                        build_object_at_version(&child, None, version),
                        child.transform(),
                    );
                }
            }
            Self::Service(items) => contents::build(a, &items),
        }
    }
}
