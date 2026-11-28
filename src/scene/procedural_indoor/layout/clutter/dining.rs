//! Service follows the chairs and booth places actually admitted, including curved
//! tabletops. Clear, drinks-only and work tables remain part of the same room.
use super::*;
use crate::scene::procedural_indoor::{
    clutter::beverages::glassware::{Form, Stock},
    clutter::napkin::Fold,
    layout::dining_places::MAX_DINING_REACH_M,
    objects::{kitchen::dishware, tables},
    placement::rotated_half,
};
use rand::seq::SliceRandom;
use std::cell::OnceCell;

/// Venue stock is shared across tables. Glassware and linen use independent draws.
pub(super) struct TableService {
    lots: [Stock; 2],
    offers: ChaCha8Rng,
    probability: f64,
    activity: IndoorLayout,
    venue: u64,
}
impl TableService {
    fn new(activity: IndoorLayout, venue: u64, table: u64) -> Self {
        let mut offers = stream(table, u64::from_le_bytes(*b"ServeH2O"));
        let mut stock = stream(
            if offers.random_bool(0.19) {
                table
            } else {
                venue
            },
            u64::from_le_bytes(*b"GlassLot"),
        );
        let (chance, stemware) = match activity {
            IndoorLayout::Restaurant => (0.54..0.94, 0.34),
            IndoorLayout::Cafe => (0.25..0.74, 0.07),
            IndoorLayout::Cafeteria => (0.18..0.64, 0.01),
            _ => (0.22..0.71, 0.12),
        };
        let lots = std::array::from_fn(|_| {
            let form = if stock.random_bool(stemware) {
                Form::Goblet
            } else {
                Form::ALL[stock.random_range(0..4)]
            };
            Stock::new(form, stock.random())
        });
        let probability = offers.random_range(chance);
        Self {
            lots,
            offers,
            probability,
            activity,
            venue,
        }
    }
    pub(super) fn offer_glass(&mut self) -> Option<Stock> {
        self.offers
            .random_bool(self.probability)
            .then(|| self.lots[usize::from(self.offers.random_bool(0.25))])
    }
}

struct NapkinStock {
    fold: Fold,
    weave: tables::LinenWeave,
    cut_m: f32,
    place_fraction: f64,
}

impl NapkinStock {
    fn for_table(service: &TableService, table: &IndoorObject) -> Option<Self> {
        let prevalence = match service.activity {
            IndoorLayout::Restaurant => 0.84,
            IndoorLayout::Cafe => 0.43,
            IndoorLayout::Cafeteria => 0.15,
            _ => 0.26,
        };
        let mut stock = stream(service.venue, u64::from_le_bytes(*b"napkinCo"));
        if !stock.random_bool(prevalence) {
            return None;
        }
        let mut table_rng = stream(table.seed, u64::from_le_bytes(*b"napkinTb"));
        let house = tables::LinenWeave::ALL[stock.random_range(0..4)];
        let weave = if table_rng.random_bool(0.28) {
            tables::dining_linen(table).map_or(house, |linen| linen.weave)
        } else {
            house
        };
        let upright = if service.activity == IndoorLayout::Restaurant {
            0.24
        } else {
            0.07
        };
        let house_fold = Fold::sample(&mut stock, upright);
        Some(Self {
            fold: if table_rng.random_bool(0.17) {
                Fold::sample(&mut table_rng, upright)
            } else {
                house_fold
            },
            weave,
            cut_m: stock.random_range(0.32..0.48),
            place_fraction: table_rng.random_range(0.42..0.98),
        })
    }
}

impl IndoorManifest {
    fn dining_venue(&self, table: &IndoorObject) -> Option<(IndoorLayout, u64)> {
        if table.kind != ObjectKind::Table || table.neighbor {
            return None;
        }
        let (activity, zone) = self
            .functional_programs()
            .iter()
            .enumerate()
            .find(|(_, program)| program.members.contains(&table.id))
            .map_or((self.layout, 0), |(index, program)| {
                (program.activity, index as u64 + 1)
            });
        matches!(
            activity,
            IndoorLayout::Cafe
                | IndoorLayout::Cafeteria
                | IndoorLayout::Restaurant
                | IndoorLayout::Kitchen
                | IndoorLayout::Kitchenette
                | IndoorLayout::Breakroom
        )
        .then_some((
            activity,
            self.seed
                .wrapping_add(zone.wrapping_mul(0x9e3779b97f4a7c15)),
        ))
    }
    pub(in crate::scene::procedural_indoor::layout) fn is_dining_table(
        &self,
        table: &IndoorObject,
    ) -> bool {
        self.dining_venue(table).is_some()
    }
    pub(super) fn table_service(&self, table: &IndoorObject) -> Option<TableService> {
        self.dining_venue(table)
            .map(|(activity, venue)| TableService::new(activity, venue, table.seed))
    }

    pub(super) fn dress_meal(
        &mut self,
        table: &IndoorObject,
        plan: &composition::ClutterProgram,
        budget: &mut SurfaceBudget,
        service: &mut TableService,
    ) {
        let mut r = stream(table.seed, 0x44494e494e47);
        if budget.full() {
            return;
        }
        let outline = OnceCell::new();
        let mut seats = self.dining_places(table, &outline);
        if seats.is_empty() {
            return;
        }
        // Drinks belong to diners. A second vessel often accompanies coffee or
        // a meal, and an occasional shared bottle can stay on the table.
        // An unseated serving table keeps its ordinary stock and clutter budget.
        let mut beverages = stream(table.seed, 0x4452494e4b53);
        let first = beverages.random_range(0.30..0.95);
        let second = beverages.random_range(0.08..0.45);
        budget.drink_limit = usize::from(beverages.random_bool(0.20));
        for _ in &seats {
            if beverages.random_bool(first) {
                budget.drink_limit += 1 + usize::from(beverages.random_bool(second));
            }
        }
        if !r.random_bool(if self.layout == IndoorLayout::Restaurant {
            0.88
        } else {
            0.65
        }) {
            return;
        }
        // A cafe set may share its glaze, service style and some place settings,
        // while a shared lunch table can contain several different collections.
        seats.shuffle(&mut r);
        let outline = outline.get_or_init(|| tables::outline(table));
        let served_fraction = r.random_range(0.32..0.98);
        let bowl_fraction = r.random_range(0.08..0.74);
        let utensil_fraction = r.random_range(0.28..0.92);
        let shared_stock = r.random_bool(0.76);
        let stock = r.random();
        let linen = NapkinStock::for_table(service, table);
        let mut linen_rng = stream(table.seed, u64::from_le_bytes(*b"napkinAt"));
        for seat in seats {
            if budget.full() {
                break;
            }
            if !r.random_bool(served_fraction) {
                continue;
            }
            let Some((edge, tangent)) = seat.booth_edge.or_else(|| station(outline, seat.position))
            else {
                continue;
            };
            let inward = tangent.perp();
            let yaw = inward.x.atan2(inward.y);
            let bowl = r.random_bool(bowl_fraction);
            let dish = items::sizing::dish(u8::from(bowl), &mut r);
            let angle = yaw + r.random_range(-0.48..0.48);
            let half = rotated_half(dish.xz(), angle);
            let dish_across = rotated_half(dish.xz(), angle - yaw).x;
            let center = edge
                + inward * (inward.abs().dot(half) + r.random_range(0.032..0.090))
                + tangent * r.random_range(-0.09..0.09);
            let mut place = |kind,
                             size: Vec3,
                             pos: Vec2,
                             yaw,
                             variant,
                             glass: Option<Stock>,
                             r: &mut ChaCha8Rng| {
                let area = size.x * size.z;
                if budget.full()
                    || !budget.permits(kind)
                    || seat.position.distance_squared(pos) > MAX_DINING_REACH_M.powi(2)
                    || budget.area + area > budget.limit
                    || (pos - plan.clear_center)
                        .abs()
                        .cmplt(plan.clear_half + rotated_half(size.xz(), yaw))
                        .all()
                {
                    return false;
                }
                let world =
                    table
                        .transform()
                        .transform_point(Vec3::new(pos.x, table.size.y, pos.y));
                let mut object = self.candidate(kind, world, size, table.yaw + yaw, r);
                if let Some(v) = variant {
                    object.variant = v;
                }
                if let Some(stock) = glass {
                    stock.apply(&mut object);
                }
                if matches!(kind, ObjectKind::Dishware | ObjectKind::TableNapkin) {
                    object.interaction_target = Some(seat.seat_id);
                }
                if shared_stock && matches!(kind, ObjectKind::Dishware | ObjectKind::Flatware) {
                    object.finish_seed = Some(stock);
                }
                if self.admit_supported_prop_with_outline(table, object, Some(outline)) {
                    budget.record(kind, size);
                    true
                } else {
                    false
                }
            };
            if !place(
                ObjectKind::Dishware,
                dish,
                center,
                angle,
                Some(if bowl {
                    dishware::SERVED_BOWL
                } else {
                    dishware::SERVED_PLATE
                }),
                None,
                &mut r,
            ) {
                continue;
            }
            if let Some(linen) = linen
                .as_ref()
                .filter(|s| linen_rng.random_bool(s.place_fraction))
            {
                let size = linen.fold.folded_size(linen.cut_m, &mut linen_rng);
                let favored_side = if linen_rng.random_bool(0.5) { -1. } else { 1. };
                for side in [favored_side, -favored_side] {
                    let angle = yaw + linen_rng.random_range(-0.47..0.47);
                    let margin = linen_rng.random_range(0.019..0.057);
                    let reach = dish_across + rotated_half(size.xz(), angle - yaw).x;
                    let pos = center
                        + tangent * side * (reach + margin)
                        + inward * linen_rng.random_range(-0.022..0.078);
                    if place(
                        ObjectKind::TableNapkin,
                        size,
                        pos,
                        angle,
                        Some(linen.fold.variant(linen.weave)),
                        None,
                        &mut linen_rng,
                    ) {
                        break;
                    }
                }
            }
            let handed = if r.random_bool(0.50) { 1. } else { -1. };
            if r.random_bool(utensil_fraction) {
                let size = items::sizing::loose(ObjectKind::Flatware, &mut r).unwrap();
                let pos = center
                    + tangent
                        * handed
                        * (tangent.abs().dot(half) + size.x * 0.5 + r.random_range(0.025..0.075))
                    + inward * r.random_range(-0.032..0.046);
                let angle = yaw + r.random_range(-0.32..0.32);
                place(ObjectKind::Flatware, size, pos, angle, None, None, &mut r);
            }
            if r.random_bool(0.57) {
                let packaged = decor::drink(&mut r);
                let glass = service.offer_glass();
                let (kind, size) =
                    glass.map_or(packaged, |sku| (ObjectKind::DrinkingGlass, sku.size));
                let pos = center
                    - tangent
                        * handed
                        * (dish.x * 0.5 + size.x * 0.5 + r.random_range(0.035..0.095))
                    + inward * r.random_range(0.04..0.16);
                let angle = r.random_range(-PI..PI);
                place(kind, size, pos, angle, None, glass, &mut r);
            }
        }
    }
}

/// Closest point and CCW tangent on the admitted top, in the table's own frame.
fn station(outline: &[Vec2], seat: Vec2) -> Option<(Vec2, Vec2)> {
    outline
        .iter()
        .zip(outline.iter().cycle().skip(1))
        .filter_map(|(&a, &b)| {
            let edge = b - a;
            let squared = edge.length_squared();
            (squared > 1e-12).then(|| {
                let point = a + edge * ((seat - a).dot(edge) / squared).clamp(0., 1.);
                (point, edge)
            })
        })
        .min_by(|(a, _), (b, _)| {
            a.distance_squared(seat)
                .total_cmp(&b.distance_squared(seat))
        })
        .map(|(point, edge)| (point, edge.normalize()))
}
