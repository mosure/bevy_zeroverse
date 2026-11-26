//! Service follows the chairs and booth places actually admitted, including curved
//! tabletops. Clear, drinks-only and work tables remain part of the same room.
use super::*;
use crate::scene::procedural_indoor::{
    layout::dining_places::MAX_DINING_REACH_M,
    objects::{kitchen::dishware, tables},
    placement::rotated_half,
};
use rand::seq::SliceRandom;
use std::cell::OnceCell;

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn is_dining_table(
        &self,
        table: &IndoorObject,
    ) -> bool {
        if table.kind != ObjectKind::Table || table.neighbor {
            return false;
        }
        let activity = self
            .functional_programs()
            .iter()
            .find(|program| program.members.contains(&table.id))
            .map_or(self.layout, |program| program.activity);
        matches!(
            activity,
            IndoorLayout::Cafe
                | IndoorLayout::Cafeteria
                | IndoorLayout::Restaurant
                | IndoorLayout::Kitchen
                | IndoorLayout::Kitchenette
                | IndoorLayout::Breakroom
        )
    }

    pub(super) fn dress_meal(
        &mut self,
        table: &IndoorObject,
        plan: &composition::ClutterProgram,
        budget: &mut SurfaceBudget,
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
            let center = edge
                + inward * (inward.abs().dot(half) + r.random_range(0.032..0.090))
                + tangent * r.random_range(-0.09..0.09);
            let mut place = |kind, size: Vec3, pos: Vec2, yaw, variant, r: &mut ChaCha8Rng| {
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
                if kind == ObjectKind::Dishware {
                    object.interaction_target = Some(seat.seat_id);
                }
                if shared_stock && matches!(kind, ObjectKind::Dishware | ObjectKind::Flatware) {
                    object.finish_seed = Some(stock);
                }
                if self.admit_supported_prop(table, object) {
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
                &mut r,
            ) {
                continue;
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
                place(ObjectKind::Flatware, size, pos, angle, None, &mut r);
            }
            if r.random_bool(0.57) {
                let (kind, size) = decor::drink(&mut r);
                let pos = center
                    - tangent
                        * handed
                        * (dish.x * 0.5 + size.x * 0.5 + r.random_range(0.035..0.095))
                    + inward * r.random_range(0.04..0.16);
                let angle = r.random_range(-PI..PI);
                place(kind, size, pos, angle, None, &mut r);
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
