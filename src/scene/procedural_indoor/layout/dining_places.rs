//! Dining places in tabletop-local metres. Banquettes face local +Z and may
//! accommodate several diners on a single continuous cushion.
use super::*;
use crate::scene::procedural_indoor::objects::{kitchen::dishware, public::banquette, tables};
use std::cell::OnceCell;

pub(crate) const MAX_DINING_REACH_M: f32 = 0.90;

#[derive(Clone, Copy)]
pub(crate) struct DiningPlace {
    pub seat_id: usize,
    pub position: Vec2,
    /// The edge directly in front of a booth diner. A movable chair can use
    /// the nearest part of the top even when facing a rounded corner.
    pub booth_edge: Option<(Vec2, Vec2)>,
}

impl IndoorManifest {
    pub(crate) fn valid_place_setting(&self, item: &IndoorObject, seat: &IndoorObject) -> bool {
        let Some(table) = item.support.and_then(|id| self.objects.get(id)) else {
            return false;
        };
        if (item.kind != ObjectKind::TableNapkin && dishware::served_style(item).is_none())
            || table.kind != ObjectKind::Table
            || seat.interaction_target != Some(table.id)
        {
            return false;
        }
        let reach = MAX_DINING_REACH_M + 1e-4;
        match seat.kind {
            ObjectKind::Chair => seat.position.xz().distance(item.position.xz()) <= reach,
            ObjectKind::Banquette => {
                let pos = table
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(item.position)
                    .xz();
                booth_places(table, seat, &tables::outline(table))
                    .any(|place| place.position.distance(pos) <= reach)
            }
            _ => false,
        }
    }

    pub(crate) fn dining_places(
        &self,
        table: &IndoorObject,
        outline: &OnceCell<Vec<Vec2>>,
    ) -> Vec<DiningPlace> {
        let inverse = table.transform().compute_affine().inverse();
        let mut places = Vec::new();
        for seat in self
            .objects
            .iter()
            .filter(|o| o.interaction_target == Some(table.id))
        {
            match seat.kind {
                ObjectKind::Chair => places.push(DiningPlace {
                    seat_id: seat.id,
                    position: inverse.transform_point3(seat.position).xz(),
                    booth_edge: None,
                }),
                ObjectKind::Banquette => places.extend(booth_places(
                    table,
                    seat,
                    outline.get_or_init(|| tables::outline(table)),
                )),
                _ => {}
            }
        }
        places
    }
}

pub(crate) fn booth_places<'a>(
    table: &IndoorObject,
    bench: &IndoorObject,
    outline: &'a [Vec2],
) -> impl Iterator<Item = DiningPlace> + 'a {
    let from_bench =
        table.transform().compute_affine().inverse() * bench.transform().compute_affine();
    let front = from_bench.transform_vector3(Vec3::Z).xz();
    let across = from_bench.transform_vector3(Vec3::X).xz();
    let origin = from_bench
        .transform_point3(Vec3::Z * (0.10 * bench.size.z))
        .xz();
    let (left, right) = outline
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
            let x = (*p - origin).dot(across);
            (lo.min(x), hi.max(x))
        });
    // Allow shoulder room at the ends of the cushion and leave the diner a
    // usable width of tabletop even where the bench extends past a short top.
    let half = 0.5 * banquette::cushion_width(bench) - 0.22;
    let lo = (-half).max(left + 0.15);
    let hi = half.min(right - 0.15);
    let mut r = stream(bench.seed, u64::from_le_bytes(*b"BoothSet"));
    let pitch = r.random_range(0.52_f32..0.66);
    let slots = if hi >= lo && (table.position.y - bench.position.y).abs() < 0.015 {
        (((hi - lo) / pitch).floor() + 1.).min(6.) as usize
    } else {
        0
    };
    let slack = (hi - lo - slots.saturating_sub(1) as f32 * pitch).max(0.);
    let first = lo + slack * r.random_range(0.12..0.88);
    let id = bench.id;
    (0..slots).filter_map(move |i| {
        let position = origin + across * (first + i as f32 * pitch);
        let (edge, tangent) = facing_edge(outline, position, front)?;
        Some(DiningPlace {
            seat_id: id,
            position,
            booth_edge: Some((edge, tangent)),
        })
    })
}

fn facing_edge(outline: &[Vec2], seat: Vec2, front: Vec2) -> Option<(Vec2, Vec2)> {
    outline
        .iter()
        .zip(outline.iter().cycle().skip(1))
        .find_map(|(&a, &b)| {
            let edge = b - a;
            let tangent = edge.try_normalize()?;
            // The convex top has a single entering edge. Reject grazes where a
            // setting would be beside a diner instead of in front of them.
            if front.dot(tangent.perp()) < 0.35 {
                return None;
            }
            let cross = front.perp_dot(edge);
            let distance = (a - seat).perp_dot(edge) / cross;
            let u = (a - seat).perp_dot(front) / cross;
            ((0.0..=0.70).contains(&distance) && (0.0..=1.0).contains(&u))
                .then_some((seat + distance * front, tangent))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;

    #[test]
    fn booth_places_follow_cushion_top_overlap_facing_and_elevation() {
        let mut table = audit::fixture(ObjectKind::Table, 4);
        table.size = Vec3::new(1.36, 0.75, 0.85);
        table.position = Vec3::new(3.2, 0.17, -2.45);
        table.yaw = 0.93;
        let outline = vec![
            Vec2::new(-0.68, -0.425),
            Vec2::new(0.68, -0.425),
            Vec2::new(0.68, 0.425),
            Vec2::new(-0.68, 0.425),
        ];
        let mut bench = audit::fixture(ObjectKind::Banquette, 97);
        bench.size = Vec3::new(2.8, 1.08, 0.70);
        bench.yaw = table.yaw;
        let pose = table.transform();
        bench.position = pose.transform_point(Vec3::new(0.33, 0., -0.96));
        for seed in 0..32 {
            bench.seed = seed;
            let places: Vec<_> = booth_places(&table, &bench, &outline).collect();
            assert!(places.len() >= 2);
            for p in &places {
                let (edge, tangent) = p.booth_edge.unwrap();
                assert!((p.position.y + 0.89).abs() < 1e-5 && (edge.y + 0.425).abs() < 1e-5);
                assert!((edge.x - p.position.x).abs() < 1e-5 && edge.x.abs() <= 0.531);
                assert!(tangent.dot(Vec2::X) > 0.99);
            }
            for pair in places.windows(2) {
                assert!((0.519..0.661).contains(&pair[0].position.distance(pair[1].position)));
            }
        }
        bench.yaw += std::f32::consts::PI;
        assert_eq!(booth_places(&table, &bench, &outline).count(), 0);
        bench.yaw = table.yaw;
        for offset in [
            Vec3::new(0., 0., -1.5),
            Vec3::new(2.5, 0., -0.96),
            Vec3::new(0., 0.17, -0.96),
        ] {
            bench.position = pose.transform_point(offset);
            assert_eq!(booth_places(&table, &bench, &outline).count(), 0);
        }
    }
}
