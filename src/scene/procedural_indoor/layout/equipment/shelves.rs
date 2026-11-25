//! Choose an installed shelf height from the usable floor/wall interval.
//! Nearby worktops need space for an operator and objects below the brackets.
use super::*;
use crate::scene::procedural_indoor::objects::{kitchen, tables};

impl IndoorManifest {
    pub(super) fn wall_shelf_base(&self, o: &IndoorObject, draw: f32) -> Option<f32> {
        let (lo, hi) = o.bounds();
        let floor = self.floor_height(o.position.xz());
        let mut use_case = stream(o.seed, u64::from_le_bytes(*b"ShelfUse"));
        let load_space = use_case.random_range(0.18..0.34);
        let highest = self.minimum_ceiling_height(lo.xz(), hi.xz()) - load_space - o.size.y;
        let shelf = o.transform();
        let probes = [-0.28, 0., 0.28].map(|x| {
            shelf
                .transform_point(Vec3::new(x * o.size.x, 0., o.size.z * 0.5))
                .xz()
        });
        let top = self
            .objects
            .iter()
            .filter(|s| {
                !s.neighbor
                    && s.support.is_none()
                    && matches!(
                        s.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::Cabinet
                            | ObjectKind::KitchenCounter
                            | ObjectKind::ServiceCounter
                    )
                    && (s.position.y - floor).abs() < 0.12
                    && s.size.y < 1.38
            })
            .filter_map(|s| {
                let inv = s.transform().compute_affine().inverse();
                let local =
                    probes.map(|q| inv.transform_point3(Vec3::new(q.x, s.position.y, q.y)).xz());
                let (lo, hi) = if s.kind == ObjectKind::KitchenCounter {
                    kitchen::counter_top_bounds(s)
                } else {
                    let half = s.size.xz() * 0.5;
                    (-half, half)
                };
                if local
                    .iter()
                    .filter(|p| p.cmpge(lo).all() && p.cmple(hi).all())
                    .count()
                    < 2
                {
                    return None;
                }
                if s.kind.has_table_outline() {
                    let outline = tables::outline(s);
                    if local
                        .iter()
                        .filter(|p| tables::supports_outline(&outline, **p, **p, 0.))
                        .count()
                        < 2
                    {
                        return None;
                    }
                }
                Some((s.position.y + s.size.y, s.kind))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let (low, high) = if let Some((h, kind)) = top {
            if matches!(
                kind,
                ObjectKind::KitchenCounter | ObjectKind::ServiceCounter
            ) {
                (h + 0.46, h + 0.86)
            } else {
                (h + 0.55, h + 1.03)
            }
        } else {
            // Elevations describe the usable top, independently of the depth
            // of its bracket. Both low display shelves and high storage fit.
            (floor + 1.02 - o.size.y, floor + 2.20 - o.size.y)
        };
        let high = high.min(highest);
        (low <= high).then(|| low.lerp(high, 0.68 * draw + 0.32 * use_case.random::<f32>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;
    #[test]
    fn wall_shelf_height_respects_worktop_and_overhead_clearance() {
        let mut scene =
            IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        let mut shelf = audit::fixture(ObjectKind::WallShelf, 1);
        shelf.position = Vec3::new(0., 1.4, 0.);
        shelf.yaw = 0.84;
        let desk_tf = shelf.transform();
        let mut counter = audit::fixture(ObjectKind::KitchenCounter, 2);
        counter.size = Vec3::new(2., 0.94, 0.62);
        counter.yaw = shelf.yaw;
        counter.position = desk_tf.transform_point(Vec3::new(0., 0., 0.22)).with_y(0.);
        let mut bounds = [f32::INFINITY, f32::NEG_INFINITY];
        for i in 0..96 {
            shelf.seed = i;
            let u = i as f32 / 95.;
            let empty = scene.wall_shelf_base(&shelf, u).unwrap();
            bounds[0] = bounds[0].min(empty + shelf.size.y);
            bounds[1] = bounds[1].max(empty + shelf.size.y);
            scene.objects.push(counter.clone());
            let fitted = scene.wall_shelf_base(&shelf, u).unwrap();
            assert!((0.46 - 1e-6..=0.86 + 1e-6).contains(&(fitted - counter.size.y)));
            assert_eq!(Some(fitted), scene.wall_shelf_base(&shelf, u));
            scene.objects.pop();
        }
        assert!(bounds[0] < 1.22 && bounds[1] > 2.00, "{bounds:?}");
        scene.envelope = None;
        scene.room_size.y = 1.40;
        scene.objects.push(counter);
        assert!(scene.wall_shelf_base(&shelf, 0.5).is_none());
    }
}
