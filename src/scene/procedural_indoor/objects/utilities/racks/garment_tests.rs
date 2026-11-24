use super::{
    dressing::{garments, GarmentKind},
    *,
};
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    objects::audit,
};
use std::collections::{BTreeMap, BTreeSet};

fn fixture(seed: u64) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::CoatRack, seed);
    o.size = dimensions(&mut stream(seed, 817));
    o
}

#[test]
fn belongings_fit_the_admitted_volume_and_reuse_independent_fabric_finishes() {
    let mut kinds = BTreeMap::new();
    let mut slots = BTreeSet::new();
    let mut empty = 0;
    for seed in 0..512 {
        let o = fixture(seed);
        let p = parameters(&o);
        let items = garments(&o, &p);
        let mut bounds = Vec::new();
        empty += usize::from(items.is_empty());
        for item in items {
            *kinds.entry(format!("{:?}", item.kind)).or_insert(0) += 1;
            slots.insert(item.finish_slot);
            let mut a = Assembly::default();
            item.build(&mut a);
            let info =
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed} {:?}: {e}", item.kind));
            assert!(
                info.triangles <= item.triangle_budget(),
                "{seed} {:?}: {}",
                item.kind,
                info.triangles
            );
            assert!(a
                .parts
                .keys()
                .all(|(_, l)| super::super::super::part_label(l) == "clothes"));
            let (min, max) = a.bounds();
            let (lo, hi) = item.footprint();
            assert!(min.xz().cmpge(lo - 0.0001).all() && max.xz().cmple(hi + 0.0001).all());
            assert!(
                min.y > p.base_height + 0.035,
                "{seed} {:?}: hem {}, base {}",
                item.kind,
                min.y,
                p.base_height
            );
            for &(a, b) in &bounds {
                assert!(!(lo.cmplt(b).all() && hi.cmpgt(a).all()));
            }
            bounds.push((lo, hi));
        }
    }
    assert!(empty > 60 && empty < 370, "{empty} bare racks");
    assert_eq!(kinds.len(), 3);
    assert!(kinds.values().all(|&n| n > 10), "{kinds:?}");
    assert!(slots.len() >= 8);
}

fn nearest_hit(g: &Geometry, ray: Vec3, up: Vec3) -> f32 {
    g.indices
        .chunks_exact(3)
        .filter_map(|idx| {
            let a = Vec3::from(g.positions[idx[0] as usize]);
            let e1 = Vec3::from(g.positions[idx[1] as usize]) - a;
            let e2 = Vec3::from(g.positions[idx[2] as usize]) - a;
            let h = up.cross(e2);
            let det = e1.dot(h);
            if det.abs() < 1e-9 {
                return None;
            }
            let q = ray - a;
            let v = q.dot(h) / det;
            let c = q.cross(e1);
            let w = c.dot(up) / det;
            let t = e2.dot(c) / det;
            (v >= -1e-4 && w >= -1e-4 && v + w <= 1.0001 && t > 0.).then_some(t)
        })
        .reduce(f32::min)
        .unwrap()
}

// Independent upward ray against the front and rear cloth at the top of a
// scarf. The innermost textile must touch the support all across its crown.
#[test]
fn scarves_turn_over_the_physical_rail_or_hook_and_keep_their_tails_separate() {
    let mut mounts = [0; 2];
    for seed in 0..192 {
        let o = fixture(seed);
        let p = parameters(&o);
        for item in garments(&o, &p)
            .iter()
            .filter(|g| g.kind == GarmentKind::Scarf)
        {
            let i = item.support_index;
            if p.rail {
                let (w, wire, radius, neck) = p.hanger(&o);
                assert!((item.anchor_m.y - (neck - w * 0.26)).abs() < 1e-6);
                assert!((item.anchor_m.z + radius).abs() + item.shoulder_width_m * 0.5 < w * 0.40);
                assert!(
                    (item.anchor_m.x - p.hanger_x[i]).abs() < 1e-6
                        && (wire - item.support_radius_m).abs() < 1e-7
                );
            } else {
                let [_, a, b] = p.hooks[i].path();
                let axis = (b - a).normalize();
                let center = (item.anchor_m - a).dot(axis);
                assert!(
                    item.anchor_m.distance(a + axis * center) < 1e-5
                        && center > item.shoulder_width_m * 0.5
                );
                assert!(center + item.shoulder_width_m * 0.5 < a.distance(b));
            }
            let mut g = Geometry::default();
            scarf::build(&mut g, item);
            let side = item.support_axis.cross(Vec3::Y).normalize();
            let up = side.cross(item.support_axis);
            for u in [-0.35, 0., 0.35] {
                let ray = item.anchor_m + item.support_axis * (item.shoulder_width_m * u);
                let nearest = nearest_hit(&g, ray, up);
                assert!((nearest - item.support_radius_m).abs() < 0.0005);
            }
            let low = g
                .positions
                .iter()
                .map(|v| Vec3::from(*v))
                .filter(|v| v.y < item.anchor_m.y - item.length_m * 0.40);
            assert!(low
                .clone()
                .any(|v| (v - item.anchor_m).dot(side) > item.support_radius_m));
            assert!(low
                .into_iter()
                .any(|v| (v - item.anchor_m).dot(side) < -item.support_radius_m));
            mounts[usize::from(p.rail)] += 1;
        }
    }
    assert!(mounts.iter().all(|&n| n > 5), "{mounts:?}");
}

#[test]
fn outerwear_is_sewn_closed_and_rests_on_both_hanger_shoulders() {
    let mut count = [0; 2];
    for seed in 0..192 {
        let o = fixture(seed);
        let p = parameters(&o);
        for g in garments(&o, &p)
            .iter()
            .filter(|g| g.kind != GarmentKind::Scarf)
        {
            let mut a = Assembly::default();
            g.build(&mut a);
            let cloth = a
                .parts
                .iter()
                .find(|((s, _), _)| *s == g.surface)
                .unwrap()
                .1;
            let mut edges = BTreeMap::new();
            for tri in cloth.indices.chunks_exact(3) {
                let points = [tri[0], tri[1], tri[2]]
                    .map(|i| cloth.positions[i as usize].map(|x| (x * 1e6).round() as i32));
                for j in 0..3 {
                    let x = points[j];
                    let y = points[(j + 1) % 3];
                    assert_ne!(x, y, "collapsed seam at {seed}");
                    *edges.entry((x.min(y), x.max(y))).or_insert(0usize) += 1;
                }
            }
            assert!(
                edges.values().all(|&n| n == 2),
                "open garment seam at {seed}"
            );
            let (width, wire, radius, neck) = p.hanger(&o);
            for side in [-1., 1.] {
                let offset = width * 0.33;
                let wire_center = Vec3::new(
                    p.hanger_x[g.support_index],
                    neck - 0.52 * offset,
                    -radius + side * offset,
                );
                // Cast onto the exterior from above so the folded lining can never mask a gap.
                let hit = nearest_hit(cloth, wire_center + Vec3::Y * 0.03, Vec3::NEG_Y);
                let gap = 0.03 - hit - wire;
                assert!(gap.abs() < 0.003, "hanger to shoulder on {seed}: {gap} m");
            }
            count[usize::from(g.kind == GarmentKind::Overcoat)] += 1;
        }
    }
    assert!(count.iter().all(|&n| n >= 8), "{count:?}");
}

#[test]
fn entry_storage_uses_the_room_side_and_preserves_the_door_approach() {
    let mut installed = 0;
    for seed in 8020..8092 {
        let room = IndoorManifest::generate_with_humans(seed, IndoorLayout::Reception, 0.72, 0, 0.)
            .unwrap();
        for rack in room
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::CoatRack && o.variant == ENTRY_RACK)
        {
            let door = room.entry_door.as_ref().unwrap();
            let local = door
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(rack.position);
            assert!(local.z < -0.36 && local.x.abs() > door.width * 0.5 + 0.23);
            let (a, b) = rack.bounds();
            assert!(room.placement_clear(rack, 0.035) && !room.entry_approach_overlaps(a, b));
            assert_eq!(
                1,
                room.objects
                    .iter()
                    .filter(|o| !o.neighbor && o.kind == ObjectKind::CoatRack)
                    .count()
            );
            installed += 1;
        }
    }
    assert!(installed >= 6, "{installed} entry installations");
}
