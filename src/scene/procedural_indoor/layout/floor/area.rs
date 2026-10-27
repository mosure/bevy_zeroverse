//! Low-pile area rugs are physical supports, not intersecting floor decals.
//! Store the exact contour once; camera/pose queries never regenerate its mesh.
use super::*;
use crate::scene::procedural_indoor::{envelope::polygon, objects};

/// Ceiling fixtures are admitted before these floor supports. Reserve this
/// possible floor rise so a later rug cannot invalidate their headroom.
pub(crate) const MAX_THICKNESS_M: f32 = 0.009;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RugSupport {
    pub object: usize,
    pub outline: Vec<Vec2>,
    pub min: Vec2,
    pub max: Vec2,
    pub bottom: f32,
    pub top: f32,
}

impl RugSupport {
    pub fn from_object(o: &IndoorObject) -> Self {
        let tf = o.transform();
        let outline: Vec<_> = objects::floor::rug_outline(o)
            .into_iter()
            .map(|p| tf.transform_point(Vec3::new(p.x, 0., p.y)).xz())
            .collect();
        Self {
            object: o.id,
            min: outline
                .iter()
                .copied()
                .fold(Vec2::splat(f32::INFINITY), Vec2::min),
            max: outline
                .iter()
                .copied()
                .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max),
            outline,
            bottom: o.position.y,
            top: o.position.y + o.size.y,
        }
    }
    pub fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all()
            && p.cmple(self.max).all()
            && polygon::contains(&self.outline, p, 0.002)
    }
    pub fn supports(&self, lo: Vec3, hi: Vec3) -> bool {
        lo.xz().cmpge(self.min).all()
            && hi.xz().cmple(self.max).all()
            && polygon::box_inside(&self.outline, lo.xz(), hi.xz(), 0.03)
    }
    pub(super) fn ceiling_clear(&self, scene: &IndoorManifest) -> bool {
        scene
            .ceiling_installation
            .as_ref()
            .is_none_or(|ceiling| ceiling.admits_floor_support(self.min, self.max, self.top))
    }
}
impl IndoorManifest {
    pub(in super::super) fn furnish_area_rugs(&mut self) {
        // Functional platforms/cabinet runs retain their original architectural
        // support. This pass targets freestanding gathering/work furniture.
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Table
                            | ObjectKind::CoffeeTable
                            | ObjectKind::Sofa
                            | ObjectKind::Desk
                    )
            })
            .cloned()
            .collect();
        let mut r = stream(self.seed, 0x4152454152554753);
        for anchor in anchors {
            if self.rug_supports.len() >= 2 || !r.random_bool(0.52) {
                continue;
            }
            // Fit the entire existing seating group, not only its tabletop.
            // Offset seats are the dominant boundary of a realistic area rug.
            let tf = anchor.transform();
            let inv = tf.compute_affine().inverse();
            let mut group_lo = Vec2::splat(f32::INFINITY);
            let mut group_hi = Vec2::splat(f32::NEG_INFINITY);
            for o in self.objects.iter().filter(|o| {
                o.id == anchor.id
                    || o.interaction_target == Some(anchor.id)
                    || (o.kind == ObjectKind::Banquette
                        && o.position.distance(anchor.position) < 2.)
            }) {
                let (a, b) = o.bounds();
                for x in [a.x, b.x] {
                    for z in [a.z, b.z] {
                        let p = inv.transform_point3(Vec3::new(x, a.y, z)).xz();
                        group_lo = group_lo.min(p);
                        group_hi = group_hi.max(p);
                    }
                }
            }
            for _ in 0..12 {
                let extra = r.random_range(0.18..0.65);
                let span = group_hi - group_lo;
                let size = Vec3::new(
                    (span.x + extra * 2.).min(7.5),
                    r.random_range(0.004..MAX_THICKNESS_M),
                    (span.y + extra * 2.).min(6.5),
                );
                let local = (group_lo + group_hi) * 0.5;
                let center = tf.transform_point(Vec3::new(local.x, 0., local.y));
                let mut rug =
                    self.floor_candidate(ObjectKind::Rug, center, size, anchor.yaw, &mut r);
                rug.solid = false;
                let (lo, hi) = rug.bounds();
                let support = RugSupport::from_object(&rug);
                // Reuse the complete architectural/portal/structure validator,
                // but furniture may occupy this rug when fully supported.
                if !self.placement_clear_internal(&rug, 0.025, true)
                    || !self.floor_support_clear(lo, hi)
                    || !support.ceiling_clear(self)
                {
                    continue;
                }
                let mut moved = std::collections::BTreeSet::new();
                let mut valid = true;
                for other in &self.objects {
                    if other.neighbor {
                        continue;
                    }
                    let (a, b) = other.bounds();
                    if lo.x < b.x
                        && hi.x > a.x
                        && lo.z < b.z
                        && hi.z > a.z
                        && a.y < hi.y
                        && b.y > lo.y
                    {
                        if !other.solid || !support.supports(a, b) || (a.y - lo.y).abs() > 0.001 {
                            valid = false;
                            break;
                        }
                        moved.insert(other.id);
                    }
                }
                if !valid || !moved.contains(&anchor.id) {
                    continue;
                }
                // Furniture just outside the rug's bounds is not lifted, but
                // its placement clearance still applies to the new rug. Testing
                // only the lifted group missed nearby plants/island circulation.
                if self.objects.iter().any(|o| {
                    o.solid
                        && !o.neighbor
                        && !moved.contains(&o.id)
                        && seating::furniture_overlap(
                            o,
                            &rug,
                            seating::furniture_margin(o, &rug, 0.),
                        )
                }) {
                    continue;
                }
                // A seated workstation moves as a group; supported props follow
                // their parent, preserving hands/table contacts and port heights.
                if self.objects.iter().any(|o| {
                    o.kind == ObjectKind::Chair
                        && o.interaction_target
                            .is_some_and(|id| moved.contains(&id) != moved.contains(&o.id))
                }) {
                    continue;
                }
                for o in &self.objects {
                    if o.support.is_some_and(|id| moved.contains(&id)) {
                        moved.insert(o.id);
                    }
                }
                let previous: Vec<_> = moved
                    .iter()
                    .map(|&id| (id, self.objects[id].position))
                    .collect();
                self.objects.push(rug.clone());
                self.rug_supports.push(support);
                for &id in &moved {
                    self.objects[id].position.y += rug.size.y;
                }
                if moved.iter().any(|&id| {
                    self.objects[id].solid && !self.placement_clear(&self.objects[id], 0.)
                }) {
                    for (id, position) in previous {
                        self.objects[id].position = position;
                    }
                    self.objects.pop();
                    self.rug_supports.pop();
                    continue;
                }
                break;
            }
        }
        // Empty adjoining lounge floor can host a compact ottoman/coffee table
        // ensemble even when a crowded existing group cannot be fully covered.
        if self.rug_supports.is_empty() {
            self.furnish_rug_nook(&mut r);
        }
    }
    fn furnish_rug_nook(&mut self, r: &mut ChaCha8Rng) {
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && matches!(o.kind, ObjectKind::Sofa | ObjectKind::CoffeeTable))
            .cloned()
            .collect();
        for anchor in anchors.into_iter().take(4) {
            for _ in 0..12 {
                let size = Vec3::new(
                    r.random_range(1.8..3.6),
                    r.random_range(0.004..MAX_THICKNESS_M),
                    r.random_range(1.6..2.8),
                );
                let side = if r.random_bool(0.5) { 1. } else { -1. };
                let p = anchor.transform().transform_point(Vec3::new(
                    r.random_range(-0.3..0.3) * anchor.size.x,
                    0.,
                    side * ((anchor.size.z + size.z) * 0.5 + 0.08),
                ));
                let mut rug = self.floor_candidate(
                    ObjectKind::Rug,
                    p,
                    size,
                    anchor.yaw + r.random_range(-0.22..0.22),
                    r,
                );
                rug.solid = false;
                if !self.placement_clear(&rug, 0.03) {
                    continue;
                }
                let support = RugSupport::from_object(&rug);
                if !support.ceiling_clear(self) {
                    continue;
                }
                self.objects.push(rug.clone());
                self.rug_supports.push(support);
                let kind = if r.random_bool(0.55) {
                    ObjectKind::CoffeeTable
                } else {
                    ObjectKind::Ottoman
                };
                let furniture = self.candidate(
                    kind,
                    p.with_y(rug.position.y + rug.size.y),
                    Vec3::new(
                        r.random_range(0.55..0.95),
                        r.random_range(0.38..0.46),
                        r.random_range(0.45..0.65),
                    ),
                    rug.yaw + r.random_range(-0.25..0.25),
                    r,
                );
                // The new covering can also invalidate neighboring furniture's
                // conservative support footprint. Check both sides of the
                // installation, not just the newly added furniture.
                if !self.placement_clear(&furniture, 0.03)
                    || self.objects.iter().any(|o| {
                        o.solid
                            && !o.neighbor
                            && !self.floor_support_clear(o.bounds().0, o.bounds().1)
                    })
                {
                    self.objects.pop();
                    self.rug_supports.pop();
                    continue;
                }
                self.objects.push(furniture);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_rugs_preserve_clearance_of_unmoved_neighbouring_plants() {
        use crate::scene::procedural_indoor::validation;
        let mut supported_rooms = 0;
        for (seed, human_density) in [(5, 0.), (5, 0.7), (924, 0.), (924, 0.7)] {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                IndoorLayout::Mixed,
                0.7,
                3,
                human_density,
            )
            .unwrap();
            assert!(scene.objects.iter().any(|o| o.kind == ObjectKind::Plant));
            // A tightly constrained seed may reject every optional rug; other
            // rooms must still admit furniture-bearing rugs through this path.
            supported_rooms += usize::from(!scene.rug_supports.is_empty());
            validation::validate_layout(&scene).unwrap();
            validation::validate_geometry(&scene).unwrap();
        }
        assert!(supported_rooms >= 2);
    }

    #[test]
    fn rug_support_replays_and_rejects_edge_overhang_and_tampered_contours() {
        let mut qualified = 0;
        // Rugs are optional and admission changes with furnishings. Exercise
        // every admitted rug in this fixed cohort without assuming one seed
        // must contain a rug at every human-density setting.
        for (seed, human_density) in [(5, 0.), (5, 0.7), (924, 0.), (924, 0.7)] {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                IndoorLayout::Mixed,
                0.7,
                0,
                human_density,
            )
            .unwrap();
            let Some(rug) = scene.rug_supports.first() else {
                continue;
            };
            qualified += 1;
            let object = scene
                .objects
                .iter()
                .find(|o| {
                    o.solid
                        && (o.position.y - rug.top).abs() < 0.001
                        && rug.supports(o.bounds().0, o.bounds().1)
                })
                .unwrap();
            assert!(scene.placement_clear(object, 0.));
            let mut shifted = object.clone();
            shifted.position.x = rug.max.x;
            assert!(!scene.floor_support_clear(shifted.bounds().0, shifted.bounds().1));
            let json = serde_json::to_vec(&scene).unwrap();
            let mut restored: IndoorManifest = serde_json::from_slice(&json).unwrap();
            assert_eq!(scene, restored);
            crate::scene::procedural_indoor::validation::validate_layout(&restored).unwrap();
            restored.rug_supports[0].top += 0.01;
            assert!(
                crate::scene::procedural_indoor::validation::validate_layout(&restored).is_err()
            );
        }
        assert!(qualified >= 2, "fixture cohort lost furniture-bearing rugs");
    }
}
