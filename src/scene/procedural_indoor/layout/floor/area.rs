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

/// A tight seating group may need the corners of a rectangular rug. Try the
/// chosen cut first, then at most one square-cornered alternative at the same
/// size and transform. Check every foot against the cut that will be rendered.
fn fit_group(
    scene: &IndoorManifest,
    rug: &mut IndoorObject,
    group: &std::collections::BTreeSet<usize>,
) -> Option<RugSupport> {
    let fits = |support: &RugSupport| {
        support.ceiling_clear(scene)
            && group.iter().all(|&id| {
                let (lo, hi) = scene.objects[id].bounds();
                support.supports(lo, hi)
            })
    };
    let support = RugSupport::from_object(rug);
    if fits(&support) {
        return Some(support);
    }
    let straight = objects::floor::rug::Cut::Straight as u32;
    if rug.variant == straight {
        return None;
    }
    rug.variant = straight;
    let support = RugSupport::from_object(rug);
    fits(&support).then_some(support)
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
                        && o.interaction_target.is_none()
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
                let span = group_hi - group_lo;
                // Independently sized borders and an off-centre furniture group:
                // rugs need not reproduce the group's bounding rectangle. The
                // exact rotated contour still has to support every lifted leg.
                let margin = Vec2::new(r.random_range(0.12..1.15), r.random_range(0.12..1.05));
                let size = Vec3::new(
                    (span.x + margin.x * 2.).min(8.5),
                    r.random_range(0.004..MAX_THICKNESS_M),
                    (span.y + margin.y * 2.).min(7.5),
                );
                let local = (group_lo + group_hi) * 0.5
                    + margin * Vec2::new(r.random_range(-0.55..0.55), r.random_range(-0.55..0.55));
                let center = tf.transform_point(Vec3::new(local.x, 0., local.y));
                let yaw = anchor.yaw + r.random_range(-0.18..0.18);
                let mut rug = self.floor_candidate(ObjectKind::Rug, center, size, yaw, &mut r);
                rug.solid = false;
                let (lo, hi) = rug.bounds();
                // Reuse the complete architectural/portal/structure validator,
                // but furniture may occupy this rug when fully supported.
                if !self.placement_clear_internal(&rug, 0.025, true)
                    || !self.floor_support_clear(lo, hi)
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
                        if !other.solid || (a.y - lo.y).abs() > 0.001 {
                            valid = false;
                            break;
                        }
                        moved.insert(other.id);
                    }
                }
                if !valid || !moved.contains(&anchor.id) {
                    continue;
                }
                let Some(support) = fit_group(self, &mut rug, &moved) else {
                    continue;
                };
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
                    matches!(o.kind, ObjectKind::Chair | ObjectKind::Banquette)
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
                let (p, yaw) = super::accent::beside(&anchor, size.xz(), r);
                let mut rug = self.floor_candidate(ObjectKind::Rug, p, size, yaw, r);
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
                            && !self.installation_supported(o, o.bounds().0, o.bounds().1)
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
    fn a_narrowly_fitted_group_uses_real_corners_and_still_rejects_overhang() {
        use crate::scene::procedural_indoor::objects::{audit, floor::rug::Cut};
        let mut scene = IndoorManifest::generate(3, IndoorLayout::Lounge, 0.5, 0).unwrap();
        scene.ceiling_installation = None;
        let mut foot = audit::fixture(ObjectKind::Ottoman, 6);
        foot.id = 0;
        foot.size = Vec3::new(0.22, 0.38, 0.22);
        foot.position = Vec3::new(0.77, 0., 0.77);
        scene.objects = vec![foot];
        let mut rug = audit::fixture(ObjectKind::Rug, 14);
        rug.size = Vec3::new(2., 0.008, 2.);
        rug.variant = Cut::Oval as u32;
        let group = std::collections::BTreeSet::from([0]);
        let (lo, hi) = scene.objects[0].bounds();
        assert!(!RugSupport::from_object(&rug).supports(lo, hi));
        let installed = fit_group(&scene, &mut rug, &group).unwrap();
        assert_eq!(rug.variant, Cut::Straight as u32);
        assert_eq!(installed, RugSupport::from_object(&rug));
        assert!(installed.supports(lo, hi));
        scene.objects[0].position.x = 0.94;
        assert!(fit_group(&scene, &mut rug, &group).is_none());
    }

    fn planted_rug_scenarios(cameras: usize) -> impl Iterator<Item = IndoorManifest> {
        let generate = move |(seed, layout, people)| {
            IndoorManifest::generate_with_humans(seed, layout, 0.7, cameras, people).unwrap()
        };
        let reported = [(5, 0.), (5, 0.7), (924, 0.), (924, 0.7)]
            .map(|(s, p)| (s, IndoorLayout::Mixed, p))
            .into_iter()
            .map(generate);
        // The reported rooms can legitimately reject every optional rug. Find
        // two distinct planted lounges with actual supported furniture as controls.
        let controls = (0..64)
            .map(|s| (s, IndoorLayout::Lounge, if s % 2 == 0 { 0.7 } else { 0. }))
            .map(generate)
            .filter(|s| {
                !s.rug_supports.is_empty() && s.objects.iter().any(|o| o.kind == ObjectKind::Plant)
            })
            .take(2);
        reported.chain(controls)
    }

    #[test]
    fn area_rugs_preserve_clearance_of_unmoved_neighbouring_plants() {
        use crate::scene::procedural_indoor::validation;
        let mut supported_seeds = std::collections::BTreeSet::new();
        for scene in planted_rug_scenarios(3) {
            assert!(scene.objects.iter().any(|o| o.kind == ObjectKind::Plant));
            if !scene.rug_supports.is_empty() {
                supported_seeds.insert(scene.seed);
            }
            validation::validate_layout(&scene).unwrap();
            validation::validate_geometry(&scene).unwrap();
        }
        assert!(supported_seeds.len() >= 2);
    }

    #[test]
    fn rug_support_replays_and_rejects_edge_overhang_and_tampered_contours() {
        let mut qualified = std::collections::BTreeSet::new();
        for scene in planted_rug_scenarios(0) {
            let Some(rug) = scene.rug_supports.first() else {
                continue;
            };
            qualified.insert(scene.seed);
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
        assert!(
            qualified.len() >= 2,
            "fixture cohort lost furniture-bearing rugs"
        );
    }
}
