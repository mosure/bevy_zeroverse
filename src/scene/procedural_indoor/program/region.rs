//! Borrowed primary reconstruction domain. Cache it once for pixel/point queries;
//! it shares the accepted envelope and support/roof functions with construction.
use super::super::{envelope::polygon, layout::IndoorManifest};
use bevy::prelude::*;

pub struct PrimaryRegion<'a> {
    scene: &'a IndoorManifest,
    pub min: Vec2,
    pub max: Vec2,
}
impl IndoorManifest {
    /// Stable reconstruction region selected after furnishing. Older archives
    /// without a recorded selection retain the largest clipped floor region.
    /// Cameras, schematic ownership and O-voxel cropping share this selection.
    pub fn primary_room_bounds(&self) -> (Vec2, Vec2) {
        if let Some(zone) = self
            .program
            .as_ref()
            .and_then(|p| p.primary_zone.and_then(|i| p.zones.get(i)))
        {
            return (zone.min, zone.max);
        }
        self.program
            .as_ref()
            .and_then(|p| {
                p.zones
                    .iter()
                    .map(|z| {
                        let area = self.envelope.as_ref().map_or_else(
                            || (z.max - z.min).element_product() as f64,
                            |e| clipped_area(&e.footprint, z.min, z.max),
                        );
                        (z, area)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
            })
            .map(|(z, _)| (z.min, z.max))
            .unwrap_or_else(|| {
                let half = self.room_size.xz() * 0.5;
                (-half, half)
            })
    }

    /// Choose once, before people/cameras, and archive the result. Area alone
    /// can select an empty allocation after furniture clearance rejected its
    /// contents, stranding CCTV in a different room from useful walkthroughs.
    /// Prefer the largest region with several substantial furnishings from at
    /// least two semantic classes; preserve the area fallback for sparse scenes.
    pub(crate) fn select_primary_room(&mut self) {
        let Some(program) = &self.program else { return };
        let selected = program
            .zones
            .iter()
            .enumerate()
            .map(|(index, zone)| {
                let mut classes = std::collections::BTreeSet::new();
                let mut objects = 0;
                for object in &self.objects {
                    let p = object.position.xz();
                    if !object.neighbor
                        && object.solid
                        && object.support.is_none()
                        && object.size.element_product() >= 0.02
                        && p.cmpgt(zone.min).all()
                        && p.cmplt(zone.max).all()
                    {
                        objects += 1;
                        classes.insert(object.kind.class_name());
                    }
                }
                let furnished = objects >= 3 && classes.len() >= 2;
                let area = self.envelope.as_ref().map_or_else(
                    || (zone.max - zone.min).element_product() as f64,
                    |e| clipped_area(&e.footprint, zone.min, zone.max),
                );
                (index, furnished, area)
            })
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)))
            .map(|v| v.0);
        self.program.as_mut().unwrap().primary_zone = selected;
    }

    pub fn primary_region(&self) -> PrimaryRegion<'_> {
        let (min, max) = self.primary_room_bounds();
        PrimaryRegion {
            scene: self,
            min,
            max,
        }
    }
}

/// Integrate clipped signed fan triangles. Oppositely oriented pieces cancel
/// outside a concave outline, including disconnected intersections. Unlike
/// clipping a convex hull, this cannot fill a notch. Fixed scratch storage keeps
/// primary-domain queries allocation-free in camera clearance hot loops.
fn clipped_area(outline: &[Vec2], lo: Vec2, hi: Vec2) -> f64 {
    let Some(origin) = outline.first() else {
        return 0.;
    };
    let mut area = 0.;
    for edge in outline[1..].windows(2) {
        let mut points = [bevy::math::DVec2::ZERO; 12];
        points[..3].copy_from_slice(&[origin.as_dvec2(), edge[0].as_dvec2(), edge[1].as_dvec2()]);
        let mut length = 3;
        for (axis, boundary, sign) in [(0, lo.x, 1.), (0, hi.x, -1.), (1, lo.y, 1.), (1, hi.y, -1.)]
        {
            let mut next = [bevy::math::DVec2::ZERO; 12];
            let mut count = 0;
            for i in 0..length {
                let a = points[i];
                let b = points[(i + 1) % length];
                let da = (a[axis] - f64::from(boundary)) * sign;
                let db = (b[axis] - f64::from(boundary)) * sign;
                if da >= 0. {
                    next[count] = a;
                    count += 1;
                }
                if (da >= 0.) != (db >= 0.) {
                    next[count] = a + (b - a) * (da / (da - db));
                    count += 1;
                }
            }
            points = next;
            length = count;
        }
        for i in 1..length.saturating_sub(1) {
            area += (points[i] - points[0]).perp_dot(points[i + 1] - points[0]) * 0.5;
        }
    }
    area.abs()
}
impl PrimaryRegion<'_> {
    /// Ray/surface intersections can land just outside a boundary through f32
    /// roundoff. This is for surface ownership only, never body/path clearance.
    pub(crate) fn contains_projection_with_tolerance(&self, p: Vec2, tolerance: f32) -> bool {
        if !p.is_finite()
            || !tolerance.is_finite()
            || tolerance < 0.
            || p.cmplt(self.min - Vec2::splat(tolerance)).any()
            || p.cmpgt(self.max + Vec2::splat(tolerance)).any()
        {
            return false;
        }
        self.scene.envelope.as_ref().is_none_or(|e| {
            polygon::contains(&e.footprint, p, 0.)
                || polygon::edges(&e.footprint).any(|(a, b)| {
                    let d = b - a;
                    let closest =
                        a + d * ((p - a).dot(d) / d.length_squared().max(1e-12)).clamp(0., 1.);
                    p.distance_squared(closest) <= tolerance * tolerance
                })
        })
    }

    /// X/Z ownership only. Does not invent floor in a concave notch.
    pub fn contains_projection(&self, p: Vec2, clearance: f32) -> bool {
        p.is_finite()
            && clearance.is_finite()
            && clearance >= 0.
            && p.cmpge(self.min + Vec2::splat(clearance)).all()
            && p.cmple(self.max - Vec2::splat(clearance)).all()
            && self
                .scene
                .envelope
                .as_ref()
                .is_none_or(|e| polygon::contains(&e.footprint, p, clearance))
    }
    /// Rendered surface ownership under sloped ceilings, soffits and stepped
    /// floors. Vertical tolerance handles the finite precision of surface hits;
    /// it never admits the sky proxy or neighboring exterior roof.
    pub fn contains_surface(&self, p: Vec3, vertical_tolerance: f32) -> bool {
        vertical_tolerance.is_finite()
            && vertical_tolerance >= 0.
            && p.is_finite()
            && self.contains_projection(p.xz(), 0.)
            && p.y >= self.scene.floor_height(p.xz()) - vertical_tolerance
            && p.y <= self.scene.ceiling_height(p.xz()) + vertical_tolerance
    }
    /// Enclosing architectural shell used by scene AABB / O-voxel crop. This is
    /// a conservative box; exact point membership is contains_surface.
    pub fn shell_bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(
                self.min.x - 0.20,
                self.scene
                    .envelope
                    .as_ref()
                    .map_or(0., |e| e.minimum_floor())
                    - 0.20,
                self.min.y - 0.20,
            ),
            Vec3::new(
                self.max.x + 0.20,
                self.scene.room_size.y + 0.14,
                self.max.y + 0.20,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::{IndoorLayout, ObjectKind};

    #[test]
    fn furnished_subroom_wins_over_larger_empty_region() {
        let mut scene =
            IndoorManifest::generate_with_humans(65574761, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        let program = scene.program.as_mut().unwrap();
        program.primary_zone = None;
        let larger = (program.zones[0].min, program.zones[0].max);
        let furnished = (program.zones[1].min, program.zones[1].max);
        let centre = (furnished.0 + furnished.1) * 0.5;
        assert_eq!(scene.primary_room_bounds(), larger);
        // Hold the selection fixture fixed while the seeded furnishing program
        // evolves. Three substantial objects of two classes make this subroom
        // useful; an empty larger room must not strand the reconstruction rig.
        let prototype = scene.objects[0].clone();
        scene.objects = [ObjectKind::Chair, ObjectKind::Chair, ObjectKind::Table]
            .into_iter()
            .enumerate()
            .map(|(id, kind)| {
                let mut object = prototype.clone();
                object.id = id;
                object.kind = kind;
                object.position = Vec3::new(centre.x, 0., centre.y + id as f32 * 0.6 - 0.6);
                object.size = Vec3::new(0.5, 0.8, 0.5);
                object.solid = true;
                object.support = None;
                object.neighbor = false;
                object
            })
            .collect();
        scene.select_primary_room();
        assert_eq!(scene.program.as_ref().unwrap().primary_zone, Some(1));
        assert_eq!(scene.primary_room_bounds(), furnished);
    }

    #[test]
    fn primary_selection_uses_furnishings_is_archived_and_stays_fixed() {
        for (seed, legacy) in [(65574761, 0), (65575086, 3), (65575252, 1)] {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let program = scene.program.as_ref().unwrap();
            // Selection follows the realized furniture, rather than a permanent
            // seed-to-zone mapping across generator versions.
            let expected = program.primary_zone.unwrap();
            let bounds = (program.zones[expected].min, program.zones[expected].max);
            assert_eq!(scene.primary_room_bounds(), bounds);
            let bytes = serde_json::to_vec(&scene).unwrap();
            let replay: IndoorManifest = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(replay.primary_room_bounds(), bounds);
            let topology = scene.room_topology();
            assert!(topology
                .regions
                .iter()
                .any(|r| r.zone == expected && r.primary));
            // Moving/removing furniture cannot silently change a captured ROI.
            scene.objects.clear();
            assert_eq!(scene.primary_room_bounds(), bounds);
            let mut archived: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            archived["program"]
                .as_object_mut()
                .unwrap()
                .remove("primary_zone");
            let archived: IndoorManifest = serde_json::from_value(archived).unwrap();
            let zone = &archived.program.as_ref().unwrap().zones[legacy];
            assert_eq!(archived.primary_room_bounds(), (zone.min, zone.max));
            // An invalid archived index is a validation error, never a panic.
            scene.program.as_mut().unwrap().primary_zone = Some(usize::MAX);
            assert!(scene.program.as_ref().unwrap().validate(&scene).is_err());
        }
    }

    #[test]
    fn empty_scenes_keep_the_clipped_area_fallback() {
        let mut scene =
            IndoorManifest::generate_with_humans(65575086, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        for object in &mut scene.objects {
            object.neighbor = true;
        }
        scene.select_primary_room();
        assert_eq!(scene.program.as_ref().unwrap().primary_zone, Some(3));
    }
    #[test]
    fn primary_room_uses_clipped_floor_area_and_matches_topology() {
        let scene =
            IndoorManifest::generate_with_humans(65_332_074, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        let topology = scene.room_topology();
        let largest = topology
            .regions
            .iter()
            .max_by(|a, b| a.area_m2.total_cmp(&b.area_m2))
            .unwrap();
        assert_eq!(
            largest.zone, 1,
            "zone zero's bounding rectangle extends outside the slanted envelope"
        );
        assert!(largest.primary);
        let zone = &scene.program.as_ref().unwrap().zones[largest.zone];
        assert_eq!(scene.primary_room_bounds(), (zone.min, zone.max));
    }

    #[test]
    fn allocation_free_clipping_matches_triangulated_concave_oracle() {
        let mut polygon = vec![
            Vec2::ZERO,
            Vec2::new(6., 0.),
            Vec2::new(6., 6.),
            Vec2::new(4., 6.),
            Vec2::new(4., 2.),
            Vec2::new(2., 2.),
            Vec2::new(2., 6.),
            Vec2::new(0., 6.),
        ];
        for _ in 0..2 {
            let triangles = polygon::triangles(&polygon);
            for x in -1..7 {
                for y in -1..7 {
                    let lo = Vec2::new(x as f32, y as f32);
                    for size in [Vec2::splat(2.5), Vec2::new(0.00001, 3.), Vec2::splat(8.)] {
                        let hi = lo + size;
                        let expected: f64 = triangles
                            .iter()
                            .map(|t| polygon::area(&polygon::clip_box(t, lo, hi)).abs() as f64)
                            .sum();
                        assert!((clipped_area(&polygon, lo, hi) - expected).abs() < 1e-5);
                    }
                }
            }
            polygon.reverse();
        }
        assert!(clipped_area(&polygon, Vec2::new(2.1, 3.), Vec2::new(3.9, 5.)) < 1e-12);
        assert_eq!(clipped_area(&[], Vec2::ZERO, Vec2::ONE), 0.);
    }

    #[test]
    fn surface_roundoff_tolerance_does_not_change_clearance_or_admit_context() {
        let mut scene =
            IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.program = None;
        scene.room_size = Vec3::new(6., 3., 6.);
        scene.envelope.as_mut().unwrap().footprint = vec![
            Vec2::new(-3., -3.),
            Vec2::new(3., -3.),
            Vec2::new(3., 0.),
            Vec2::ZERO,
            Vec2::new(0., 3.),
            Vec2::new(-3., 3.),
        ];
        let r = scene.primary_region();
        for point in [Vec2::new(-3.00001, 0.), Vec2::new(0.00001, 1.)] {
            assert!(!r.contains_projection(point, 0.));
            assert!(r.contains_projection_with_tolerance(point, 1e-4));
        }
        for point in [
            Vec2::new(-3.01, 0.),
            Vec2::new(0.01, 1.),
            Vec2::splat(2.),
            Vec2::splat(f32::NAN),
        ] {
            assert!(!r.contains_projection_with_tolerance(point, 1e-4));
        }
        assert!(!r.contains_projection(Vec2::new(-2.99, 0.), 0.1));
    }
    #[test]
    fn membership_uses_local_floor_roof_and_excludes_context() {
        for seed in 0..32 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let region = scene.primary_region();
            for r in scene.room_topology().regions.iter().filter(|r| r.primary) {
                for t in &r.triangles {
                    let p = (t[0] + t[1] + t[2]) / 3.;
                    let floor = scene.floor_height(p);
                    let roof = scene.ceiling_height(p);
                    assert!(region.contains_surface(Vec3::new(p.x, floor, p.y), 0.01));
                    assert!(region.contains_surface(Vec3::new(p.x, roof, p.y), 0.01));
                    assert!(!region.contains_surface(Vec3::new(p.x, roof + 0.1, p.y), 0.01));
                    assert!(!region.contains_surface(Vec3::new(p.x, floor - 0.1, p.y), 0.01));
                }
            }
            assert!(!region.contains_surface(Vec3::splat(f32::NAN), 0.01));
            assert!(!region.contains_surface(Vec3::new(0., 100., 0.), 0.01));
        }
    }
}
