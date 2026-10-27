use super::*;
use crate::scene::procedural_indoor::layout::segment_hits_box;

impl EnvelopeProgram {
    /// Guard exposed level changes using the same geometry in the camera oracle.
    pub fn tier_guards(&self) -> Vec<(Vec3, Vec3)> {
        let mut segments = Vec::new();
        for p in &self.floor_patches {
            if p.height < 0.61 {
                continue;
            }
            let corners = [
                p.min,
                Vec2::new(p.max.x, p.min.y),
                p.max,
                Vec2::new(p.min.x, p.max.y),
            ];
            for (a, b) in polygon::edges(&corners) {
                let d = b - a;
                let outside = (a + b) * 0.5 + Vec2::new(d.y, -d.x).normalize() * 0.01;
                if p.height - self.floor_height(outside) < 0.61 {
                    continue;
                }
                segments.push((Vec3::new(a.x, p.height, a.y), Vec3::new(b.x, p.height, b.y)));
            }
        }
        segments
    }
    pub fn structural_boxes(&self, size: Vec3) -> Vec<(Vec3, Vec3)> {
        let mut boxes = self
            .tier_guards()
            .into_iter()
            .map(|(a, b)| {
                if let Some(g) = &self.tier_guard {
                    return g.section.bounds(g.height, a, b);
                }
                (
                    a.min(b) - Vec3::new(0.025, 0., 0.025),
                    // The top rail is centered at 1.03 m with a 22 mm radius.
                    a.max(b) + Vec3::new(0.025, 1.055, 0.025),
                )
            })
            .collect::<Vec<_>>();
        for p in &self.soffits {
            boxes.push((
                Vec3::new(p.min.x, p.height, p.min.y),
                Vec3::new(p.max.x, size.y + 0.14, p.max.y),
            ));
        }
        for p in &self.pillars {
            // Include the projecting base collar and the high edge of a cap
            // cut by the roof plane, not just the nominal shaft dimensions.
            let radius = p.radius + 0.025;
            let top = self.ceiling_height(size, p.center)
                + (self.ceiling_drop / size.xz()).abs().element_sum() * radius;
            boxes.push((
                Vec3::new(p.center.x - radius, 0., p.center.y - radius),
                Vec3::new(p.center.x + radius, top, p.center.y + radius),
            ));
        }
        if let Some(m) = &self.mezzanine {
            boxes.push((
                Vec3::new(m.deck.min.x, m.deck.height - m.thickness, m.deck.min.y),
                Vec3::new(m.deck.max.x, m.deck.height, m.deck.max.y),
            ));
            boxes.extend(m.columns());
            boxes.extend(m.beams());
            boxes.extend(m.stair_boxes());
            for (a, b) in m.rails() {
                if let Some(c) = &m.construction {
                    boxes.push(c.guard.bounds(m.rail_height, a, b));
                } else {
                    // Conservative continuous clearance for thin glass guard panels.
                    boxes.push((
                        a.min(b) - Vec3::new(0.035, 0., 0.035),
                        a.max(b) + Vec3::new(0.035, m.rail_height + 0.03, 0.035),
                    ));
                }
            }
        }
        boxes
    }

    /// Lens clearance includes the maximum inward sill/shade projection (20 cm).
    pub fn volume_clear(&self, size: Vec3, p: Vec3, radius: f32) -> bool {
        polygon::contains(&self.footprint, p.xz(), radius + 0.20)
            && p.y >= self.floor_height(p.xz()) + radius
            && p.y <= self.ceiling_height(size, p.xz()) - radius
    }
    pub fn segment_clear(&self, size: Vec3, a: Vec3, b: Vec3, radius: f32) -> bool {
        if !polygon::segment_inside(&self.footprint, a.xz(), b.xz(), radius + 0.20)
            || !self.volume_clear(size, a, radius)
            || !self.volume_clear(size, b, radius)
        {
            return false;
        }
        if self.floor_patches.is_empty() && self.soffits.is_empty() {
            return true;
        }
        // Floor heights are piecewise constant: split at every patch boundary,
        // then test the exact interval extrema instead of fixed-distance samples.
        let mut times = vec![0., 1.];
        for f in self.floor_patches.iter().chain(&self.soffits) {
            for axis in [0, 2] {
                let d = b[axis] - a[axis];
                if d.abs() < 1e-8 {
                    continue;
                }
                for edge in [f.min[axis / 2], f.max[axis / 2]] {
                    let t = (edge - a[axis]) / d;
                    if t > 0. && t < 1. {
                        times.push(t);
                    }
                }
            }
        }
        times.sort_by(f32::total_cmp);
        times.windows(2).all(|t| {
            let middle = a.lerp(b, (t[0] + t[1]) * 0.5).xz();
            let h = self.floor_height(middle);
            let ceiling = self
                .soffits
                .iter()
                .filter(|p| p.contains(middle))
                .map(|p| p.height)
                .fold(f32::INFINITY, f32::min);
            a.lerp(b, t[0]).y.min(a.lerp(b, t[1]).y) >= h + radius
                && a.lerp(b, t[0]).y.max(a.lerp(b, t[1]).y) <= ceiling - radius
        })
    }

    /// Annotation-opaque wall/glass shell, exact piecewise floors and planar
    /// ceiling. Used by camera proposals; rendered pixels remain the oracle.
    pub fn ray_hit(
        &self,
        size: Vec3,
        door_x: f32,
        door_width: f32,
        door_height: f32,
        origin: Vec3,
        ray: Vec3,
    ) -> Option<(f32, &'static str)> {
        let mut hit: Option<(f32, &'static str)> = None;
        let mut accept = |t: f32, label| {
            if t > 1e-5 && hit.is_none_or(|h| t < h.0) {
                hit = Some((t, label));
            }
        };
        if ray.y.abs() > 1e-8 {
            for y in std::iter::once(0.).chain(self.floor_patches.iter().map(|p| p.height)) {
                let t = (y - origin.y) / ray.y;
                let p = (origin + ray * t).xz();
                if polygon::contains(&self.footprint, p, 0.)
                    && (self.floor_height(p) - y).abs() < 1e-5
                {
                    accept(t, "floor");
                }
            }
        }
        let slope = -self.ceiling_drop / size.xz();
        let h = self.roof_height(size, origin.xz());
        let denominator = ray.y - slope.dot(ray.xz());
        if denominator.abs() > 1e-8 {
            let t = (h - origin.y) / denominator;
            if polygon::contains(&self.footprint, (origin + ray * t).xz(), 0.) {
                accept(t, "ceiling");
            }
        }
        for patch in &self.soffits {
            let lo = Vec3::new(patch.min.x, patch.height, patch.min.y);
            let hi = Vec3::new(patch.max.x, size.y + 0.14, patch.max.y);
            let mut enter = f32::NEG_INFINITY;
            let mut exit = f32::INFINITY;
            for axis in 0..3 {
                if ray[axis].abs() < 1e-8 {
                    if origin[axis] < lo[axis] || origin[axis] > hi[axis] {
                        exit = f32::NEG_INFINITY;
                    }
                } else {
                    let a = (lo[axis] - origin[axis]) / ray[axis];
                    let b = (hi[axis] - origin[axis]) / ray[axis];
                    enter = enter.max(a.min(b));
                    exit = exit.min(a.max(b));
                }
            }
            if enter <= exit {
                accept(if enter > 0. { enter } else { exit }, "ceiling");
            }
        }
        for (i, (a, b)) in polygon::edges(&self.footprint).enumerate() {
            let d = b - a;
            let denominator = ray.xz().perp_dot(d);
            if denominator.abs() < 1e-8 {
                continue;
            }
            let t = (a - origin.xz()).perp_dot(d) / denominator;
            let p = origin + ray * t;
            let u = (p.xz() - a).dot(d) / d.length_squared();
            if (0.0..=1.0).contains(&u)
                && p.y >= 0.
                && p.y <= self.ceiling_height(size, p.xz())
                && !(self.shared_edge(i, size)
                    && (p.x - door_x).abs() < door_width * 0.5 - 0.022
                    && p.y < door_height - 0.025)
            {
                accept(t, "wall");
            }
        }
        for patch in &self.floor_patches {
            let corners = [
                patch.min,
                Vec2::new(patch.max.x, patch.min.y),
                patch.max,
                Vec2::new(patch.min.x, patch.max.y),
            ];
            for (a, b) in polygon::edges(&corners) {
                let d = b - a;
                let denominator = ray.xz().perp_dot(d);
                if denominator.abs() < 1e-8 {
                    continue;
                }
                let t = (a - origin.xz()).perp_dot(d) / denominator;
                let p = origin + ray * t;
                let u = (p.xz() - a).dot(d) / d.length_squared();
                let other =
                    self.floor_height((a + b) * 0.5 + Vec2::new(d.y, -d.x).normalize() * 0.001);
                if (0.0..=1.0).contains(&u)
                    && p.y >= patch.height.min(other)
                    && p.y <= patch.height.max(other)
                    && (patch.height - other).abs() > 1e-5
                {
                    accept(t, "floor");
                }
            }
        }
        hit
    }

    /// Conservative motion-planning barriers for a model whose root support is
    /// a level plane. Stair climbing is not silently synthesized by ARDY.
    pub fn motion_barriers(&self, size: Vec3) -> Vec<(Vec3, Vec3)> {
        // Structural solids are already supplied by scene.camera_obstacles().
        let mut boxes = Vec::new();
        for (a, b) in polygon::edges(&self.footprint) {
            // Axis-aligned walls are exact single boxes; bounded short boxes
            // approximate only oblique segments, without filling their AABB.
            let d = b - a;
            let count = if d.x.abs().min(d.y.abs()) < 1e-5 {
                1
            } else {
                (d.length() / 0.35).ceil() as usize
            };
            for i in 0..count {
                let p = a.lerp(b, i as f32 / count as f32);
                let q = a.lerp(b, (i + 1) as f32 / count as f32);
                if (p.y - size.z * 0.5).abs() < 1e-4 {
                    continue;
                }
                boxes.push((
                    Vec3::new(p.x.min(q.x) - 0.04, -1., p.y.min(q.y) - 0.04),
                    Vec3::new(p.x.max(q.x) + 0.04, size.y, p.y.max(q.y) + 0.04),
                ));
            }
        }
        for p in &self.floor_patches {
            if p.height.abs() > 0.001 {
                boxes.push((
                    Vec3::new(p.min.x, -1., p.min.y),
                    Vec3::new(p.max.x, size.y, p.max.y),
                ));
            }
        }
        boxes
    }

    pub fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        if let Some(g) = &self.tier_guard {
            g.validate()?;
        }
        polygon::validate(&self.footprint)?;
        let size = scene.room_size;
        if !self.ceiling_drop.is_finite()
            || self.footprint.iter().any(|&p| {
                self.ceiling_height(size, p) < 2.49
                    || p.abs().cmpgt(size.xz() * 0.5 + Vec2::splat(1e-4)).any()
            })
        {
            return Err("invalid envelope roof".into());
        }
        let mut wall_edges = std::collections::BTreeSet::new();
        for (i, p) in self.soffits.iter().enumerate() {
            if !p.min.is_finite()
                || !p.max.is_finite()
                || !(2.5..=size.y).contains(&p.height)
                || (p.max - p.min).min_element() < 0.1
                || !polygon::box_inside(&self.footprint, p.min, p.max, 0.1)
                || self.soffits[i + 1..]
                    .iter()
                    .any(|q| q.overlaps(p.min, p.max))
            {
                return Err("invalid ceiling support patch".into());
            }
        }
        for w in &self.walls {
            if w.edge >= self.footprint.len()
                || self.shared_edge(w.edge, size)
                || !wall_edges.insert(w.edge)
            {
                return Err("invalid or duplicated envelope wall".into());
            }
            if let Some(f) = &w.facade {
                let a = self.footprint[w.edge];
                let b = self.footprint[(w.edge + 1) % self.footprint.len()];
                let span = a.distance(b);
                let height = self
                    .ceiling_height(size, a)
                    .min(self.ceiling_height(size, b));
                super::super::architecture::facade::ExteriorProgram {
                    exposure_probability: 1.,
                    facades: vec![f.clone()],
                }
                .validate(Vec3::new(span, height, span))?;
            }
        }
        if wall_edges.len() + 1 != self.footprint.len() {
            return Err("incomplete exterior envelope".into());
        }
        for (i, p) in self.floor_patches.iter().enumerate() {
            if !p.height.is_finite()
                || p.height.abs()
                    > if scene.layout == super::super::layout::IndoorLayout::LectureHall {
                        2.1
                    } else {
                        0.6
                    }
                || (scene.layout == super::super::layout::IndoorLayout::LectureHall
                    && self.ceiling_height(size, (p.min + p.max) * 0.5) - p.height < 2.2)
                || !polygon::box_inside(&self.footprint, p.min, p.max, 0.1)
                || (p.max - p.min).min_element() < 0.1
                || self.floor_patches[i + 1..]
                    .iter()
                    .any(|q| q.overlaps(p.min, p.max))
            {
                return Err("invalid floor level patch".into());
            }
        }
        for p in &self.pillars {
            if let Some(profile) = &p.profile {
                profile.validate()?;
            }
            if !p.radius.is_finite()
                || !(0.1..=0.4).contains(&p.radius)
                || ![4, 24, 32].contains(&p.sides)
                || !polygon::contains(&self.footprint, p.center, p.radius + 0.1)
            {
                return Err("invalid interior pillar".into());
            }
        }
        if let Some(m) = &self.mezzanine {
            m.validate(self, size)?;
        }
        for (lo, hi) in self.structural_boxes(size) {
            if !lo.is_finite() || !hi.is_finite() || hi.cmple(lo).any() {
                return Err("invalid architectural solid".into());
            }
        }
        // Every portal's low central passage must remain unobstructed.
        if let Some(p) = &scene.program {
            for door in &p.partitions {
                let c = door.position(door.door_center, 1.0);
                if self
                    .structural_boxes(size)
                    .iter()
                    .any(|(lo, hi)| segment_hits_box(c, c, *lo, *hi))
                {
                    return Err("architectural feature blocks a portal".into());
                }
            }
        }
        Ok(())
    }
}
