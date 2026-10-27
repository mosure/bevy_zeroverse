//! Shelf contents are packed in the actual cavity, independently on each shelf.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum StoredProgram {
    Dish(dishware::DishProgram),
    Package(packaging::PackageProgram),
}
impl StoredProgram {
    fn half_extents(&self, yaw: f32) -> Vec2 {
        match self {
            Self::Dish(p) => p.half_extents(yaw),
            Self::Package(p) => p.half_extents(yaw),
        }
    }
    fn height(&self) -> f32 {
        match self {
            Self::Dish(p) => p.height + p.wall * (p.count - 1) as f32,
            Self::Package(p) => p.body.y,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct StoredItem {
    pub position: Vec3,
    pub size: Vec3,
    pub yaw: f32,
    pub pantry: bool,
    pub style: u8,
    pub count: u32,
    pub half_extents: Vec2,
    pub construction: StoredProgram,
    pub finish_seed: u64,
}
impl StoredItem {
    pub fn footprint(&self) -> Footprint {
        Footprint {
            center: self.position.xz(),
            half: self.half_extents,
            yaw: self.yaw,
        }
    }
    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.position).with_rotation(Quat::from_rotation_y(self.yaw))
    }
}

pub(crate) fn shelf_contents(
    o: &IndoorObject,
    level: u32,
    bounds: (Vec2, Vec2),
    y: f32,
    clearance: f32,
) -> Vec<StoredItem> {
    let mut r = stream(o.seed, 8020 + level as u64);
    let pattern = SurfacePattern::sample(&mut r);
    let span = bounds.1 - bounds.0;
    let capacity = ((span.x * span.y / 0.035).ceil() as usize).clamp(1, 6);
    let target = r.random_range(1..=capacity);
    let mut items: Vec<StoredItem> = Vec::with_capacity(target);
    let collection_finish = r.random_bool(0.55).then(|| r.random::<u64>());
    for _ in 0..target {
        let pantry = o.kind == ObjectKind::Refrigerator
            || (o.kind == ObjectKind::Pantry && r.random_bool(0.78));
        let style: u8 = r.random_range(0..if pantry { 6 } else { 5 });
        let size = if o.kind == ObjectKind::Refrigerator {
            let w = r.random_range(0.075..0.23);
            Vec3::new(w, r.random_range(0.085..0.30), w * r.random_range(0.8..1.0))
        } else if pantry {
            let w = r.random_range(0.075..0.25);
            Vec3::new(w, r.random_range(0.07..0.29), w * r.random_range(0.72..1.0))
        } else {
            super::super::super::items::sizing::dish(style, &mut r)
        };
        let seed = r.random();
        let count = r.random_range(1..=5);
        let construction = if pantry {
            StoredProgram::Package(packaging::parameters(size, style, seed))
        } else {
            StoredProgram::Dish(dishware::parameters(size, style, count, seed))
        };
        let finish_seed = collection_finish.unwrap_or_else(|| r.random());
        // Keep physical dimensions. Oversize supplies are omitted, rather than
        // squeezed to the cavity size or accumulated at a clamped maximum.
        if construction.height() + 0.012 > clearance {
            continue;
        }
        for _ in 0..20 {
            let yaw = pattern.yaw(&mut r);
            // Admission uses exactly the program later built, including handles
            // and the protruding label skin. No second approximate shape model.
            let half = construction.half_extents(yaw);
            let Some(p) = pattern.propose_half(bounds, half, yaw, 0.008, &mut r) else {
                continue;
            };
            if items.iter().any(|i| p.overlaps(i.footprint(), 0.008)) {
                continue;
            }
            items.push(StoredItem {
                position: Vec3::new(p.center.x, y, p.center.y),
                size,
                yaw,
                pantry,
                style,
                count,
                half_extents: half,
                construction,
                finish_seed,
            });
            break;
        }
    }
    items
}

pub(crate) fn build(a: &mut Assembly, items: &[StoredItem]) {
    for item in items {
        match &item.construction {
            StoredProgram::Package(p) => packaging::build(a, item.transform(), p, item.finish_seed),
            StoredProgram::Dish(p) => dishware::build(a, item.transform(), p, item.finish_seed),
        }
    }
}
