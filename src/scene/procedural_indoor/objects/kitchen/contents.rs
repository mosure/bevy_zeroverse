//! Shelf contents are packed in the actual cavity, independently on each shelf.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint, SurfacePattern};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct StoredItem {
    pub position: Vec3,
    pub size: Vec3,
    pub yaw: f32,
    pub pantry: bool,
    pub style: u8,
    pub count: u32,
    pub half_extents: Vec2,
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
    for _ in 0..target {
        let pantry = o.kind == ObjectKind::Pantry && r.random_bool(0.78);
        let style: u8 = r.random_range(0..if pantry { 3 } else { 5 });
        let size = if o.kind == ObjectKind::Refrigerator {
            let w = r.random_range(0.075..0.23);
            Vec3::new(w, r.random_range(0.085..0.30), w * r.random_range(0.8..1.0))
        } else if pantry {
            let w = r.random_range(0.075..0.25);
            Vec3::new(w, r.random_range(0.07..0.29), w * r.random_range(0.72..1.0))
        } else {
            super::super::super::items::sizing::dish(style, &mut r)
        };
        // Keep physical dimensions. Oversize supplies are omitted, rather than
        // squeezed to the cavity size or accumulated at a clamped maximum.
        if size.y + 0.012 > clearance {
            continue;
        }
        for _ in 0..20 {
            let yaw = pattern.yaw(&mut r);
            let half = if o.kind == ObjectKind::Refrigerator {
                if style.is_multiple_of(2) {
                    Vec2::splat(size.x.min(size.z) * 0.30)
                } else {
                    rotated_half(size.xz() * 0.72, yaw)
                }
            } else if pantry {
                if style == 0 {
                    rotated_half(size.xz() * Vec2::new(0.67, 0.60), yaw)
                } else {
                    Vec2::splat(size.x.min(size.z) * 0.28 + 0.002)
                }
            } else {
                Vec2::splat(size.x.min(size.z) * 0.43 + 0.006)
            };
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
                count: r.random_range(1..=5),
                half_extents: half,
            });
            break;
        }
    }
    items
}

pub(crate) fn build(a: &mut Assembly, items: &[StoredItem]) {
    for item in items {
        if item.pantry {
            props::pantry_item(a, item.transform(), item.size, item.style);
        } else {
            props::dish_at(a, item.transform(), item.size, item.style, item.count);
        }
    }
}
