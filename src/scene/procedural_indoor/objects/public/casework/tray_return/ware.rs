//! A molded tray or wash tub has a hollow well, a bearing lip and two feet.
//! The inner floor sets the repeat height of a nested stack of serving trays.
use super::*;
use crate::scene::procedural_indoor::{geometry::Geometry, materials::variants};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WareKind {
    ServingTray,
    BusTub,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReturnWare {
    pub kind: WareKind,
    pub size_m: Vec2,
    pub height_m: f32,
    pub lip_m: f32,
    pub gauge_m: f32,
    pub foot_m: f32,
    pub flare_m: f32,
    pub roundness: f32,
    pub stock: Surface,
    pub finish_slot: usize,
}
impl ReturnWare {
    pub(super) fn fit(o: &IndoorObject, available: Vec2) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"TrayMold"));
        let kind = if r.random_bool(0.25) {
            WareKind::BusTub
        } else {
            WareKind::ServingTray
        };
        let (length, width, height_m) = match kind {
            WareKind::ServingTray => (
                r.random_range(0.37..0.49),
                r.random_range(0.255..0.355),
                r.random_range(0.018..0.034),
            ),
            WareKind::BusTub => (
                r.random_range(0.39..0.52),
                r.random_range(0.29..0.375),
                r.random_range(0.095..0.152),
            ),
        };
        let size_m = if r.random_bool(0.48) {
            Vec2::new(length, width)
        } else {
            Vec2::new(width, length)
        };
        // Keep the catalogue aspect when a shallow station needs a smaller item.
        let size_m = size_m * (available / size_m).min_element().clamp(0., 1.);
        let gauge_m = r.random_range(0.0017_f32..0.0029);
        let foot_m = r.random_range(0.0045_f32..0.008);
        let lip_m = r
            .random_range(0.010_f32..0.019)
            .min(size_m.min_element() * 0.08);
        let flare_m = if kind == WareKind::ServingTray {
            // The upper body's exterior must clear the lower body's inner wall
            // when its two feet bear on the lower tray's floor.
            let rise = height_m - 1.6 * gauge_m - foot_m;
            r.random_range(0.014_f32..0.031)
                .max(rise * 1.4 * gauge_m / foot_m)
        } else {
            r.random_range(0.022..0.046)
        };
        let finish_seed = o.finish_seed.unwrap_or(o.seed) ^ u64::from_le_bytes(*b"TrayCoat");
        let mut finish = stream(finish_seed, 1);
        let stock = if finish.random_bool(0.19) {
            Surface::Metal
        } else {
            Surface::Plastic
        };
        Self {
            kind,
            size_m,
            height_m,
            lip_m,
            gauge_m,
            foot_m,
            flare_m,
            roundness: r.random_range(4.3..12.),
            stock,
            finish_slot: variants::object_slot(stock, finish_seed, None),
        }
    }

    pub fn under_lip_m(&self) -> f32 {
        self.height_m - 1.6 * self.gauge_m
    }
    pub fn nesting_pitch_m(&self) -> f32 {
        self.foot_m + self.gauge_m
    }

    pub(super) fn mesh(&self) -> Geometry {
        let mut g = Geometry::default();
        let half = self.size_m * 0.5;
        let body = half - Vec2::splat(self.lip_m);
        let base = body - Vec2::splat(self.flare_m);
        let under = self.under_lip_m();
        let rolled = self.gauge_m * 0.42;
        let outer = half - Vec2::splat(rolled);
        g.housing(
            &[
                (self.foot_m, base),
                (under, body),
                (under, outer),
                (under + rolled, half),
                (self.height_m - rolled, half),
                (self.height_m, outer),
                (self.height_m, body - Vec2::splat(self.gauge_m)),
                (self.nesting_pitch_m(), base - Vec2::splat(self.gauge_m)),
            ],
            self.roundness,
            Transform::IDENTITY,
        );
        for side in [-1., 1.] {
            g.cuboid(
                Vec3::new(0.011, self.foot_m, base.y * 1.20),
                self.foot_m * 0.22,
                Transform::from_xyz(side * base.x * 0.55, self.foot_m * 0.5, 0.),
            );
        }
        g
    }

    pub(super) fn install(&self, a: &mut Assembly, loads: &[Load]) {
        let template = self.mesh();
        let mesh = a.part(self.stock, &format!("dishware#finish{}", self.finish_slot));
        let count: usize = loads.iter().map(|l| l.count).sum();
        let vertices = template.positions.len();
        mesh.positions.reserve(count * vertices);
        mesh.normals.reserve(count * vertices);
        mesh.uvs.reserve(count * vertices);
        mesh.indices.reserve(count * template.indices.len());
        for load in loads {
            let rotation = Quat::from_rotation_y(load.yaw);
            let start = mesh.positions.len();
            mesh.positions.extend(
                template
                    .positions
                    .iter()
                    .map(|&p| (rotation * Vec3::from(p) + load.origin).to_array()),
            );
            mesh.normals.extend(
                template
                    .normals
                    .iter()
                    .map(|&n| (rotation * Vec3::from(n)).to_array()),
            );
            for layer in 0..load.count {
                let offset = start + layer * vertices;
                if layer > 0 {
                    mesh.positions.extend_from_within(start..start + vertices);
                    for p in &mut mesh.positions[offset..] {
                        p[1] += layer as f32 * self.nesting_pitch_m();
                    }
                    mesh.normals.extend_from_within(start..start + vertices);
                }
                mesh.uvs.extend_from_slice(&template.uvs);
                mesh.indices
                    .extend(template.indices.iter().map(|&i| offset as u32 + i));
            }
        }
    }
}
