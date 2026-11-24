//! Handed writing tops share one installation transform with their underframe.
use super::*;
use crate::scene::procedural_indoor::placement::rotated_half;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Outline {
    Clipped,
    Radiused,
    Paddle,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Tablet {
    pub(crate) left_handed: bool,
    pub(super) half: Vec2,
    pub(super) center: Vec3,
    pub(super) thickness: f32,
    cutback: f32,
    pub(crate) outline: Outline,
    pub(crate) yaw: f32,
    pub(crate) face_stock: Surface,
    pub(crate) edge_stock: Surface,
    corner_setback: f32,
    shoulder: f32,
    bevel: f32,
}

impl Tablet {
    pub(super) fn sample(o: &IndoorObject, shell_stock: Surface, r: &mut impl Rng) -> Self {
        let left_handed = stream(o.seed, u64::from_le_bytes(*b"WriteArm")).random_bool(0.15);
        let side = if left_handed { -1. } else { 1. };
        let half = Vec2::new(
            o.size.x * r.random_range(0.143..0.183),
            o.size.z * r.random_range(0.225..0.278),
        );
        let height = r.random_range(0.685..0.743);
        let forward = o.size.z * r.random_range(0.13..0.165);
        let thickness = r.random_range(0.015..0.024);
        let cutback = r.random_range(0.46..0.94);
        let mut edge = stream(o.seed, u64::from_le_bytes(*b"DeskWing"));
        let outline = match edge.random_range(0..10) {
            0..=2 => Outline::Clipped,
            3..=5 => Outline::Radiused,
            _ => Outline::Paddle,
        };
        // A fixed elbow rest or a slightly turned pivoting top. Fit the rotated
        // blank inside the admitted chair footprint before mounting the bracket.
        let yaw = if edge.random_bool(0.24) {
            0.
        } else {
            side * edge.random_range(-0.17..0.24)
        };
        let extent = rotated_half(2. * half, yaw);
        let center = Vec3::new(
            side * (0.472 * o.size.x - extent.x),
            height,
            -forward.min(0.48 * o.size.z - extent.y),
        );
        // Tops can be purchased separately from the molded seating. A room's
        // product finish is shared, while moving the tablet cannot change it.
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"DeskSkin"),
        );
        let face_stock = match finish.random_range(0..10) {
            0..=3 => shell_stock,
            4..=7 => Surface::Wood,
            _ => Surface::Plastic,
        };
        let edge_stock = if face_stock == Surface::Wood && finish.random_bool(0.46) {
            Surface::Plastic // impact-resistant perimeter on the veneered panel
        } else {
            face_stock
        };
        Self {
            left_handed,
            half,
            center,
            thickness,
            cutback,
            outline,
            yaw,
            face_stock,
            edge_stock,
            corner_setback: half.min_element() * edge.random_range(0.17..0.48),
            shoulder: edge.random_range(-0.22..0.36),
            bevel: edge.random_range(0.0016..0.0035),
        }
    }

    pub(super) fn side(self) -> f32 {
        if self.left_handed {
            -1.
        } else {
            1.
        }
    }

    pub(super) fn projected_half(self) -> Vec2 {
        rotated_half(2. * self.half, self.yaw)
    }

    fn pose(self) -> Transform {
        Transform::from_translation(self.center).with_rotation(Quat::from_rotation_y(self.yaw))
    }

    pub(super) fn mounting_plate(self) -> (Transform, Vec3) {
        let pose = self.pose();
        let contact = pose.transform_point(Vec3::new(
            self.side() * self.half.x * 0.40,
            -self.thickness * 0.5 - 0.007,
            self.half.y * 0.48,
        ));
        (
            pose.with_translation(contact),
            Vec3::new(self.half.x * 0.73, 0.016, self.half.y * 0.40),
        )
    }

    pub(super) fn build(self, a: &mut Assembly) {
        let x = self.half.x;
        let z = self.half.y;
        let back_inner = -x + self.cutback * x;
        let rect = [
            Vec2::new(-x, -z),
            Vec2::new(x, -z),
            Vec2::new(x, z),
            Vec2::new(-x, z),
        ];
        let clipped = [
            rect[0],
            rect[1],
            rect[2],
            Vec2::new(back_inner, z),
            Vec2::new(-x, z * self.shoulder),
        ];
        // A paddle also tapers on the outside of the elbow. The writing end
        // stays broad while the rear section clears the user's shoulder.
        let paddle = [
            rect[0],
            rect[1],
            Vec2::new(x, 0.15 * z),
            Vec2::new(x * (0.89 - 0.16 * self.cutback), z),
            clipped[3],
            clipped[4],
        ];
        let (corners, subdivisions): (&[Vec2], usize) = match self.outline {
            Outline::Clipped => (&clipped, 1),
            Outline::Radiused => (&rect, 3),
            Outline::Paddle => (&paddle, 3),
        };
        // Quadratic corner fillets have tangent entry and exit. The setbacks on
        // adjacent corners jointly consume at most 84% of any edge.
        let mut points = [Vec2::ZERO; 24];
        let len = corners.len() * (subdivisions + 1);
        for (i, &tip) in corners.iter().enumerate() {
            let prev = corners[(i + corners.len() - 1) % corners.len()] - tip;
            let next = corners[(i + 1) % corners.len()] - tip;
            let start = tip + prev * (self.corner_setback / prev.length()).min(0.42);
            let end = tip + next * (self.corner_setback / next.length()).min(0.42);
            for step in 0..=subdivisions {
                let t = step as f32 / subdivisions as f32;
                let p = start.lerp(tip, t).lerp(tip.lerp(end, t), t);
                let j = i * (subdivisions + 1) + step;
                points[if self.left_handed { len - 1 - j } else { j }] =
                    Vec2::new(p.x * self.side(), p.y);
            }
        }
        let outline = &points[..len];
        let (bevel, inset) = Geometry::slab_profile(outline, self.thickness, self.bevel);
        let pose = self.pose();
        let face = a.part(self.face_stock, "chair");
        for height in [-0.5 * self.thickness, 0.5 * self.thickness] {
            face.slab_cap(outline, inset, Vec2::ZERO, height, &pose, |v| v);
        }
        a.part(self.edge_stock, "chair")
            .slab_edge(outline, self.thickness, (bevel, inset), &pose);
    }
}
