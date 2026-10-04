//! Clothing regions live in the rest body, then follow the full Anny skinning.
//! Anatomical heights define waist/cuffs/necklines; rig ownership alone does not.
pub(super) mod details;
pub(super) mod fit;
pub(super) mod legs;
mod program;
pub use program::GarmentProgram;
pub(super) mod drape;

use super::{HumanAssembly, HumanSurface, IndoorHuman};
use bevy::prelude::*;

pub(super) struct GarmentCut {
    pub waist: f32,
    pub neck: f32,
    pub chest: f32,
    pub depth_center: f32,
    pub fit: fit::TorsoFit,
    pub legs: [legs::LegFit; 2],
    pub shoe_top: f32,
    pub cuffs: [(Vec3, Vec3); 2],
    pub leg_cuffs: [(Vec3, Vec3); 2],
    pub program: GarmentProgram,
}
impl GarmentCut {
    pub fn surface(&self, h: &IndoorHuman, p: Vec3, bone: &str) -> HumanSurface {
        if bone.starts_with("eye.") {
            return HumanSurface::Eye;
        }
        self.surface_with_skin_field(h, p, self.skin_field(p, bone))
    }

    pub fn skin_field(&self, p: Vec3, bone: &str) -> f32 {
        if is_arm(bone) || is_hand(bone) {
            let (point, normal) = self.cuffs[usize::from(bone.ends_with(".R"))];
            let field = (p - point).dot(normal);
            if is_hand(bone) {
                field.max(0.001)
            } else {
                field
            }
        } else {
            self.program.neckline(p, self.neck, self.depth_center)
        }
    }

    fn surface_with_skin_field(&self, h: &IndoorHuman, p: Vec3, field: f32) -> HumanSurface {
        if field > 0.0 {
            return HumanSurface::Skin;
        }
        if p.y < self.shoe_top {
            return HumanSurface::Shoes;
        }
        if p.y < self.waist {
            return if self.leg_field(p) > 0.0 {
                HumanSurface::Skin
            } else {
                HumanSurface::Trousers
            };
        }
        let front = p.z < self.depth_center;
        let y = ((p.y - self.waist) / (self.neck - self.waist)).clamp(0.0, 1.0);
        if h.outfit.open_front()
            && front
            && p.x.abs() < self.program.placket_width + y.powi(2) * self.program.collar_width
        {
            return HumanSurface::Shirt;
        }
        HumanSurface::Top
    }

    /// Split faces at garment cuts instead of assigning an entire coarse quad
    /// from its first bone. Both sides retain identical interpolated positions,
    /// normals and face-corner UVs; cuffs and collars cannot develop cracks.
    pub fn append(
        &self,
        h: &IndoorHuman,
        bone: &str,
        triangle: [GarmentVertex; 3],
        mesh: &mut HumanAssembly,
    ) {
        if bone.starts_with("eye.") {
            emit(&triangle, HumanSurface::Eye, mesh);
            return;
        }
        let surface = self.surface_with_skin_field(h, triangle[0].rest, triangle[0].skin_field);
        let center = triangle.iter().map(|v| v.rest).sum::<Vec3>() / 3.0;
        let center_field = triangle.iter().map(|v| v.skin_field).sum::<f32>() / 3.0;
        // Almost all faces lie wholly within one garment. Only boundary faces
        // need polygon allocations or clipping; keep dense Anny builds cheap.
        if triangle
            .iter()
            .all(|v| self.surface_with_skin_field(h, v.rest, v.skin_field) == surface)
            && self.surface_with_skin_field(h, center, center_field) == surface
        {
            emit(&triangle, surface, mesh);
            return;
        }
        // Each source vertex owns its skin field. Adjacent triangles share that
        // value even when their first corner belongs to a different rig bone.
        let (below_neck, skin) = split_values(&triangle, |v| v.skin_field, Some((mesh, 0.0022)));
        emit(&skin, HumanSurface::Skin, mesh);
        let (legs, top) = split(&below_neck, |p| p.y - self.waist, Some((mesh, 0.0011)));
        let (shoes, legs) = split(&legs, |p| p.y - self.shoe_top, None);
        let (trousers, bare_legs) = split(&legs, |p| self.leg_field(p), Some((mesh, 0.0014)));
        emit(&bare_legs, HumanSurface::Skin, mesh);
        emit(&shoes, HumanSurface::Shoes, mesh);
        emit(&trousers, HumanSurface::Trousers, mesh);
        if h.outfit.open_front() {
            let (shirt, jacket) = split(
                &top,
                |p| {
                    let y = ((p.y - self.waist) / (self.neck - self.waist)).clamp(0.0, 1.0);
                    (p.x.abs() - self.program.placket_width - y.powi(2) * self.program.collar_width)
                        .max(p.z - self.depth_center)
                },
                Some((mesh, 0.0011)),
            );
            emit(&shirt, HumanSurface::Shirt, mesh);
            emit(&jacket, HumanSurface::Top, mesh);
        } else {
            emit(&top, HumanSurface::Top, mesh);
        }
    }

    fn leg_field(&self, p: Vec3) -> f32 {
        let (point, normal) = self.leg_cuffs[usize::from(p.x > 0.0)];
        (p - point).dot(normal)
    }

    /// Ease smooths anatomical contours into a hanging torso silhouette. It is
    /// evaluated before skinning, so it also works on leaning/seated people.
    pub fn ease(&self, p: Vec3, surface: HumanSurface) -> Vec3 {
        if matches!(surface, HumanSurface::Top | HumanSurface::Shirt) {
            self.fit.delta(p, self.waist, self.chest + 0.03, self.neck)
        } else if surface == HumanSurface::Trousers {
            let side = usize::from(p.x > 0.0);
            self.legs[side].delta(p, self.leg_cuffs[side], self.waist)
        } else {
            Vec3::ZERO
        }
    }
}

fn is_arm(bone: &str) -> bool {
    ["upperarm", "lowerarm", "wrist", "finger", "thumb"]
        .iter()
        .any(|prefix| bone.starts_with(prefix))
}

fn is_hand(bone: &str) -> bool {
    ["wrist", "finger", "thumb", "metacarpal"]
        .iter()
        .any(|prefix| bone.starts_with(prefix))
}

#[derive(Clone, Copy)]
pub(super) struct GarmentVertex {
    pub rest: Vec3,
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub skin_field: f32,
}

pub(super) fn split(
    poly: &[GarmentVertex],
    field: impl Fn(Vec3) -> f32,
    seam: Option<(&mut HumanAssembly, f32)>,
) -> (Vec<GarmentVertex>, Vec<GarmentVertex>) {
    split_values(poly, |v| field(v.rest), seam)
}

fn split_values(
    poly: &[GarmentVertex],
    field: impl Fn(GarmentVertex) -> f32,
    seam: Option<(&mut HumanAssembly, f32)>,
) -> (Vec<GarmentVertex>, Vec<GarmentVertex>) {
    let mut inside = Vec::with_capacity(6);
    let mut outside = Vec::with_capacity(6);
    let mut crossings = Vec::with_capacity(2);
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let da = field(a);
        let db = field(b);
        if da <= 0.0 {
            inside.push(a);
        }
        if da >= 0.0 {
            outside.push(a);
        }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
            // Canonical edge orientation makes adjacent faces produce exactly
            // the same boundary positions, independent of triangle winding.
            let (a, b, da, db) = if a.rest.to_array() < b.rest.to_array() {
                (a, b, da, db)
            } else {
                (b, a, db, da)
            };
            let t = (da as f64 / (da as f64 - db as f64)) as f32;
            let v = GarmentVertex {
                rest: a.rest.lerp(b.rest, t),
                position: a.position.lerp(b.position, t),
                normal: a.normal.lerp(b.normal, t).normalize_or(a.normal),
                uv: a.uv.lerp(b.uv, t),
                skin_field: a.skin_field + (b.skin_field - a.skin_field) * t,
            };
            inside.push(v);
            outside.push(v);
            crossings.push(v);
        }
    }
    if let (Some((mesh, radius)), [a, b]) = (seam, crossings.as_slice()) {
        mesh.part(HumanSurface::Seam).rod(
            a.position + a.normal * 0.0012,
            b.position + b.normal * 0.0012,
            radius,
        );
    }
    (inside, outside)
}

pub(super) fn emit(poly: &[GarmentVertex], surface: HumanSurface, mesh: &mut HumanAssembly) {
    if poly.len() < 3 {
        return;
    }
    let g = mesh.part(surface);
    let base = g.positions.len() as u32;
    for v in poly {
        g.positions.push(v.position.to_array());
        g.normals.push(v.normal.to_array());
        g.uvs.push(v.uv.to_array());
    }
    for i in 1..poly.len() as u32 - 1 {
        let a = poly[0].position;
        let b = poly[i as usize].position;
        let c = poly[i as usize + 1].position;
        if (b - a).cross(c - a).length_squared() > 1e-18 {
            g.indices.extend([base, base + i, base + i + 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palms_and_fingers_are_skin_even_below_the_waist() {
        let h = crate::scene::procedural_indoor::humans::sample_person(
            0,
            0,
            Vec3::ZERO,
            0.0,
            crate::scene::procedural_indoor::humans::HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        let cut = GarmentCut {
            legs: std::array::from_fn(|_| {
                legs::LegFit::new(&[], Vec3::Y, Vec3::ZERO, &GarmentProgram::default())
            }),
            waist: 1.0,
            neck: 1.5,
            chest: 1.4,
            depth_center: 0.0,
            fit: fit::TorsoFit::new(&[], 1.0, 1.4),
            shoe_top: 0.15,
            cuffs: [(Vec3::ZERO, Vec3::X); 2],
            leg_cuffs: [(Vec3::ZERO, Vec3::NEG_Y); 2],
            program: GarmentProgram::default(),
        };
        for side in ["L", "R"] {
            for bone in [
                "metacarpal1",
                "metacarpal4",
                "wrist",
                "finger1-1",
                "finger5-3",
            ] {
                let bone = format!("{bone}.{side}");
                let p = Vec3::new(-0.2, 0.8, -0.05);
                assert_eq!(cut.surface(&h, p, &bone), HumanSurface::Skin);
                assert_eq!(cut.ease(p, cut.surface(&h, p, &bone)), Vec3::ZERO);
                let tri = [p, p + Vec3::X * 0.01, p + Vec3::Y * 0.01].map(|p| GarmentVertex {
                    rest: p,
                    position: p,
                    normal: Vec3::Z,
                    uv: p.truncate(),
                    skin_field: cut.skin_field(p, &bone),
                });
                let mut mesh = HumanAssembly::default();
                cut.append(&h, &bone, tri, &mut mesh);
                assert_eq!(mesh.parts.len(), 1);
                assert_eq!(mesh.parts[&HumanSurface::Skin].positions.len(), 3);
            }
        }
        assert_eq!(
            cut.surface(&h, Vec3::new(0.0, 1.6, -0.1), "oris05"),
            HumanSurface::Skin
        );
    }

    #[test]
    fn sleeve_material_boundaries_do_not_depend_on_first_face_bone() {
        let h = super::super::sample_person(
            0,
            0,
            Vec3::ZERO,
            0.,
            super::super::HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        let cut = GarmentCut {
            legs: std::array::from_fn(|_| {
                legs::LegFit::new(&[], Vec3::Y, Vec3::ZERO, &GarmentProgram::default())
            }),
            waist: 1.0,
            neck: 1.5,
            chest: 1.4,
            depth_center: 0.,
            fit: fit::TorsoFit::new(&[], 1.0, 1.4),
            shoe_top: 0.15,
            cuffs: [
                (Vec3::new(-0.45, 1.2, 0.), Vec3::NEG_X),
                (Vec3::new(0.45, 1.2, 0.), Vec3::X),
            ],
            leg_cuffs: [(Vec3::ZERO, Vec3::NEG_Y); 2],
            program: GarmentProgram::default(),
        };
        let triangle = [
            Vec3::new(0.44, 1.20, -0.05),
            Vec3::new(0.46, 1.20, -0.05),
            Vec3::new(0.44, 1.24, -0.05),
        ]
        .map(|p| GarmentVertex {
            rest: p,
            position: p,
            normal: Vec3::Z,
            uv: p.truncate(),
            skin_field: cut.skin_field(p, "upperarm01.R"),
        });
        let mut expected = HumanAssembly::default();
        cut.append(&h, "upperarm01.R", triangle, &mut expected);
        assert!(expected.parts.contains_key(&HumanSurface::Top));
        assert!(expected.parts.contains_key(&HumanSurface::Skin));
        for first_bone in ["spine01", "lowerarm01.R", "wrist.R", "metacarpal1.R"] {
            let mut mesh = HumanAssembly::default();
            cut.append(&h, first_bone, triangle, &mut mesh);
            assert_eq!(mesh.parts.len(), expected.parts.len());
            for (surface, g) in &expected.parts {
                let actual = &mesh.parts[surface];
                assert_eq!(actual.positions, g.positions, "{first_bone}/{surface:?}");
                assert_eq!(actual.normals, g.normals);
                assert_eq!(actual.uvs, g.uvs);
                assert_eq!(actual.indices, g.indices);
            }
        }
    }

    #[test]
    fn garment_cut_preserves_area_and_shared_edges() {
        let v = |x, y| GarmentVertex {
            rest: Vec3::new(x, y, 0.0),
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::new(x, y),
            skin_field: 0.0,
        };
        let vertices = [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)];
        let mut boundary = Vec::new();
        let mut area = 0.0;
        for indices in [[0, 1, 2], [0, 2, 3]] {
            let triangle = indices.map(|i| vertices[i]);
            let (a, b) = split(&triangle, |p| p.y - 0.37, None);
            for poly in [a, b] {
                for v in &poly {
                    if (v.position.y - 0.37).abs() < 1e-6 {
                        boundary.push(v.position);
                    }
                    assert!(v.uv.distance(v.position.truncate()) < 1e-6);
                }
                for i in 1..poly.len() - 1 {
                    area += (poly[i].position - poly[0].position)
                        .cross(poly[i + 1].position - poly[0].position)
                        .z
                        * 0.5;
                }
            }
        }
        assert!((area - 1.0).abs() < 1e-6);
        assert_eq!(boundary.iter().filter(|p| p.x == p.y).count(), 4);
    }
}
