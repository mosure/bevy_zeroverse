//! Clothing regions live in the rest body, then follow the full Anny skinning.
//! Anatomical heights define waist/cuffs/necklines; rig ownership alone does not.
use super::{HumanAssembly, HumanOutfit, HumanSurface, IndoorHuman};
use bevy::prelude::*;

pub(super) struct GarmentCut {
    pub waist: f32,
    pub neck: f32,
    pub chest: f32,
    pub torso_half_width: f32,
    pub depth_center: f32,
    pub torso_half_depth: f32,
    pub shoe_top: f32,
    pub cuffs: [(Vec3, Vec3); 2],
}
impl GarmentCut {
    pub fn surface(&self, h: &IndoorHuman, p: Vec3, bone: &str) -> HumanSurface {
        if bone.starts_with("eye.") {
            return HumanSurface::Eye;
        }
        if is_hand(bone) {
            return HumanSurface::Skin;
        }
        if is_arm(bone) {
            let (point, normal) = self.cuffs[usize::from(bone.ends_with(".R"))];
            return if (p - point).dot(normal) <= 0.0 {
                HumanSurface::Top
            } else {
                HumanSurface::Skin
            };
        }
        if p.y > self.neck {
            return HumanSurface::Skin;
        }
        if p.y < self.shoe_top {
            return HumanSurface::Shoes;
        }
        if p.y < self.waist {
            return HumanSurface::Trousers;
        }
        let front = p.z < self.depth_center;
        let y = ((p.y - self.waist) / (self.neck - self.waist)).clamp(0.0, 1.0);
        if h.outfit == HumanOutfit::Blazer && front && p.x.abs() < 0.012 + y.powi(2) * 0.075 {
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
        let surface = self.surface(h, triangle[0].rest, bone);
        let center = triangle.iter().map(|v| v.rest).sum::<Vec3>() / 3.0;
        // Almost all faces lie wholly within one garment. Only boundary faces
        // need polygon allocations or clipping; keep dense Anny builds cheap.
        if triangle
            .iter()
            .all(|v| self.surface(h, v.rest, bone) == surface)
            && self.surface(h, center, bone) == surface
        {
            emit(&triangle, surface, mesh);
            return;
        }
        if bone.starts_with("eye.") || is_hand(bone) {
            emit(&triangle, self.surface(h, triangle[0].rest, bone), mesh);
            return;
        }
        if is_arm(bone) {
            let (point, normal) = self.cuffs[usize::from(bone.ends_with(".R"))];
            let (cloth, skin) = split(&triangle, |p| (p - point).dot(normal), Some(mesh));
            emit(&cloth, HumanSurface::Top, mesh);
            emit(&skin, HumanSurface::Skin, mesh);
            return;
        }
        let (below_neck, skin) = split(&triangle, |p| p.y - self.neck, Some(mesh));
        emit(&skin, HumanSurface::Skin, mesh);
        let (legs, top) = split(&below_neck, |p| p.y - self.waist, Some(mesh));
        let (shoes, trousers) = split(&legs, |p| p.y - self.shoe_top, None);
        emit(&shoes, HumanSurface::Shoes, mesh);
        emit(&trousers, HumanSurface::Trousers, mesh);
        if h.outfit == HumanOutfit::Blazer {
            let (shirt, jacket) = split(
                &top,
                |p| {
                    let y = ((p.y - self.waist) / (self.neck - self.waist)).clamp(0.0, 1.0);
                    (p.x.abs() - 0.012 - y.powi(2) * 0.075).max(p.z - self.depth_center)
                },
                Some(mesh),
            );
            emit(&shirt, HumanSurface::Shirt, mesh);
            emit(&jacket, HumanSurface::Top, mesh);
        } else {
            emit(&top, HumanSurface::Top, mesh);
        }
    }

    /// Ease smooths anatomical contours into a hanging torso silhouette. It is
    /// evaluated before skinning, so it also works on leaning/seated people.
    pub fn ease(&self, p: Vec3, surface: HumanSurface) -> Vec3 {
        if !matches!(surface, HumanSurface::Top | HumanSurface::Shirt)
            || p.y > self.chest
            || p.y < self.waist
            || p.x.abs() > self.torso_half_width * 1.05
        {
            return Vec3::ZERO;
        }
        let t = ((p.y - self.waist) / (self.chest - self.waist)).clamp(0.0, 1.0);
        let rx = self.torso_half_width * (0.84 + t * 0.16);
        let rz = self.torso_half_depth * (0.92 + t * 0.08);
        let q = Vec2::new(p.x, p.z - self.depth_center);
        // A rounded hanging cross-section spans the bust instead of following
        // every skin depression. An ellipse made shirts look painted onto skin.
        let section = (q / Vec2::new(rx, rz)).abs();
        let radius = (section.x.powi(4) + section.y.powi(4)).sqrt().sqrt();
        if radius < 0.2 || radius >= 1.0 {
            return Vec3::ZERO;
        }
        let correction = q * (1.0 / radius - 1.0);
        Vec3::new(correction.x, 0.0, correction.y).clamp_length_max(0.05)
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
}

fn split(
    poly: &[GarmentVertex],
    field: impl Fn(Vec3) -> f32,
    seam: Option<&mut HumanAssembly>,
) -> (Vec<GarmentVertex>, Vec<GarmentVertex>) {
    let mut inside = Vec::with_capacity(6);
    let mut outside = Vec::with_capacity(6);
    let mut crossings = Vec::with_capacity(2);
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let da = field(a.rest);
        let db = field(b.rest);
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
            };
            inside.push(v);
            outside.push(v);
            crossings.push(v);
        }
    }
    if let (Some(mesh), [a, b]) = (seam, crossings.as_slice()) {
        mesh.part(HumanSurface::Seam).rod(
            a.position + a.normal * 0.0012,
            b.position + b.normal * 0.0012,
            0.0011,
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
        g.indices.extend([base, base + i, base + i + 1]);
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
            waist: 1.0,
            neck: 1.5,
            chest: 1.4,
            torso_half_width: 0.25,
            depth_center: 0.0,
            torso_half_depth: 0.12,
            shoe_top: 0.15,
            cuffs: [(Vec3::ZERO, Vec3::X); 2],
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
    fn garment_cut_preserves_area_and_shared_edges() {
        let v = |x, y| GarmentVertex {
            rest: Vec3::new(x, y, 0.0),
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::new(x, y),
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
