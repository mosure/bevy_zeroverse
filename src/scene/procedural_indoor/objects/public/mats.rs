//! Floor-contact practice surfaces. Hinges and edge bindings occupy their own
//! sections of the mat so a grazing camera sees the same seams in RGB and depth.
use super::*;
pub use crate::scene::procedural_indoor::materials::mat_stock::Stock;
use std::f32::consts::FRAC_PI_2;

pub mod props;
mod rack;
pub(super) use rack::build as build_rack;
pub use rack::{MatRack, MatRoll};
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 294;

pub(super) fn dimensions(r: &mut impl Rng) -> Vec3 {
    let h = match r.random_range(0..100) {
        0..=54 => r.random_range(0.004..0.008),
        55..=84 => r.random_range(0.008..0.017),
        _ => r.random_range(0.017..0.034),
    };
    Vec3::new(r.random_range(0.56..0.91), h, r.random_range(1.61..2.14))
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PracticeMat {
    pub panels: usize,
    pub corner_m: f32,
    pub hinge_gap_m: f32,
    pub binding_m: f32,
    pub facing: Stock,
}

impl PracticeMat {
    fn panel_depth(&self, size: Vec3) -> f32 {
        (size.z - (self.panels - 1) as f32 * self.hinge_gap_m) / self.panels as f32
    }

    /// A rectangle fully bearing on the flat top of the last panel. Reflect
    /// it in Z for the other end. The middle third stays clear for practice.
    pub(crate) fn prop_area(&self, size: Vec3) -> (Vec2, Vec2) {
        let half = Vec2::new(size.x, self.panel_depth(size)) * 0.5;
        let outline = rounded_outline(half, self.corner_m);
        let inset = Geometry::slab_profile(&outline, size.y, (size.y * 0.17).min(0.003)).1;
        let x = (half.x - self.corner_m) * inset;
        let mid = size.z * 0.5 - half.y;
        (
            Vec2::new(-x, (mid - half.y * inset).max(size.z / 6.)),
            Vec2::new(x, mid + half.y * inset),
        )
    }

    pub(crate) fn supports_prop(&self, size: Vec3, lo: Vec2, hi: Vec2, margin: f32) -> bool {
        let (a, b) = self.prop_area(size);
        let end = if lo.y >= 0. {
            (lo.y, hi.y)
        } else {
            (-hi.y, -lo.y)
        };
        lo.x >= a.x + margin
            && hi.x <= b.x - margin
            && end.0 >= a.y + margin
            && end.1 <= b.y - margin
    }

    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"MatBuild"));
        let folded = r.random_bool(if o.size.y >= 0.015 { 0.62 } else { 0.15 });
        let panels = if folded { r.random_range(2..=5) } else { 1 };
        let corner_m = r
            .random_range(0.009..0.095_f32)
            .min((o.size.z / panels as f32).min(o.size.x) * 0.20);
        let binding_m = if r.random_bool(if folded { 0.65 } else { 0.27 }) {
            r.random_range(0.008..0.021)
        } else {
            0.
        };
        Self {
            panels,
            corner_m,
            binding_m,
            hinge_gap_m: if folded {
                (o.size.y * r.random_range(0.42..0.86)).clamp(0.003, 0.013)
            } else {
                0.
            },
            facing: Stock::for_mat(o, folded),
        }
    }
}

pub(super) fn rounded_outline(half: Vec2, radius: f32) -> [Vec2; 16] {
    static CORNERS: std::sync::LazyLock<[(Vec2, Vec2); 16]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|i| {
            let quadrant = i / 4;
            let direction = Vec2::from_angle((quadrant as f32 + (i % 4) as f32 / 3.) * FRAC_PI_2);
            let sign = [
                Vec2::ONE,
                Vec2::new(-1., 1.),
                Vec2::NEG_ONE,
                Vec2::new(1., -1.),
            ][quadrant];
            (sign, direction)
        })
    });
    CORNERS.map(|(sign, dir)| sign * (half - Vec2::splat(radius)) + radius * dir)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let mat = PracticeMat::for_object(o);
    let s = o.size;
    let panel_depth = mat.panel_depth(s);
    let half = Vec2::new(s.x, panel_depth) * 0.5;
    let corners = rounded_outline(half, mat.corner_m);
    let profile = Geometry::slab_profile(&corners, s.y, (s.y * 0.17).min(0.003));
    let binding = (mat.binding_m > 0.).then(|| {
        rounded_outline(
            half * profile.1 - Vec2::splat(mat.binding_m),
            (mat.corner_m * profile.1 - mat.binding_m).max(0.001),
        )
    });
    let (face, label) = mat.facing.key();
    let class = o.kind.class_name();
    for i in 0..mat.panels {
        let z = -s.z * 0.5 + panel_depth * 0.5 + i as f32 * (panel_depth + mat.hinge_gap_m);
        let tf = Transform::from_xyz(0., s.y * 0.5, z);
        let rubber = a.part(Surface::Rubber, class);
        rubber.slab_edge(&corners, s.y, profile, &tf);
        rubber.slab_cap(&corners, profile.1, Vec2::ZERO, -s.y * 0.5, &tf, |p| p);
        if let Some(inner) = &binding {
            let at = |p: Vec2| Vec3::new(p.x, s.y, p.y + z);
            for j in 0..corners.len() {
                let k = (j + 1) % corners.len();
                rubber.metric_quad(
                    [
                        at(corners[j] * profile.1),
                        at(corners[k] * profile.1),
                        at(inner[k]),
                        at(inner[j]),
                    ],
                    Vec3::Y,
                );
            }
        }
        let facing = a.part(face, label);
        if let Some(inner) = &binding {
            facing.slab_cap(inner, 1., Vec2::ZERO, s.y * 0.5, &tf, |p| p + Vec2::Y * z);
        } else {
            facing.slab_cap(&corners, profile.1, Vec2::ZERO, s.y * 0.5, &tf, |p| {
                p + Vec2::Y * z
            });
        }
        if i + 1 < mat.panels {
            // Only the flexible web crosses the gap. Its upper face is recessed
            // and its underside clears the floor, avoiding a duplicate bottom face.
            a.box_part(
                Surface::Rubber,
                class,
                Vec3::new(0., s.y * 0.40, z + (panel_depth + mat.hinge_gap_m) * 0.5),
                Vec3::new(s.x - 2. * mat.corner_m, s.y * 0.17, mat.hinge_gap_m),
                0.,
            );
        }
    }
}
