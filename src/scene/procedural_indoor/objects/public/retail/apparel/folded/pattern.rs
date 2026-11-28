use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FoldedCut {
    Tee,
    Pullover,
    ButtonDown,
    Trousers,
}

impl FoldedCut {
    pub(super) fn sample(r: &mut ChaCha8Rng) -> Self {
        match r.random_range(0..10) {
            0..=3 => Self::Tee,
            4..=5 => Self::Pullover,
            6..=7 => Self::ButtonDown,
            _ => Self::Trousers,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FoldedGarment {
    pub cut: FoldedCut,
    pub surface: Surface,
    pub dye_slot: usize,
    pub binding_slot: usize,
    /// Nominal width, compressed thickness and folded length.
    pub size_m: Vec3,
    /// Fraction of each side turned inward at the open edge.
    pub side_tuck: Vec2,
    pub open_edge_recess_m: f32,
    /// Signed left-to-right variation in the depth of the exposed return.
    pub edge_skew: f32,
    /// Neck opening or patch pocket in the top garment's X/Z plane.
    pub feature_center_m: Vec2,
    pub feature_half_size_m: Vec2,
    pub detail_rise_m: f32,
}

impl FoldedGarment {
    pub(super) fn sample(cut: FoldedCut, r: &mut ChaCha8Rng) -> Self {
        let (width, thickness, length, rise) = match cut {
            FoldedCut::Tee => (0.23..0.34, 0.014..0.025, 0.27..0.37, 0.0018..0.0040),
            FoldedCut::Pullover => (0.265..0.37, 0.031..0.052, 0.30..0.41, 0.0035..0.0075),
            FoldedCut::ButtonDown => (0.24..0.34, 0.020..0.034, 0.30..0.41, 0.007..0.014),
            FoldedCut::Trousers => (0.195..0.29, 0.021..0.038, 0.30..0.425, 0.0024..0.0048),
        };
        let size_m = Vec3::new(
            r.random_range(width),
            r.random_range(thickness),
            r.random_range(length),
        );
        let detail_rise_m = r.random_range(rise);
        let tuck = match cut {
            FoldedCut::Trousers => 0.10..0.24,
            FoldedCut::ButtonDown => 0.035..0.13,
            _ => 0.005..0.095,
        };
        let (cx, cz, rx, rz, alt) = match cut {
            FoldedCut::Tee => (-0.03..0.03, -0.285..-0.221, 0.17..0.253, 0.087..0.128, 0.34),
            FoldedCut::Pullover => (-0.03..0.03, -0.278..-0.218, 0.19..0.264, 0.089..0.139, 0.81),
            FoldedCut::ButtonDown => (
                -0.021..0.021,
                -0.275..-0.230,
                0.165..0.205,
                0.079..0.102,
                0.19,
            ),
            FoldedCut::Trousers => (-0.13..0.13, 0.00..0.12, 0.18..0.252, 0.14..0.20, 0.66),
        };
        let surface = if r.random_bool(alt) {
            Surface::FabricAlt
        } else {
            Surface::Fabric
        };
        let dye_slot = garment_stock::sample_slot(r);
        let binding_slot =
            if matches!(cut, FoldedCut::Tee | FoldedCut::Pullover) && r.random_bool(0.27) {
                garment_stock::contrasting_slot(dye_slot, r)
            } else {
                dye_slot
            };
        Self {
            cut,
            surface,
            dye_slot,
            binding_slot,
            size_m,
            detail_rise_m,
            side_tuck: Vec2::new(r.random_range(tuck.clone()), r.random_range(tuck)),
            edge_skew: r.random_range(-0.48..0.48),
            feature_center_m: size_m.xz() * Vec2::new(r.random_range(cx), r.random_range(cz)),
            feature_half_size_m: size_m.xz() * Vec2::new(r.random_range(rx), r.random_range(rz)),
            open_edge_recess_m: (size_m.y * r.random_range(0.19..0.73)).min(size_m.z * 0.055),
        }
    }

    /// Conservative horizontal extent of the largest rotated copy. Fold the
    /// outline and its sewn details to the same scale inside a narrow bay.
    pub(super) fn fit(&mut self, available: Vec2, turn: f32) -> Vec2 {
        let extent =
            (self.size_m.xz() + turn * Vec2::new(self.size_m.z, self.size_m.x)) * MAX_GRADE;
        let fit = (available / extent).min_element().min(1.);
        self.size_m.x *= fit;
        self.size_m.z *= fit;
        self.feature_center_m *= fit;
        self.feature_half_size_m *= fit;
        self.open_edge_recess_m *= fit;
        extent * fit
    }

    pub(super) fn install(&self, a: &mut Assembly, layers: &[FoldedLayer], exposed: bool) {
        let mut item = Assembly::default();
        garment::build(&mut item, self, exposed);
        for ((surface, label), g) in item.parts {
            let dest = a.part(surface, &label);
            dest.positions.reserve(g.positions.len() * layers.len());
            dest.normals.reserve(g.normals.len() * layers.len());
            dest.uvs.reserve(g.uvs.len() * layers.len());
            dest.indices.reserve(g.indices.len() * layers.len());
            for layer in layers {
                append_facing(
                    dest,
                    &g,
                    layer.base_m,
                    layer.yaw_rad.sin_cos(),
                    layer.size_grade,
                );
            }
        }
    }
}
