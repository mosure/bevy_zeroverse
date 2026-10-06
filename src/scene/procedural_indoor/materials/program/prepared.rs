//! Map-owned invariant coefficients; never part of the serialized material program.
use super::super::{
    ceramic::PreparedGlaze,
    mineral::{MineralFields, MineralRecipe},
    paint::PreparedPaint,
    textile::PreparedTextile,
    timber::WoodFinish,
};

// One context lives on the stack per atlas, never per texel or in a collection.
// Keep its fixed coefficients inline rather than add another map allocation.
#[allow(clippy::large_enum_variant)]
pub(in super::super) enum PreparedMaterial {
    Paint(PreparedPaint),
    Glaze(PreparedGlaze),
    Wood(PreparedWood),
    Mineral(PreparedMineral),
    Textile(PreparedTextile),
}

pub(in super::super) struct PreparedWood {
    base: [f32; 3],
    stain: [f32; 3],
    bleach: [f32; 3],
}
impl PreparedWood {
    pub(in super::super) fn new(w: &WoodFinish, color: [f32; 3]) -> Self {
        Self {
            base: color.map(super::super::srgb_to_linear),
            stain: w.stain_color.map(super::super::srgb_to_linear),
            bleach: [0.88, 0.87, 0.82].map(super::super::srgb_to_linear),
        }
    }
    pub(in super::super) fn stain(&self, amount: f32) -> [f32; 3] {
        linear_mix(self.base, self.stain, amount)
    }
    pub(in super::super) fn bleach(&self, color: [f32; 3], amount: f32) -> [f32; 3] {
        // The input is the preceding mix's sRGB result. Retain that round trip;
        // collapsing the mixtures in linear space changes f32 rounding.
        linear_mix(color.map(super::super::srgb_to_linear), self.bleach, amount)
    }
}

pub(in super::super) struct PreparedMineral {
    base: [f32; 3],
    aggregate: [[f32; 3]; 2],
    vein: [f32; 3],
    fields: Option<MineralFields>,
}
impl PreparedMineral {
    pub(in super::super) fn new(m: &MineralRecipe, color: [f32; 3]) -> Self {
        Self {
            base: color.map(super::super::srgb_to_linear),
            aggregate: m
                .aggregate_color
                .map(|c| c.map(super::super::srgb_to_linear)),
            vein: m.vein_color.map(super::super::srgb_to_linear),
            fields: None,
        }
    }
    pub(in super::super) fn cache_fields(
        mut self,
        m: &MineralRecipe,
        r: &super::MaterialRecipe,
    ) -> Self {
        self.fields = Some(MineralFields::new(m, r, |dye| self.aggregate(dye)));
        self
    }
    #[cfg(test)]
    pub(in super::super) fn set_test_fields(&mut self, fields: MineralFields) {
        self.fields = Some(fields);
    }
    pub(in super::super) fn fields(&self) -> Option<&MineralFields> {
        self.fields.as_ref()
    }
    pub(in super::super) fn matrix(&self, gain: f32) -> [f32; 3] {
        self.base
            .map(|c| super::super::linear_to_srgb((c * gain).clamp(0., 1.)))
    }
    pub(in super::super) fn aggregate(&self, amount: f32) -> [f32; 3] {
        linear_mix(self.aggregate[0], self.aggregate[1], amount)
    }
    pub(in super::super) fn vein(&self, color: [f32; 3], amount: f32) -> [f32; 3] {
        linear_mix(color.map(super::super::srgb_to_linear), self.vein, amount)
    }
    pub(in super::super) fn vein_linear(&self) -> [f32; 3] {
        self.vein
    }
    pub(in super::super) fn mixed_linear(
        color: [f32; 3],
        endpoint: [f32; 3],
        amount: f32,
    ) -> [f32; 3] {
        linear_mix(color, endpoint, amount)
    }
}

fn linear_mix(a: [f32; 3], b: [f32; 3], amount: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| super::super::linear_to_srgb(a[i] * (1. - amount) + b[i] * amount))
}
