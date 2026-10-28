//! Metric recipe evaluation, map sizing and PBR map generation.
use super::super as materials;
use super::*;

impl MaterialRecipe {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(c) = &self.construction {
            c.validate()?;
            if materials::construction::labels(self.surface, 0).is_none() {
                return Err("construction provenance requires a supported material role".into());
            }
        }
        if let Some(m) = &self.manufactured {
            m.validate()?;
            if !matches!(
                self.surface,
                Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Rubber
            ) {
                return Err("manufactured finish requires a metal/polymer role".into());
            }
        }
        if self
            .detail_scale
            .is_some_and(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
        {
            return Err("material detail scale must be finite and in [0,1]".into());
        }
        if let Some(f) = &self.flooring {
            f.validate()?;
            if self.surface != Surface::Floor {
                return Err("floor installation requires a floor role".into());
            }
        }
        if let Some(c) = &self.carpet {
            c.validate()?;
            if !matches!(
                self.surface,
                Surface::Floor | Surface::Fabric | Surface::FabricAlt
            ) {
                return Err("carpet requires a floor/textile role".into());
            }
        }
        if !self
            .color
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.).contains(v))
            || !self.roughness.is_finite()
            || !(0.0..=1.).contains(&self.roughness)
            || !self.period_m.is_finite()
            || self.period_m <= 0.
            || !self.relief_m.is_finite()
            || !(0.0..0.02).contains(&self.relief_m)
            || [
                self.warp,
                self.contrast,
                self.weathering,
                self.mineral_mix,
                self.weave_mix,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || self.phase.iter().any(|v| !v.is_finite())
            || !self.grain_frequency.is_finite()
            || !(1.0..=256.).contains(&self.grain_frequency)
            || !self.cross_frequency.is_finite()
            || !(1.0..=64.).contains(&self.cross_frequency)
            || self.panel_count.iter().any(|v| !(1..=64).contains(v))
            || !self.joint_width.is_finite()
            || !(0.0..=0.02).contains(&self.joint_width)
        {
            return Err("invalid metric material recipe".into());
        }
        if self.leaf.is_some()
            && (!matches!(
                self.surface,
                Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
            ) || self.period_m != 1.)
            || self.coating.is_some()
                && !matches!(
                    self.surface,
                    Surface::Paint | Surface::Accent | Surface::Ceiling | Surface::Ceramic
                )
            || self.wood.is_some()
                && !matches!(
                    self.surface,
                    Surface::Wood | Surface::WoodEdge | Surface::Floor
                )
            || self.mineral.is_some()
                && !matches!(
                    self.surface,
                    Surface::Concrete | Surface::Terracotta | Surface::Soil | Surface::Floor
                )
            || self.textile.is_some()
                && !matches!(
                    self.surface,
                    Surface::Fabric | Surface::FabricAlt | Surface::Floor
                )
            || self.leather.is_some() && self.surface != Surface::Leather
            || self.cuticle.is_some() && self.surface != Surface::Succulent
        {
            return Err("substrate program does not match material role".into());
        }
        if let Some(m) = &self.mineral {
            m.validate()?;
            if m.casting.is_some() && self.surface != Surface::Concrete {
                return Err("concrete casting requires a concrete role".into());
            }
        }
        if let Some(c) = &self.coating {
            c.validate()?;
            if c.wall.is_some() && !matches!(self.surface, Surface::Paint | Surface::Accent) {
                return Err("wall covering requires a wall material role".into());
            }
            if c.glaze.is_some() && self.surface != Surface::Ceramic
                || c.application.is_some() && self.surface == Surface::Ceramic
            {
                return Err("glaze/wall application does not match material role".into());
            }
        }
        if let Some(w) = &self.wood {
            w.validate()?;
        }
        if let Some(l) = &self.leaf {
            l.validate()?;
        }
        if let Some(c) = &self.cuticle {
            c.validate()?;
        }
        if let Some(t) = &self.textile {
            t.validate()?;
        }
        if let Some(l) = &self.leather {
            l.validate()?;
        }
        if let Some(l) = &self.layers {
            l.validate()?;
        }
        Ok(())
    }
    /// Bounded resolution: floor atlases cover many independently cut panels.
    pub fn map_size(&self, floor_style: u32) -> u32 {
        if self.carpet.is_some()
            || self.surface == Surface::Floor && floor_style != 1
            || self.surface == Surface::Concrete
                && self.mineral.as_ref().is_some_and(|m| m.casting.is_some())
        {
            512
        } else {
            256
        }
    }
    /// Production substrate maps with sRGB color, tangent-space normal, and
    /// linear R=AO/G=perceptual roughness/B=metalness channels, including mips.
    pub fn maps(&self, floor_style: u32) -> [bevy::prelude::Image; 3] {
        materials::mapped_images(
            materials::texture_maps(
                self.surface,
                floor_style,
                self.seed,
                self.roughness,
                Some(self),
            ),
            self.map_size(floor_style),
            Some(self),
        )
    }
    pub(super) fn specialize(&mut self) {
        let mut rng = stream(self.seed, 0x5355425354524154);
        self.manufactured =
            materials::manufactured::ManufacturedFinish::sample(self.seed, self.surface);
        if let Some(m) = &self.manufactured {
            let polish = rng.random_range(0.0_f32..1.);
            let (roughness, relief) = match self.surface {
                Surface::Chrome => (
                    0.055 + (1. - polish) * 0.26,
                    0.0000003 + (1. - polish) * 0.000003,
                ),
                Surface::Metal => (
                    0.24 + (1. - polish) * 0.50,
                    0.000006 + (1. - polish) * 0.000050 + m.peening * 0.000040,
                ),
                Surface::Rubber => (
                    0.68 + (1. - polish) * 0.28,
                    0.000020 + (1. - polish) * 0.00012,
                ),
                _ => (
                    0.18 + (1. - polish) * 0.62,
                    0.000003 + (1. - polish) * 0.000060,
                ),
            };
            self.roughness = roughness;
            self.relief_m = relief;
            self.period_m = rng.random_range(0.065..0.24);
        }
        if matches!(
            self.surface,
            Surface::Fabric | Surface::FabricAlt | Surface::Floor
        ) {
            let t = materials::textile::TextileRecipe::sample(self.seed);
            if self.surface != Surface::Floor {
                self.period_m = rng.random_range(0.0012..0.0035) * t.yarns[1] as f32;
                self.relief_m = rng.random_range(0.00006..0.00022);
                self.roughness = (0.74 + t.fuzz * 0.20 - t.lustre * 0.30).clamp(0.40, 0.98);
                if rng.random_bool(0.55) {
                    let n = rng.random_range(0.08..0.84);
                    let warmth = rng.random_range(-0.03..0.06);
                    self.color = [
                        n,
                        (n * (1. - warmth)).min(0.95),
                        (n * (1. - warmth * 1.7)).min(0.95),
                    ];
                }
            }
            self.textile = Some(t);
        }
        if self.surface == Surface::Leather {
            let l = materials::leather::LeatherRecipe::sample(self.seed);
            self.period_m = rng.random_range(0.0008..0.0024) * l.cells[1] as f32;
            self.relief_m = rng.random_range(0.000015..0.00010) * (1. - l.polish * 0.55);
            self.roughness = (0.52 - l.polish * 0.26 + l.nap * 0.36).clamp(0.20, 0.91);
            // Upholstery may be dyed neutral, cream or chromatic as well as brown.
            if rng.random_bool(0.35) {
                let c = bevy::prelude::Color::hsl(
                    rng.random_range(0.0..360.),
                    rng.random_range(0.02..0.38),
                    rng.random_range(0.09..0.75),
                )
                .to_srgba();
                self.color = [c.red, c.green, c.blue];
            }
            self.leather = Some(l);
        }
        if matches!(
            self.surface,
            Surface::Concrete | Surface::Floor | Surface::Terracotta | Surface::Soil
        ) {
            let m = materials::mineral::MineralRecipe::sample(self.seed, self.surface);
            if self.surface == Surface::Concrete {
                self.period_m = rng.random_range(0.35..0.90);
                self.relief_m = rng.random_range(0.00006..0.00045) * (1. - m.polish * 0.85);
                self.roughness = 0.88 - m.polish * 0.60;
            } else if matches!(self.surface, Surface::Terracotta | Surface::Soil) {
                self.period_m = rng.random_range(0.06..0.22);
                self.relief_m = rng.random_range(0.00008..0.0005);
            }
            self.mineral = Some(m);
        }
        if matches!(
            self.surface,
            Surface::Paint | Surface::Accent | Surface::Ceiling | Surface::Ceramic
        ) {
            let mut c = materials::coating::CoatingRecipe::sample(self.seed, self.surface);
            self.period_m = if self.surface == Surface::Ceramic {
                rng.random_range(0.085..0.22)
            } else {
                rng.random_range(0.18..0.42)
            };
            self.roughness = if self.surface == Surface::Ceramic {
                0.80 - c.gloss * 0.72
            } else {
                0.95 - c.gloss * 0.72
            };
            self.relief_m = if self.surface == Surface::Ceramic {
                rng.random_range(0.000012..0.000065)
            } else {
                materials::coating::plaster_relief(c.texture_mix)
            };
            if let Some(w) = &c.wall {
                // A wall covering needs a mesoscale domain. Its fibrous relief
                // is millimetric; it must not inherit heavy sprayed plaster.
                self.period_m = rng.random_range(0.70..1.65);
                self.relief_m *= 0.20;
                c.texture_mix *= 0.25;
                c.roller *= 0.35;
                self.roughness =
                    (0.72 + (w.weights[1] + w.weights[2]) * 0.18 - w.burnish).clamp(0.45, 0.96);
            }
            self.coating = Some(c);
        }
        if matches!(
            self.surface,
            Surface::Wood | Surface::WoodEdge | Surface::Floor
        ) {
            self.wood = Some(materials::timber::WoodFinish::sample(self.seed));
        }
        if self.surface == Surface::Bark {
            self.period_m = rng.random_range(0.08..0.25);
            self.relief_m = rng.random_range(0.00025..0.0012);
        }
        if matches!(
            self.surface,
            Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
        ) {
            let l = materials::botanical::LeafRecipe::sample(self.seed, self.surface);
            self.period_m = 1.;
            self.relief_m = rng.random_range(0.000015..0.000065);
            self.roughness = 0.68 - l.wax * 0.26;
            self.leaf = Some(l);
        }
        if self.surface == Surface::Succulent {
            let c = materials::botanical::CuticleRecipe::sample(self.seed);
            self.period_m = rng.random_range(0.055..0.18);
            self.relief_m = rng.random_range(0.000008..0.000030);
            self.roughness = 0.62 - c.wax * 0.25;
            self.cuticle = Some(c);
        }
    }
    /// Bounded structure programs shared by the twelve finish slots.
    pub fn variant(&self, slot: usize) -> Self {
        let mut result = self.clone();
        let group = slot % materials::variants::structure_count(self.surface);
        if group != 0
            && matches!(
                self.surface,
                Surface::Ceramic
                    | Surface::Metal
                    | Surface::Chrome
                    | Surface::Plastic
                    | Surface::Fabric
                    | Surface::FabricAlt
                    | Surface::Leather
                    | Surface::Wood
                    | Surface::WoodEdge
                    | Surface::Leaf
                    | Surface::LeafLight
                    | Surface::LeafVariegated
                    | Surface::Succulent
            )
        {
            result.seed = stream(self.seed, 0x535452554354 + group as u64).random();
            result.specialize();
            result.color = self.color;
            if matches!(
                self.surface,
                Surface::Metal | Surface::Chrome | Surface::Plastic
            ) {
                // Finish families share pigment while their manufactured relief,
                // polish and repeat scale vary. Extra maps remain lazily selected.
                let mut finish = stream(result.seed, 0x46494e495348);
                let polish: f32 = finish.random_range(0.0..1.0);
                let (rough, relief) = match self.surface {
                    Surface::Chrome => (
                        0.07 + (1. - polish) * 0.21,
                        0.000001 + (1. - polish) * 0.000010,
                    ),
                    Surface::Metal => (
                        0.22 + (1. - polish) * 0.36,
                        0.000008 + (1. - polish) * 0.000040,
                    ),
                    _ => (
                        0.24 + (1. - polish) * 0.42,
                        0.000006 + (1. - polish) * 0.000060,
                    ),
                };
                result.roughness = rough;
                result.relief_m = relief;
                result.period_m *= finish.random_range(0.72..1.30);
                result.phase = [finish.random(), finish.random()];
            }
            if let Some(w) = &self.wood {
                result.wood = Some(materials::timber::WoodFinish::sample(
                    stream(w.seed, 0x46494e495348 + group as u64).random(),
                ));
            }
            if self.construction.is_some() {
                materials::construction::sample(&mut result, 0);
            } else if !matches!(self.surface, Surface::Wood | Surface::WoodEdge) {
                // Legacy specialization replaces relief except on timber;
                // construction resolution applies its own intervention once.
                result.relief_m *= self.detail_scale.unwrap_or(1.);
            }
        }
        result
    }
    pub(in crate::scene::procedural_indoor::materials) fn apply_pbr(
        &self,
        mat: &mut bevy::prelude::StandardMaterial,
    ) {
        if let Some(t) = &self.textile {
            if matches!(self.surface, Surface::Fabric | Surface::FabricAlt) {
                t.apply(self, mat);
            }
        }
        if let Some(l) = &self.leather {
            l.apply(mat);
        }
        if let Some(c) = &self.coating {
            c.apply(mat);
        }
        if let Some(w) = &self.wood {
            if self.surface != Surface::Floor {
                w.apply(mat);
            }
        }
        if let Some(l) = &self.leaf {
            l.apply(mat);
        }
        if let Some(c) = &self.cuticle {
            c.apply(mat);
        }
    }
    /// Colored substrate maps contain absolute sRGB reflectance. Upholstery and
    /// modern ceramic glazes retain relative maps for independent pigment tinting.
    pub(in crate::scene::procedural_indoor::materials) fn absolute_color(
        &self,
        floor_style: u32,
    ) -> bool {
        self.carpet.is_some()
            || self.leaf.is_some()
            || self.cuticle.is_some()
            || self.coating.as_ref().is_some_and(|c| c.glaze.is_none())
            || self.surface == Surface::Bark
            || matches!(self.surface, Surface::Wood | Surface::WoodEdge) && self.wood.is_some()
            || matches!(
                self.surface,
                Surface::Concrete | Surface::Terracotta | Surface::Soil
            ) && self.mineral.is_some()
            || self.surface == Surface::Floor
                && (floor_style == 0 && self.wood.is_some()
                    || floor_style == 2 && self.mineral.is_some())
    }
    /// Base-color multiplier paired with `maps`. Mineral/coating/foliage and
    /// stained timber maps carry absolute color; upholstery maps retain tinting.
    pub fn texture_base_color(&self, floor_style: u32) -> [f32; 3] {
        if self.absolute_color(floor_style) {
            [1.; 3]
        } else {
            self.color
        }
    }
    pub fn period_uv(&self) -> bevy::prelude::Vec2 {
        let across = if matches!(self.surface, Surface::Wood | Surface::WoodEdge) {
            (self.period_m * 0.32).clamp(0.22, 0.65)
        } else if let Some(l) = &self.leather {
            self.period_m * l.stretch
        } else {
            self.period_m
        };
        bevy::prelude::Vec2::new(across, self.period_m)
    }

    /// Integer repetitions preserve tileability while constraining the actual
    /// plank/tile dimensions rather than drawing arbitrary counts per texture.
    pub fn floor_repetitions(&self, floor_style: u32) -> [f32; 2] {
        if let Some(f) = &self.flooring {
            return f.cells.map(|n| n as f32);
        }
        let width = if floor_style == 0 {
            0.10 + self.panel_count[0] as f32 * 0.010
        } else {
            0.30 + self.panel_count[0] as f32 * 0.055
        };
        let length = if floor_style == 0 {
            0.75 + self.panel_count[1] as f32 * 0.20
        } else {
            0.30 + self.panel_count[1] as f32 * 0.10
        };
        [
            (self.period_m / width).round().max(1.0),
            (self.period_m / length).round().max(1.0),
        ]
    }
    /// Tileable multi-scale layers, evaluated in a physical repeat domain.
    pub fn evaluate(&self, u: f32, v: f32, floor_style: u32) -> (f32, f32, f32) {
        let t = self.texel(u, v, floor_style);
        (
            (t.color[0] + t.color[1] + t.color[2]) / 3.,
            t.height,
            t.roughness,
        )
    }
    pub(in crate::scene::procedural_indoor::materials) fn texel(
        &self,
        u: f32,
        v: f32,
        floor_style: u32,
    ) -> Texel {
        self.texel_prepared(u, v, floor_style, None)
    }
    pub(in crate::scene::procedural_indoor::materials) fn prepare_texels(
        &self,
        floor_style: u32,
    ) -> Option<PreparedMaterial> {
        if self.leaf.is_some() || self.carpet.is_some() {
            return None;
        }
        if let Some(c) = &self.coating {
            // Preserve the existing glaze-before-paint precedence even when an
            // authored recipe contains both optional application programs.
            if let Some(g) = &c.glaze {
                return materials::ceramic::PreparedGlaze::new(g, c, self)
                    .map(PreparedMaterial::Glaze);
            }
            return c
                .application
                .as_ref()
                .map(|a| PreparedMaterial::Paint(a.prepare(self).cache_cells(self.seed)));
        }
        if let Some(m) = &self.mineral {
            if self.surface != Surface::Floor || floor_style == 2 {
                return Some(PreparedMaterial::Mineral(
                    PreparedMineral::new(m, self.color).cache_fields(m, self),
                ));
            }
        }
        if self.surface == Surface::Bark {
            return None;
        }
        if let Some(w) = self
            .wood
            .as_ref()
            .filter(|_| self.surface != Surface::Floor || floor_style == 0)
        {
            return Some(PreparedMaterial::Wood(PreparedWood::new(w, self.color)));
        }
        if (matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || self.surface == Surface::Floor && floor_style == 1)
            && self.textile.as_ref().is_none_or(|t| !t.knit)
        {
            return Some(PreparedMaterial::Textile(
                materials::textile::PreparedTextile::new(self),
            ));
        }
        None
    }
    pub(in crate::scene::procedural_indoor::materials) fn texel_prepared(
        &self,
        u: f32,
        v: f32,
        floor_style: u32,
        prepared: Option<&PreparedMaterial>,
    ) -> Texel {
        // Leaf UVs run from petiole to tip. Rotating/phasing a repeating wall
        // texture here would move the midrib away from the geometric fold.
        if let Some(l) = &self.leaf {
            return l.evaluate(self, [u, v]);
        }
        // Knit crossovers can have coincident nearest yarns. Wrapping the
        // integer tile offset before rotation/phase keeps opposite tile edges
        // bit-identical instead of letting (1 + phase) - 1 round to a different
        // yarn. Bake texels already lie in [0,1), so archived map bytes and the
        // GPU bake domain remain unchanged.
        let (u, v) = if self.carpet.is_none() && self.textile.as_ref().is_some_and(|t| t.knit) {
            let wrap = |v: f32| {
                if (0.0..1.).contains(&v) {
                    v
                } else {
                    v.rem_euclid(1.)
                }
            };
            (wrap(u), wrap(v))
        } else {
            (u, v)
        };
        let (u, v) = self.layers.as_ref().map_or((u, v), |l| l.rotate(u, v));
        let u = (u + self.phase[0]).rem_euclid(1.0);
        let v = (v + self.phase[1]).rem_euclid(1.0);
        if let Some(c) = &self.carpet {
            return c.evaluate(self, [u, v]);
        }
        if let Some(c) = &self.cuticle {
            return c.evaluate(self, [u, v]);
        }
        if let Some(c) = &self.coating {
            if let Some(g) = &c.glaze {
                return if let Some(PreparedMaterial::Glaze(p)) = prepared {
                    p.evaluate(g, c, self, [u, v])
                } else {
                    g.evaluate(c, self, [u, v])
                };
            }
            if let (Some(PreparedMaterial::Paint(p)), Some(a)) = (prepared, c.application.as_ref())
            {
                return p.evaluate(a, c, self, [u, v]);
            }
            return c.evaluate(self, [u, v]);
        }
        if let Some(m) = &self.mineral {
            if self.surface != Surface::Floor || floor_style == 2 {
                return if let Some(PreparedMaterial::Mineral(p)) = prepared {
                    m.evaluate_prepared(self, [u, v], floor_style, Some(p))
                } else {
                    m.evaluate(self, [u, v], floor_style)
                };
            }
        }
        if self.surface == Surface::Bark {
            return materials::timber::bark(self, [u, v]);
        }
        if let Some(w) = &self.wood {
            if self.surface != Surface::Floor || floor_style == 0 {
                let mut t = if let Some(PreparedMaterial::Wood(p)) = prepared {
                    w.evaluate_prepared(self, [u, v], Some(p))
                } else {
                    w.evaluate(self, [u, v])
                };
                if self.surface == Surface::Floor {
                    self.floor_joints([u, v], &mut t);
                    if let Some(f) = &self.flooring {
                        f.apply(self, [u, v], &mut t);
                    }
                }
                return t;
            }
        }
        if matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || self.surface == Surface::Floor && floor_style == 1
        {
            if let Some(PreparedMaterial::Textile(p)) = prepared {
                return p.evaluate(self, [u, v]);
            }
            if let Some(t) = &self.textile {
                if self.surface != Surface::Floor {
                    return t.evaluate(self, [u, v]);
                }
                let mut t = t.clone();
                if t.pile == 0. {
                    t.pile = 0.85;
                }
                return t.evaluate(self, [u, v]);
            }
            let mut t = materials::textile::TextileRecipe::sample(self.seed);
            if self.surface == Surface::Floor {
                t.pile = 0.85;
            }
            return t.evaluate(self, [u, v]);
        }
        if self.surface == Surface::Leather {
            return self.leather.as_ref().map_or_else(
                || materials::leather::LeatherRecipe::sample(self.seed).evaluate(self, [u, v]),
                |l| l.evaluate(self, [u, v]),
            );
        }
        let noise = |nx, ny, s| periodic_noise(u, v, nx, ny, self.seed.wrapping_add(s));
        let macro_n = noise(3, 3, 1);
        let meso = noise(17, 19, 2);
        let micro = noise(97, 91, 3);
        let wood = matches!(
            self.surface,
            Surface::Wood | Surface::WoodEdge | Surface::Bark
        ) || (self.surface == Surface::Floor && floor_style == 0);
        let cloth = matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || (self.surface == Surface::Floor && floor_style == 1);
        let value =
            materials::microstructure::value(self, [u, v], [macro_n, meso, micro], floor_style);
        let stain = ((macro_n - 0.5) * 3.0).max(0.0) * self.weathering;
        let mut shade = if wood {
            // Leave headroom for light earlywood instead of clipping almost the
            // entire grain to white. Base tint remains the species/palette.
            0.88 + (value - 0.5) * self.contrast * 1.6 - stain
        } else {
            0.96 + (value - 0.5) * self.contrast - stain
        };
        let mut height = (value - 0.5) * self.relief_m;
        let finish_variation = if self.surface == Surface::Chrome {
            0.035
        } else if matches!(
            self.surface,
            Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Paper
        ) {
            0.14
        } else if wood || cloth {
            0.26
        } else {
            0.34
        };
        let mut roughness = self.roughness
            + (value - 0.5) * finish_variation
            + (macro_n - 0.5) * self.weathering
            + stain * 0.3;
        if matches!(
            self.surface,
            Surface::Leather | Surface::Wood | Surface::WoodEdge
        ) {
            // Pores and creases scatter more broadly than their polished tops.
            roughness += (0.5 - value) * 0.28;
        }
        if self.surface == Surface::Floor && floor_style != 1 {
            let [nx, ny] = self.floor_repetitions(floor_style);
            let strip = (u * nx).floor() as u32;
            let stagger = if floor_style == 0 {
                (strip % 2) as f32 * 0.5
            } else {
                0.0
            };
            let x = (u * nx).fract();
            let y = (v * ny + stagger).fract();
            let distance =
                (x.min(1.0 - x) * self.period_m / nx).min(y.min(1.0 - y) * self.period_m / ny);
            let filter_width = self.joint_width + self.period_m / 256.0;
            let seam =
                (1.0 - distance / filter_width).clamp(0.0, 1.0) * (self.joint_width / filter_width);
            // Staggered boards cross the texture boundary halfway along their
            // length. Wrap their identity as well as UVs to avoid a false joint.
            let board = materials::hash(
                strip,
                (v * ny + stagger).floor() as u32 % ny as u32,
                self.seed,
            );
            shade = shade * (1.0 - 0.22 * seam) + (board - 0.5) * self.contrast;
            height -= seam * self.joint_width * 0.25 * self.detail_scale.unwrap_or(1.);
            roughness += seam * 0.12;
        }
        let mut finish = [shade, height, roughness];
        if let Some(layers) = &self.layers {
            layers.apply(
                self,
                [u, v],
                [macro_n, meso, micro],
                &mut finish,
                floor_style,
            );
        }
        Texel {
            color: [finish[0].clamp(0.25, 1.); 3],
            height: finish[1],
            roughness: finish[2].clamp(0.045, 1.),
            occlusion: 1.,
        }
    }
    pub(super) fn floor_joints(&self, [u, v]: [f32; 2], t: &mut Texel) {
        if let Some(f) = &self.flooring {
            let cut = f.cut(self, [u, v]);
            let seam = f.seam(self, &cut, 0);
            t.color = materials::field::tint(t.color, 1. - seam * 0.22);
            t.height -= seam * self.joint_width * 0.25 * self.detail_scale.unwrap_or(1.);
            t.roughness = (t.roughness + seam * 0.12).min(1.);
            if let Some(stock) = &f.stock {
                stock.joint(seam, t);
            }
            return;
        }
        let [nx, ny] = self.floor_repetitions(0);
        let strip = (u * nx).floor() as u32;
        let stagger = (strip % 2) as f32 * 0.5;
        let x = (u * nx).fract();
        let y = (v * ny + stagger).fract();
        let d = (x.min(1. - x) * self.period_m / nx).min(y.min(1. - y) * self.period_m / ny);
        let width = self.joint_width + self.period_m / self.map_size(0) as f32;
        let seam = (1. - d / width).clamp(0., 1.) * self.joint_width / width;
        let board = materials::hash(
            strip,
            (v * ny + stagger).floor() as u32 % ny as u32,
            self.seed,
        );
        t.color = materials::field::tint(
            t.color,
            1. - seam * 0.22 + (board - 0.5) * self.contrast * 0.35,
        );
        t.height -= seam * self.joint_width * 0.25 * self.detail_scale.unwrap_or(1.);
        t.roughness = (t.roughness + seam * 0.12).min(1.);
    }
}
