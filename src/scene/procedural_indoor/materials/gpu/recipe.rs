//! Explicit, versioned CPU/WGSL recipe ABI. Unsupported programs keep their CPU implementation.
use super::super::{program::MaterialRecipe, Surface};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Recipe {
    pub meta: [u32; 4],
    pub surface: [u32; 4],
    pub metric: [f32; 4],
    pub phase: [f32; 4],
    pub topology: [u32; 4],
    pub cloth0: [f32; 4],
    pub cloth1: [f32; 4],
    pub cloth2: [f32; 4],
    pub tint0: [f32; 4],
    pub tint1: [f32; 4],
    pub dye_runs: [f32; 4],
    pub dye_phase: [f32; 4],
    pub dye_tint: [f32; 4],
    pub leather0: [f32; 4],
    pub leather1: [f32; 4],
    pub carpet: [f32; 4],
    pub carpet_meta: [u32; 4],
    pub carpet0: [f32; 4],
    pub carpet1: [f32; 4],
    pub carpet2: [f32; 4],
    pub floor_meta: [u32; 4],
    pub floor0: [f32; 4],
    pub floor1: [f32; 4],
    pub floor_tint: [f32; 4],
}

impl Recipe {
    pub(super) fn new(r: &MaterialRecipe, floor: u32) -> Option<Self> {
        // Match texel_prepared's precedence. Never silently bypass an authored program.
        if r.map_size(floor) > super::kernel::MAX_SIZE
            || r.validate().is_err()
            || r.leaf.is_some()
            || r.cuticle.is_some()
            || r.coating.is_some()
            || r.carpet.is_none()
                && r.mineral.is_some()
                && (r.surface != Surface::Floor || floor == 2)
            || r.carpet.is_none() && r.wood.is_some() && (r.surface != Surface::Floor || floor == 0)
        {
            return None;
        }
        let cloth = matches!(r.surface, Surface::Fabric | Surface::FabricAlt)
            || r.surface == Surface::Floor && floor == 1;
        let generic = matches!(
            r.surface,
            Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Rubber | Surface::Paper
        );
        if !cloth && r.surface != Surface::Leather && !generic {
            return None;
        }
        let period = r.period_uv();
        let mut p = Self {
            meta: [3, r.map_size(floor), r.seed as u32, (r.seed >> 32) as u32],
            surface: [
                r.surface as u32,
                r.layers.as_ref().map_or(0, |l| u32::from(l.quarter_turn)),
                0,
                0,
            ],
            metric: [period.x, period.y, r.relief_m, r.roughness],
            phase: [r.phase[0], r.phase[1], r.weathering, r.contrast],
            ..bytemuck::Zeroable::zeroed()
        };
        if let Some(c) = &r.carpet {
            p.meta[0] = 4;
            p.carpet = [c.contrast, c.heather, c.pile, c.line_width];
            p.carpet_meta = [c.pattern, c.repeats[0], c.repeats[1], 0];
            p.carpet0[..3].copy_from_slice(&c.palette[0]);
            p.carpet0[3] = c.edge_variation;
            p.carpet1[..3].copy_from_slice(&c.palette[1]);
            p.carpet2[..3].copy_from_slice(&c.palette[2]);
            if let Some(f) = &r.flooring {
                use super::super::flooring::FloorLayout;
                p.floor_meta = [
                    1,
                    u32::from(f.layout == FloorLayout::BrokenBond),
                    f.cells[0],
                    f.cells[1],
                ];
                p.floor0 = [f.piece_tone, f.piece_warmth, f.piece_roughness, f.mottling];
                p.floor1 = [f.accent_fraction, f.tile_definition, r.joint_width, 0.];
                p.floor_tint[..3].copy_from_slice(&f.accent_tint);
                // Basket is a timber installation, never a carpet bake.
                if f.layout == FloorLayout::Basket {
                    return None;
                }
            }
        } else if cloth {
            let mut t = r
                .textile
                .clone()
                .unwrap_or_else(|| super::super::textile::TextileRecipe::sample(r.seed));
            if r.surface == Surface::Floor && t.pile == 0. {
                t.pile = 0.85;
            }
            p.meta[0] = if t.knit { 2 } else { 0 };
            p.surface[2..].copy_from_slice(&t.yarns);
            p.topology = [
                t.repeat,
                t.float_length,
                t.bundle,
                t.advance | (u32::from(t.herringbone) << 16),
            ];
            p.cloth0 = [t.width[0], t.width[1], t.slub, t.crimp];
            p.cloth1 = [t.twist, t.fuzz, t.lustre, t.dye_variation];
            p.cloth2 = [t.pile, 0., 0., 0.];
            if let Some(l) = &r.layers {
                p.cloth2[1..].copy_from_slice(&[
                    l.stripe_strength,
                    l.bands[0] as f32,
                    l.bands[1] as f32,
                ]);
            }
            p.tint0[..3].copy_from_slice(&t.yarn_tint[0]);
            p.tint1[..3].copy_from_slice(&t.yarn_tint[1]);
            if let Some(d) = &t.dye_program {
                p.dye_runs = [d.runs[0] as f32, d.runs[1] as f32, d.duty[0], d.duty[1]];
                p.dye_phase = [d.phase[0], d.phase[1], d.strength[0], d.strength[1]];
                p.dye_tint[..3].copy_from_slice(&d.secondary_tint);
                p.dye_tint[3] = 1.;
            }
        } else if r.surface == Surface::Leather {
            let l = r
                .leather
                .clone()
                .unwrap_or_else(|| super::super::leather::LeatherRecipe::sample(r.seed));
            p.meta[0] = 1;
            p.metric[0] = r.period_m * l.stretch;
            p.surface[2..].copy_from_slice(&l.stretched_cells());
            p.leather0 = [l.roundness, l.pore_depth, l.crease_strength, l.mottling];
            p.leather1 = [l.nap, 0., 0., 0.];
        }
        Some(p)
    }
}
