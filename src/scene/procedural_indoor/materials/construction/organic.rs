use super::*;

pub(super) fn apply(r: &mut MaterialRecipe, p: &ConstructionSample) {
    if let Some(l) = &mut r.leaf {
        l.vein_pairs = p.count([9., 16., 7., 21.], [3, 24]);
        l.vein_rake = p.mix([0.22, 0.65, 0.40, 0.75]);
        l.vein_curve = -0.24 + p.variation * 0.48;
        l.vein_width = p.mix([0.002, 0.003, 0.0025, 0.008]);
        l.midrib_width = p.mix([0.005, 0.009, 0.006, 0.017]);
        l.wax = p.mix([0.94, 0.04, 0.35, 0.18]);
        l.chlorophyll_variation = p.mix([0.06, 0.18, 0.28, 0.14]);
        l.patch_cells = [
            p.count([5., 8., 2., 6.], [1, 16]),
            p.count([8., 12., 4., 14.], [1, 16]),
        ];
        l.variegation = if r.surface == Surface::LeafVariegated {
            p.mix([0.24, 0.38, 0.92, 0.48])
        } else {
            p.weights[2] * 0.25
        };
        retain_variation!(
            l,
            super::super::botanical::LeafRecipe::sample(r.seed, r.surface),
            vein_rake,
            vein_curve,
            vein_width,
            midrib_width,
            wax,
            chlorophyll_variation,
            variegation
        );
        // Waxier/thicker leaves transmit less; never synthesize luminous backs.
        l.transmission = 0.018 + (1. - l.wax) * 0.055;
        r.roughness = 0.70 - l.wax * 0.42;
        r.relief_m = p.mix([0.000015, 0.000025, 0.000020, 0.000075]);
    } else if let Some(c) = &mut r.cuticle {
        c.wax = p.mix([0.12, 0.94, 0.20, 0.32]);
        c.bloom = p.mix([0.08, 0.01, 0.88, 0.18]);
        c.pore_cells = [
            p.count([48., 56., 32., 12.], [8, 64]),
            p.count([52., 48., 36., 18.], [8, 64]),
        ];
        c.pore_contrast = p.mix([0.035, 0.018, 0.06, 0.20]);
        c.chlorophyll_variation = p.mix([0.06, 0.025, 0.11, 0.18]);
        retain_variation!(
            c,
            super::super::botanical::CuticleRecipe::sample(r.seed),
            wax,
            bloom,
            pore_contrast,
            chlorophyll_variation
        );
        r.roughness = 0.68 - c.wax * 0.30 + c.bloom * 0.18;
        r.relief_m = p.mix([0.000014, 0.000008, 0.000024, 0.000045]);
        r.period_m = 0.10 * p.scale;
    } else if let Some(w) = &mut r.wood {
        w.stain_strength = p.mix([0.08, 0.24, 0.03, 0.87]);
        w.bleach = p.mix([0.03, 0.01, 0.72, 0.0]);
        w.pore_fill = p.mix([0.62, 0.04, 0.20, 0.92]);
        w.clearcoat = p.mix([0.20, 0.015, 0.04, 0.85]);
        w.coat_roughness = p.mix([0.24, 0.38, 0.45, 0.075]);
        w.ring_contrast = p.mix([0.40, 1.15, 0.70, 0.55]);
        retain_variation!(
            w,
            super::super::timber::WoodFinish::sample(w.seed),
            stain_strength,
            bleach,
            pore_fill,
            clearcoat,
            coat_roughness,
            ring_contrast
        );
        r.grain_frequency = p.mix([24., 64., 38., 18.]) / p.scale;
        r.warp = 0.025 + p.variation * 0.10;
        r.relief_m = p.mix([0.000035, 0.00022, 0.00012, 0.000025]);
        r.roughness = p.mix([0.43, 0.76, 0.84, 0.21]);
        if r.surface != Surface::Floor {
            r.period_m = p.mix([0.75, 1.3, 0.9, 1.05]) * p.scale;
        }
    } else if r.surface == Surface::Bark {
        r.period_m = p.mix([0.07, 0.25, 0.14, 0.20]) * p.scale;
        r.relief_m = p.mix([0.00025, 0.0017, 0.00045, 0.0011]);
        r.roughness = p.mix([0.78, 0.98, 0.85, 0.94]);
    }
}
