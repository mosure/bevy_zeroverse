use super::*;

pub(super) fn apply(r: &mut MaterialRecipe, p: &ConstructionSample) {
    if let Some(l) = &mut r.leather {
        l.cells = [
            p.count([48., 20., 60., 36.], [12, 64]),
            p.count([42., 25., 54., 32.], [12, 64]),
        ];
        l.roundness = p.mix([1.2, 0.45, 1.7, 0.85]);
        l.stretch = 0.65 + p.variation * 1.05;
        l.polish = p.mix([0.40, 0.12, 0.01, 0.88]);
        l.nap = p.mix([0.02, 0.05, 0.92, 0.0]) * (1. - l.polish).powi(2);
        l.coat = p.mix([0.08, 0.24, 0., 0.85]) * (1. - l.nap);
        l.coat_roughness = 0.10 + (1. - l.polish) * 0.32;
        l.pore_depth = p.mix([0.06, 0.30, 0.12, 0.02]);
        l.crease_strength = p.mix([0.10, 0.24, 0.04, 0.025]) * (0.3 + p.variation * 0.7);
        l.mottling = p.mix([0.24, 0.10, 0.30, 0.035]);
        retain_variation!(
            l,
            super::super::leather::LeatherRecipe::sample(r.seed),
            roundness,
            stretch,
            polish,
            pore_depth,
            crease_strength,
            mottling,
            coat_roughness
        );
        // Couple the final polish to nap/coating after independent finish
        // variation, so a residual cannot produce shiny suede.
        l.nap = p.mix([0.02, 0.05, 0.92, 0.0]) * (1. - l.polish).powi(2);
        l.coat = p.mix([0.08, 0.24, 0., 0.85]) * (1. - l.nap);
        r.period_m = p.mix([0.085, 0.13, 0.06, 0.095]) * p.scale;
        r.relief_m = p.mix([0.000035, 0.00010, 0.000055, 0.000015]);
        r.roughness = (0.52 - l.polish * 0.28 + l.nap * 0.35).clamp(0.20, 0.94);
        return;
    }
    if let Some(t) = &mut r.textile {
        let family = p.dominant();
        // Keep the full seeded topology support within each construction. Four
        // process priors must not turn arbitrary weaves into four fixed stamps.
        let mut rng = stream(r.seed, 0x5745415645505249);
        t.repeat = rng.random_range(2..=7);
        t.bundle = rng.random_range(1..=3);
        t.float_length = if family == 0 && rng.random_bool(0.55) {
            1
        } else {
            rng.random_range(1..t.repeat)
        };
        t.advance = rng.random_range(1..t.repeat);
        t.herringbone = rng.random_bool(if family == 1 { 0.6 } else { 0.15 });
        t.knit = family == 3;
        let unit = t.repeat * t.bundle * 2;
        t.yarns = [p.mix([28., 42., 24., 30.]), p.mix([24., 36., 24., 40.])].map(|n| {
            ((n / unit as f32).round() as u32).clamp(16u32.div_ceil(unit), 48 / unit) * unit
        });
        t.width = [
            p.mix([0.58, 0.82, 0.95, 0.73]),
            p.mix([0.67, 0.86, 0.91, 0.78]),
        ];
        t.slub = p.mix([0.33, 0.025, 0.45, 0.08]) * (0.5 + p.variation * 0.5);
        t.crimp = p.mix([0.25, 0.16, 0.42, 0.35]);
        t.fuzz = p.mix([0.30, 0.06, 0.90, 0.42]);
        t.lustre = p.mix([0.10, 0.70, 0.04, 0.20]);
        t.pile = p.mix([0., 0., 0.85, 0.]);
        t.dye_variation = p.mix([0.25, 0.045, 0.32, 0.09]);
        let original = super::super::textile::TextileRecipe::sample(r.seed);
        for (width, original) in t.width.iter_mut().zip(original.width) {
            *width = *width * 0.80 + original * 0.20;
        }
        retain_variation!(t, original, slub, crimp, fuzz, lustre, dye_variation);
        r.period_m = p.mix([0.0028, 0.0015, 0.0042, 0.0021]) * t.yarns[1] as f32 * p.scale;
        r.relief_m = p.mix([0.00010, 0.000045, 0.00030, 0.00016]);
        r.roughness = (0.74 + t.fuzz * 0.20 - t.lustre * 0.30).clamp(0.40, 0.98);
    }
}

pub(super) fn carpet(r: &mut MaterialRecipe, p: &ConstructionSample) {
    if let Some(c) = &mut r.carpet {
        // Rug dye composition is independent of the pile substrate. Preserve
        // its full pattern/arrangement support rather than replacing it with
        // wall-to-wall carpet installation priors.
        if c.rug.is_none() {
            c.pattern = [0, 1, 2, 0][p.dominant()];
            c.repeats = [
                p.count([6., 3., 8., 12.], [1, 16]),
                p.count([9., 12., 7., 14.], [1, 16]),
            ];
            c.contrast = p.mix([0.14, 0.40, 0.30, 0.055]) * (0.5 + 0.5 * p.variation);
            c.line_width = p.mix([0.12, 0.05, 0.22, 0.30]);
            c.edge_variation = 0.003 + p.variation * 0.022;
        }
        c.heather = p.mix([0.26, 0.10, 0.13, 0.055]);
        c.pile = p.mix([0.20, 0.72, 0.48, 0.12]);
        retain_variation!(
            c,
            super::super::carpet::CarpetDesign::sample(r.seed, r.color, c.rug.is_some()),
            heather,
            pile
        );
        r.relief_m = p.mix([0.00025, 0.00055, 0.00035, 0.00012]);
        r.roughness = p.mix([0.90, 0.85, 0.88, 0.97]);
        // Whole-rug composition, palette, border and installation stay intact.
    }
}
