use super::*;

pub(super) fn apply(r: &mut MaterialRecipe, p: &ConstructionSample, _floor: u32) {
    let Some(m) = &mut r.mineral else {
        return;
    };
    m.aggregate_cells = [
        p.count([80., 18., 48., 64.], [12, 96]),
        p.count([72., 22., 36., 60.], [12, 96]),
    ];
    m.aggregate_radius = p.mix([0.14, 0.43, 0.20, 0.30]);
    m.binder_variation = p.mix([0.025, 0.09, 0.12, 0.045]);
    if let Some(c) = &mut m.casting {
        let ground = p.weights[3];
        m.polish = (p.mix([0.18, 0.02, 0.36, 0.88]) + (p.variation - 0.5) * 0.60).clamp(0., 1.);
        m.aggregate_exposure = p.mix([0.03, 0.08, 0.12, 0.82]);
        m.porosity = p.mix([0.015, 0.07, 0.035, 0.005]);
        c.formwork = p.mix([0.02, 0.96, 0.05, 0.01]);
        c.board_width_m = ((0.055 + p.variation * 0.22) * p.scale).clamp(0.04, 0.5);
        c.form_relief_m = p.mix([0.000015, 0.00032, 0.00006, 0.00001]);
        c.trowel = p.mix([0.12, 0.02, 0.95, 0.16]);
        c.trowel_scale_m = (0.025 + 0.10 * p.variation) * p.scale;
        c.cure_variation = p.mix([0.04, 0.18, 0.12, 0.025]);
        c.sand_exposure = p.mix([0.06, 0.25, 0.18, 0.02]);
        c.bughole_density = p.mix([0.025, 0.20, 0.08, 0.003]);
        c.bughole_radius_m = (0.0005 + p.variation * 0.003) * p.scale;
        c.bleed = (1. - ground) * p.variation * 0.08;
        retain_variation!(
            c,
            super::super::mineral::MineralRecipe::sample(r.seed, r.surface)
                .casting
                .unwrap(),
            formwork,
            board_width_m,
            form_relief_m,
            trowel,
            trowel_scale_m,
            cure_variation,
            sand_exposure,
            bughole_density,
            bughole_radius_m,
            bleed
        );
        r.period_m = (0.4 + p.variation * 0.5) * p.scale;
        // Casting/finishing roughness is independent of aggregate color. Keep
        // polished concrete fine while allowing a varied sanded/cast surface.
        r.relief_m = (0.00020 + p.variation * 0.00085) * (1. - m.polish * 0.94);
        r.roughness = 0.90 - m.polish * 0.62;
    } else if r.surface == Surface::Floor {
        m.polish = (p.mix([0.18, 0.58, 0.75, 0.95]) + (p.variation - 0.5) * 0.60).clamp(0., 1.);
        m.aggregate_exposure = p.mix([0.04, 0.90, 0.03, 0.14]);
        m.marble_mix = p.mix([0.02, 0.01, 0.88, 0.20]);
        m.porosity = p.mix([0.065, 0.008, 0.012, 0.002]);
        m.vein_cells = [
            p.count([3., 7., 2., 8.], [1, 16]),
            p.count([5., 9., 6., 12.], [1, 16]),
        ];
        m.vein_width = 0.003 + p.variation * 0.020;
        m.vein_warp = 0.04 + p.variation * 0.40;
        m.vein_strength = 0.25 + p.variation * 0.50;
        retain_variation!(
            m,
            super::super::mineral::MineralRecipe::sample(r.seed, r.surface),
            aggregate_exposure,
            porosity,
            vein_width,
            vein_warp,
            vein_strength,
            binder_variation
        );
        r.relief_m = 0.00002 + (1. - m.polish) * 0.00020;
        r.roughness = (0.84 - m.polish * 0.65).clamp(0.14, 0.92);
        // Installation scale and cut geometry remain exactly the authored floor.
    } else {
        let soil = r.surface == Surface::Soil;
        m.aggregate_exposure = p.mix(if soil {
            [0.06, 0.85, 0.60, 0.08]
        } else {
            [0.02, 0.58, 0.04, 0.35]
        });
        m.porosity = p.mix([0.10, 0.32, 0.07, 0.23]);
        m.polish = if soil { 0. } else { p.weights[2] * 0.55 };
        r.period_m = p.mix([0.10, 0.22, 0.085, 0.15]) * p.scale;
        r.relief_m = p.mix([0.00008, 0.00060, 0.00003, 0.00030]);
        r.roughness = if soil {
            0.70 + (1. - p.variation) * 0.29
        } else {
            0.94 - m.polish * 0.42
        };
    }
}
