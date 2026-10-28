use super::*;

pub(super) fn apply(r: &mut MaterialRecipe, p: &ConstructionSample) {
    if let Some(m) = &mut r.manufactured {
        m.cells = [
            p.count([110., 32., 105., 20.], [2, 128]),
            p.count([96., 40., 3., 24.], [2, 128]),
        ];
        m.structure = p.mix([0.35, 0.80, 0.95, 0.90]);
        m.peening = p.mix([0., 0.72, 0.04, 0.35]);
        retain_variation!(
            m,
            super::super::manufactured::ManufacturedFinish::sample(r.seed, r.surface).unwrap(),
            structure,
            peening
        );
        let (rough, relief) = match r.surface {
            Surface::Chrome => (
                [0.06, 0.20, 0.32, 0.46],
                [0.0000004, 0.000002, 0.000003, 0.000004],
            ),
            Surface::Metal => (
                [0.27, 0.58, 0.37, 0.81],
                [0.000008, 0.000065, 0.000018, 0.00012],
            ),
            Surface::Rubber => (
                [0.64, 0.82, 0.91, 0.98],
                [0.00002, 0.00013, 0.00020, 0.00007],
            ),
            _ => (
                [0.16, 0.52, 0.86, 0.36],
                [0.000003, 0.00009, 0.000025, 0.000015],
            ),
        };
        // Polish is an independent finish axis within a manufacturing process;
        // a peened or molded substrate is not always the same satin/gloss grade.
        let minimum = if r.surface == Surface::Chrome {
            0.045
        } else if r.surface == Surface::Rubber {
            0.50
        } else {
            0.08
        };
        r.roughness = (p.mix(rough) + (p.variation - 0.5) * 0.40).clamp(minimum, 0.99);
        r.relief_m = p.mix(relief);
        r.period_m = p.mix([0.08, 0.18, 0.13, 0.26]) * p.scale;
        return;
    }
    if r.surface == Surface::Paper {
        r.period_m = p.mix([0.07, 0.05, 0.18, 0.25]) * p.scale;
        r.relief_m = p.mix([0.000014, 0.000003, 0.000055, 0.00012]);
        r.roughness = p.mix([0.82, 0.34, 0.94, 0.98]);
        r.contrast = p.mix([0.012, 0.008, 0.035, 0.09]) * r.detail_scale.unwrap_or(1.);
        return;
    }
    let Some(c) = &mut r.coating else {
        return;
    };
    let original = super::super::coating::CoatingRecipe::sample(r.seed, r.surface);
    if let Some(g) = &mut c.glaze {
        // Matte and glossy glazes both occur within each fired-body process.
        // Keep that independently sampled range rather than fixing it by pole.
        c.gloss = original.gloss;
        c.clearcoat = c.gloss * p.mix([0.32, 0.12, 0.50, 0.28]);
        c.coat_roughness = 0.055 + (1. - c.gloss) * 0.34;
        c.crackle = p.mix([0., 0.015, 0.05, 0.70]);
        g.reactive_mix = p.mix([0.015, 0.12, 0.92, 0.24]);
        g.thickness_variation = p.mix([0.03, 0.12, 0.85, 0.26]);
        g.flow = p.mix([0.04, 0.08, 0.80, 0.22]);
        g.throwing = p.mix([0.015, 0.42, 0.25, 0.62]);
        g.speckle_density = p.mix([0.002, 0.62, 0.08, 0.04]);
        g.speckle_radius_m = p.mix([0.0001, 0.0008, 0.00035, 0.00018]) * p.scale;
        g.body_grain_m = (p.mix([0.0003, 0.0017, 0.00065, 0.0009]) * p.scale).clamp(0.0002, 0.004);
        g.cloud_scale_m = (p.mix([0.012, 0.022, 0.045, 0.018]) * p.scale).clamp(0.003, 0.10);
        g.crack_spacing_m = p.mix([0.009, 0.028, 0.018, 0.007]) * p.scale;
        g.turning_pitch_m = p.mix([0.002, 0.008, 0.004, 0.006]) * p.scale;
        retain_variation!(
            g,
            original.glaze.as_ref().unwrap(),
            reactive_mix,
            thickness_variation,
            flow,
            throwing,
            speckle_density,
            speckle_radius_m,
            body_grain_m,
            cloud_scale_m,
            crack_spacing_m,
            turning_pitch_m
        );
        r.period_m = 0.12 + p.variation * 0.10;
        r.relief_m = p.mix([0.000008, 0.00008, 0.000035, 0.000055]);
        r.roughness = 0.80 - c.gloss * 0.72;
        return;
    }
    c.texture_mix = p.mix([0.03, 0.40, 0.88, 0.38]);
    c.knockdown = p.mix([0., 0.06, 0.90, 0.32]);
    c.roller = p.mix([0.65, 0.10, 0.06, 0.02]);
    c.trowel = p.mix([0.01, 0.025, 0.18, 0.90]);
    c.gloss = p.mix([0.48, 0.12, 0.035, 0.20]);
    c.pinholes = p.mix([0.005, 0.035, 0.12, 0.05]);
    c.pigment_variation = p.mix([0.008, 0.025, 0.04, 0.12]);
    if let Some(a) = &mut c.application {
        a.spray_spacing_m = p.mix([0.003, 0.004, 0.008, 0.005]) * p.scale;
        a.roller_spacing_m = p.mix([0.001, 0.002, 0.004, 0.0025]) * p.scale;
        a.roller_stretch = 1.1 + p.variation * 4.5;
        a.orange_peel = p.mix([0.025, 0.90, 0.16, 0.02]);
        a.orange_peel_m = p.mix([0.0007, 0.0018, 0.0025, 0.001]) * p.scale;
        a.trowel_scale_m = (p.mix([0.02, 0.03, 0.045, 0.09]) * p.scale).clamp(0.008, 0.15);
        a.lap_variation = p.mix([0.015, 0.025, 0.04, 0.14]);
        a.repair_mix = p.variation.powi(3) * 0.12;
        retain_variation!(
            a,
            original.application.as_ref().unwrap(),
            spray_spacing_m,
            roller_spacing_m,
            roller_stretch,
            orange_peel,
            orange_peel_m,
            trowel_scale_m,
            lap_variation,
            repair_mix
        );
    }
    retain_variation!(
        c,
        &original,
        texture_mix,
        knockdown,
        roller,
        trowel,
        gloss,
        pinholes,
        pigment_variation
    );
    // Paint sheen is separately specified from application texture in practice.
    c.gloss = original.gloss;
    r.period_m = p.mix([0.20, 0.26, 0.42, 0.65]) * p.scale;
    r.relief_m = super::super::coating::plaster_relief(c.texture_mix);
    r.roughness = 0.95 - c.gloss * 0.72;
    if let Some(w) = &mut c.wall {
        if r.surface == Surface::Accent {
            w.weights = p.weights;
            w.fiber_cells = [
                p.count([28., 52., 72., 36.], [4, 96]),
                p.count([24., 10., 65., 48.], [4, 96]),
            ];
            w.bands = p.count([3., 5., 7., 12.], [1, 16]);
            w.band_duty = 0.05 + p.variation * 0.60;
            w.pigment = p.mix([0.22, 0.48, 0.36, 0.70]);
            w.relief_m = p.mix([0.00005, 0.00030, 0.00042, 0.000025]);
            w.roughness_variation = p.mix([0.12, 0.14, 0.18, 0.05]);
            w.burnish = p.weights[0] * p.variation * 0.28;
            r.period_m = 0.95 * p.scale;
        }
        r.relief_m *= 0.20;
        c.texture_mix *= 0.25;
        c.roller *= 0.35;
    }
    if r.surface == Surface::Ceiling {
        // Large overhead fields remain plausible paint/plaster, not heavy stucco.
        r.relief_m *= 0.45;
        c.pigment_variation *= 0.5;
    }
}
