//! Bounded burner packing and an independent control strip. The admitted plan
//! drives gas/induction geometry and exported metadata, including empty space.
use super::*;
mod gas;
mod glass;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Burner {
    pub center: Vec2,
    pub radius_m: f32,
    pub grate_radius_m: f32,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GasGrate {
    pub fingers: u32,
    pub frame_segments: u32,
    pub height_fraction: f32,
    pub tip_fraction: f32,
    pub width_fraction: f32,
    pub phase_radians: f32,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct CooktopProgram {
    pub burners: Vec<Burner>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gas_grate: Option<GasGrate>,
    pub controls: Vec<Vec2>,
    pub knob_radius_m: f32,
    pub pointer_angles: Vec<f32>,
}

pub fn parameters(seed: u64, size: Vec3, p: &KitchenProgram) -> CooktopProgram {
    let mut r = stream(seed, 0x4255524e455253);
    let scale = size.x.min(size.z);
    let knob_radius = scale * r.random_range(0.020..0.032);
    let border = scale * 0.045;
    let mut lo = -size.xz() * 0.5 + Vec2::splat(border);
    let mut hi = size.xz() * 0.5 - Vec2::splat(border);
    // Reserve the real knob strip in metric units, including wide appliances
    // whose control-axis extent is larger than their short dimension.
    match p.controls {
        0 => hi.y = hi.y.min(size.z * 0.435 - knob_radius - 0.012),
        1 => lo.y = lo.y.max(-size.z * 0.435 + knob_radius + 0.012),
        _ => hi.x = hi.x.min(size.x * 0.435 - knob_radius - 0.012),
    }
    let count = r.random_range(2..=5);
    let mut burners: Vec<Burner> = Vec::with_capacity(count);
    let phase = r.random_range(0.0..1.);
    // Stratification gives usable stove layouts without identical metric pitch,
    // fixed four-burner counts or radius being determined by burner index.
    for i in 0..64 {
        if burners.len() == count {
            break;
        }
        let radius = scale * r.random_range(0.075..0.145);
        let reserve = radius * if p.gas { 1.30 } else { 1. } + 0.008;
        let a = lo + Vec2::splat(reserve);
        let b = hi - Vec2::splat(reserve);
        if a.cmpge(b).any() {
            continue;
        }
        let q = if i < 4 {
            let cell = (i + (phase * 4.) as usize) % 4;
            Vec2::new((cell % 2) as f32, (cell / 2) as f32) * 0.65
                + Vec2::new(r.random_range(0.0..0.35), r.random_range(0.0..0.35))
        } else {
            Vec2::new(r.random(), r.random())
        };
        let center = a + (b - a) * q;
        if burners
            .iter()
            .any(|old| center.distance(old.center) < reserve + old.grate_radius_m + 0.012)
        {
            continue;
        }
        burners.push(Burner {
            center,
            radius_m: radius,
            grate_radius_m: reserve,
        });
    }
    let n = burners.len();
    let span = r.random_range(0.50..0.77);
    let phase = r.random_range(-0.06..0.06);
    let controls = (0..n)
        .map(|i| {
            let t = ((i as f32 + 0.5) / n as f32 - 0.5) * span + phase;
            match p.controls {
                0 => Vec2::new(size.x * t, size.z * 0.435),
                1 => Vec2::new(size.x * t, -size.z * 0.435),
                _ => Vec2::new(size.x * 0.435, size.z * t),
            }
        })
        .collect();
    let gas_grate = p.gas.then(|| {
        let mut stock = stream(seed, 0x4341_5354_4952_4f4e);
        let fingers = stock.random_range(3..=5);
        GasGrate {
            fingers,
            height_fraction: stock.random_range(0.40..0.60),
            tip_fraction: stock.random_range(0.54..0.68),
            width_fraction: stock.random_range(0.09..0.16),
            phase_radians: stock.random_range(-0.60..0.60),
            // Both the faceted and rounded castings have a vertex at every foot.
            frame_segments: fingers * if stock.random_bool(0.52) { 2 } else { 8 },
        }
    });
    CooktopProgram {
        burners,
        gas_grate,
        controls,
        knob_radius_m: knob_radius,
        pointer_angles: (0..n).map(|_| r.random_range(-2.6..2.6)).collect(),
    }
}

pub(super) fn build(a: &mut Assembly, s: Vec3, base: Vec3, p: &KitchenProgram, seed: u64) {
    let plan = parameters(seed, s, p);
    if let Some(grate) = &plan.gas_grate {
        gas::build(a, s, base, &plan, grate);
    } else {
        glass::build(a, s, base, &plan.burners);
    }
    for (&c, &angle) in plan.controls.iter().zip(&plan.pointer_angles) {
        let center = base + Vec3::new(c.x, s.y * 0.40, c.y);
        a.part(
            if p.hardware == Surface::WoodEdge {
                // Cabinet pulls may be timber. The control on the hot deck is metal.
                Surface::Metal
            } else {
                p.hardware
            },
            "stove",
        )
        .cylinder(
            plan.knob_radius_m,
            s.y * 0.32,
            Transform::from_translation(center),
        );
        let rotation = Quat::from_rotation_y(angle);
        a.part(Surface::Ink, "stove").cuboid(
            Vec3::new(plan.knob_radius_m * 0.16, 0.0007, plan.knob_radius_m * 0.55),
            0.,
            Transform::from_translation(
                center
                    + Vec3::Y * (s.y * 0.16 + 0.00035)
                    + rotation * Vec3::Z * (plan.knob_radius_m * 0.47),
            )
            .with_rotation(rotation),
        );
    }
}
