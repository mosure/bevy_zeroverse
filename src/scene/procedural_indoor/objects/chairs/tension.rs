//! Upholstery stretched over a shaped perimeter. The sampled construction owns
//! its actual coverage in RGB, ray annotations and articulated chair geometry.
use super::*;

mod cloth;
mod cords;
#[cfg(test)]
mod tests;

const TEXTILE_VERSION: u32 = 206;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WebbingDirection {
    Horizontal,
    Vertical,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensionInfill {
    OpenCord,
    WovenSling {
        gauge_m: f32,
        ease_m: f32,
    },
    Webbing {
        direction: WebbingDirection,
        nominal_pitch_m: f32,
        fill_fraction: f32,
        gauge_m: f32,
        ease_m: f32,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize)]
pub struct TensionBack {
    #[serde(flatten)]
    pub infill: TensionInfill,
    pub table_order: bool,
    pub rail_radius_m: f32,
    pub rail_surface: Surface,
    pub cover_surface: Surface,
}

impl ChairProgram {
    /// Sample the cut and rail once, using a common order for coordinated
    /// seating and an instance seed for independently purchased replacements.
    pub fn tension_back(&self, o: &IndoorObject, version: u32) -> Option<TensionBack> {
        use TensionInfill::*;
        if self.back_construction != 0
            || o.kind != ObjectKind::Chair
            || !matches!(o.variant, 0..=5 | 9)
        {
            return None;
        }
        let cord = TensionBack {
            infill: OpenCord,
            table_order: false,
            rail_radius_m: 0.012,
            rail_surface: Surface::Plastic,
            cover_surface: self.back_surface,
        };
        if version < TEXTILE_VERSION {
            return Some(cord);
        }
        let order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| stream(o.seed, 0x534c_494e_474f_5244).random_bool(0.78));
        let product = order.map_or(o.seed, |(lot, table)| {
            lot ^ (table as u64).rotate_left(17) ^ u64::from(o.variant).rotate_left(49)
        });
        let mut r = stream(product, 0x534c_494e_4742_414b);
        let draw = r.random::<f32>();
        let infill = if draw < 0.54 {
            WovenSling {
                gauge_m: r.random_range(0.0020..0.0048),
                ease_m: r.random_range(0.003..0.014),
            }
        } else if draw < 0.90 {
            Webbing {
                direction: if r.random_bool(0.67) {
                    WebbingDirection::Horizontal
                } else {
                    WebbingDirection::Vertical
                },
                nominal_pitch_m: r.random_range(0.052..0.106),
                fill_fraction: r.random_range(0.48..0.85),
                gauge_m: r.random_range(0.0018..0.0042),
                ease_m: r.random_range(0.002..0.012),
            }
        } else {
            return Some(TensionBack {
                table_order: order.is_some(),
                ..cord
            });
        };
        let mut rail = stream(product, 0x534c_494e_4752_494d);
        Some(TensionBack {
            infill,
            table_order: order.is_some(),
            rail_radius_m: rail.random_range(0.0085..0.012),
            rail_surface: if rail.random_bool(0.42) {
                self.frame_surface
            } else {
                Surface::Plastic
            },
            // Use the seat's woven upholstery where available. Timber, leather
            // and moulded seats can be paired with a contrasting textile back.
            cover_surface: match (self.seat_surface, self.back_surface) {
                (s @ (Surface::Fabric | Surface::FabricAlt), _) => s,
                (_, Surface::Fabric) => Surface::Fabric,
                _ => Surface::FabricAlt,
            },
        })
    }
}

struct Frame {
    width: f32,
    height: f32,
    curvature: f32,
    lumbar: f32,
    roundness: f32,
    transform: Transform,
}

impl Frame {
    fn local(&self, x: f32, v: f32) -> Vec3 {
        Vec3::new(
            x * self.width * 0.5,
            v * self.height,
            self.curvature * (1.0 - x * x) + v * self.height * 0.13 - self.lumbar * (PI * v).sin(),
        )
    }

    fn point(&self, x: f32, v: f32) -> Vec3 {
        self.transform.transform_point(self.local(x, v))
    }

    /// Half-chord of the frame's superellipse at a given signed ordinate.
    fn chord(&self, ordinate: f32) -> f32 {
        (1. - ordinate.abs().powf(self.roundness))
            .max(0.)
            .powf(1. / self.roundness)
    }

    fn rim(&self) -> [Vec3; 65] {
        let shape = |v: f32| {
            if v.abs() < 1e-6 {
                0.
            } else {
                v.signum() * v.abs().powf(2. / self.roundness)
            }
        };
        std::array::from_fn(|i| {
            let t = TAU * (i % 64) as f32 / 64.;
            self.point(shape(t.cos()), 0.5 + shape(t.sin()) * 0.5)
        })
    }
}

pub(super) fn build(
    a: &mut Assembly,
    p: &ChairProgram,
    size: Vec2,
    transform: Transform,
    stock: TensionBack,
) -> [Vec3; 2] {
    let frame = Frame {
        width: size.x,
        height: size.y,
        transform,
        curvature: p.curvature,
        lumbar: p.lumbar,
        roundness: p.mesh_roundness,
    };
    let rim = frame.rim();
    a.part(stock.rail_surface, "chair")
        .tube(&rim, stock.rail_radius_m, 8);
    let cover = a.part(stock.cover_surface, "chair");
    match stock.infill {
        TensionInfill::OpenCord => cords::build(cover, &frame, p.mesh_pitch_m),
        TensionInfill::WovenSling { gauge_m, ease_m } => {
            cloth::sling(cover, &frame, gauge_m, ease_m)
        }
        TensionInfill::Webbing {
            direction,
            nominal_pitch_m,
            fill_fraction,
            gauge_m,
            ease_m,
        } => cloth::webbing(
            cover,
            &frame,
            direction,
            nominal_pitch_m,
            fill_fraction,
            gauge_m,
            ease_m,
        ),
    }
    [rim[44], rim[52]]
}
