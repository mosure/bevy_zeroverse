//! Dimensioned washing station. Geometry, accessory admission and metrics share
//! these realized parameters; finish changes never move the bowl or tap.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};

#[derive(Clone, Debug, Serialize)]
pub struct Basin {
    pub center: Vec2,
    pub size: Vec2,
    pub floor_m: f32,
    pub rim_m: f32,
    pub wall_m: f32,
    pub taper: Vec2,
    pub roundness: f32,
    pub drain: Vec2,
    pub drain_radius: f32,
}
impl Basin {
    pub fn inner_half(&self) -> Vec2 {
        self.size * 0.5 - Vec2::splat(self.rim_m)
    }
    pub fn floor_half(&self) -> Vec2 {
        self.inner_half() * self.taper
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Accessory {
    pub kind: ObjectKind,
    pub position: Vec3,
    pub size: Vec3,
    pub yaw: f32,
    pub seed: u64,
    pub pump: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct BasinDish {
    pub position: Vec3,
    pub yaw: f32,
    pub seed: u64,
    pub dish: dishware::DishProgram,
}
#[derive(Clone, Debug, Serialize)]
pub struct SinkProgram {
    pub worktop_m: f32,
    pub aperture_min: Vec2,
    pub aperture_max: Vec2,
    pub basins: Vec<Basin>,
    pub basin_finish: Surface,
    pub faucet_finish: Surface,
    pub faucet_points: Vec<Vec3>,
    pub faucet_radius_m: f32,
    pub mount: Vec3,
    pub handles: Vec<Vec3>,
    pub accessories: Vec<Accessory>,
    pub dishes: Vec<BasinDish>,
}

pub fn parameters(o: &IndoorObject, p: &KitchenProgram) -> SinkProgram {
    let mut r = stream(o.seed, 0x53494e4b464954);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x53494e4b46494e);
    let s = o.size;
    let (d, z) = body(o, p);
    let top = s.y * 0.69;
    let width = s.x
        * if p.draining_board {
            r.random_range(0.49..0.64)
        } else {
            r.random_range(0.66..0.82)
        };
    let board_side = if r.random_bool(0.5) { 1. } else { -1. };
    let cx = if p.draining_board {
        -board_side * (s.x - width) * r.random_range(0.25..0.36)
    } else {
        (s.x - width) * r.random_range(-0.20..0.20)
    };
    let depth = d * r.random_range(0.46..0.64);
    let cz = z + d * r.random_range(0.02..0.10);
    let lo = Vec2::new(cx - width * 0.5, cz - depth * 0.5);
    let hi = Vec2::new(cx + width * 0.5, cz + depth * 0.5);
    let split = if p.basins == 2 {
        r.random_range(0.38..0.62)
    } else {
        1.
    };
    let mut basins = Vec::new();
    let mut x = lo.x;
    for i in 0..p.basins {
        let w = width * if i == 0 { split } else { 1. - split };
        let rim = r.random_range(0.012_f32..0.024).min(w * 0.10);
        let inner = Vec2::new(w, depth) * 0.5 - Vec2::splat(rim);
        let taper = Vec2::new(r.random_range(0.65..0.91), r.random_range(0.66..0.92));
        let center = Vec2::new(x + w * 0.5, cz);
        let floor_half = inner * taper;
        basins.push(Basin {
            center,
            size: Vec2::new(w, depth),
            floor_m: top - r.random_range(0.13..0.235),
            rim_m: rim,
            wall_m: r.random_range(0.004..0.009),
            taper,
            roundness: r.random_range(2.4..8.0),
            drain: center
                + floor_half * Vec2::new(r.random_range(-0.45..0.45), r.random_range(-0.44..0.44)),
            drain_radius: floor_half.min_element() * r.random_range(0.13..0.22),
        });
        x += w;
    }
    let target = &basins[r.random_range(0..basins.len())];
    let back = lo.y - (lo.y - (z - d * 0.5)) * r.random_range(0.43..0.62);
    let mount = Vec3::new(
        (target.center.x + r.random_range(-0.10..0.10) * target.size.x)
            .clamp(-s.x * 0.5 + 0.13, s.x * 0.5 - 0.13),
        top + 0.009,
        back,
    );
    let tip = Vec3::new(
        target.center.x + r.random_range(-0.12..0.12) * target.size.x,
        0.,
        target.center.y + r.random_range(-0.10..0.10) * depth,
    );
    let rise = (s.y - top) * r.random_range(0.52..0.88) * if p.faucet == 2 { 0.74 } else { 1. };
    let travel = Vec3::new(tip.x - mount.x, 0., tip.z - mount.z);
    let faucet_points = match p.faucet {
        1 | 2 => vec![
            mount,
            mount + Vec3::Y * rise,
            mount + travel + Vec3::Y * rise,
            mount + travel + Vec3::Y * rise * 0.45,
        ],
        _ => {
            // A semicircular gooseneck has bounded curvature even with a tall
            // stem and short reach. Stretching a sine arch did not preserve it.
            let radius = travel.length() * 0.5;
            let height = rise.max(radius + 0.075);
            let direction = travel.normalize();
            let mut points = vec![mount, mount + Vec3::Y * (height - radius)];
            for i in 1..=24 {
                let t = PI * i as f32 / 24.;
                points.push(
                    mount
                        + direction * radius * (1. - t.cos())
                        + Vec3::Y * (height - radius + radius * t.sin()),
                );
            }
            points.push(mount + travel + Vec3::Y * (height - radius - 0.025));
            points
        }
    };
    let radius = r.random_range(0.008..0.014);
    let spread = r.random_range(0.063..0.091);
    let handles = if p.faucet_handles == 1 {
        vec![mount - Vec3::X * spread, mount + Vec3::X * spread]
    } else {
        vec![mount + Vec3::X * spread * if r.random_bool(0.5) { 1. } else { -1. }]
    };
    let mut accessories = Vec::new();
    let aperture = Footprint {
        center: (lo + hi) * 0.5,
        half: (hi - lo) * 0.5 + Vec2::splat(0.008),
        yaw: 0.,
    };
    let mut occupied = vec![
        aperture,
        Footprint {
            center: mount.xz(),
            half: Vec2::new(spread + 0.048, 0.045),
            yaw: 0.,
        },
    ];
    let mut intents = vec![
        (ObjectKind::SoapDispenser, p.hand_soap, true),
        (ObjectKind::SoapDispenser, p.dish_soap, false),
        (ObjectKind::Sponge, p.sponge, false),
        (ObjectKind::DryingRack, p.draining_board, false),
    ];
    use rand::seq::SliceRandom;
    intents.shuffle(&mut r);
    for (kind, enabled, pump) in intents {
        if !enabled {
            continue;
        }
        let size = match kind {
            ObjectKind::Sponge => Vec3::new(
                r.random_range(0.075..0.12),
                r.random_range(0.018..0.032),
                r.random_range(0.045..0.075),
            ),
            ObjectKind::DryingRack => Vec3::new(
                s.x * r.random_range(0.16..0.23),
                (s.y - top) * r.random_range(0.35..0.62),
                depth * r.random_range(0.64..0.85),
            ),
            _ => Vec3::new(
                r.random_range(0.048..0.072),
                r.random_range(0.13..0.25),
                r.random_range(0.052..0.080),
            ),
        };
        let seed = r.random();
        for _ in 0..24 {
            let yaw = r.random_range(-PI..PI)
                * if kind == ObjectKind::DryingRack {
                    0.08
                } else {
                    1.
                };
            let half = rotated_half(size.xz(), yaw);
            let a = Vec2::new(-s.x * 0.5, z - d * 0.5) + half + Vec2::splat(0.012);
            let b = Vec2::new(s.x * 0.5, z + d * 0.5) - half - Vec2::splat(0.012);
            if a.cmpge(b).any() {
                continue;
            }
            let position = Vec2::new(r.random_range(a.x..b.x), r.random_range(a.y..b.y));
            let q = Footprint {
                center: position,
                half,
                yaw: 0.,
            };
            if occupied.iter().any(|&other| q.overlaps(other, 0.006)) {
                continue;
            }
            occupied.push(q);
            accessories.push(Accessory {
                kind,
                position: Vec3::new(position.x, top, position.y),
                size,
                yaw,
                seed,
                pump,
            });
            break;
        }
    }
    let mut dishes = Vec::new();
    if p.sink_dishes > 0 {
        for basin in &basins {
            let half = basin.floor_half();
            let seed = r.random();
            let diameter = half.min_element() * r.random_range(0.80..1.15);
            let dish = dishware::parameters(
                Vec3::new(diameter, (top - basin.floor_m) * 0.65, diameter),
                r.random_range(0..=4),
                p.sink_dishes,
                seed,
            );
            let yaw = r.random_range(-PI..PI);
            let extent = dish.half_extents(yaw);
            let free = half * 0.68 - extent;
            if free.min_element() <= 0. {
                continue;
            }
            let center = basin.center
                + free * Vec2::new(r.random_range(-0.9..0.9), r.random_range(-0.9..0.9));
            dishes.push(BasinDish {
                position: Vec3::new(center.x, basin.floor_m + 0.001, center.y),
                yaw,
                seed,
                dish,
            });
        }
    }
    SinkProgram {
        worktop_m: top,
        aperture_min: lo,
        aperture_max: hi,
        basins,
        basin_finish: [Surface::Metal, Surface::Ceramic, Surface::Concrete][f.random_range(0..3)],
        faucet_finish: [Surface::Chrome, Surface::Metal, Surface::Ink][f.random_range(0..3)],
        faucet_points,
        faucet_radius_m: radius,
        mount,
        handles,
        accessories,
        dishes,
    }
}
