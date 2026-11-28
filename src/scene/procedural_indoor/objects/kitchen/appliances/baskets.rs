//! Dishwasher inserts drain through real openings. Dishes and utensils rest
//! on the same stock that is rendered into the optical and geometry passes.
use super::*;
use interior::Basket;
use rand::seq::SliceRandom;
use supplies::rack::{RackPlate, RackProgram};

pub(super) const INTRODUCED: u32 = 323;
const HALF_WIDTH: f32 = 0.47;
const HALF_DEPTH: f32 = 0.43;

#[derive(Serialize)]
struct Utensil {
    lane: usize,
    size: Vec3,
    position: Vec3,
    yaw: f32,
    seed: u64,
}

pub(super) struct WashBasket {
    rack: RackProgram,
    radius: f32,
    cross_z: Vec<f32>,
    rim_height: f32,
    tine_splay: f32,
    cutlery: bool,
    utensils: Vec<Utensil>,
}

pub(super) fn slots(basket: &Basket, stocked: bool) -> RackProgram {
    supplies::rack::parameters(
        basket.size,
        if stocked && !basket.cutlery { 6 } else { 0 },
        basket.seed,
    )
}

impl WashBasket {
    pub(super) fn sample(mut rack: RackProgram, basket: &Basket, stocked: bool) -> Self {
        let mut r = stream(basket.seed, u64::from_le_bytes(*b"WashGrid"));
        let size = rack.size;
        let radius = if basket.cutlery {
            rack.wire.min(0.0024)
        } else {
            rack.wire.min(rack.tray * 0.25)
        };
        let depth = size.z * HALF_DEPTH;
        let intervals = ((2. * depth / r.random_range(0.040..0.063)).ceil() as usize).clamp(8, 14);
        let cross_z: Vec<_> = (0..=intervals)
            .map(|i| depth * (2. * i as f32 / intervals as f32 - 1.))
            .collect();
        let crown = 4. * radius;
        for plate in &mut rack.plates {
            seat_plate(plate, &cross_z, crown);
        }
        let rim_height = if basket.cutlery {
            size.y * r.random_range(0.30..0.49)
        } else {
            rack.frame_height * r.random_range(0.53..0.84)
        };
        let tine_splay =
            (r.random_range(0.028..0.085) * size.z).min(0.82 * (cross_z[1] - cross_z[0]));
        let mut utensils = Vec::new();
        if basket.cutlery && stocked {
            let mut lanes: Vec<_> = (0..rack.rail_x.len() - 1).collect();
            lanes.shuffle(&mut r);
            let count = r.random_range(0..=lanes.len().min(6));
            for lane in lanes.into_iter().take(count) {
                let left = rack.rail_x[lane];
                let pitch = rack.rail_x[lane + 1] - left;
                let length = r.random_range(0.145..0.210_f32).min(size.z * 0.70);
                let width = r.random_range(0.026..0.040_f32).min(pitch * 0.58);
                let play = (spine_width(pitch) * 0.31 / length).min(0.033);
                let yaw = r.random_range(-play..play) + if r.random_bool(0.5) { PI } else { 0. };
                let travel = size.z * 0.39 - length * 0.5;
                utensils.push(Utensil {
                    lane,
                    size: Vec3::new(width, r.random_range(0.006..0.011), length),
                    position: Vec3::new(left + pitch * 0.5, crown, r.random_range(-travel..travel)),
                    yaw,
                    seed: r.random(),
                });
            }
        }
        Self {
            rack,
            radius,
            cross_z,
            rim_height,
            tine_splay,
            cutlery: basket.cutlery,
            utensils,
        }
    }

    pub(super) fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "wire_radius_m": self.radius, "rim_height_m": self.rim_height,
            "cross_wires_z_m": self.cross_z, "lane_boundaries_x_m": self.rack.rail_x,
            "plate_slots": self.rack.plates.iter().map(|p| p.slot).collect::<Vec<_>>(),
            "flatware": self.utensils
        })
    }

    fn wirework(&self, g: &mut Geometry, origin: Vec3) {
        let size = self.rack.size;
        let r = self.radius;
        let x = size.x * HALF_WIDTH;
        let z = size.z * HALF_DEPTH;
        let at = |x, y, z| origin + Vec3::new(x, y, z);
        let bed = if self.cutlery { r } else { 3. * r };
        for &v in &self.cross_z {
            g.planar_tube(&[at(-x, bed, v), at(x, bed, v)], r, 8, Vec3::Y);
        }
        let rim = self.rim_height;
        g.planar_tube(
            &[
                at(-x, rim, -z),
                at(x, rim, -z),
                at(x, rim, z),
                at(-x, rim, z),
                at(-x, rim, -z),
            ],
            r,
            8,
            Vec3::Y,
        );
        for side in [-x, x] {
            for end in [-z, z] {
                g.planar_tube(&[at(side, bed, end), at(side, rim, end)], r, 8, Vec3::X);
            }
        }
        if self.cutlery {
            // Each stamped runner carries the handle and eating end along one
            // lane. Cross wires underneath join the runners to the slide rails.
            for lane in self.rack.rail_x.windows(2) {
                g.cuboid(
                    Vec3::new(spine_width(lane[1] - lane[0]), 2. * r, 2. * z),
                    r * 0.24,
                    Transform::from_translation(at((lane[0] + lane[1]) * 0.5, 3. * r, 0.)),
                );
            }
        } else {
            for fraction in [-0.43, -0.215, 0., 0.215, 0.43] {
                let px = size.x * fraction;
                g.planar_tube(&[at(px, r, -z), at(px, r, z)], r, 8, Vec3::Y);
            }
        }
        let a = self.cross_z[1];
        let b = self.cross_z[self.cross_z.len() - 2];
        let tip = if self.cutlery {
            rim * 0.82
        } else {
            self.rack.frame_height
        };
        for &px in &self.rack.rail_x {
            g.planar_tube(
                &[
                    at(px, tip, a - self.tine_splay),
                    at(px, bed, a),
                    at(px, bed, b),
                    at(px, tip, b + self.tine_splay),
                ],
                r,
                8,
                Vec3::X,
            );
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, basket: &Basket, version: u32) {
        let stock = o.finish_seed.unwrap_or(o.seed);
        let coated = stream(stock, u64::from_le_bytes(*b"RackWire")).random_bool(0.62);
        let surface = if coated {
            Surface::Plastic
        } else {
            Surface::Metal
        };
        let slot =
            crate::scene::procedural_indoor::materials::variants::object_slot(surface, stock, None);
        self.wirework(
            a.part(surface, &format!("dishwasher#finish{slot}")),
            basket.position,
        );
        let tf = Transform::from_translation(basket.position);
        for plate in &self.rack.plates {
            dishware::build(
                a,
                tf.mul_transform(plate.transform()),
                &plate.dish,
                plate.finish_seed,
            );
        }
        for utensil in &self.utensils {
            let item = IndoorObject {
                kind: ObjectKind::Flatware,
                size: utensil.size,
                seed: utensil.seed,
                finish_seed: Some(basket.seed),
                ..o.clone()
            };
            let mut p = flatware::parameters_at(&item, version);
            // A loaded machine samples dishwasher-safe bare or moulded handles.
            if p.fitted_grip
                .as_ref()
                .is_some_and(|v| v.material == Surface::Wood)
            {
                p.fitted_grip = None;
            }
            let mut child = Assembly::default();
            flatware::build_sampled(&mut child, &item, &p);
            append_at(
                a,
                child,
                Transform::from_translation(basket.position + utensil.position)
                    .with_rotation(Quat::from_rotation_y(utensil.yaw)),
            );
        }
    }
}

fn spine_width(pitch: f32) -> f32 {
    (pitch * 0.23).min(0.012)
}

/// The slot sampler fits plates inside ±0.37 depths. Snapping their lowest
/// mesh vertex to one of at least eight intervals moves them by at most
/// 0.054 depths, inside the ±0.43-depth rim. Every sampled plate is retained.
fn seat_plate(plate: &mut RackPlate, wires: &[f32], crown: f32) {
    let rotation = plate.rotation();
    let outline = plate.dish.outline();
    let section = plate.dish.profile();
    let vertical = rotation.inverse() * Vec3::Y;
    let (edge, _, _) = outline
        .meridians()
        .min_by(|a, b| a.0.dot(vertical.xz()).total_cmp(&b.0.dot(vertical.xz())))
        .unwrap();
    let radial = edge.dot(vertical.xz());
    let (radius, y) = section
        .iter()
        .copied()
        .min_by(|(ra, ya), (rb, yb)| {
            (ra * radial + ya * vertical.y).total_cmp(&(rb * radial + yb * vertical.y))
        })
        .unwrap();
    let contact = rotation * Vec3::new(edge.x * radius, y, edge.y * radius);
    let target = plate.position.z + contact.z;
    let pitch = wires[1] - wires[0];
    let nearest = ((target - wires[0]) / pitch).round() as usize;
    plate.position.z = wires[nearest.min(wires.len() - 1)] - contact.z;
    plate.position.y = crown - contact.y;
}

#[cfg(test)]
mod tests;
