//! A draining tray with real slots. Plate poses fit between the wire uprights;
//! the same program is used by loose racks, sinks and dishwasher baskets.
use super::*;
use rand::seq::SliceRandom;

#[derive(Clone, Debug, Serialize)]
pub struct RackPlate {
    pub slot: usize,
    pub position: Vec3,
    pub yaw: f32,
    pub lean: f32,
    pub dish: dishware::DishProgram,
    pub finish_seed: u64,
}
impl RackPlate {
    pub(crate) fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_z(FRAC_PI_2 + self.lean)
    }
    pub(crate) fn transform(&self) -> Transform {
        Transform::from_translation(self.position).with_rotation(self.rotation())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct RackProgram {
    pub size: Vec3,
    pub tray: f32,
    pub wire: f32,
    pub frame_height: f32,
    pub rail_x: Vec<f32>,
    pub finish: Surface,
    pub hardware: Surface,
    pub plates: Vec<RackPlate>,
}
pub fn parameters(size: Vec3, maximum_plates: u32, seed: u64) -> RackProgram {
    let mut r = stream(seed, 0x5241434b534c4f54);
    let size = size * Vec3::new(r.random_range(0.91..1.0), 1., r.random_range(0.89..1.0));
    let tray = size.y * r.random_range(0.035..0.065);
    let wire = (size.x.min(size.z) * r.random_range(0.009..0.014)).min(0.0035);
    let span = size.x * r.random_range(0.83..0.92);
    let slots = ((span / r.random_range(0.035..0.065)) as usize).clamp(2, 10);
    let pitch = span / slots as f32;
    let rail_x: Vec<_> = (0..=slots)
        .map(|i| -span * 0.5 + i as f32 * pitch)
        .collect();
    let mut order: Vec<_> = (0..slots).collect();
    order.shuffle(&mut r);
    let count = if maximum_plates == 0 {
        0
    } else {
        r.random_range(1..=maximum_plates.min(slots as u32))
    };
    let mut plates = Vec::new();
    for slot in order.into_iter().take(count as usize) {
        let max_radius = ((size.y - tray) * 0.5).min(size.z * 0.36);
        let diameter = 2. * max_radius * r.random_range(0.72..0.98);
        let dish_seed = r.random();
        let mut dish = dishware::parameters(
            Vec3::new(diameter / 0.86, diameter * 0.16, diameter / 0.86),
            0,
            1,
            dish_seed,
        );
        // Even tessellation puts the supporting radial extremum at a vertex.
        dish.segments = 32;
        let points = profile_points(&dish);
        for _ in 0..12 {
            let yaw = r.random_range(-0.09..0.09);
            let lean = r.random_range(-0.13..0.13);
            let rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_z(FRAC_PI_2 + lean);
            let (lo, hi) = rotated_bounds(&points, rotation);
            let left = rail_x[slot] + wire + 0.001 - lo.x;
            let right = rail_x[slot + 1] - wire - 0.001 - hi.x;
            if left >= right || hi.z - lo.z > size.z * 0.73 {
                continue;
            }
            let z_limit = size.z * 0.37 - lo.z.abs().max(hi.z.abs());
            let position = Vec3::new(
                r.random_range(left..right),
                tray - lo.y,
                r.random_range(-1.0..1.0) * z_limit,
            );
            if position.y + hi.y > size.y {
                continue;
            }
            plates.push(RackPlate {
                slot,
                position,
                yaw,
                lean,
                dish,
                finish_seed: dish_seed,
            });
            break;
        }
    }
    RackProgram {
        size,
        tray,
        wire,
        frame_height: size.y * r.random_range(0.30..0.54),
        rail_x,
        plates,
        finish: if r.random_bool(0.76) {
            Surface::Plastic
        } else {
            Surface::Metal
        },
        hardware: if r.random_bool(0.6) {
            Surface::Metal
        } else {
            Surface::Chrome
        },
    }
}
/// Bounds of the emitted polygonal profile, not a proxy disk. Used for slot
/// admission and exact tray contact (also for oval plates).
fn profile_points(p: &dishware::DishProgram) -> Vec<Vec3> {
    let angles: Vec<_> = (0..p.segments)
        .map(|i| {
            let t = TAU * i as f32 / p.segments as f32;
            Vec2::new(t.cos() * p.aspect.x, t.sin() * p.aspect.y)
        })
        .collect();
    p.profile()
        .into_iter()
        .flat_map(|(radius, y)| {
            angles
                .iter()
                .map(move |v| Vec3::new(radius * v.x, y, radius * v.y))
        })
        .collect()
}
fn rotated_bounds(points: &[Vec3], rotation: Quat) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &v in points {
        let v = rotation * v;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}
pub(crate) fn build(a: &mut Assembly, tf: Transform, p: &RackProgram, seed: u64) {
    let finish = |s| {
        format!(
            "dishware#finish{}",
            super::super::super::super::materials::variants::object_slot(s, seed, None)
        )
    };
    a.part(p.finish, &finish(p.finish)).cuboid(
        Vec3::new(p.size.x, p.tray, p.size.z),
        p.tray * 0.20,
        tf.mul_transform(Transform::from_xyz(0., p.tray * 0.5, 0.)),
    );
    for &x in &p.rail_x {
        let points = [
            Vec3::new(x, p.tray, -p.size.z * 0.41),
            Vec3::new(x, p.frame_height, -p.size.z * 0.30),
            Vec3::new(x, p.frame_height, p.size.z * 0.30),
            Vec3::new(x, p.tray, p.size.z * 0.41),
        ]
        .map(|v| tf.transform_point(v));
        a.part(p.hardware, &finish(p.hardware))
            .tube(&points, p.wire, 10);
    }
    for side in [-1., 1.] {
        let points = [
            Vec3::new(-p.size.x * 0.47, p.tray + p.wire, side * p.size.z * 0.41),
            Vec3::new(p.size.x * 0.47, p.tray + p.wire, side * p.size.z * 0.41),
        ]
        .map(|v| tf.transform_point(v));
        a.part(p.hardware, &finish(p.hardware))
            .tube(&points, p.wire, 10);
    }
    for plate in &p.plates {
        dishware::build(
            a,
            tf.mul_transform(plate.transform()),
            &plate.dish,
            plate.finish_seed,
        );
    }
}
