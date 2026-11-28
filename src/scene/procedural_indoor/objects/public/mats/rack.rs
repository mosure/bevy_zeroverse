//! Open storage rack with horizontally wound mats. The annular ends keep a
//! real through-bore; the front view, normals and range annotations agree.
use super::*;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MatRoll {
    pub center: Vec3,
    pub length: f32,
    pub radius: f32,
    pub bore: f32,
    pub sheet_m: f32,
    pub recess_m: f32,
    pub stock: Stock,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatRack {
    pub frame: Surface,
    pub tiers: usize,
    pub slots: usize,
    pub shelf_thickness: f32,
    pub rolls: Vec<MatRoll>,
}

const POST: f32 = 0.034;
const PLINTH: f32 = 0.082;

impl MatRack {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"MatShelf"));
        let s = o.size;
        let tiers = ((s.y - PLINTH) / r.random_range(0.26..0.39))
            .floor()
            .clamp(2., 4.) as usize;
        let slots = ((s.x - 2. * POST) / r.random_range(0.195..0.26))
            .floor()
            .clamp(2., 7.) as usize;
        let dx = (s.x - 2. * POST) / slots as f32;
        let dy = (s.y - PLINTH) / tiers as f32;
        let shelf_thickness = r.random_range(0.016..0.027);
        let fill = r.random_range(0.38..0.94);
        let kept = r.random_range(0..tiers * slots);
        let mut rolls = Vec::new();
        for row in 0..tiers {
            for slot in 0..slots {
                if row * slots + slot != kept && !r.random_bool(fill) {
                    continue;
                }
                // A 1.6–2.1 m sheet wound around a hollow core.
                let bore = r.random_range(0.013..0.029);
                let thickness = r.random_range(0.0038..0.010);
                let radius = (bore * bore
                    + thickness * r.random_range(1.61_f32..2.14) / std::f32::consts::PI)
                    .sqrt()
                    .min(dx * 0.45)
                    .min((dy - shelf_thickness) * 0.47);
                let length = (s.z - 0.030).min(r.random_range(0.59..0.77));
                let z = r.random_range(-0.5..0.5) * (s.z - length - 0.012);
                let x = (slot as f32 + 0.5 - slots as f32 * 0.5) * dx
                    + r.random_range(-0.35..0.35) * (dx - 2. * radius);
                rolls.push(MatRoll {
                    center: Vec3::new(x, PLINTH + row as f32 * dy + radius, z),
                    length,
                    radius,
                    bore,
                    sheet_m: thickness,
                    recess_m: r.random_range(0.003..0.019),
                    stock: Stock::rubber(r.random_range(0..3)),
                });
            }
        }
        Self {
            frame: if r.random_bool(0.64) {
                Surface::Wood
            } else {
                Surface::Metal
            },
            tiers,
            slots,
            shelf_thickness,
            rolls,
        }
    }
}

impl MatRack {
    fn frame_parts(&self, s: Vec3, mut emit: impl FnMut(Vec3, Vec3, f32)) {
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                emit(
                    Vec3::new(
                        side * (s.x - POST) * 0.5,
                        s.y * 0.5,
                        end * (s.z - POST) * 0.5,
                    ),
                    Vec3::new(POST, s.y, POST),
                    0.003,
                );
            }
            // Side rails tie the front and rear posts; mats cannot roll off either side.
            for row in 0..self.tiers {
                let y = PLINTH + row as f32 * (s.y - PLINTH) / self.tiers as f32 + 0.065;
                emit(
                    Vec3::new(side * (s.x - POST) * 0.5, y, 0.),
                    Vec3::new(POST * 0.66, 0.064, s.z - 2. * POST),
                    0.002,
                );
            }
        }
        for row in 0..self.tiers {
            let top = PLINTH + row as f32 * (s.y - PLINTH) / self.tiers as f32;
            emit(
                Vec3::new(0., top - self.shelf_thickness * 0.5, 0.),
                Vec3::new(s.x - 2. * POST, self.shelf_thickness, s.z - 0.008),
                0.002,
            );
        }
    }

    pub(crate) fn view_boxes(&self, size: Vec3, mut emit: impl FnMut(Vec3, Vec3)) {
        self.frame_parts(size, |center, size, _| emit(center, size * 0.5));
        for roll in &self.rolls {
            emit(
                roll.center,
                Vec3::new(roll.radius, roll.radius, roll.length * 0.5),
            );
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let rack = MatRack::for_object(o);
    rack.frame_parts(o.size, |center, size, bevel| {
        a.box_part(rack.frame, o.kind.class_name(), center, size, bevel)
    });
    for roll in rack.rolls {
        roll.build(a);
    }
}

impl MatRoll {
    fn build(self, a: &mut Assembly) {
        let (surface, stock) = self.stock.key();
        let body = a.part(surface, stock);
        let end = self.length * 0.5;
        let r = self.radius;
        let tf = Transform::from_translation(self.center)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2));
        body.lathe(
            &[
                (self.bore, -end),
                (r - 0.0015, -end),
                (r, -end + 0.0015),
                (r, end),
            ],
            20,
            tf,
        );
        body.lathe(
            &[(self.bore, end - self.recess_m), (self.bore, -end)],
            20,
            tf,
        );

        let cut = a.part(surface, self.stock.edge_label());
        let start = cut.positions.len();
        cut.lathe(&[(r, end), (self.bore, end - self.recess_m)], 20, tf);
        let diameter =
            2. * crate::scene::procedural_indoor::materials::mat_stock::CHART_TURNS * self.sheet_m;
        for (uv, p) in cut.uvs[start..].iter_mut().zip(&cut.positions[start..]) {
            *uv = ((Vec2::new(p[0], p[1]) - self.center.xy()) / diameter + 0.5).to_array();
        }
    }
}
