//! Resolve garment gravity, support and neighbour clearance in rack-local metres.
use super::*;
use crate::scene::procedural_indoor::materials::garment_stock;
use rand_chacha::ChaCha8Rng;

pub const INTRODUCED: u32 = 220;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum GarmentKind {
    Scarf,
    Jacket,
    Overcoat,
    Shirt,
    Dress,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Garment {
    pub kind: GarmentKind,
    pub support_index: usize,
    /// Centre and direction of the physical hanger or peg carrying the cloth.
    pub anchor_m: Vec3,
    pub support_axis: Vec3,
    pub support_radius_m: f32,
    pub shoulder_width_m: f32,
    pub width_m: f32,
    pub length_m: f32,
    pub half_depth_m: f32,
    pub tail_fraction: f32,
    pub fold_phase: f32,
    pub fold_count: u8,
    pub closure: u8,
    pub surface: Surface,
    pub finish_slot: usize,
}

impl Garment {
    /// Conservative mesh costs for the sewn cut, edge turns and fasteners.
    /// The object regression test also checks the complete rack budget.
    pub(super) fn triangle_budget(&self) -> usize {
        match self.kind {
            GarmentKind::Scarf => 832,
            _ if self.closure == 0 => 1600,
            _ => 1800,
        }
    }

    pub(super) fn footprint(&self) -> (Vec2, Vec2) {
        let half = if self.kind == GarmentKind::Scarf {
            let side = self.support_axis.cross(Vec3::Y).normalize();
            self.support_axis.xz().abs() * (self.width_m * 0.5)
                + side.xz().abs() * (self.support_radius_m + self.half_depth_m + 0.006)
                + Vec2::splat(self.support_radius_m + 0.002)
        } else {
            Vec2::new(self.half_depth_m + 0.009, self.width_m * 0.5)
        };
        (self.anchor_m.xz() - half, self.anchor_m.xz() + half)
    }

    pub(in crate::scene::procedural_indoor::objects) fn build(&self, a: &mut Assembly) {
        let label = format!("clothes#finish{}", self.finish_slot);
        if self.kind == GarmentKind::Scarf {
            super::scarf::build(a.part(self.surface, &label), self);
        } else {
            super::outerwear::build(a, self, &label);
        }
    }
}

pub fn garments(o: &IndoorObject, p: &RackProgram) -> Vec<Garment> {
    let mut load = stream(o.seed, 0x5241_434b_4c4f_4144);
    if o.size.y < 1.0 || load.random_bool(0.18) {
        return Vec::new();
    }
    let probability = if p.rail {
        load.random_range(0.38..0.91)
    } else {
        // A stand may have seven hardware pegs. Filling most with the same
        // kind of scarf makes ordinary offices resemble a retail display.
        load.random_range(0.16..0.50)
    };
    let count = if p.rail {
        p.hanger_x.len()
    } else {
        p.hooks.len()
    };
    let mut out: Vec<Garment> = Vec::new();
    // Reserve the remainder of the established 10k rack budget for hardware.
    // Sample supports in construction order (their X or azimuth is random).
    let mut available_triangles = 6600;
    for i in 0..count {
        let mut r = stream(o.seed, 0x5241_434b_4954_454d ^ i as u64);
        if !r.random_bool(probability) {
            continue;
        }
        let garment = if p.rail {
            rail_item(o, p, i, &mut r)
        } else {
            peg_scarf(p, i, &mut r)
        };
        let Some(mut g) = garment else { continue };
        let (lo, hi) = g.footprint();
        let limit = o.size.xz() * 0.5 - 0.003;
        if g.triangle_budget() > available_triangles
            || lo.cmplt(-limit).any()
            || hi.cmpgt(limit).any()
            || out.iter().any(|other| {
                let (a, b) = other.footprint();
                lo.cmplt(b + 0.003).all() && hi.cmpgt(a - 0.003).all()
            })
        {
            continue;
        }
        // Individual belongings do not inherit the pole's paint colour or one
        // another's dye lots. Use only existing bounded material-cache slots.
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            0x434c_4f54_4846_494e ^ i as u64,
        );
        g.surface = if g.kind == GarmentKind::Jacket && finish.random_bool(0.19) {
            Surface::Leather
        } else if finish.random_bool(0.48) {
            Surface::FabricAlt
        } else {
            Surface::Fabric
        };
        g.finish_slot = garment_stock::sample_slot(&mut finish);
        available_triangles -= g.triangle_budget();
        out.push(g);
    }
    out
}

fn scarf(
    anchor: Vec3,
    axis: Vec3,
    radius: f32,
    crown_width: f32,
    available: f32,
    i: usize,
    r: &mut ChaCha8Rng,
) -> Option<Garment> {
    let wanted = r.random_range(0.43_f32..0.92);
    let width = (crown_width * r.random_range(1.22..1.90)).min(0.17);
    // On a tilted peg the lower selvage begins below the support centre.
    // Reserve that drop and the full bias hem before choosing the tail length.
    let length = wanted.min(available - axis.y.abs() * width * 0.5 - 0.008);
    if crown_width < 0.025 || length < 0.34 {
        return None;
    }
    Some(Garment {
        kind: GarmentKind::Scarf,
        support_index: i,
        anchor_m: anchor,
        support_axis: axis,
        support_radius_m: radius,
        shoulder_width_m: crown_width,
        width_m: width,
        length_m: length,
        half_depth_m: r.random_range(0.009..0.021),
        tail_fraction: r.random_range(0.48..0.92),
        fold_phase: r.random_range(-PI..PI),
        fold_count: r.random_range(2..=4),
        closure: 0,
        surface: Surface::Fabric,
        finish_slot: 0,
    })
}

fn peg_scarf(p: &RackProgram, i: usize, r: &mut ChaCha8Rng) -> Option<Garment> {
    let [_, elbow, tip] = p.hooks[i].path();
    let axis = tip - elbow;
    let t = r.random_range(0.38_f32..0.58);
    let anchor = elbow.lerp(tip, t);
    let width = (axis.length() * 2. * t.min(1. - t) - p.upright * 2.2).min(0.10);
    scarf(
        anchor,
        axis.normalize(),
        p.upright * 0.65,
        width,
        anchor.y - p.base_height - 0.06,
        i,
        r,
    )
}

fn rail_item(o: &IndoorObject, p: &RackProgram, i: usize, r: &mut ChaCha8Rng) -> Option<Garment> {
    let (w, wire, radius, neck) = p.hanger(o);
    let x = p.hanger_x[i];
    let clear = p
        .hanger_x
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, other)| (x - other).abs() * 0.5 - 0.005)
        .fold(p.rail_width * 0.5 - x.abs() - p.upright - 0.006, f32::min);
    let shoulder = w * 1.015;
    let sleeve_space = (o.size.z * 0.5 - radius - 0.005 - shoulder * 0.5).min(0.075);
    if shoulder >= 0.29 && clear > 0.026 && sleeve_space > 0.024 && r.random_bool(0.74) {
        let long = r.random_bool(0.36);
        let length = (if long {
            r.random_range(0.79_f32..1.12)
        } else {
            r.random_range(0.51_f32..0.78)
        })
        .min(neck - p.base_height - 0.06);
        return (length > 0.45).then(|| Garment {
            kind: if long {
                GarmentKind::Overcoat
            } else {
                GarmentKind::Jacket
            },
            support_index: i,
            anchor_m: Vec3::new(x, neck, -radius),
            support_axis: Vec3::Z,
            support_radius_m: wire,
            shoulder_width_m: shoulder,
            width_m: shoulder + 2. * sleeve_space * r.random_range(0.76..0.98),
            length_m: length,
            half_depth_m: r.random_range(0.029_f32..0.053).min(clear - 0.009),
            tail_fraction: r.random_range(0.58..0.88),
            fold_phase: r.random_range(-PI..PI),
            fold_count: r.random_range(2..=4),
            closure: if long { 1 } else { r.random_range(0..=1) },
            surface: Surface::Fabric,
            finish_slot: 0,
        });
    }
    // The lower bar of the actual bent-wire triangle, away from its corners.
    let anchor = Vec3::new(
        x,
        neck - w * 0.26,
        -radius + w * r.random_range(-0.12..0.12),
    );
    let g = scarf(
        anchor,
        Vec3::Z,
        wire,
        r.random_range(0.047..0.095_f32).min(w * 0.35),
        anchor.y - p.base_height - 0.07,
        i,
        r,
    )?;
    (g.half_depth_m + wire * 2. + 0.009 < clear).then_some(g)
}
