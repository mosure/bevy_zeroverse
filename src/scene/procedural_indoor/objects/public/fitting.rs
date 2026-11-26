//! Freestanding changing cubicles. Local +Z faces the shop aisle. The two-bay
//! kit shares one partition; accessories and curtain travel stay inside each bay.
use super::*;
use crate::scene::procedural_indoor::materials::garment_stock;
use rand_chacha::ChaCha8Rng;

mod curtain;
#[cfg(test)]
mod tests;

const FIXTURE: &str = "retail_fixture";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FittingRoomSpec {
    pub panel_thickness: f32,
    pub enclosure_finish: Surface,
    pub curtain_stock: Surface,
    pub dye_slot: usize,
    pub rail_height: f32,
    pub hem_height: f32,
    pub bays: Vec<CurtainBay>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CurtainBay {
    /// Inner partition faces in local metres.
    pub left: f32,
    pub right: f32,
    pub gathered_left: bool,
    /// Uncovered width, measured from the curtain edge to the opposite jamb.
    pub clear_opening: f32,
    pub pleats: usize,
    pub bench_left: bool,
    pub bench_height: f32,
    pub bench_depth: f32,
    pub mirror_width: f32,
    pub mirror_bottom: f32,
    pub mirror_height: f32,
}

pub(super) fn dimensions(r: &mut ChaCha8Rng) -> Vec3 {
    let count = if r.random_bool(0.48) { 2. } else { 1. };
    Vec3::new(
        count * r.random_range(1.28..1.74),
        r.random_range(2.10..2.43),
        r.random_range(1.40..1.91),
    )
}

/// Replay the local construction in metres from an archived fitting room instance.
pub fn fitting_room(o: &IndoorObject) -> Option<FittingRoomSpec> {
    if o.kind != ObjectKind::FittingRoom {
        return None;
    }
    let mut r = stream(o.seed, u64::from_le_bytes(*b"CurtBay "));
    let mut finishes = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"FitStock"),
    );
    let panel = r.random_range(0.024..0.046);
    let count = if o.size.x >= 2.4 { 2 } else { 1 };
    let span = (o.size.x - (count + 1) as f32 * panel) / count as f32;
    let mut bays = Vec::with_capacity(count);
    for i in 0..count {
        let left = -o.size.x * 0.5 + panel + i as f32 * (span + panel);
        let opening = match r.random_range(0..10) {
            0..=2 => r.random_range(0.0..0.026),
            3..=5 => r.random_range(0.08..0.61),
            _ => r.random_range(0.66..0.84),
        };
        bays.push(CurtainBay {
            left,
            right: left + span,
            gathered_left: r.random(),
            clear_opening: span * opening,
            pleats: r.random_range(6..=10),
            bench_left: r.random(),
            bench_height: r.random_range(0.419..0.49),
            bench_depth: r.random_range(0.29..0.39),
            mirror_width: span * r.random_range(0.41..0.60),
            mirror_bottom: r.random_range(0.17..0.36),
            mirror_height: r.random_range(1.39..1.64),
        });
    }
    Some(FittingRoomSpec {
        panel_thickness: panel,
        enclosure_finish: [Surface::Wood, Surface::Paint, Surface::Accent]
            [finishes.random_range(0..3)],
        curtain_stock: if finishes.random_bool(0.52) {
            Surface::Fabric
        } else {
            Surface::FabricAlt
        },
        dye_slot: garment_stock::sample_slot(&mut finishes),
        rail_height: o.size.y - r.random_range(0.029..0.075),
        hem_height: r.random_range(0.055..0.15),
        bays,
    })
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let f = fitting_room(o).expect("changing cubicle");
    let s = o.size;
    let t = f.panel_thickness;
    let back = -s.z * 0.5;
    a.box_part(
        f.enclosure_finish,
        FIXTURE,
        Vec3::new(0., s.y * 0.5, back + t * 0.5),
        Vec3::new(s.x, s.y, t),
        0.003,
    );
    let wall_length = s.z - t;
    let curtain_label = format!("curtain#finish{}", f.dye_slot);
    for (i, b) in f.bays.iter().enumerate() {
        let mut end_panel = |x| {
            a.box_part(
                f.enclosure_finish,
                FIXTURE,
                Vec3::new(x, s.y * 0.5, t * 0.5),
                Vec3::new(t, s.y, wall_length),
                0.003,
            );
        };
        if i == 0 {
            end_panel(b.left - t * 0.5);
        }
        end_panel(b.right + t * 0.5);
        let rail_z = s.z * 0.5 - 0.118;
        let rail = |x| Vec3::new(x, f.rail_height, rail_z);
        a.part(Surface::Metal, FIXTURE)
            .rod(rail(b.left), rail(b.right), 0.012);
        curtain::build(a, &curtain_label, &f, b, rail_z);
        accessories(a, s, f.panel_thickness, b);
    }
}

fn accessories(a: &mut Assembly, s: Vec3, panel: f32, b: &CurtainBay) {
    let back = -s.z * 0.5 + panel;
    let (edge, inward) = if b.bench_left {
        (b.left, 1.)
    } else {
        (b.right, -1.)
    };
    let bench_z = back + s.z * 0.345;
    let bench_length = s.z * 0.51;
    let x = edge + inward * b.bench_depth * 0.5;
    let seat = 0.031;
    a.box_part(
        Surface::Wood,
        "bench",
        Vec3::new(x, b.bench_height - seat * 0.5, bench_z),
        Vec3::new(b.bench_depth, seat, bench_length),
        0.009,
    );
    for z in [bench_z - bench_length * 0.33, bench_z + bench_length * 0.33] {
        let inner = edge + inward * 0.014;
        let outer = edge + inward * b.bench_depth * 0.78;
        let y = b.bench_height - seat;
        a.box_part(
            Surface::Metal,
            "bench",
            Vec3::new(inner, y - 0.09, z),
            Vec3::new(0.028, 0.18, 0.034),
            0.002,
        );
        a.part(Surface::Metal, "bench").rod(
            Vec3::new(inner, y - 0.148, z),
            Vec3::new(outer, y - 0.008, z),
            0.012,
        );
    }
    // The mirror backing attaches directly to the rear wall. A projecting face
    // leaves a narrow frame visible on all sides without coincident surfaces.
    let mx = (b.left + b.right) * 0.5 + inward * (b.right - b.left) * 0.12;
    let my = b.mirror_bottom + b.mirror_height * 0.5;
    a.box_part(
        Surface::Metal,
        "mirror",
        Vec3::new(mx, my, back + 0.010),
        Vec3::new(b.mirror_width + 0.028, b.mirror_height + 0.028, 0.020),
        0.004,
    );
    a.box_part(
        Surface::Chrome,
        crate::scene::procedural_indoor::materials::glass::mirror::FACE,
        Vec3::new(mx, my, back + 0.023),
        Vec3::new(b.mirror_width, b.mirror_height, 0.006),
        0.002,
    );
    let z = bench_z + bench_length * 0.24;
    let hook = Vec3::new(edge + inward * 0.065, b.bench_height + 1.025, z);
    a.part(Surface::Metal, FIXTURE)
        .rod(Vec3::new(edge, hook.y, z), hook, 0.011);
    a.part(Surface::Metal, FIXTURE)
        .rod(hook, hook + Vec3::new(inward * 0.012, 0.047, 0.), 0.011);
}
