//! Hollow dishware and supported kitchen supplies; bounded mesh counts per item.
use super::*;
#[derive(Serialize)]
pub struct PropProgram {
    pub style: u8,
    pub count: u32,
    pub horizontal: bool,
    pub pump: bool,
    pub basket: bool,
}
pub fn program(o: &IndoorObject) -> PropProgram {
    let mut r = stream(o.seed, 8030);
    PropProgram {
        style: r.random_range(0..5),
        count: r.random_range(2..=5),
        horizontal: r.random_bool(0.5),
        pump: r.random_bool(0.60),
        basket: r.random_bool(0.5),
    }
}

pub(super) fn rack(a: &mut Assembly, base: Vec3, size: Vec3, plates: u32, seed: u64) {
    let p = supplies::rack::parameters(size, plates, seed);
    supplies::rack::build(a, Transform::from_translation(base), &p, seed);
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let s = o.size;
    match o.kind {
        ObjectKind::SoapDispenser => {
            super::supplies::dispenser::build(a, Vec3::ZERO, s, program(o).pump, o.seed)
        }
        ObjectKind::Sponge => supplies::sponge::build(a, o),
        ObjectKind::DryingRack => {
            let program = supplies::rack::parameters(s, program(o).count, o.seed);
            supplies::rack::build(
                a,
                Transform::IDENTITY,
                &program,
                o.finish_seed.unwrap_or(o.seed),
            );
        }
        ObjectKind::Dishware => {
            let program = dishware::for_object(o, &program(o));
            dishware::build(
                a,
                Transform::IDENTITY,
                &program,
                o.finish_seed.unwrap_or(o.seed),
            );
            if let Some(meal) = serving::sample(o) {
                serving::build(a, o, &program, meal);
            }
        }
        ObjectKind::Flatware => flatware::build(a, o),
        ObjectKind::CuttingBoard => super::supplies::board(a, o),
        ObjectKind::KitchenTowel => super::supplies::towel(a, o),
        ObjectKind::KnifeBlock => super::supplies::knives::build(a, o),
        ObjectKind::UtensilHolder => supplies::utensils::build(a, o),
        ObjectKind::PaperTowel => supplies::paper_towel::build(a, o),
        ObjectKind::FruitBowl => produce::build(a, o, version),
        ObjectKind::Bread => super::supplies::bread::build(a, o),
        _ => unreachable!("non kitchen prop"),
    }
}

/// Folded around an actual handle: two fabric faces, narrow thickness and open hem.
pub(super) fn hanging_towel(a: &mut Assembly, center: Vec3, width: f32, length: f32) {
    let g = a.part(Surface::FabricAlt, "towel");
    for side in [-1., 1.] {
        for row in 0..20 {
            for col in 0..8 {
                let at = |x: usize, y: usize| {
                    let u = x as f32 / 8.;
                    let v = y as f32 / 20.;
                    center
                        + Vec3::new(
                            (u - 0.5) * width,
                            -v * length,
                            side * (0.006 + 0.003 * (u * TAU * 3.).sin() * v),
                        )
                };
                let points = [
                    at(col, row),
                    at(col + 1, row),
                    at(col + 1, row + 1),
                    at(col, row + 1),
                ];
                let n = (points[1] - points[0])
                    .cross(points[3] - points[0])
                    .normalize()
                    * side;
                let start = g.positions.len() as u32;
                for (i, p) in points.iter().enumerate() {
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs.push([
                        (col + usize::from(i == 1 || i == 2)) as f32 * width / 8.,
                        (row + usize::from(i >= 2)) as f32 * length / 20.,
                    ]);
                }
                g.indices.extend(if side > 0. {
                    [start, start + 1, start + 2, start, start + 2, start + 3]
                } else {
                    [start, start + 2, start + 1, start, start + 3, start + 2]
                });
            }
        }
    }
}
