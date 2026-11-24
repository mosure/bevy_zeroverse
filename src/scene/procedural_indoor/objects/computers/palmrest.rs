//! One closed chassis with a cut for the touch surface. The deck, reveal and
//! pad meet at disjoint edges, including the radiused corners.
use super::*;
use crate::scene::procedural_indoor::materials::input_housing::Housing;

fn outline(half: Vec2, radius: f32, offset: Vec2) -> [Vec2; 12] {
    rounded_outline(half, radius.min(half.min_element() * 0.45), offset)
}

pub(super) fn rounded_outline(half: Vec2, r: f32, offset: Vec2) -> [Vec2; 12] {
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    std::array::from_fn(|i| {
        let arc = [Vec2::X, Vec2::splat(diagonal), Vec2::Y][i % 3];
        let (corner, unit) = match i / 3 {
            0 => (Vec2::ONE, arc),
            1 => (Vec2::new(-1., 1.), Vec2::new(-arc.y, arc.x)),
            2 => (Vec2::NEG_ONE, -arc),
            _ => (Vec2::new(1., -1.), Vec2::new(arc.y, -arc.x)),
        };
        offset + corner * (half - r) + unit * r
    })
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &LaptopProgram,
    stock: &Housing,
    depth: f32,
    base_z: f32,
    bank: Option<&io::Bank>,
) {
    let mut r = stream(o.seed, 0x50414c4d52455354);
    let stock_label = stock.case_label(o);
    let pad_label = stock.pad_label();
    let chassis = a.part(stock.surface, &stock_label);
    let half = Vec2::new(o.size.x, depth) * 0.5;
    let outer = outline(half, p.corner_m, Vec2::ZERO);
    // With a numeric bank, center the pad nearer the typing block.
    let cx = o.size.x
        * (if p.keyboard_columns >= 19 {
            r.random_range(-0.105..-0.045)
        } else {
            r.random_range(-0.038..0.024)
        });
    let center = Vec2::new(cx, depth * r.random_range(0.27..0.30));
    let pad_half = Vec2::new(
        o.size.x * p.trackpad_fraction * 0.5,
        depth * r.random_range(0.096..0.125),
    );
    let gap = r.random_range(0.00050..0.00105);
    let recess = r.random_range(0.00035..0.00090);
    let radius = r.random_range(0.0025..0.0065);
    let lip = outline(pad_half + gap, radius + gap, center);
    let pad = outline(pad_half, radius, center);
    let tf = Transform::from_xyz(0., p.chassis_m * 0.5 + 0.003, base_z);
    let y = p.chassis_m * 0.5;
    let pt = |p: Vec2, y: f32| tf.transform_point(Vec3::new(p.x, y, p.y));
    let profile = Geometry::slab_profile(&outer, p.chassis_m, p.corner_m.min(0.002));
    chassis.slab_cap(&outer, profile.1, Vec2::ZERO, -y, &tf, |v| v);
    for i in 0..outer.len() {
        let j = (i + 1) % outer.len();
        chassis.metric_quad(
            [
                pt(outer[i] * profile.1, y),
                pt(outer[j] * profile.1, y),
                pt(lip[j], y),
                pt(lip[i], y),
            ],
            Vec3::Y,
        );
        chassis.metric_quad(
            [
                pt(lip[i], y),
                pt(lip[j], y),
                pt(pad[j], y - recess),
                pt(pad[i], y - recess),
            ],
            Vec3::Y,
        );
    }
    if let Some(bank) = bank {
        bank.deck_edges(chassis, &outer, p.chassis_m, profile, &tf);
        bank.pockets(a, o.kind.class_name(), &pad_label);
    } else {
        chassis.slab_edge(&outer, p.chassis_m, profile, &tf);
    }
    a.part(Surface::Plastic, &pad_label)
        .slab_cap(&pad, 1., center, y - recess, &tf, |v| v - center);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;
    use std::collections::BTreeMap;

    #[test]
    fn etched_insert_and_case_make_one_closed_surface_with_an_actual_recess() {
        for seed in 0..40 {
            let mut o = audit::fixture(ObjectKind::Laptop, seed);
            if seed % 2 == 0 {
                o.variant &= !io::FITTED_IO;
            }
            let p = parameters(&o);
            let stock = Housing::for_object(&o);
            let mut a = Assembly::default();
            let deck = Deck::fit(&o, &p);
            let bank = p.io_bank(&o);
            build(
                &mut a,
                &o,
                &p,
                &stock,
                deck.depth,
                deck.rear + deck.depth * 0.5,
                bank.as_ref(),
            );
            audit::check(&o, &a).unwrap();
            // The same polymer batch also contains the fitted edge inserts.
            let pad: Vec<_> = a.parts[&(Surface::Plastic, stock.pad_label())]
                .positions
                .iter()
                .filter(|v| v[0].abs() < o.size.x * 0.4)
                .collect();
            assert_eq!(pad.len(), 13);
            let y = pad[0][1];
            assert!((0.00034..0.00091).contains(&(p.chassis_m + 0.003 - y)));
            assert!(pad.iter().all(|p| (p[1] - y).abs() < 1e-7));
            // Weld the material/chart seams and verify two opposing uses of
            // every edge. An overlapping original top face fails this check.
            let mut edges = BTreeMap::new();
            for g in a.parts.values() {
                for t in g.indices.as_chunks::<3>().0 {
                    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                        let mut e = [i, j]
                            .map(|k| g.positions[t[k] as usize].map(|v| (v * 1e7).round() as i32));
                        let dir = if e[0] < e[1] {
                            1
                        } else {
                            e.swap(0, 1);
                            -1
                        };
                        let pair = edges.entry(e).or_insert((0, 0));
                        pair.0 += 1;
                        pair.1 += dir;
                    }
                }
            }
            assert!(edges.values().all(|e| *e == (2, 0)), "seed {seed}");
        }
    }
}
