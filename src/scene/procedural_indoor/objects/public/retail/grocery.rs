//! Gondola shelves keep neighboring facings in product lots, with real gaps
//! when stock is depleted. The printed wall *is* the side of the package.
use super::*;
use crate::scene::procedural_indoor::{
    materials::package_print, objects::kitchen::produce::crate_lot,
};

pub(super) mod chilled;
pub(super) mod gondola;
mod package;
pub(super) mod produce;
use package::Pack;

/// The stocking layouts guarantee that all four sampled products appear.
/// The material pass can replay their print lots without constructing meshes.
#[derive(Clone, Copy)]
struct Sku {
    pack: Pack,
    lot: usize,
}

fn assortment(o: &IndoorObject) -> [Sku; 4] {
    if o.variant & REFRIGERATED != 0 {
        use rand::seq::SliceRandom;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ColdSKUs"));
        let stock = [Pack::Gable, Pack::Tub, Pack::Bottle];
        let department = stock[r.random_range(0..stock.len())];
        let mut packs = [
            department,
            department,
            stock[r.random_range(0..3)],
            stock[r.random_range(0..3)],
        ];
        packs.shuffle(&mut r);
        return packs.map(|pack| Sku {
            pack,
            lot: r.random_range(0..package_print::LOTS),
        });
    }
    let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopSKUs"));
    let department = [Pack::Carton, Pack::Can, Pack::Bottle][r.random_range(0..3)];
    let local_share = r.random_range(0.28..0.77);
    std::array::from_fn(|_| Sku {
        pack: if r.random_bool(local_share) {
            department
        } else {
            [Pack::Carton, Pack::Can, Pack::Bottle][r.random_range(0..3)]
        },
        lot: r.random_range(0..package_print::LOTS),
    })
}

pub(crate) fn print_finishes(o: &IndoorObject) -> [(Surface, usize); 4] {
    assortment(o).map(|sku| (sku.pack.face().0, package_print::retail_finish(sku.lot)))
}

/// The four SKU meshes live only during construction of this fixture. Copies
/// retain their source UVs, normals and indices in ordinary renderer batches.
fn facing(a: &mut Assembly, template: &Assembly, base: Vec3, yaw: f32) {
    let rotation = yaw.sin_cos();
    for ((surface, label), g) in &template.parts {
        append_facing(a.part(*surface, label), g, base, rotation, 1.);
    }
}

/// Material and geometry are constructed once per product lot, then appended
/// directly to their annotation and material batches at the stocked positions.
fn build_lots(a: &mut Assembly, products: [(Sku, Vec3); 4], lots: [Vec<(Vec3, f32, f32)>; 4]) {
    for ((sku, size), instances) in products.into_iter().zip(lots) {
        let mut template = Assembly::default();
        sku.pack.build(
            &mut template,
            Vec3::ZERO,
            size,
            0.,
            &package_print::retail_label(sku.pack.face().1, sku.lot),
        );
        append_stock(a, template, &instances);
    }
}

/// A stocked product has one mesh per shape/finish, irrespective of its facing count.
fn append_stock(a: &mut Assembly, template: Assembly, instances: &[(Vec3, f32, f32)]) {
    for ((surface, label), g) in template.parts {
        let dest = a.part(surface, &label);
        dest.positions.reserve(g.positions.len() * instances.len());
        dest.normals.reserve(g.normals.len() * instances.len());
        dest.uvs.reserve(g.uvs.len() * instances.len());
        dest.indices.reserve(g.indices.len() * instances.len());
        for &(base, sin, cos) in instances {
            append_facing(dest, &g, base, (sin, cos), 1.);
        }
    }
}

pub(super) fn legacy_shelves(
    a: &mut Assembly,
    o: &IndoorObject,
    levels: usize,
    facings: usize,
    stock: f32,
    frame: Surface,
    r: &mut ChaCha8Rng,
) {
    let s = o.size;
    let assortment = assortment(o);
    let plinth = 0.105;
    let thickness = r.random_range(0.018..0.029);
    board(
        a,
        frame,
        FIXTURE,
        Vec3::Y * (plinth * 0.5),
        Vec3::new(s.x - 0.08, plinth, s.z - 0.09),
    );
    // Solid central backing, with a cantilevered shelf to each side. The two
    // faces share a datum and stop at the backing instead of cutting through it.
    let spine = 0.036;
    let run = s.y - plinth;
    board(
        a,
        frame,
        FIXTURE,
        Vec3::Y * (plinth + run * 0.5),
        Vec3::new(s.x - 0.05, run, spine),
    );
    let interval = run / levels as f32;
    let pitch = (s.x - 0.10) / facings as f32;
    let templates = assortment.map(|Sku { pack, lot }| {
        let (surface, class) = pack.face();
        let size = pack.size(
            pitch,
            interval - thickness,
            &mut stream(o.seed, 0x534b_5500 + surface as u64 * 16 + lot as u64),
        );
        let mut mesh = Assembly::default();
        pack.build(
            &mut mesh,
            Vec3::ZERO,
            size,
            0.,
            &package_print::retail_label(class, lot),
        );
        (size, mesh)
    });
    for i in 0..levels {
        let y = plinth + thickness + i as f32 * interval;
        let projection = (s.z - spine) * 0.5 * (1. - 0.18 * i as f32 / (levels - 1) as f32);
        for (face, sign) in [-1., 1.].into_iter().enumerate() {
            board(
                a,
                frame,
                FIXTURE,
                Vec3::new(0., y - thickness * 0.5, sign * (spine + projection) * 0.5),
                Vec3::new(s.x, thickness, projection),
            );
            let mut column = 0;
            while column < facings {
                let index = if column == 0 {
                    (2 * i + face) % assortment.len()
                } else {
                    r.random_range(0..assortment.len())
                };
                let (size, mesh) = &templates[index];
                let group = r.random_range(2..=5);
                let end = (column + group).min(facings);
                // Adjacent copies belong to one SKU. Restocking fronts a group
                // and leaves a little depth and angular variation on individual packs.
                let edge =
                    (spine * 0.5 + projection - r.random_range(0.018..0.058) - size.z * 0.55)
                        .max(spine * 0.5 + size.z * 0.58);
                for slot in column..end {
                    if slot != 0 && !r.random_bool(f64::from(stock)) {
                        continue;
                    }
                    let x = -s.x * 0.5 + 0.05 + pitch * (slot as f32 + r.random_range(0.46..0.54));
                    let z = edge - r.random_range(0.0..0.018);
                    let yaw = if sign > 0. { 0. } else { PI };
                    let copies = if size.z * 1.85 < z - spine * 0.5 && r.random_bool(0.48) {
                        2
                    } else {
                        1
                    };
                    for depth in 0..copies {
                        facing(
                            a,
                            mesh,
                            Vec3::new(x, y, sign * (z - depth as f32 * (size.z + 0.015))),
                            yaw + r.random_range(-0.075..0.075),
                        );
                    }
                }
                column = end;
            }
        }
    }
}

pub(super) fn legacy_produce(
    a: &mut Assembly,
    s: Vec3,
    crates: usize,
    fill: f32,
    frame: Surface,
    r: &mut ChaCha8Rng,
) {
    let top = s.y - 0.16;
    table_base(a, s, top, frame);
    let pitch = (s.x - 0.045) / crates as f32;
    for i in 0..crates {
        let cx = (i as f32 + 0.5 - crates as f32 * 0.5) * pitch;
        let width = pitch - 0.018;
        let depth = s.z - 0.036;
        let t = 0.014;
        let rim = r.random_range(0.074..0.106);
        let floor = top + t;
        board(
            a,
            Surface::Wood,
            FIXTURE,
            Vec3::new(cx, top + t * 0.5, 0.),
            Vec3::new(width, t, depth),
        );
        for side in [-1., 1.] {
            board(
                a,
                Surface::Wood,
                FIXTURE,
                Vec3::new(cx + side * (width - t) * 0.5, floor + rim * 0.5, 0.),
                Vec3::new(t, rim, depth),
            );
            board(
                a,
                Surface::Wood,
                FIXTURE,
                Vec3::new(cx, floor + rim * 0.5, side * (depth - t) * 0.5),
                Vec3::new(width - 2. * t, rim, t),
            );
        }
        let room = Vec2::new(width, depth) - Vec2::splat(2. * t + 0.004);
        let stock = crate_lot(r);
        // Loose produce gathers in irregular touching groups. Bound the work
        // and check actual fruit radii before accepting any position.
        let wanted = (room.element_product() * fill / 0.012)
            .round()
            .clamp(4., 30.) as usize;
        let mut placed: Vec<(Vec2, f32)> = Vec::with_capacity(wanted);
        for _ in 0..wanted {
            let (rad, mesh) = &stock[r.random_range(0..stock.len())];
            let limit = room * 0.5 - *rad;
            if limit.min_element() <= 0. {
                continue;
            }
            for _ in 0..8 {
                let q = if !placed.is_empty() && r.random_bool(0.58) {
                    let (near, other) = placed[r.random_range(0..placed.len())];
                    near + Vec2::from_angle(r.random_range(-PI..PI))
                        * (rad + other + r.random_range(0.001..0.022))
                } else {
                    Vec2::new(
                        r.random_range(-limit.x..limit.x),
                        r.random_range(-limit.y..limit.y),
                    )
                };
                if q.abs().cmpgt(limit).any()
                    || placed
                        .iter()
                        .any(|(p, other)| p.distance_squared(q) < (rad + other).powi(2))
                {
                    continue;
                }
                placed.push((q, *rad));
                facing(
                    a,
                    mesh,
                    Vec3::new(cx + q.x, floor, q.y),
                    r.random_range(-PI..PI),
                );
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retail_facing_instances_match_direct_package_geometry_and_mapping() {
        for pack in [
            Pack::Carton,
            Pack::Can,
            Pack::Bottle,
            Pack::Gable,
            Pack::Tub,
        ] {
            let size = pack.size(0.17, 0.26, &mut stream(19, 12));
            let label = package_print::retail_label(pack.face().1, 0);
            let mut template = Assembly::default();
            pack.build(&mut template, Vec3::ZERO, size, 0., &label);
            for angle in [-1.77, 0.13, 2.91] {
                let origin = Vec3::new(0.75, 1.27, -0.38);
                let mut stock = Assembly::default();
                let mut direct = Assembly::default();
                facing(&mut stock, &template, origin, angle);
                pack.build(&mut direct, origin, size, angle, &label);
                assert_eq!(stock.parts.len(), direct.parts.len());
                for (key, g) in &stock.parts {
                    let reference = &direct.parts[key];
                    assert_eq!(g.uvs, reference.uvs);
                    assert_eq!(g.indices, reference.indices);
                    for (x, y) in g
                        .positions
                        .iter()
                        .zip(&reference.positions)
                        .chain(g.normals.iter().zip(&reference.normals))
                    {
                        assert!((Vec3::from_array(*x) - Vec3::from_array(*y)).length() < 0.000001);
                    }
                }
            }
        }
    }
}
