//! Contents of an admitted dining place setting, in the hollow dish's local
//! frame. Inventory, appliances and stacked dishes use the bare dish builder.
use super::{dishware::DishProgram, *};
mod bakery;
pub(super) mod soup;
pub(crate) use bakery::biscuit;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Serving {
    Empty,
    Toast,
    Fruit,
    Soup,
    Biscuit,
}

pub(crate) fn sample(o: &IndoorObject) -> Option<Serving> {
    let style = dishware::served_style(o)?;
    let mut r = stream(o.seed, 0x4d45_414c_504c_4154);
    Some(if r.random_bool(0.37) {
        Serving::Empty
    } else if style == 0 && r.random_bool(0.64) {
        let diameter = o.size.x.min(o.size.z) * 0.8;
        if diameter < 0.13
            || r.random_bool(f64::from(((0.215 - diameter) / 0.095).clamp(0.05, 0.92)))
        {
            Serving::Biscuit
        } else {
            Serving::Toast
        }
    } else if style == 1 && r.random_bool(0.66) {
        Serving::Soup
    } else {
        Serving::Fruit
    })
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    d: &DishProgram,
    meal: Serving,
    version: u32,
) {
    let mut r = stream(o.seed, 0x5345_5256_464f_4f44);
    // Use the inscribed circle of the actual faceted, possibly oval foot.
    // It is a conservative support surface at every yaw, including 10-gons.
    let flat = d.serving_floor_radius() * d.aspect.min_element() * (PI / d.segments as f32).cos();
    let floor = d.interior_floor();
    match meal {
        Serving::Empty => {}
        Serving::Toast => bakery::toast(a, flat, floor, o.size.y, &mut r),
        Serving::Biscuit => bakery::biscuit(a, flat, floor, o.size.y, &mut r),
        Serving::Fruit => {
            let count = r.random_range(3..=7);
            let stock_radius = (flat * r.random_range(0.18..0.25)).min(0.0145);
            let mut fruit: Vec<(Vec2, f32)> = Vec::with_capacity(count);
            let stock = if r.random_bool(0.53) {
                "food#finish100"
            } else {
                "food#finish101"
            };
            for _ in 0..count {
                let radius = stock_radius * r.random_range(0.76..1.06);
                let center = if fruit.is_empty() {
                    flat * Vec2::new(r.random_range(-0.32..0.32), r.random_range(-0.32..0.32))
                } else {
                    // Grow an irregular touching cluster from its existing
                    // neighbors. Bound work and keep all fruit on the foot.
                    let Some(center) = (0..10).find_map(|_| {
                        let (parent, other) = fruit[r.random_range(0..fruit.len())];
                        let v = parent
                            + Vec2::from_angle(r.random_range(-PI..PI))
                                * (radius + other)
                                * r.random_range(1.001..1.10);
                        (v.length() + radius <= flat
                            && fruit
                                .iter()
                                .all(|(p, rr)| v.distance_squared(*p) >= (radius + rr).powi(2)))
                        .then_some(v)
                    }) else {
                        continue;
                    };
                    center
                };
                fruit.push((center, radius));
                let height = (radius * r.random_range(0.77..1.12)).min((o.size.y - floor) * 0.47);
                a.part(Surface::Ceramic, stock).small_ellipsoid(
                    Vec3::new(radius, height, radius * r.random_range(0.75..1.)),
                    Transform::from_xyz(center.x, floor + height, center.y)
                        .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI))),
                );
            }
        }
        Serving::Soup if version >= soup::INTRODUCED => soup::Soup::sample(o, d).build(a, d),
        Serving::Soup => {
            let fill = d.liquid_height(r.random_range(0.18..0.74));
            let radius = d.liquid_radius(fill);
            d.liquid_surface(a.part(Surface::Ceramic, "food#finish105"), radius, fill);
            let radius = d.inscribed_radius(radius);
            let count = r.random_range(0..=4);
            let phase = r.random_range(0.0..TAU);
            for i in 0..count {
                let angle = phase + TAU * (i as f32 + r.random_range(-0.10..0.10)) / count as f32;
                let distance = radius * r.random_range(0.25..0.48);
                let side = (radius * r.random_range(0.11..0.19)).min(0.015);
                let h = side * 0.67;
                a.part(Surface::Terracotta, "food#finish103").cuboid(
                    Vec3::new(side, h, side * r.random_range(0.68..1.)),
                    h * 0.11,
                    Transform::from_xyz(
                        distance * angle.cos() * d.aspect.x,
                        fill + h * 0.17,
                        distance * angle.sin() * d.aspect.y,
                    )
                    .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI))),
                );
            }
        }
    }
}
