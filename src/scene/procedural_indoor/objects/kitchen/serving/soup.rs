//! A serving is stationary in its dish. The stock lot, ladle and garnish have
//! separate seeds so a shared ceramic set need not imply identical contents.
use super::*;
use crate::scene::procedural_indoor::materials::broth::{self, Recipe};
pub(in super::super) use broth::INTRODUCED;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Ingredient {
    Herb,
    Crouton,
    Bean,
    Vegetable,
}

#[derive(Debug, Serialize)]
struct Piece {
    ingredient: Ingredient,
    center_m: Vec2,
    half_size_m: Vec3,
    yaw: f32,
}

impl Piece {
    fn radius(&self) -> f32 {
        self.half_size_m.xz().length()
    }

    fn emit(&self, a: &mut Assembly, fill: f32) {
        let h = self.half_size_m;
        // Partially immerse every piece. There is no second horizontal garnish plane.
        let tf = Transform::from_xyz(self.center_m.x, fill + h.y * 0.12, self.center_m.y)
            .with_rotation(Quat::from_rotation_y(self.yaw));
        match self.ingredient {
            Ingredient::Crouton | Ingredient::Vegetable => {
                let (surface, label, bevel) = if self.ingredient == Ingredient::Crouton {
                    (Surface::Terracotta, "food#finish103", 0.33)
                } else {
                    (Surface::Ceramic, "food#finish102", 0.51)
                };
                a.part(surface, label)
                    .cuboid(h * 2., h.min_element() * bevel, tf);
            }
            Ingredient::Herb | Ingredient::Bean => {
                let tf = Transform {
                    rotation: tf.rotation * Quat::from_rotation_z(-PI * 0.5),
                    scale: Vec3::new(h.y, h.x, h.z),
                    ..tf
                };
                if self.ingredient == Ingredient::Herb {
                    // A pointed blade with a raised midrib and a closed underside.
                    a.part(Surface::Paper, broth::HERB_LABEL).lathe(
                        &[
                            (0., -1.),
                            (0.67, -0.60),
                            (1., -0.14),
                            (0.72, 0.46),
                            (0., 1.),
                        ],
                        6,
                        tf,
                    );
                } else {
                    // The softer ends use an asymmetric smooth longitudinal profile.
                    // Ninety triangles suffice even for the densest serving.
                    use crate::scene::procedural_indoor::geometry::{
                        ProfileJoints, RevolvedOutline,
                    };
                    RevolvedOutline::round(9, 0., 0).sweep(
                        a.part(Surface::Paper, "food#finish104"),
                        &[
                            (0., -1.),
                            (0.59, -0.82),
                            (0.93, -0.44),
                            (1., 0.12),
                            (0.84, 0.63),
                            (0.48, 0.88),
                            (0., 1.),
                        ],
                        ProfileJoints::Smooth,
                        tf,
                    );
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Soup {
    pub(crate) recipe: Recipe,
    pub(crate) fill_height_m: f32,
    #[serde(skip)]
    radius: f32,
    chart_angle: f32,
    chart_fraction: f32,
    chart_offset: Vec2,
    pieces: Vec<Piece>,
}

impl Soup {
    pub(crate) fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    pub(crate) fn sample(o: &IndoorObject, d: &DishProgram) -> Self {
        // Keep the calibrated fill and all dish, place and camera draws stable.
        let fill = d.liquid_height(stream(o.seed, 0x5345_5256_464f_4f44).random_range(0.18..0.74));
        let radius = d.liquid_radius(fill);
        let mut r = stream(o.seed, 0x4c41_444c_4553_4f55);
        let kitchen = o.finish_seed.unwrap_or(o.seed);
        let house = stream(kitchen, 0x534f_5550_4d45_4e55).random_range(0..4);
        let recipe = Recipe::ALL[if r.random_bool(0.52) {
            house
        } else {
            r.random_range(0..4)
        }];
        let chart_angle = r.random_range(-PI..PI);
        let chart_fraction = r.random_range(0.68..0.97);
        let travel = (1. - chart_fraction) * 0.5;
        let chart_offset = Vec2::new(
            r.random_range(-travel..travel),
            r.random_range(-travel..travel),
        );
        let mut soup = Self {
            recipe,
            fill_height_m: fill,
            radius,
            chart_angle,
            chart_fraction,
            chart_offset,
            pieces: Vec::new(),
        };
        if r.random_bool(if recipe == Recipe::GoldenStock {
            0.18
        } else {
            0.14
        }) {
            return soup;
        }

        // Use the narrowest part of the *inner* wall at the depth occupied by
        // floating pieces. Retain the actual aspect ratio for oval vessels.
        let half_depth = 0.006_f32
            .min((fill - d.interior_floor()) * 0.46)
            .min((d.height - fill) * 0.46);
        let inner = d.inscribed_radius(
            radius
                .min(d.liquid_radius(fill - half_depth))
                .min(d.liquid_radius(fill + half_depth)),
        ) * (PI / d.segments as f32).cos();
        let small_axis = d.aspect.min_element();
        let stock = match recipe {
            Recipe::GoldenStock => [Ingredient::Bean, Ingredient::Vegetable, Ingredient::Herb],
            _ => [Ingredient::Crouton, Ingredient::Herb, Ingredient::Vegetable],
        };
        let primary = stock[r.random_range(0..2)];
        let secondary = stock[r.random_range(0..3)];
        let count = r.random_range(2..=9);
        let cluster = Vec2::from_angle(r.random_range(-PI..PI)) * r.random_range(0.0..0.64);
        let spread = r.random_range(0.24..0.58);
        let clustered = r.random_bool(0.55);
        soup.pieces.reserve(count);
        for _ in 0..count {
            let ingredient = if r.random_bool(0.70) {
                primary
            } else {
                secondary
            };
            let (length, width, depth) = match ingredient {
                Ingredient::Herb => (
                    r.random_range(0.0035..0.0095),
                    r.random_range(0.30..0.57),
                    r.random_range(0.07..0.14),
                ),
                Ingredient::Crouton => (
                    r.random_range(0.0042..0.0105),
                    r.random_range(0.59..1.),
                    r.random_range(0.38..0.76),
                ),
                Ingredient::Vegetable => (
                    r.random_range(0.0036..0.0088),
                    r.random_range(0.47..0.89),
                    r.random_range(0.39..0.62),
                ),
                Ingredient::Bean => (
                    r.random_range(0.0043..0.0081),
                    r.random_range(0.53..0.82),
                    r.random_range(0.37..0.66),
                ),
            };
            let length = f32::min(length, inner * small_axis * 0.20);
            let h = Vec3::new(
                length,
                (length * depth).min(half_depth / 1.12),
                length * width,
            );
            let mut piece = Piece {
                ingredient,
                half_size_m: h,
                center_m: Vec2::ZERO,
                yaw: r.random_range(-PI..PI),
            };
            let radius = piece.radius();
            let limit = inner - radius / small_axis;
            for _ in 0..7 {
                let disk = Vec2::from_angle(r.random_range(-PI..PI)) * r.random::<f32>().sqrt();
                let point = if clustered {
                    cluster + spread * disk
                } else {
                    disk
                };
                if point.length_squared() > 1. {
                    continue;
                }
                let center = point * limit * d.aspect;
                if soup
                    .pieces
                    .iter()
                    .all(|p| center.distance_squared(p.center_m) >= (radius + p.radius()).powi(2))
                {
                    piece.center_m = center;
                    soup.pieces.push(piece);
                    break;
                }
            }
        }
        soup
    }

    pub(super) fn build(&self, a: &mut Assembly, d: &DishProgram) {
        let g = a.part(Surface::Ceramic, broth::LABEL);
        let start = g.positions.len();
        d.liquid_surface(g, self.radius, self.fill_height_m);
        let points = &g.positions[start..];
        let extent = points
            .iter()
            .map(|p| Vec2::new(p[0], p[2]).length_squared())
            .fold(0., f32::max)
            .sqrt();
        let scale = self.chart_fraction / (2. * extent);
        let rotation = Vec2::from_angle(self.chart_angle);
        for (&p, uv) in points.iter().zip(&mut g.uvs[start..]) {
            let local = rotation.rotate(Vec2::new(p[0], -p[2])) * scale + self.chart_offset + 0.5;
            *uv = self.recipe.uv(local);
        }
        for piece in &self.pieces {
            piece.emit(a, self.fill_height_m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{
        audit::fixture, build_object_at_version, kitchen,
    };
    use std::collections::BTreeMap;

    #[test]
    fn servings_match_the_inner_wall_and_keep_distinct_orders_in_one_dish_set() {
        let mut recipes = [0usize; 4];
        let mut kinds = [0usize; 4];
        let mut counts = [0usize; 10];
        let mut radial = [0usize; 3];
        let mut soup_triangles = 0;
        for seed in 0..1280 {
            let mut o = fixture(ObjectKind::Dishware, seed);
            o.finish_seed = Some(41);
            o.variant = dishware::SERVED_BOWL;
            o.size =
                crate::scene::procedural_indoor::items::sizing::dish(1, &mut stream(seed, 321));
            if sample(&o) != Some(Serving::Soup) {
                continue;
            }
            let d = dishware::object_parameters(&o, INTRODUCED);
            let p = Soup::sample(&o, &d);
            recipes[p.recipe as usize] += 1;
            counts[p.piece_count()] += 1;
            let old = build_object_at_version(&o, None, INTRODUCED - 1);
            let new = build_object_at_version(&o, None, INTRODUCED);
            let liquid = &new.parts[&(Surface::Ceramic, broth::LABEL.to_string())];
            let old_liquid = &old.parts[&(Surface::Ceramic, "food#finish105".into())];
            assert_eq!(old_liquid.positions, liquid.positions);
            assert_eq!(old_liquid.indices, liquid.indices);
            assert_eq!(old_liquid.normals, liquid.normals);
            assert!(liquid.uvs.iter().all(|&uv| {
                let uv = Vec2::from_array(uv);
                uv.cmpge(p.recipe.uv(Vec2::ZERO).into()).all()
                    && uv.cmple(p.recipe.uv(Vec2::ONE).into()).all()
            }));
            for (key, geometry) in &old.parts {
                if !key.1.starts_with("food#") {
                    let same = &new.parts[key];
                    assert_eq!(geometry.positions, same.positions);
                    assert_eq!(geometry.indices, same.indices);
                    assert_eq!(geometry.uvs, same.uvs);
                }
            }
            assert!(kitchen::metadata_for_version(&o, INTRODUCED - 1)
                .get("soup")
                .is_none());
            assert_eq!(
                kitchen::metadata_for_version(&o, INTRODUCED)["soup"],
                serde_json::json!(&p)
            );
            let mut by_material = Assembly::default();
            for (i, piece) in p.pieces.iter().enumerate() {
                kinds[piece.ingredient as usize] += 1;
                for other in &p.pieces[..i] {
                    assert!(
                        piece.center_m.distance(other.center_m) + 1e-7
                            >= piece.radius() + other.radius()
                    );
                }
                let distance = (piece.center_m / d.aspect).length() / d.inscribed_radius(p.radius);
                radial[if distance < 0.25 {
                    0
                } else if distance < 0.62 {
                    1
                } else {
                    2
                }] += 1;
                let mut isolated = Assembly::default();
                piece.emit(&mut isolated, p.fill_height_m);
                let (lo, hi) = isolated.bounds();
                assert!(lo.y < p.fill_height_m && hi.y > p.fill_height_m && hi.y < d.height);
                for g in isolated.parts.values() {
                    for &v in &g.positions {
                        let v = Vec3::from_array(v);
                        let clearance = d.inscribed_radius(d.liquid_radius(v.y))
                            * (PI / d.segments as f32).cos();
                        assert!((v.xz() / d.aspect).length() < clearance + 1e-6);
                    }
                }
                piece.emit(&mut by_material, p.fill_height_m);
            }
            let triangles: usize = new
                .parts
                .iter()
                .filter(|((_, l), _)| l.starts_with("food#"))
                .map(|(_, g)| g.indices.len() / 3)
                .sum();
            assert!(
                triangles <= 1200,
                "food triangle budget exceeded at {seed}: {triangles}"
            );
            soup_triangles += triangles;
            for (key, expected) in by_material.parts {
                assert_eq!(expected.positions, new.parts[&key].positions);
            }
        }
        assert!(recipes.iter().all(|&n| n > 22) && kinds.iter().all(|&n| n > 40));
        assert!(counts[0] > 15 && counts[8] + counts[9] > 10 && radial.iter().all(|&n| n > 45));
        let receipt = BTreeMap::from([
            ("recipes", serde_json::json!(recipes)),
            ("pieces_by_ingredient", serde_json::json!(kinds)),
            ("pieces_by_normalized_radius", serde_json::json!(radial)),
            ("garnish_counts", serde_json::json!(counts)),
            ("total_food_triangles", serde_json::json!(soup_triangles)),
        ]);
        if let Some(d) = std::env::var_os("ZEROVERSE_OBJECT_REPORT") {
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                std::path::Path::new(&d).join("soup.json"),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
        }
    }
}
