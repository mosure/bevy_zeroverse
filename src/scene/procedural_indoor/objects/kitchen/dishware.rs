//! One hollow, dimensioned dish program for loose dishes, shelves and appliances.
use super::*;
use crate::scene::procedural_indoor::geometry::RevolvedOutline;
mod bowl;
mod plate;
pub use bowl::BowlForm;
#[cfg(test)]
mod tests;
pub use plate::PlateForm;

// Distinguish individual place settings from legacy loose or stacked dishes.
pub(crate) const SERVED_PLATE: u32 = 16;
pub(crate) const SERVED_BOWL: u32 = 17;

pub(crate) fn served_style(o: &IndoorObject) -> Option<u8> {
    match (o.kind, o.variant) {
        (ObjectKind::Dishware, SERVED_PLATE) => Some(0),
        (ObjectKind::Dishware, SERVED_BOWL) => Some(1),
        _ => None,
    }
}

pub(crate) fn object_parameters(o: &IndoorObject, version: u32) -> DishProgram {
    for_object(o, &props::program(o), version)
}

pub(super) fn for_object(o: &IndoorObject, p: &props::PropProgram, version: u32) -> DishProgram {
    if let Some(style) = served_style(o) {
        let mut dish = parameters_at(o.size, style, 1, o.seed, version);
        // Dish sets share a finish while their outlines and rim profiles vary.
        dish.surface = Surface::Ceramic;
        dish
    } else {
        parameters_at(o.size, p.style, p.count, o.seed, version)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DishProgram {
    pub style: u8,
    pub count: u32,
    pub radius: f32,
    pub aspect: Vec2,
    pub height: f32,
    pub wall: f32,
    pub foot: f32,
    pub shoulder: f32,
    pub shoulder_height: f32,
    pub mouth: f32,
    pub handle_reach: f32,
    pub handle_radius: f32,
    pub segments: u32,
    pub surface: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plate: Option<PlateForm>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bowl: Option<BowlForm>,
}
pub fn parameters(size: Vec3, style: u8, count: u32, seed: u64) -> DishProgram {
    parameters_at(
        size,
        style,
        count,
        seed,
        bevy_zeroverse_capture::GENERATOR_VERSION,
    )
}
pub(super) fn parameters_at(
    size: Vec3,
    style: u8,
    count: u32,
    seed: u64,
    version: u32,
) -> DishProgram {
    let mut r = stream(seed, 0x4449534850524f47);
    let count = if style == 0 { count.clamp(1, 7) } else { 1 };
    let radius = size.x.min(size.z)
        * if style <= 1 {
            r.random_range(0.36..0.43)
        } else {
            r.random_range(0.23..0.2666)
        };
    let height = if style == 0 {
        (radius * r.random_range(0.11_f32..0.20)).min(size.y * 0.66)
    } else {
        size.y * r.random_range(0.69..0.94)
    };
    let wall = r
        .random_range(0.0016_f32..0.0045)
        .min(height * 0.25)
        .min(size.y / (count + 2) as f32);
    let aspect = if style <= 1 && r.random_bool(0.38) {
        Vec2::new(1., r.random_range(0.64..0.94))
    } else {
        Vec2::ONE
    };
    let handle_reach = r.random_range(0.34..0.66);
    // The tube must be thinner than the bend's local curvature radius,
    // including shallow teacups at the supported aspect-ratio extremes.
    let handle_radius = (radius * r.random_range(0.055_f32..0.10))
        .min(0.005)
        .min(height * 0.035)
        .min((height * 0.27).powi(2) / (radius * handle_reach) * 0.30);
    let mut dish = DishProgram {
        style,
        count,
        radius,
        aspect,
        height: height.min(size.y - wall * (count - 1) as f32),
        wall,
        foot: if style <= 1 {
            r.random_range(0.38..0.68)
        } else {
            r.random_range(0.64..0.92)
        },
        shoulder: r.random_range(0.76..0.96),
        shoulder_height: r.random_range(0.26..0.56),
        mouth: if style <= 1 {
            1.
        } else {
            r.random_range(0.80..1.0)
        },
        handle_reach,
        handle_radius,
        segments: if r.random_bool(0.18) {
            r.random_range(10..=16)
        } else {
            32
        },
        surface: if style == 4 {
            Surface::ContainerGlass
        } else {
            [
                Surface::Ceramic,
                Surface::Ceramic,
                Surface::Ceramic,
                Surface::Plastic,
                Surface::Metal,
            ][r.random_range(0..5)]
        },
        plate: None,
        bowl: None,
    };
    if style == 0 {
        dish.plate = Some(PlateForm::sample(&mut dish, size, seed));
    } else if style == 1 && version >= bowl::INTRODUCED {
        dish.bowl = Some(BowlForm::sample(&mut dish, seed));
    }
    dish
}
impl DishProgram {
    pub(super) fn outline(&self) -> RevolvedOutline {
        if let Some(bowl) = &self.bowl {
            return RevolvedOutline::superellipse(
                self.segments,
                bowl.outline_exponent,
                self.aspect * bowl.outline_scale(),
            );
        }
        let exponent = self.plate.as_ref().map_or(2., |form| form.outline_exponent);
        RevolvedOutline::superellipse(self.segments, exponent, self.aspect)
    }

    /// The top of the flat surface that supports food or the next plate's foot.
    pub(crate) fn interior_floor(&self) -> f32 {
        let lift = self.bowl.as_ref().map_or_else(
            || self.plate.as_ref().map_or(0., |p| p.foot_height_m),
            |b| b.foot_height_m,
        );
        self.wall + lift
    }

    /// Radius of the supported flat well, before accounting for aspect and polygon facets.
    pub(super) fn serving_floor_radius(&self) -> f32 {
        if let Some(bowl) = &self.bowl {
            bowl.floor_radius(self) * bowl.outline_scale()
        } else {
            self.radius * self.foot - if self.style == 1 { self.wall } else { 0. }
        }
    }

    pub(super) fn liquid_height(&self, level: f32) -> f32 {
        let middle = self.height * self.shoulder_height + self.wall * 0.25;
        middle + (self.height - middle) * level
    }

    /// Radius in the bowl's outline coordinates at a real elevation in metres.
    pub(super) fn liquid_radius(&self, fill: f32) -> f32 {
        if let Some(bowl) = &self.bowl {
            bowl.cavity_radius(self, fill)
        } else {
            let middle = self.height * self.shoulder_height + self.wall * 0.25;
            let t = (fill - middle) / (self.height - middle);
            self.radius - self.wall + (self.mouth - 1.) * self.radius * t
        }
    }

    pub(super) fn inscribed_radius(&self, radius: f32) -> f32 {
        self.bowl
            .as_ref()
            .map_or(radius, |b| radius * b.outline_scale())
    }

    /// Close the horizontal surface on exactly the same faceted inner wall.
    pub(super) fn liquid_surface(&self, g: &mut Geometry, radius: f32, fill: f32) {
        let profile = [(radius, fill), (0., fill)];
        if self.bowl.is_some() {
            self.outline().sweep(
                g,
                &profile,
                crate::scene::procedural_indoor::geometry::ProfileJoints::Creased,
                Transform::IDENTITY,
            );
        } else {
            g.lathe(
                &profile,
                self.segments,
                Transform::from_scale(Vec3::new(self.aspect.x, 1., self.aspect.y)),
            );
        }
    }

    pub(crate) fn stack_height(&self) -> f32 {
        self.height + self.interior_floor() * (self.count - 1) as f32
    }

    pub(crate) fn half_extents(&self, yaw: f32) -> Vec2 {
        if let Some(plate) = &self.plate {
            return plate.half_extents(self, yaw);
        }
        if self.style != 2 && self.aspect == Vec2::ONE {
            return Vec2::splat(self.radius);
        }
        let mut half = self.aspect * self.radius;
        if self.style == 2 {
            half.x = self.radius * (1. + self.handle_reach) + self.handle_radius;
        }
        crate::scene::procedural_indoor::placement::rotated_half(half * 2., yaw)
    }
    pub(crate) fn profile(&self) -> Vec<(f32, f32)> {
        if let Some(bowl) = &self.bowl {
            return bowl.profile(self);
        }
        if let Some(plate) = &self.plate {
            return plate.section(self).to_vec();
        }
        let (r, h, w) = (self.radius, self.height, self.wall);
        vec![
            (0., 0.),
            (r * self.foot, 0.),
            (r, h * self.shoulder_height),
            (r * self.mouth, h),
            (r * self.mouth - w, h),
            (r - w, h * self.shoulder_height + w * 0.25),
            (r * self.foot - w, w),
            (0., w),
        ]
    }
    fn outer_radius(&self, y: f32) -> f32 {
        let middle = self.height * self.shoulder_height;
        if y <= middle {
            self.radius * (self.foot + (1. - self.foot) * y / middle)
        } else {
            self.radius * (1. + (self.mouth - 1.) * (y - middle) / (self.height - middle))
        }
    }
    pub(crate) fn handle(&self) -> Vec<Vec3> {
        let a = self.outer_radius(self.height * 0.22) * 0.97;
        let b = self.outer_radius(self.height * 0.76) * 0.97;
        (0..=16)
            .map(|i| {
                let t = i as f32 / 16.;
                let y = self.height * (0.49 - 0.27 * (PI * t).cos());
                Vec3::new(
                    a + (b - a) * t + self.radius * self.handle_reach * (PI * t).sin(),
                    y,
                    0.,
                )
            })
            .collect()
    }
}
pub(crate) fn build(a: &mut Assembly, tf: Transform, p: &DishProgram, finish_seed: u64) {
    if let Some(bowl) = &p.bowl {
        return bowl.build(a, tf, p, finish_seed);
    }
    if let Some(plate) = &p.plate {
        return plate.build(a, tf, p, finish_seed);
    }
    let slot = super::super::super::materials::variants::object_slot(p.surface, finish_seed, None);
    let label = format!("dishware#finish{slot}");
    let profile = p.profile();
    let t = tf.mul_transform(Transform::from_scale(Vec3::new(p.aspect.x, 1., p.aspect.y)));
    a.part(p.surface, &label).lathe(&profile, p.segments, t);
    if p.style == 2 {
        let points: Vec<_> = p
            .handle()
            .into_iter()
            .map(|v| tf.transform_point(v))
            .collect();
        a.part(p.surface, &label).tube(&points, p.handle_radius, 10);
    }
}
