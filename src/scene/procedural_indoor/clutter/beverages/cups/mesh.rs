use super::*;
use crate::scene::procedural_indoor::materials::drinkware::KRAFT_SLOT;

/// Outside of the physical wall, in (radius, height) coordinates. Drinks,
/// sleeve shoulders and handle sockets interpolate these tessellated stations.
pub(super) struct Section {
    outer: Vec<(f32, f32)>,
}

impl Section {
    pub(super) fn new(f: &CupForm, p: &BeverageProgram) -> Self {
        let r = f.body_radius_m;
        let upper = f.height_m - f.lip_m;
        let mut outer = Vec::with_capacity(13);
        outer.extend([(r * 0.78, 0.), (r, f.heel_m)]);
        let mouth = r * p.taper;
        if f.handle_gauge_m.is_some() {
            let belly_y = f.heel_m + (upper - f.heel_m) * p.belly_height;
            let belly = r * p.belly_fraction;
            for (start, end, low, high) in [
                (r, belly, f.heel_m, belly_y),
                (belly, mouth, belly_y, upper),
            ] {
                for i in 1..=5 {
                    let t = i as f32 * 0.2;
                    outer.push((
                        start + (end - start) * t * t * (3. - 2. * t),
                        low + (high - low) * t,
                    ));
                }
            }
        } else {
            // Formed paper has straight conical sides between its heel and rolled mouth.
            outer.push((mouth, upper));
        }
        Self { outer }
    }

    pub(super) fn mouth(&self) -> f32 {
        self.outer.last().unwrap().0
    }

    pub(super) fn radius_at(&self, y: f32) -> f32 {
        self.radius_and_slope(y).0
    }

    pub(super) fn radius_and_slope(&self, y: f32) -> (f32, f32) {
        let index = self
            .outer
            .partition_point(|&(_, height)| height < y)
            .clamp(1, self.outer.len() - 1);
        let [(a, lo), (b, hi)] = [self.outer[index - 1], self.outer[index]];
        let slope = (b - a) / (hi - lo);
        (a + slope * (y.clamp(lo, hi) - lo), slope)
    }

    pub(super) fn draw(&self, a: &mut Assembly, f: &CupForm, label: &str) {
        let tf = f.transform();
        let section = outline(f.sides);
        let body = if f.handle_gauge_m.is_some() {
            Surface::Ceramic
        } else {
            Surface::Paper
        };
        let foot = self.outer[0].0;
        let recess = f.floor_m * 0.39;
        let mut path = Vec::with_capacity(self.outer.len() * 2 + 10);
        path.extend([(0., recess), (foot * 0.80, recess * 0.84)]);
        path.extend_from_slice(&self.outer);
        if let Some((sleeve, cut)) = f
            .sleeve
            .as_ref()
            .and_then(|s| s.cut.as_ref().map(|c| (s, c)))
        {
            let band = Vec2::new(sleeve.low_fraction, sleeve.high_fraction) * f.height_m;
            let upper = path.pop().expect("paper cup has a mouth station");
            section.sweep(a.part(body, label), &path, ProfileJoints::Smooth, tf);
            cut.draw(a, self, f, sleeve, band);
            path.clear();
            path.push(upper);
        } else if let Some(sleeve) = &f.sleeve {
            let low = f.height_m * sleeve.low_fraction;
            let high = f.height_m * sleeve.high_fraction;
            let r0 = self.radius_at(low);
            let r1 = self.radius_at(high);
            let t = sleeve.thickness_m;
            // Replace the outside under the sleeve. There is one visible skin,
            // with two real cut edges and no floating or interpenetrating wrap.
            path.pop();
            path.push((r0, low));
            section.sweep(a.part(body, label), &path, ProfileJoints::Smooth, tf);
            let kraft = format!("{label}#finish{KRAFT_SLOT}");
            let edges = a.part(Surface::Paper, &kraft);
            section.sweep(
                edges,
                &[(r0, low), (r0 + t, low)],
                ProfileJoints::Creased,
                tf,
            );
            section.sweep(
                edges,
                &[(r1 + t, high), (r1, high)],
                ProfileJoints::Creased,
                tf,
            );
            let face = if sleeve.printed {
                a.part(Surface::Art, label)
            } else {
                edges
            };
            let start = face.positions.len();
            section.sweep(
                face,
                &[(r0 + t, low), (r1 + t, high)],
                ProfileJoints::Creased,
                tf,
            );
            if sleeve.printed {
                let cols = f.sides as usize + 1;
                for (i, uv) in face.uvs[start..].iter_mut().enumerate() {
                    *uv = [(i % cols) as f32 / f.sides as f32, (i / cols) as f32];
                }
            }
            path.clear();
            path.extend([(r1, high), *self.outer.last().unwrap()]);
        }
        let top = f.height_m;
        let r = self.mouth();
        path.extend([
            (r + f.lip_m * 0.4, top - f.lip_m * 0.5),
            (r, top),
            (r - f.wall_m * 0.58, top),
            (r - f.wall_m, top - f.lip_m),
        ]);
        path.extend(
            self.outer
                .iter()
                .rev()
                .skip(1)
                .filter(|&&(_, y)| y > f.floor_m)
                .map(|&(r, y)| (r - f.wall_m, y)),
        );
        path.extend([
            (self.radius_at(f.floor_m) - f.wall_m, f.floor_m),
            (0., f.floor_m),
        ]);
        section.sweep(a.part(body, label), &path, ProfileJoints::Smooth, tf);
    }

    pub(super) fn drink(
        &self,
        a: &mut Assembly,
        f: &CupForm,
        p: &BeverageProgram,
        brew: Brew,
        label: &str,
    ) {
        let flat = f.height_m * p.fill_fraction;
        let meniscus = (f.wall_m * 0.45).clamp(0.00026, 0.0011);
        let edge = flat + meniscus;
        let rim = self.radius_at(edge) - f.wall_m;
        let inside = (self.radius_at(flat) - f.wall_m).min(rim);
        // The wetted boundary shares the cup's polygon, including flat-faceted mugs.
        // Only the free surface is visible through this opaque vessel.
        outline(f.sides).sweep(
            a.part(Surface::Drink, &format!("{label}#finish{}", brew.slot())),
            &[(rim, edge), (inside - 2. * meniscus, flat), (0., flat)],
            ProfileJoints::Smooth,
            f.transform(),
        );
    }
}

impl Saucer {
    pub(super) fn draw(&self, a: &mut Assembly, label: &str) {
        let r = self.radius_m;
        let t = self.seat_y_m;
        let h = self.rim_y_m;
        let foot = self.well_radius_m * self.foot_fraction;
        outline(32).sweep(
            a.part(
                Surface::Ceramic,
                &format!("{label}#finish{}", self.finish_slot),
            ),
            &[
                (0., t * 0.57),
                (foot * 0.82, t * 0.53),
                (foot, 0.),
                (foot * 1.10, t * 0.46),
                (self.well_radius_m, t * 0.48),
                (r * 0.95, h - t * 0.51),
                (r, h - t * 0.25),
                (r * 0.99, h),
                (r * 0.94, h - t * 0.08),
                (self.well_radius_m, t),
                (0., t),
            ],
            ProfileJoints::Smooth,
            Transform::from_xyz(self.center_m.x, 0., self.center_m.y),
        );
    }
}
