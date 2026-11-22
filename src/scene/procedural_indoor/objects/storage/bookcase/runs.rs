//! Books lean onto a divider or a weighted stop, then onto the preceding book.
use super::{contents::Content, *};

#[derive(Clone, Debug)]
pub(crate) struct Bookend {
    offset: Vec3,
    size: Vec3,
    direction: f32,
    surface: Surface,
}

impl Bookend {
    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, c: &Content) {
        let half = self.size * 0.5;
        // A tapered outside edge keeps the bearing face vertical and the foot wide.
        let section = [
            Vec2::new(-half.x, -half.y),
            Vec2::new(half.x, -half.y),
            Vec2::new(half.x, half.y),
            Vec2::new(-0.18 * half.x, half.y),
        ];
        let face = if self.direction > 0. { 0. } else { PI };
        let tf = Transform::from_translation(c.position + self.offset)
            .with_rotation(Quat::from_rotation_y(face) * Quat::from_rotation_x(-FRAC_PI_2));
        a.part(self.surface, o.kind.class_name())
            .profile_slab(&section, self.size.z, 0.0004, tf);
    }
}

/// A section in coordinates measured away from the stop. The bearing corner rests on y=0.
#[derive(Clone, Copy)]
struct Section {
    foot: f32,
    width: f32,
    height: f32,
    angle: f32,
}

impl Section {
    fn next_foot(self, height: f32, angle: f32) -> f32 {
        let (sin, cos) = self.angle.sin_cos();
        let top = self.height * cos + self.width * sin;
        let contact_y = top.min(height * angle.cos());
        // The preceding outer edge and the next inner edge are closest here
        // because each successive book tilts at least as far toward the stop.
        self.foot + self.width / cos + contact_y * (angle.tan() - self.angle.tan()) + 0.0002
    }
}

pub(super) fn arrange(
    o: &IndoorObject,
    p: &BookcaseProgram,
    bay_index: usize,
    level: usize,
    books: &mut [Content],
) {
    if books.len() < 2 {
        return;
    }
    let mut rng = stream(
        o.seed,
        0x5348_454c_4652_554e + (bay_index * 16 + level) as u64,
    );
    // Matching editions share trim size and jacket design, but can have different page counts.
    if rng.random_bool(0.48) {
        let n = rng.random_range(2..=books.len());
        let start = rng.random_range(0..=books.len() - n);
        let series = &mut books[start..start + n];
        let height = series[0].size.y;
        let depth = series
            .iter()
            .map(|c| c.size.z)
            .fold(f32::INFINITY, f32::min);
        let edition = rng.random();
        for book in series {
            book.edition = Some(edition);
            book.size.y = height;
            book.size.z = depth;
        }
    }
    if !rng.random_bool(0.51) {
        return;
    }
    let bay = &p.bays[bay_index];
    let base = bay.levels[level] + 0.5 * p.panel_m;
    let headroom = bay.levels[level + 1] - 0.5 * p.panel_m - base - 0.006;
    let span = bay.right - bay.left;
    let direction = if rng.random_bool(0.5) { 1. } else { -1. };
    let with_bookend = p.open_frame || rng.random_bool(0.38);
    let stop_width: f32 = if with_bookend {
        rng.random_range(0.038..0.084)
    } else {
        0.
    };
    let stop_height = if with_bookend {
        (books[0].size.y * rng.random_range(0.51..0.84)).min(0.22)
    } else {
        headroom
    };
    let mut sections: Vec<Section> = Vec::with_capacity(books.len());
    let mut angle: f32 = rng.random_range(0.06..0.29);
    let mut lo = -stop_width;
    let mut hi: f32 = 0.;
    for book in books.iter() {
        let width = book.size.x;
        let height = book.size.y;
        // Gravity must load the stop. With the centre of mass on the other
        // side of the bearing corner, a thick book would settle back upright.
        angle = angle.max((width / height).atan() + 0.012);
        if angle > 0.47 {
            return;
        }
        let (sin, cos) = angle.sin_cos();
        if height * cos + width * sin > headroom {
            return;
        }
        let foot = match sections.last() {
            Some(prev) => prev.next_foot(height, angle),
            None => stop_height.min(height * cos) * angle.tan() + 0.0002,
        };
        lo = lo.min(foot - height * sin);
        hi = hi.max(foot + width * cos);
        sections.push(Section {
            foot,
            width,
            height,
            angle,
        });
        angle = (angle + rng.random_range(0.004..0.049)).min(0.47);
    }
    // Retain the original upright run in a tight compartment. A divider must
    // be at the edge of the bay; a portable stop and its books can shift together.
    let anchor = if with_bookend {
        let first = 0.003 - lo;
        let last = span - 0.003 - hi;
        if first >= last {
            return;
        }
        rng.random_range(first..last)
    } else {
        if lo < -0.0001 || hi >= span - 0.003 {
            return;
        }
        0.
    };
    let anchor_x = if direction > 0. {
        bay.left + anchor
    } else {
        bay.right - anchor
    };
    let rear = -o.size.z * 0.5 + if p.closed_back { p.panel_m } else { 0. };
    let front = o.size.z * 0.5;
    let deepest = books.iter().map(|c| c.size.z).fold(0., f32::max);
    let min_spine = rear + deepest + 0.008;
    let max_spine = front - 0.008;
    if min_spine >= max_spine {
        return;
    }
    let spine = rng.random_range(min_spine..max_spine);
    for (book, section) in books.iter_mut().zip(sections) {
        let (sin, cos) = section.angle.sin_cos();
        book.lean = direction * section.angle;
        book.yaw = 0.;
        book.position = Vec3::new(
            anchor_x + direction * (section.foot + 0.5 * section.width * cos),
            base + 0.5 * section.width * sin,
            spine + rng.random_range(-0.006..0.006) - 0.5 * book.size.z,
        );
    }
    if with_bookend {
        let first = &mut books[0];
        let size = Vec3::new(
            stop_width,
            stop_height,
            first.size.z * rng.random_range(0.60..0.92),
        );
        let center = Vec3::new(
            anchor_x - direction * stop_width * 0.5,
            base + stop_height * 0.5,
            first.position.z,
        );
        first.bookend = Some(Bookend {
            size,
            offset: center - first.position,
            direction,
            surface: [Surface::Wood, Surface::Metal, Surface::Plastic][rng.random_range(0..3)],
        });
    }
}
