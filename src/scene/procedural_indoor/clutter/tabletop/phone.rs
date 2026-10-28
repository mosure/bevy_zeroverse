//! Handset construction, independent of screen content and finish identity.
//! Dimensions describe the admitted object; details are fitted before meshing.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PhoneProgram {
    pub span: Vec2,
    pub body_height: f32,
    pub roundness: f32,
    pub face_up: bool,
    /// Bare enclosure, protective bumper, or padded back cover.
    pub case: u8,
    pub shell: Surface,
    pub back: Surface,
    pub rim_width: f32,
    pub screen_span: Vec2,
    pub screen_z: f32,
    pub face_height: f32,
    pub camera_layout: u8,
    pub island_center: Vec2,
    pub island_span: Vec2,
    pub island_roundness: f32,
    pub island_height: f32,
    pub lens_count: usize,
    pub lens_centers: [Vec2; 4],
    pub lens_radius: f32,
    pub button_z: f32,
    pub button_length: f32,
    pub buttons: usize,
    pub button_side: f32,
}

pub fn parameters(o: &IndoorObject) -> PhoneProgram {
    let prior = super::parameters(o);
    let s = o.size;
    let mut r = stream(o.seed, 0x50484f4e45534850);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x50484f4e4546494e);
    let span = s.xz() * Vec2::new(r.random_range(0.90..0.98), r.random_range(0.91..0.98));
    let case = r.random_range(0..3);
    // Leave room for the cover, island and optical rims before constructing
    // anything. The padded cover uses that reserve rather than growing bounds.
    let body_height = s.y
        * (prior
            .case_fraction
            .min(if prior.phone_face_up { 0.93 } else { 0.87 })
            - if case == 2 { 0.04 } else { 0. });
    let camera_layout = r.random_range(0..4);
    let (mut island_span, center, island_roundness) = match camera_layout {
        0 => (
            span * Vec2::new(r.random_range(0.32..0.44), r.random_range(0.18..0.25)),
            Vec2::new(-span.x * 0.18, -span.y * 0.25),
            r.random_range(3.0..8.0),
        ),
        1 => (
            span * Vec2::new(r.random_range(0.73..0.89), r.random_range(0.12..0.18)),
            Vec2::new(0., -span.y * r.random_range(0.26..0.34)),
            r.random_range(4.0..12.0),
        ),
        2 => {
            let d = span.x * r.random_range(0.52..0.72);
            (Vec2::splat(d), Vec2::new(0., -span.y * 0.25), 2.)
        }
        _ => (
            span * Vec2::new(r.random_range(0.19..0.28), r.random_range(0.24..0.34)),
            Vec2::new(-span.x * 0.25, -span.y * 0.22),
            r.random_range(2.0..4.0),
        ),
    };
    // Fit the full camera enclosure onto the rounded rear cover. Resolve this
    // before packing lenses: no later mesh rescale can invalidate their layout.
    let max_z = center.y.abs() + island_span.y * 0.5;
    let allowed_x = span.x
        * 0.47
        * (1. - (max_z / (span.y * 0.47)).powf(prior.phone_roundness))
            .max(0.01)
            .powf(1. / prior.phone_roundness)
        * 0.97;
    let fit = ((allowed_x - center.x.abs()) * 2. / island_span.x).clamp(0.1, 1.);
    island_span.x *= fit;
    if camera_layout == 2 {
        island_span.y *= fit;
    }
    let lens_count: usize = r.random_range(1..=4);
    let columns = if camera_layout == 1 {
        lens_count
    } else if camera_layout == 3 {
        1
    } else {
        lens_count.min(2)
    };
    let rows = lens_count.div_ceil(columns);
    let cell = island_span / Vec2::new(columns as f32, rows as f32);
    let lens_radius = cell.min_element() * r.random_range(0.23..0.32);
    let lens_centers = std::array::from_fn(|i| {
        if i >= lens_count {
            return center;
        }
        // Keep the last odd lens centered, rather than a redundant empty cell.
        let row = i / columns;
        let row_count = (lens_count - row * columns).min(columns);
        center
            + Vec2::new(
                (i % columns) as f32 - (row_count - 1) as f32 * 0.5,
                row as f32 - (rows - 1) as f32 * 0.5,
            ) * cell
                * 0.80
    });
    PhoneProgram {
        span,
        body_height,
        roundness: prior.phone_roundness,
        face_up: prior.phone_face_up,
        case,
        shell: if case == 0 {
            [Surface::Metal, Surface::Chrome, Surface::Plastic][f.random_range(0..3)]
        } else {
            [Surface::Rubber, Surface::Plastic][f.random_range(0..2)]
        },
        back: if case == 2 {
            [Surface::Leather, Surface::FabricAlt][f.random_range(0..2)]
        } else {
            [Surface::Metal, Surface::Plastic][f.random_range(0..2)]
        },
        rim_width: if case == 0 {
            0.
        } else {
            span.min_element() * r.random_range(0.008..0.017)
        },
        screen_span: span
            * Vec2::new(
                (1. - prior.bezel_fraction).min(0.92),
                1. - prior.phone_top_border - prior.phone_bottom_border,
            ),
        screen_z: span.y * (prior.phone_top_border - prior.phone_bottom_border) * 0.5,
        face_height: s.y * if case == 2 { 0.05 } else { 0.025 },
        camera_layout,
        island_center: center,
        island_span,
        island_roundness,
        island_height: s.y * r.random_range(0.022..0.052),
        lens_count,
        lens_centers,
        lens_radius,
        button_z: span.y * r.random_range(-0.24..0.06),
        button_length: span.y * r.random_range(0.045..0.095),
        buttons: r.random_range(1..=3),
        button_side: if r.random_bool(0.5) { -1. } else { 1. },
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let h = p.body_height;
    let half = p.span * 0.5;
    a.part(p.shell, l).housing(
        &[
            (0., half * 0.98),
            (h * 0.16, half),
            (h * 0.88, half),
            (h, half * 0.98),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    // Visible cover and body intersect internally; there is no floating glass
    // plane or overlapping exposed top face. UVs are resolved by build_object.
    let face_top = h + p.face_height;
    let (surface, extent, z) = if p.face_up {
        (Surface::PhoneScreen, p.screen_span * 0.5, p.screen_z)
    } else {
        (p.back, half * 0.94, 0.)
    };
    a.part(surface, l).housing(
        &[(h - p.face_height * 0.2, extent), (face_top, extent)],
        p.roundness,
        Transform::from_xyz(0., 0., z),
    );
    if p.rim_width > 0. {
        // A real raised protective edge, with an annular underside as well.
        // Solid loft caps would overlap the rear cover and fill the aperture.
        let outer = half * 0.98;
        let inner = outer - Vec2::splat(p.rim_width);
        a.part(p.shell, l).housing_loop(
            &[
                (h - p.face_height * 0.3, outer),
                (face_top + p.face_height * 0.2, outer),
                (face_top + p.face_height * 0.2, inner),
                (h - p.face_height * 0.3, inner),
                (h - p.face_height * 0.3, outer),
            ],
            p.roundness,
            Transform::IDENTITY,
        );
    }
    if p.face_up {
        // A front camera stays on its display, including short/tall bezels.
        a.part(Surface::Ink, l).cylinder(
            p.span.x * 0.018,
            p.face_height * 0.3,
            Transform::from_xyz(
                0.,
                face_top + p.face_height * 0.10,
                z - p.screen_span.y * 0.44,
            ),
        );
    } else {
        let top = face_top + p.island_height;
        a.part(p.shell, l).housing(
            &[
                (face_top - p.face_height * 0.2, p.island_span * 0.5),
                (top, p.island_span * 0.48),
            ],
            p.island_roundness,
            Transform::from_xyz(p.island_center.x, 0., p.island_center.y),
        );
        for center in &p.lens_centers[..p.lens_count] {
            // Raised lens rim with a dark, recessed optical face.
            let at = Transform::from_xyz(center.x, top, center.y);
            a.part(Surface::Metal, l).lathe(
                &[
                    (p.lens_radius * 0.72, -p.face_height * 0.1),
                    (p.lens_radius, -p.face_height * 0.1),
                    (p.lens_radius, p.face_height),
                    (p.lens_radius * 0.72, p.face_height),
                    (p.lens_radius * 0.72, -p.face_height * 0.1),
                ],
                16,
                at,
            );
            a.part(Surface::Ink, l).cylinder(
                p.lens_radius * 0.75,
                p.face_height * 0.6,
                Transform::from_xyz(center.x, top + p.face_height * 0.25, center.y),
            );
        }
    }
    for (side, count) in [(p.button_side, p.buttons), (-p.button_side, 1)] {
        for i in 0..count {
            let z = p.button_z + p.button_length * 1.28 * (i as f32 - (count - 1) as f32 * 0.5);
            // Follow the case section; short buttons cannot float off a rounded corner.
            let x = half.x * (1. - (z / half.y).abs().powf(p.roundness)).powf(1. / p.roundness);
            a.box_part(
                Surface::Metal,
                l,
                Vec3::new(side * (x - o.size.x * 0.003), h * 0.54, z),
                Vec3::new(o.size.x * 0.015, h * 0.20, p.button_length),
                h * 0.04,
            );
        }
    }
    a.box_part(
        Surface::Ink,
        l,
        Vec3::new(0., h * 0.49, half.y - o.size.z * 0.001),
        Vec3::new(p.span.x * 0.16, h * 0.22, o.size.z * 0.004),
        h * 0.06,
    );
}

#[cfg(test)]
mod tests;
