use super::*;
use crate::scene::procedural_indoor::layout::{ArchitectureStyle, NEIGHBOR_DEPTH};

/// Rectangle complement, merged along each row. No intact wall is left behind
/// glazing, including when openings form ribbons or multiple vertical bands.
pub fn solid_rectangles(min: Vec2, max: Vec2, holes: &[WindowOpening]) -> Vec<(Vec2, Vec2)> {
    let mut xs = vec![min.x, max.x];
    let mut ys = vec![min.y, max.y];
    for o in holes {
        if o.min.cmplt(max).all() && o.max.cmpgt(min).all() {
            xs.extend([o.min.x.clamp(min.x, max.x), o.max.x.clamp(min.x, max.x)]);
            ys.extend([o.min.y.clamp(min.y, max.y), o.max.y.clamp(min.y, max.y)]);
        }
    }
    for edges in [&mut xs, &mut ys] {
        edges.sort_by(f32::total_cmp);
        edges.dedup_by(|a, b| (*a - *b).abs() < 0.00001);
    }
    let mut result = Vec::new();
    for y in ys.windows(2) {
        let mut start = None;
        for (i, x) in xs.windows(2).enumerate() {
            let p = Vec2::new((x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5);
            let solid = !holes
                .iter()
                .any(|o| p.cmpgt(o.min).all() && p.cmplt(o.max).all());
            if solid {
                start.get_or_insert(x[0]);
            }
            if (!solid || i + 2 == xs.len()) && start.is_some() {
                let left = start.take().unwrap();
                let right = if solid { x[1] } else { x[0] };
                result.push((Vec2::new(left, y[0]), Vec2::new(right, y[1])));
            }
        }
    }
    result
}

fn part(
    a: &mut Assembly,
    tf: Transform,
    surface: Surface,
    label: &str,
    center: Vec3,
    size: Vec3,
    bevel: f32,
) {
    if size.min_element() <= 0.0001 {
        return;
    }
    a.part(surface, label).cuboid(
        size,
        bevel,
        Transform::from_translation(tf.transform_point(center)).with_rotation(tf.rotation),
    );
}

/// Existing surface panels/wainscot are clipped by the same rough openings.
pub fn clipped_wall_box(
    a: &mut Assembly,
    scene: &IndoorManifest,
    side: FacadeSide,
    surface: Surface,
    position: Vec3,
    size: Vec3,
    bevel: f32,
) {
    let Some(f) = scene.exterior.as_ref().and_then(|e| e.facade(side)) else {
        a.box_part(surface, "wall", position, size, bevel);
        return;
    };
    let p = side.local(scene.room_size, position);
    let size = if side == FacadeSide::Rear {
        size
    } else {
        Vec3::new(size.z, size.y, size.x)
    };
    for (lo, hi) in solid_rectangles(
        p.truncate() - size.truncate() * 0.5,
        p.truncate() + size.truncate() * 0.5,
        &f.openings,
    ) {
        part(
            a,
            side.transform(scene.room_size),
            surface,
            "wall",
            Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, p.z),
            Vec3::new(hi.x - lo.x, hi.y - lo.y, size.z),
            bevel,
        );
    }
}

pub(in super::super) fn shell(a: &mut Assembly, scene: &IndoorManifest) {
    let Some(exterior) = &scene.exterior else {
        return legacy::shell(a, scene);
    };
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    let (hx, hz, t) = (w * 0.5, d * 0.5, 0.18);
    for side in FacadeSide::ALL {
        let f = exterior.facade(side);
        let tf = side.transform(scene.room_size);
        let span = side.span(scene.room_size);
        let holes = f.map_or(&[][..], |f| f.openings.as_slice());
        if side == FacadeSide::Rear
            && f.is_none()
            && scene.architecture_style == ArchitectureStyle::Classic
        {
            super::super::details::rear_niche(a, scene);
        } else {
            let end = span * 0.5 + if side == FacadeSide::Rear { t } else { 0. };
            for (lo, hi) in solid_rectangles(Vec2::new(-end, 0.), Vec2::new(end, h), holes) {
                part(
                    a,
                    tf,
                    if scene.architecture_style == ArchitectureStyle::Industrial {
                        Surface::Concrete
                    } else {
                        Surface::Paint
                    },
                    "wall",
                    Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, -t * 0.5),
                    Vec3::new(hi.x - lo.x, hi.y - lo.y, t),
                    0.,
                );
            }
        }
        // Base trim also stops at full-height apertures.
        for (lo, hi) in solid_rectangles(
            Vec2::new(-span * 0.5, 0.),
            Vec2::new(span * 0.5, 0.13),
            holes,
        ) {
            part(
                a,
                tf,
                Surface::WoodEdge,
                "wall",
                Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, 0.014),
                Vec3::new(hi.x - lo.x, hi.y - lo.y, 0.024),
                0.002,
            );
        }
        if let Some(f) = f {
            glazing(a, scene, f);
        }
    }
    // The neighbor remains enclosed behind the internal +Z glass partition.
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(0., h * 0.5, hz + NEIGHBOR_DEPTH + t * 0.5),
        Vec3::new(w + 2. * t, h, t),
        0.,
    );
    a.box_part(
        Surface::WoodEdge,
        "wall",
        Vec3::new(0., 0.065, hz + NEIGHBOR_DEPTH - 0.014),
        Vec3::new(w, 0.13, 0.024),
        0.002,
    );
    for sign in [-1., 1.] {
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new(sign * (hx + t * 0.5), h * 0.5, hz + NEIGHBOR_DEPTH * 0.5),
            Vec3::new(t, h, NEIGHBOR_DEPTH),
            0.,
        );
        a.box_part(
            Surface::WoodEdge,
            "wall",
            Vec3::new(sign * (hx - 0.014), 0.065, hz + NEIGHBOR_DEPTH * 0.5),
            Vec3::new(0.024, 0.13, NEIGHBOR_DEPTH),
            0.002,
        );
    }
}

pub(super) fn glazing(a: &mut Assembly, scene: &IndoorManifest, f: &Facade) {
    let tf = f.side.transform(scene.room_size);
    glazing_at(a, tf, f);
}

pub(crate) fn glazing_at(a: &mut Assembly, tf: Transform, f: &Facade) {
    let fw = f.frame_width;
    let bar = fw * 0.66;
    for o in &f.openings {
        let c = (o.min + o.max) * 0.5;
        let size = o.max - o.min;
        for x in [o.min.x + fw * 0.5, o.max.x - fw * 0.5] {
            part(
                a,
                tf,
                f.frame,
                "window",
                Vec3::new(x, c.y, -f.recess),
                Vec3::new(fw, size.y, f.frame_depth),
                0.0025,
            );
        }
        for y in [o.min.y + fw * 0.5, o.max.y - fw * 0.5] {
            part(
                a,
                tf,
                f.frame,
                "window",
                Vec3::new(c.x, y, -f.recess),
                Vec3::new(size.x - 2. * fw - 0.002, fw, f.frame_depth),
                0.0025,
            );
        }
        let lo = o.min + Vec2::splat(fw);
        let hi = o.max - Vec2::splat(fw);
        let pitch = (hi.x - lo.x) / o.columns as f32;
        for col in 1..o.columns {
            part(
                a,
                tf,
                f.frame,
                "window",
                Vec3::new(lo.x + col as f32 * pitch, c.y, -f.recess),
                Vec3::new(bar, hi.y - lo.y - 0.002, f.frame_depth * 0.88),
                0.002,
            );
        }
        let mut levels = vec![lo.y];
        if o.transom > 0. {
            levels.push(lo.y + (hi.y - lo.y) * o.transom);
        }
        levels.push(hi.y);
        for col in 0..o.columns {
            let left = lo.x + col as f32 * pitch + if col == 0 { 0. } else { bar * 0.5 };
            let right =
                lo.x + (col + 1) as f32 * pitch - if col + 1 == o.columns { 0. } else { bar * 0.5 };
            if o.transom > 0. {
                part(
                    a,
                    tf,
                    f.frame,
                    "window",
                    Vec3::new((left + right) * 0.5, levels[1], -f.recess),
                    Vec3::new(right - left - 0.002, bar, f.frame_depth * 0.88),
                    0.002,
                );
            }
            for (i, row) in levels.windows(2).enumerate() {
                let bottom = row[0] + if i == 0 { 0. } else { bar * 0.5 };
                let top = row[1] - if i + 2 == levels.len() { 0. } else { bar * 0.5 };
                part(
                    a,
                    tf,
                    Surface::Glass,
                    "window",
                    Vec3::new(
                        (left + right) * 0.5,
                        (bottom + top) * 0.5,
                        -f.recess - 0.004,
                    ),
                    Vec3::new(right - left + 0.004, top - bottom + 0.004, 0.008),
                    0.,
                );
            }
        }
        if o.min.y > 0.25 {
            part(
                a,
                tf,
                Surface::WoodEdge,
                "window",
                Vec3::new(c.x, o.min.y - 0.010, (f.sill_projection - 0.174) * 0.5),
                Vec3::new(size.x + 0.03, 0.028, 0.174 + f.sill_projection),
                0.003,
            );
        }
        if o.operable {
            let p = Vec3::new(
                o.max.x - fw * 0.5,
                (o.min.y + 1.1).min(c.y),
                -f.recess + f.frame_depth * 0.5 + 0.013,
            );
            part(
                a,
                tf,
                Surface::Chrome,
                "window",
                p,
                Vec3::new(0.012, 0.095, 0.021),
                0.004,
            );
        }
        if f.shade == Shade::None || f.shade_coverage < 0.02 {
            continue;
        }
        part(
            a,
            tf,
            Surface::Metal,
            "blinds",
            Vec3::new(c.x, o.max.y + 0.005, 0.09),
            Vec3::new(size.x, 0.055, 0.058),
            0.003,
        );
        let height = (size.y - 0.08) * f.shade_coverage;
        if f.shade == Shade::Roller {
            part(
                a,
                tf,
                Surface::FabricAlt,
                "blinds",
                Vec3::new(c.x, o.max.y - 0.04 - height * 0.5, 0.092),
                Vec3::new(size.x - 0.03, height, 0.002),
                0.,
            );
            part(
                a,
                tf,
                Surface::Metal,
                "blinds",
                Vec3::new(c.x, o.max.y - 0.04 - height, 0.092),
                Vec3::new(size.x - 0.025, 0.016, 0.014),
                0.003,
            );
        } else {
            let count = (height / 0.064).floor().clamp(1., 72.) as usize;
            for i in 0..count {
                let p = tf.transform_point(Vec3::new(c.x, o.max.y - 0.07 - i as f32 * 0.064, 0.10));
                a.part(Surface::WoodEdge, "blinds").cuboid(
                    Vec3::new(size.x - 0.025, 0.003, 0.063),
                    0.,
                    Transform::from_translation(p)
                        .with_rotation(tf.rotation * Quat::from_rotation_x(f.shade_tilt)),
                );
            }
        }
    }
}

pub(crate) fn backdrop(a: &mut Assembly, scene: &IndoorManifest) {
    let Some(exterior) = &scene.exterior else {
        return legacy::backdrop(a, scene);
    };
    let Vec3 { x: w, z: d, .. } = scene.room_size;
    let (hx, hz) = (w * 0.5, d * 0.5);
    // Nonoverlapping ground strips: no coplanar duplicate at glazed corners and
    // no backdrop geometry inside the primary-room O-voxel region.
    let (left, right, back, front) = (-hx - 0.5, hx + 0.5, -hz - 0.5, hz + NEIGHBOR_DEPTH + 0.5);
    for (lo, hi) in [
        (
            Vec2::new(left - 18., back - 18.),
            Vec2::new(left, front + 18.),
        ),
        (
            Vec2::new(right, back - 18.),
            Vec2::new(right + 18., front + 18.),
        ),
        (Vec2::new(left, back - 18.), Vec2::new(right, back)),
        (Vec2::new(left, front), Vec2::new(right, front + 18.)),
    ] {
        a.box_part(
            Surface::Concrete,
            "floor",
            Vec3::new((lo.x + hi.x) * 0.5, -0.17, (lo.y + hi.y) * 0.5),
            Vec3::new(hi.x - lo.x, 0.20, hi.y - lo.y),
            0.,
        );
    }
    for f in &exterior.facades {
        let tf = f.side.transform(scene.room_size);
        let mut rng = stream(scene.seed, 3120 + f.side as u64);
        let span = f.side.span(scene.room_size);
        for i in 0..7 {
            let u = (i as f32 - 3.) * (span + 10.) / 6.;
            let height = rng.random_range(4.5..16.);
            let depth = rng.random_range(9.0..15.);
            let width = rng.random_range(2.4..4.4);
            part(
                a,
                tf,
                Surface::Concrete,
                "other_structure",
                Vec3::new(u, height * 0.5, -depth),
                Vec3::new(width, height, 3.6),
                0.012,
            );
            for level in 0..(height / 2.5) as usize {
                for col in [-1., 1.] {
                    part(
                        a,
                        tf,
                        Surface::Glass,
                        "window",
                        Vec3::new(
                            u + col * width * 0.23,
                            1.3 + level as f32 * 2.5,
                            -depth + 1.81,
                        ),
                        Vec3::new(width * 0.29, 1.15, 0.02),
                        0.,
                    );
                }
            }
        }
    }
}
