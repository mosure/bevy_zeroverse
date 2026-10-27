//! Cabinet fronts fit sampled compartments; operated leaves retain ownership.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ShelfProgram) {
    let s = o.size;
    let t = p.panel_m;
    let l = o.kind.class_name();
    let face = p.body_depth_m * 0.5;
    let top = s.y - t - 0.004;
    if p.front == 4 {
        for y in [p.bottom_m + t * 0.5 + 0.005, top - 0.005] {
            a.box_part(
                Surface::Metal,
                l,
                Vec3::new(0., y, face - 0.010),
                Vec3::new(s.x - 2. * t - 0.004, 0.008, 0.012),
                0.001,
            );
        }
    }
    for (index, bay) in p.bays.iter().enumerate() {
        let width = bay.right - bay.left - 0.006;
        let mut x = (bay.left + bay.right) * 0.5;
        if p.front == 3 {
            let bottom = bay.levels[program::open_levels(p, bay)] + t * 0.5;
            if top <= bottom + 0.012 {
                continue;
            }
            let center = Vec3::new(x, (bottom + top) * 0.5, face - 0.023);
            a.box_part(
                p.fronts,
                l,
                center,
                Vec3::new(width, top - bottom, 0.020),
                0.003,
            );
            handle(a, p, center.with_z(face - 0.013), width);
            continue;
        }
        if bay.drawers {
            for v in bay.levels.windows(2) {
                let bottom = v[0] + 0.003;
                let upper = (v[1] - 0.003).min(top);
                let center = Vec3::new(x, (bottom + upper) * 0.5, face - 0.023);
                a.box_part(
                    p.fronts,
                    l,
                    center,
                    Vec3::new(width, upper - bottom, 0.020),
                    0.003,
                );
                handle(
                    a,
                    p,
                    Vec3::new(
                        x,
                        bottom + (upper - bottom) * p.handle_height_fraction,
                        face - 0.013,
                    ),
                    width,
                );
            }
            continue;
        }
        let seed = o.seed.wrapping_add(index as u64);
        // Sliding rails use the available run, not an assumed equal bay width.
        let sliding_travel = if p.front == 4 && index == 0 {
            (s.x * 0.5 - t - (bay.right - 0.003)).max(0.)
        } else {
            0.
        };
        x += sliding_travel * p.open_fraction;
        let recess = if p.front == 4 && index == 0 {
            0.027
        } else if p.front == 5 {
            0.010
        } else {
            0.
        };
        let bottom = p.bottom_m + t * 0.5 + 0.006;
        let height = top - bottom;
        let mut leaf = Assembly::default();
        let center = Vec3::new(x, (top + bottom) * 0.5, face - 0.023 - recess);
        leaf.box_part(p.fronts, l, center, Vec3::new(width, height, 0.020), 0.003);
        if p.front == 5 {
            let count = (width / p.slat_pitch_m).round().clamp(3., 24.) as usize;
            let pitch = width / count as f32;
            for i in 0..count {
                leaf.box_part(
                    p.fronts,
                    l,
                    Vec3::new(
                        x + (i as f32 + 0.5) * pitch - width * 0.5,
                        center.y,
                        face - 0.008 - recess,
                    ),
                    Vec3::new(pitch * p.slat_fill, height - 0.012, 0.010),
                    0.001,
                );
            }
        }
        // Attach hardware to the actual front skin (or to a slatted backing).
        let handle_face = face - 0.013 - recess;
        handle(
            &mut leaf,
            p,
            Vec3::new(
                x + width * 0.28,
                bottom + height * p.handle_height_fraction,
                handle_face,
            ),
            width.min(0.5),
        );
        let hinge = Vec3::new(x - width * 0.5 + 0.005, 0., face - 0.012 - recess);
        let q = Quat::from_rotation_y(-p.opening_radians);
        for g in leaf.parts.values_mut() {
            for v in &mut g.positions {
                *v = (hinge + q * (Vec3::from_array(*v) - hinge)).to_array();
            }
            for n in &mut g.normals {
                *n = (q * Vec3::from_array(*n)).to_array();
            }
        }
        use super::super::super::motion::{construction as motion, MotionKind};
        if a.motion_selected(MotionKind::Cabinet, seed) {
            let (angle, travel) = if p.front == 4 {
                (
                    0.,
                    Vec3::X
                        * sliding_travel
                        * (motion::endpoint(seed, p.open_fraction, 0.85) - p.open_fraction),
                )
            } else {
                (
                    -(motion::endpoint(seed, p.opening_radians, 1.15) - p.opening_radians),
                    Vec3::ZERO,
                )
            };
            if angle.abs() + travel.length() > 0.001 {
                let mut joint =
                    motion::rigid(MotionKind::Cabinet, seed, hinge, Vec3::Y, angle, travel);
                joint.frame = Transform::from_translation(hinge).with_rotation(q);
                a.articulated(leaf, joint);
                continue;
            }
        }
        a.append_assembly(leaf);
    }
}

fn handle(a: &mut Assembly, p: &ShelfProgram, center: Vec3, span: f32) {
    let label = "cabinet_hardware";
    let width = (span * p.handle_span_fraction).min(span * 0.40);
    let reach = if p.front == 5 { 0.020 } else { 0.010 };
    match p.handle {
        0 => {
            for side in [-1., 1.] {
                let base = center + Vec3::X * width * side * 0.5;
                a.part(Surface::Metal, label)
                    .rod(base, base + Vec3::Z * reach, 0.0025);
            }
            a.part(Surface::Metal, label).rod(
                center - Vec3::X * width * 0.5 + Vec3::Z * reach,
                center + Vec3::X * width * 0.5 + Vec3::Z * reach,
                0.0025,
            );
        }
        1 => {
            a.part(Surface::Chrome, label)
                .rod(center, center + Vec3::Z * (reach - 0.003), 0.003);
            a.part(Surface::Chrome, label).ellipsoid(
                Vec3::new(0.008, 0.008, 0.005),
                Transform::from_translation(center + Vec3::Z * (reach - 0.003)),
            );
        }
        2 => a.box_part(
            Surface::Ink,
            label,
            center + Vec3::Z * 0.0006,
            Vec3::new(width, 0.009, 0.001),
            0.001,
        ),
        _ => {}
    }
}
