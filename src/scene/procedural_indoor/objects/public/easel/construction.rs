use super::*;
use crate::scene::procedural_indoor::materials::canvas_stock;

const CLASS: &str = "easel";

struct Frame<'a> {
    a: &'a mut Assembly,
    cuts: TimberCuts,
    stock: Surface,
    transform: Transform,
}

impl Frame<'_> {
    fn member(&mut self, surface: Surface, local: Vec3, size: Vec3, axis: GrainAxis) {
        self.cuts.cuboid(
            self.a.part(surface, CLASS),
            surface,
            size,
            0.0016,
            self.transform * Transform::from_translation(local),
            axis,
        );
    }

    // The two endpoints are the centres of the sawn ends, including raked rear stays.
    fn rail(&mut self, start: Vec3, end: Vec3, width: f32, depth: f32) {
        let delta = end - start;
        let tf = Transform::from_translation((start + end) * 0.5)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()));
        self.cuts.cuboid(
            self.a.part(self.stock, CLASS),
            self.stock,
            Vec3::new(width, delta.length(), depth),
            0.0015,
            tf,
            GrainAxis::Y,
        );
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &Easel) {
    let tf = p.rail_frame(o);
    let w = p.stile_width_m;
    let d = p.stock_depth_m;
    let half = (p.base_width_m - w) * 0.5;
    let mast =
        (o.size.y - tf.translation.y - 0.5 * d * p.lean_radians.sin()) / p.lean_radians.cos();
    let foot = 0.037;
    let mut f = Frame {
        a,
        cuts: TimberCuts::new(o.finish_seed.unwrap_or(o.seed), 0),
        stock: p.frame_finish,
        transform: tf,
    };
    let at = |x: f32, h: f32| tf.transform_point(Vec3::new(x, h, 0.));
    let pad = |a: &mut Assembly, pt: Vec3| {
        a.box_part(
            Surface::Rubber,
            CLASS,
            pt.with_y(foot * 0.5),
            Vec3::new(w, foot, d * 1.18),
            0.002,
        )
    };
    match p.stand {
        Stand::Tripod => {
            let peak = mast * 0.78;
            for side in [-1., 1.] {
                let toe = at(side * half, 0.).with_y(foot);
                pad(f.a, toe);
                f.rail(toe, at(side * (w * 0.72), peak), w, d);
            }
            let heel = Vec3::new(0., foot, -o.size.z * 0.5 + d * 0.7);
            pad(f.a, heel);
            let hinge = at(0., peak) - tf.rotation * Vec3::Z * d;
            f.rail(heel, hinge, w * 0.94, d);
            // A spreader carries the central mast and limits the rear leg's opening.
            let level = 0.24 * mast;
            let spread = half + (w * 0.72 - half) * level / peak;
            f.member(
                f.stock,
                Vec3::new(0., level, -d * 0.85),
                Vec3::new(spread * 2. + w, w * 1.12, d * 0.70),
                GrainAxis::X,
            );
            let stay = heel.lerp(hinge, 0.27);
            f.a.part(Surface::Metal, CLASS).rod(
                at(0., level) - tf.rotation * Vec3::Z * d,
                stay,
                0.005,
            );
            f.member(
                Surface::Metal,
                Vec3::new(0., peak, -d * 0.12),
                Vec3::new(w * 2.7, w * 0.9, d * 0.6),
                GrainAxis::X,
            );
        }
        Stand::HFrame => {
            let rail_y = 0.050;
            for side in [-1., 1.] {
                let x = side * half;
                f.cuts.cuboid(
                    f.a.part(f.stock, CLASS),
                    f.stock,
                    Vec3::new(w, rail_y, o.size.z),
                    0.002,
                    Transform::from_xyz(x, rail_y * 0.5, 0.),
                    GrainAxis::Z,
                );
                let upright = at(x, mast * 0.76);
                f.rail(at(x, 0.).with_y(rail_y), upright, w, d);
                f.rail(
                    Vec3::new(x, rail_y, -o.size.z * 0.46),
                    at(x, mast * 0.57) - tf.rotation * Vec3::Z * d,
                    w * 0.64,
                    d * 0.68,
                );
            }
            for height in [mast * 0.19, mast * 0.72] {
                f.member(
                    f.stock,
                    Vec3::new(0., height, -d * 0.875),
                    Vec3::new(p.base_width_m - 0.008, w * 1.35, d * 0.75),
                    GrainAxis::X,
                );
            }
        }
    }
    // Through-blocks join the crossbars to the sliding central mast.
    let ties: &[f32] = match p.stand {
        Stand::Tripod => &[0.24, 0.78],
        Stand::HFrame => &[0.19, 0.72],
    };
    for h in ties {
        f.member(
            f.stock,
            Vec3::new(0., mast * h, 0.),
            Vec3::new(w * 1.12, w * 0.86, d),
            GrainAxis::X,
        );
    }
    let start = mast * 0.12;
    // The leading edge of the inclined mast is its highest corner in world space.
    let end = mast - d * p.lean_radians.tan();
    f.member(
        f.stock,
        Vec3::new(0., (start + end) * 0.5, d),
        Vec3::new(w * 1.38, end - start, d),
        GrainAxis::Y,
    );
    let back = p.canvas_back_z();
    let loaded = p.canvas != Canvas::Empty;
    let tray_width = if loaded {
        (p.canvas_width_m * 0.88).max(p.base_width_m * 0.65)
    } else {
        p.base_width_m * 0.77
    };
    let shelf_thickness = 0.020;
    f.member(
        f.stock,
        Vec3::new(
            0.,
            p.tray_position_m - shelf_thickness * 0.5,
            back + p.tray_depth_m * 0.5,
        ),
        Vec3::new(tray_width, shelf_thickness, p.tray_depth_m),
        GrainAxis::X,
    );
    // A raised outer lip holds brushes, leaving the canvas resting against the mast.
    f.member(
        f.stock,
        Vec3::new(0., p.tray_position_m + 0.006, back + p.tray_depth_m - 0.004),
        Vec3::new(tray_width, 0.012, 0.008),
        GrainAxis::X,
    );
    let upper = p.tray_position_m
        + if loaded {
            p.canvas_height_m
        } else {
            mast * 0.27
        };
    let throat = if loaded {
        p.canvas_depth_m + 0.006
    } else {
        0.038
    };
    f.member(
        f.stock,
        Vec3::new(0., upper + 0.014, back + throat * 0.5),
        Vec3::new(w * 2.4, 0.028, throat),
        GrainAxis::X,
    );
    // A positive spacer keeps the turning knob clear of the mast and the canvas.
    let axle = tf.transform_point(Vec3::new(w * 0.69, p.tray_position_m - 0.03, d));
    f.a.part(Surface::Metal, CLASS)
        .rod(axle, axle + Vec3::X * 0.020, 0.005);
    f.a.part(Surface::Rubber, CLASS)
        .rod(axle + Vec3::X * 0.020, axle + Vec3::X * 0.032, 0.014);
    if loaded {
        canvas(&mut f, p, o.id);
    }
}

fn canvas(f: &mut Frame, p: &Easel, id: usize) {
    let w = p.canvas_width_m;
    let h = p.canvas_height_m;
    let at = |x: f32, y: f32, z: f32| {
        f.transform
            .transform_point(Vec3::new(x, y + p.tray_position_m, z + p.canvas_back_z()))
    };
    let n = f.transform.rotation * Vec3::Z;
    let d = p.canvas_depth_m;
    let rect = [
        Vec2::new(-w * 0.5, 0.),
        Vec2::new(w * 0.5, 0.),
        Vec2::new(w * 0.5, h),
        Vec2::new(-w * 0.5, h),
    ];
    let world = |idx: usize, z: f32| {
        let xy = rect[idx];
        at(xy.x, xy.y, z)
    };
    let painting;
    let (front, key) = match p.canvas {
        Canvas::Painted => {
            painting = crate::scene::procedural_indoor::materials::art::studies::label(id);
            (Surface::Art, painting.as_str())
        }
        Canvas::Primed => (Surface::Paper, canvas_stock::PRIMED),
        Canvas::Linen => (Surface::Paper, canvas_stock::RAW),
        Canvas::Empty => return,
    };
    let uv = if p.canvas == Canvas::Painted {
        [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]
    } else {
        [[0., h], [w, h], [w, 0.], [0., 0.]]
    };
    f.a.part(front, key).vertex_quad(
        [world(0, d), world(1, d), world(2, d), world(3, d)],
        [n; 4],
        uv,
    );
    let g = f.a.part(Surface::Paper, canvas_stock::RAW);
    // Folded edges share the front perimeter. The linen back sits inside the stretcher,
    // so the timber has visible depth and only one sheet occupies each outer face.
    for i in 0..4 {
        let j = (i + 1) % 4;
        let points = [world(i, 0.), world(j, 0.), world(j, d), world(i, d)];
        let outward = (points[1] - points[0])
            .cross(points[3] - points[0])
            .normalize();
        let length = rect[j].distance(rect[i]);
        g.vertex_quad(
            points,
            [outward; 4],
            [[0., 0.], [length, 0.], [length, d], [0., d]],
        );
    }
    g.vertex_quad(
        [
            world(1, d - 0.002),
            world(0, d - 0.002),
            world(3, d - 0.002),
            world(2, d - 0.002),
        ],
        [-n; 4],
        [[w, h], [0., h], [0., 0.], [w, 0.]],
    );
    let rim = (w.min(h) * 0.085).clamp(0.026, 0.048);
    // Inset the stretcher from the folded fabric on every edge.
    let depth = d - 0.003;
    let z = p.canvas_back_z() + depth * 0.5;
    let mut stretcher = |pos, size, axis| {
        f.cuts.cuboid(
            f.a.part(Surface::Wood, "picture"),
            Surface::Wood,
            size,
            0.0016,
            f.transform * Transform::from_translation(pos),
            axis,
        )
    };
    for x in [-1., 1.] {
        let pos = Vec3::new(
            x * (w * 0.5 - rim * 0.5 - 0.002),
            p.tray_position_m + h * 0.5,
            z,
        );
        stretcher(pos, Vec3::new(rim, h - 0.004, depth), GrainAxis::Y);
    }
    for y in [rim * 0.5 + 0.002, h - rim * 0.5 - 0.002] {
        stretcher(
            Vec3::new(0., p.tray_position_m + y, z),
            Vec3::new(w - 2. * rim - 0.004, rim, depth),
            GrainAxis::X,
        );
    }
}
