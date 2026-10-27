//! Projection screen construction and the exact optical target surface.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScreenProgram {
    pub framed: bool,
    pub drop_fraction: f32,
    pub case_height_m: f32,
    pub case_depth_m: f32,
    pub cap_width_m: f32,
    pub sheet_width_m: f32,
    pub sheet_center_x_m: f32,
    pub sheet_bottom_m: f32,
    pub sheet_top_m: f32,
    pub sheet_front_m: f32,
    pub bar_radius_m: f32,
    pub bar_width_m: f32,
    pub surface_lo: Vec3,
    pub surface_hi: Vec3,
    pub housing: Surface,
    pub trim: Surface,
    pub mount_x_m: [f32; 2],
}
impl ScreenProgram {
    pub fn target_local(&self) -> Vec3 {
        (self.surface_lo + self.surface_hi) * 0.5
    }
    pub fn image_width_m(&self) -> f32 {
        self.surface_hi.x - self.surface_lo.x
    }
}

pub fn parameters(o: &IndoorObject) -> ScreenProgram {
    let s = o.size;
    let mut r = stream(o.seed, 0x50524f4a5343524e);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5343524e46494e49);
    let framed = r.random_bool(0.32);
    let drop = if framed {
        1.
    } else {
        r.random_range(0.52..0.995)
    };
    let case_height = r
        .random_range(0.032_f32..0.080)
        .min(s.z * 0.85)
        .min(s.y * 0.15);
    let case_depth = s.z * r.random_range(0.60..0.88);
    let cap = (s.x * r.random_range(0.018..0.055)).min(0.075);
    let width = if framed {
        s.x
    } else {
        (s.x - 2. * cap) * r.random_range(0.87..0.99)
    };
    let x = if framed {
        0.
    } else {
        (s.x - 2. * cap - width) * r.random_range(-0.44..0.44)
    };
    let top = if framed {
        s.y
    } else {
        // The roll exits inside its housing; a gap here would leave the
        // hanging sheet visibly unsupported.
        s.y - case_height + 0.001
    };
    let bottom = top * (1. - drop);
    let bar = r.random_range(0.006_f32..0.014).min(s.z * 0.14);
    // End caps extend beyond the sheet rather than competing with its side
    // faces on the same plane. Reserve both ends inside the placement envelope.
    let bar_width = width
        + 2. * r
            .random_range(0.004_f32..0.014)
            .min((s.x - width) * 0.5 - x.abs() - 0.001)
            .max(0.);
    let front = if framed {
        -s.z * 0.5 + s.z * r.random_range(0.18..0.42)
    } else {
        case_depth * r.random_range(0.05..0.30)
    };
    let side = width * r.random_range(0.012..0.050);
    let lower = (top - bottom) * r.random_range(0.018..0.065) + if framed { 0. } else { 2. * bar };
    let upper = (top - bottom) * r.random_range(0.015..0.055);
    ScreenProgram {
        framed,
        drop_fraction: drop,
        case_height_m: case_height,
        case_depth_m: case_depth,
        cap_width_m: cap,
        sheet_width_m: width,
        sheet_center_x_m: x,
        sheet_bottom_m: bottom,
        sheet_top_m: top,
        sheet_front_m: front,
        bar_radius_m: bar,
        bar_width_m: bar_width,
        surface_lo: Vec3::new(x - width * 0.5 + side, bottom + lower, front + 0.0008),
        surface_hi: Vec3::new(x + width * 0.5 - side, top - upper, front + 0.0008),
        housing: [Surface::Metal, Surface::Plastic, Surface::WoodEdge][f.random_range(0..3)],
        trim: [Surface::Metal, Surface::Chrome, Surface::Plastic][f.random_range(0..3)],
        mount_x_m: [
            -s.x * r.random_range(0.23..0.43),
            s.x * r.random_range(0.23..0.43),
        ],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    if p.framed {
        let depth = p.sheet_front_m + s.z * 0.5;
        a.box_part(
            p.housing,
            l,
            Vec3::new(0., s.y * 0.5, -s.z * 0.5 + depth * 0.5),
            Vec3::new(s.x, s.y, depth),
            0.003,
        );
    } else {
        let h = p.case_height_m;
        let depth = p.case_depth_m;
        let cap = p.cap_width_m;
        let y = s.y - h * 0.5;
        a.box_part(
            p.housing,
            l,
            Vec3::new(0., y, 0.),
            Vec3::new(s.x - cap * 2., h, depth),
            h * 0.14,
        );
        for side in [-1., 1.] {
            a.box_part(
                p.trim,
                l,
                Vec3::new(side * (s.x - cap) * 0.5, y, 0.),
                Vec3::new(cap, h, depth),
                cap.min(h) * 0.12,
            );
        }
        for x in p.mount_x_m {
            a.box_part(
                p.trim,
                l,
                Vec3::new(x, y, (-s.z - depth) * 0.25),
                Vec3::new(cap.min(0.045), h * 0.65, (s.z - depth) * 0.5),
                0.001,
            );
        }
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(
                p.sheet_center_x_m,
                (p.sheet_bottom_m + p.sheet_top_m) * 0.5,
                p.sheet_front_m - 0.0015,
            ),
            Vec3::new(p.sheet_width_m, p.sheet_top_m - p.sheet_bottom_m, 0.003),
            0.0005,
        );
        a.part(p.trim, l).cylinder(
            p.bar_radius_m,
            p.bar_width_m,
            Transform::from_xyz(
                p.sheet_center_x_m,
                p.sheet_bottom_m + p.bar_radius_m,
                p.sheet_front_m - 0.0015,
            )
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
    }
    face(a, Surface::Paper, l, p.surface_lo, p.surface_hi);
}
