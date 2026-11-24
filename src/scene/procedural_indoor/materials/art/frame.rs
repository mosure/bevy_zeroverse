//! Metric stock and aperture specification shared by the artwork mesh and audits.
use super::*;
use crate::scene::procedural_indoor::layout::IndoorObject;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MouldingShape {
    Flat,
    Bevel,
    Cove,
    Roundover,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Moulding {
    pub shape: MouldingShape,
    /// Drop from the highest rail face to its inside lip.
    pub rise_m: f32,
    pub arris_m: f32,
    pub curvature: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct FrameProgram {
    /// 0: stretched canvas; 1/4: timber; 2: black; 3/5: metal frames.
    pub style: u32,
    pub border_width: f32,
    pub mat_width: f32,
    pub shadow_gap: f32,
    pub picture_inset: f32,
    pub stepped: bool,
    pub outer_fraction: f32,
    pub recess_fraction: f32,
    pub mat_scale: Vec2,
    pub picture_bias: Vec2,
    pub moulding: Option<Moulding>,
    pub mat_thickness_m: f32,
    pub mat_bevel_m: f32,
    pub float_depth_m: f32,
    /// Printed edge continuation on a stretched or floated work.
    pub image_wrap: bool,
}
pub fn frame_parameters(o: &IndoorObject) -> FrameProgram {
    let mut rng = stream(o.seed, 3094);
    let style = o.variant % 6;
    let mut section = stream(o.seed, 3095);
    let mat_scale = Vec2::new(
        section.random_range(0.65..1.35),
        section.random_range(0.65..1.50),
    );
    let picture_bias = Vec2::new(
        section.random_range(-0.55..0.55),
        section.random_range(-0.35..0.65),
    );
    let stepped = style != 0 && section.random_bool(0.56);
    let outer_fraction = section.random_range(0.30..0.70);
    let recess_fraction = section.random_range(0.18..0.62);
    let short = o.size.x.min(o.size.y);
    let border_width = if style == 0 {
        0.
    } else if style == 5 {
        rng.random_range(0.006..0.013_f32).min(short * 0.06)
    } else {
        rng.random_range(0.008..0.04_f32).min(short * 0.095)
    };
    let mat_width = if style == 0 || style == 4 || rng.random_bool(0.36) {
        0.
    } else {
        rng.random_range(0.010..0.13_f32).min(short * 0.18)
    };
    let shadow_gap = if style == 4 {
        rng.random_range(0.008..0.023_f32).min(short * 0.06)
    } else {
        0.
    };
    let mut mount = stream(o.seed, 0x5049_4354_4d4f_554e);
    let picture_inset = if style == 0 {
        0.
    } else {
        let shallow = rng.random_range(0.003..0.008_f32).min(o.size.z * 0.20);
        let deep = (o.size.z * 0.62).min(0.048);
        shallow + (deep - shallow).max(0.) * mount.random::<f32>().powi(2)
    };
    let mat_thickness_m = if mat_width > 0. {
        mount
            .random_range(0.0011..0.0036_f32)
            .min(picture_inset * 0.57)
    } else {
        0.
    };
    let narrowest_margin = (mat_width * mat_scale * (Vec2::ONE - picture_bias.abs())).min_element();
    let mat_bevel_m =
        (mat_thickness_m * mount.random_range(0.68..1.34)).min(narrowest_margin * 0.48);
    let float_depth_m = if style == 4 {
        ((o.size.z - picture_inset) * mount.random_range(0.22..0.79)).min(0.046)
    } else {
        0.
    };
    let moulding = (style != 0).then(|| {
        let shape = match mount.random_range(0..8) {
            0 | 1 => MouldingShape::Flat,
            2 | 3 => MouldingShape::Bevel,
            4 | 5 if style != 5 => MouldingShape::Cove,
            6 | 7 if style != 5 => MouldingShape::Roundover,
            _ => MouldingShape::Bevel,
        };
        Moulding {
            shape,
            rise_m: (o.size.z * recess_fraction)
                .min((picture_inset - mat_thickness_m) * 0.86)
                .min(border_width * mount.random_range(0.34..0.98)),
            arris_m: mount
                .random_range(0.0005..0.0026_f32)
                .min(border_width * 0.12)
                .min(o.size.z * 0.06),
            curvature: mount.random_range(1.52..2.80),
        }
    });
    FrameProgram {
        style,
        border_width,
        mat_width,
        shadow_gap,
        picture_inset,
        stepped,
        outer_fraction,
        recess_fraction,
        mat_scale,
        picture_bias,
        mat_thickness_m,
        mat_bevel_m,
        float_depth_m,
        moulding,
        image_wrap: matches!(style, 0 | 4) && mount.random_bool(0.58),
    }
}
