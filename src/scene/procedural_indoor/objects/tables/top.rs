//! Sheet faces and their edge treatment follow the same support outline.
use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish;
use std::borrow::Cow;

pub(super) enum TopStock {
    Timber(TimberProgram),
    Sheet {
        label: Cow<'static, str>,
        timber_lip: bool,
        cut_origin: Vec2,
    },
}

pub(super) fn slab_profile(o: &IndoorObject, p: &TableProgram, outline: &[Vec2]) -> (f32, f32) {
    if o.variant != SHARED_ROW_VARIANT {
        return Geometry::slab_profile(outline, p.top_thickness, p.edge_bevel);
    }
    // Cut leaves can have a long straight joint opposite a densely sampled curved end.
    // Limit the actual displacement of every top vertex to the tool's bevel width.
    // A scale based on the closest vertex can open a centimetre-wide groove here.
    let bevel = p.edge_bevel.min(p.top_thickness * 0.45);
    let radius = outline
        .iter()
        .map(|v| v.length_squared())
        .fold(0., f32::max)
        .sqrt();
    (bevel, 1. - bevel / radius.max(2. * bevel))
}

impl TopStock {
    pub(super) fn sample(o: &IndoorObject, p: &TableProgram) -> Self {
        if p.top_surface == Surface::Wood {
            Self::Timber(TimberProgram::sample(product_seed(o)))
        } else {
            Self::Sheet {
                label: worktop_finish::table_label(o, p.top_surface),
                timber_lip: p.top_surface == Surface::Plastic
                    && stream(o.finish_seed.unwrap_or(o.seed), 0x4544_4745_4c49_5053)
                        .random_bool(0.36),
                cut_origin: if o.variant == SHARED_ROW_VARIANT {
                    let mut r = stream(o.seed, u64::from_le_bytes(*b"SheetCut"));
                    Vec2::new(r.random_range(0.0..2.4), r.random_range(0.0..2.4))
                } else {
                    Vec2::ZERO
                },
            }
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &TableProgram,
        outline: &[Vec2],
        tf: Transform,
        leaf: usize,
    ) {
        match self {
            Self::Timber(wood) => wood.build(
                a.part(Surface::Wood, o.kind.class_name()),
                o,
                p,
                outline,
                tf,
                leaf,
            ),
            Self::Sheet {
                label,
                timber_lip,
                cut_origin,
            } => {
                let slab = a.part(p.top_surface, label);
                let profile = slab_profile(o, p, outline);
                // A modular top is cut in local leaf coordinates. Keep the print
                // in table coordinates so every leaf does not repeat its centre.
                for y in [-0.5, 0.5] {
                    slab.slab_cap(
                        outline,
                        profile.1,
                        Vec2::ZERO,
                        y * p.top_thickness,
                        &tf,
                        |v| v + tf.translation.xz() + *cut_origin,
                    );
                }
                if *timber_lip {
                    let edge = a.part(Surface::Wood, o.kind.class_name());
                    let start = edge.uvs.len();
                    edge.slab_edge(outline, p.top_thickness, profile, &tf);
                    for uv in &mut edge.uvs[start..] {
                        let [along, height] = *uv;
                        *uv = [height, -along];
                    }
                } else {
                    slab.slab_edge(outline, p.top_thickness, profile, &tf);
                }
            }
        }
    }
}
