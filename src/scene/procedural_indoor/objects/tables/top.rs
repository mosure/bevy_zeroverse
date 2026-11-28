//! Sheet faces and their edge treatment follow the same support outline.
use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish;
use std::borrow::Cow;

pub(super) enum TopStock {
    Timber(TimberProgram),
    Sheet {
        label: Cow<'static, str>,
        cut: SheetCut,
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
    pub(super) fn sample(o: &IndoorObject, p: &TableProgram, version: u32) -> Self {
        if p.top_surface == Surface::Wood {
            Self::Timber(TimberProgram::sample(product_seed(o)))
        } else {
            Self::Sheet {
                label: worktop_finish::table_label(o, p.top_surface),
                cut: SheetCut::for_version(o, p.top_surface, version),
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
            Self::Sheet { label, cut } => {
                let cut = cut.for_leaf(o.seed, leaf);
                let slab = a.part(p.top_surface, label);
                let profile = slab_profile(o, p, outline);
                let rotation = Mat2::from_angle(cut.heading_rad);
                for (sign, origin) in [(-0.5, cut.back_origin_m), (0.5, cut.face_origin_m)] {
                    slab.slab_cap(
                        outline,
                        profile.1,
                        Vec2::ZERO,
                        sign * p.top_thickness,
                        &tf,
                        |v| rotation * (v + tf.translation.xz()) + origin,
                    );
                }
                let edge = if let Some(table) = p.classroom {
                    a.part(table.band, o.kind.class_name())
                } else if cut.edge == SheetEdge::TimberLip {
                    a.part(Surface::Wood, o.kind.class_name())
                } else {
                    slab
                };
                cut.edge_uvs(edge, outline, p.top_thickness, profile, &tf, rotation);
            }
        }
    }

    /// Separate substrate below the worktop. Its underside is visible from low cameras.
    pub(super) fn backing(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        outline: &[Vec2],
        underside: f32,
        surface: Surface,
        version: u32,
    ) {
        let tf = Transform::from_xyz(0., underside - 0.006, 0.);
        match self {
            Self::Timber(_) => a
                .part(Surface::WoodEdge, o.kind.class_name())
                .profile_slab(outline, 0.012, 0.003, tf),
            Self::Sheet { label, cut } => {
                let part = a.part(surface, label);
                if version < REGISTERED_SHEET_VERSION {
                    part.profile_slab(outline, 0.012, 0.003, tf);
                    return;
                }
                let profile = Geometry::slab_profile(outline, 0.012, 0.003);
                let rotation = Mat2::from_angle(cut.heading_rad);
                for y in [-0.006, 0.006] {
                    part.slab_cap(outline, profile.1, Vec2::ZERO, y, &tf, |p| {
                        rotation * p + cut.back_origin_m
                    });
                }
                part.slab_edge_with_uv(outline, 0.012, profile, &tf, |_, along, rings| {
                    rings.map(|(y, _)| Vec2::new(along, y) + cut.band_origin_m)
                });
            }
        }
    }
}
