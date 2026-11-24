//! Loose throws are cut and filled independently from fixed upholstery. Each
//! accepted pose rests on the installed seat and can bear against a real back pad.
use super::*;
use crate::scene::procedural_indoor::materials::variants;
mod cut;
pub(super) mod legacy;
mod placement;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 232;

#[derive(Debug, Clone, Copy, serde::Serialize)]
struct Cloth {
    surface: Surface,
    slot: usize,
}
impl Cloth {
    fn label(self) -> &'static str {
        const LABELS: [&str; variants::COUNT] = [
            "pillow#finish0#uv_local",
            "pillow#finish1#uv_local",
            "pillow#finish2#uv_local",
            "pillow#finish3#uv_local",
            "pillow#finish4#uv_local",
            "pillow#finish5#uv_local",
            "pillow#finish6#uv_local",
            "pillow#finish7#uv_local",
            "pillow#finish8#uv_local",
            "pillow#finish9#uv_local",
            "pillow#finish10#uv_local",
            "pillow#finish11#uv_local",
        ];
        LABELS[self.slot]
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
struct Stock {
    face: Cloth,
    reverse: Cloth,
    edge: Cloth,
    // Cuts from different areas of the bolt keep checks and stripes in the
    // same scale without centering the identical motif on every cushion.
    face_origin_m: Vec2,
    reverse_origin_m: Vec2,
    edge_origin_m: Vec2,
}

impl Stock {
    fn sample(o: &IndoorObject, p: &SofaProgram, index: usize) -> Self {
        let owner = o.finish_seed.unwrap_or(o.seed);
        let mut group = stream(owner, u64::from_le_bytes(*b"SofaDyes"));
        let matched = Cloth {
            surface: p.upholstery,
            slot: variants::object_slot(p.upholstery, o.seed, o.finish_seed),
        };
        let accent = Cloth {
            surface: if group.random_bool(0.73) {
                Surface::FabricAlt
            } else {
                Surface::Fabric
            },
            slot: group.random_range(0..variants::COUNT),
        };
        let mut r = stream(owner, u64::from_le_bytes(*b"ThrowDye") ^ index as u64);
        let face = match r.random_range(0..100) {
            0..=24 => matched,
            25..=73 => accent,
            _ => Cloth {
                surface: accent.surface,
                slot: (accent.slot + r.random_range(1..variants::COUNT)) % variants::COUNT,
            },
        };
        let reverse = if r.random_bool(0.26) { matched } else { face };
        let edge = if r.random_bool(0.46) {
            if face.slot == accent.slot {
                matched
            } else {
                accent
            }
        } else {
            face
        };
        let mut bolt = stream(o.seed, u64::from_le_bytes(*b"PanelCut") ^ index as u64);
        let mut origin = || Vec2::new(bolt.random_range(0.0..2.0), bolt.random_range(0.0..3.0));
        Self {
            face,
            reverse,
            edge,
            face_origin_m: origin(),
            reverse_origin_m: origin(),
            edge_origin_m: origin(),
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &SofaProgram,
    widths: &[f32],
    version: u32,
) {
    if !p.pillows || o.kind != ObjectKind::Sofa {
        return;
    }
    if version < INTRODUCED {
        for g in legacy::build(o, p, widths) {
            a.part(Surface::FabricAlt, "pillow").append(g);
        }
        return;
    }
    placement::install(o, p, widths, |installed, form| {
        form.sew(
            a,
            installed.stock,
            Transform::from_translation(installed.center_m).with_rotation(installed.rotation),
        );
    });
}
pub(crate) fn metadata(
    o: &IndoorObject,
    p: &SofaProgram,
    widths: &[f32],
) -> Vec<placement::Installed> {
    let mut items = Vec::new();
    if p.pillows && o.kind == ObjectKind::Sofa {
        placement::install(o, p, widths, |item, _| items.push(item));
    }
    items
}

impl placement::Installed {
    pub(crate) fn edge_name(&self) -> &'static str {
        match self.cut.edge {
            cut::Edge::Knife => "knife",
            cut::Edge::Welt => "welt",
            cut::Edge::Boxed => "boxed",
            cut::Edge::Flanged => "flanged",
        }
    }
    pub(crate) fn width_m(&self) -> f32 {
        self.cut.size.x
    }
}
