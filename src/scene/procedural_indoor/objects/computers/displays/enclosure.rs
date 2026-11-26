//! One closed display housing. The active face, bezel, case edge and rear
//! tooling shoulder meet at shared edges, including on bent panels.
use super::*;
mod bend;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum GlassFit {
    Flush,
    Inset,
    Proud,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct PanelSection {
    pub glass_fit: GlassFit,
    /// Signed distance from the front of the bezel to the optical face.
    pub glass_offset_m: f32,
    pub lip_m: f32,
    pub edge_bevel_m: f32,
    pub corner_m: f32,
    pub front_shell_fraction: f32,
    /// Fraction of the rear ledge formed into a slope around the electronics.
    pub shoulder_fraction: f32,
}

impl PanelSection {
    pub fn sample(o: &IndoorObject, d: &DisplayGeometry) -> Self {
        let mut stock = stream(o.seed, u64::from_le_bytes(*b"PanelDie"));
        let glass_fit = match stock.random_range(0..10) {
            0..=3 => GlassFit::Flush,
            4..=8 => GlassFit::Inset,
            _ => GlassFit::Proud,
        };
        let glass_offset_m = match glass_fit {
            GlassFit::Flush => 0.,
            GlassFit::Inset => -stock.random_range(0.0007_f32..0.0038).min(d.panel_m * 0.23),
            GlassFit::Proud => stock.random_range(0.00035..0.00125),
        };
        Self {
            glass_fit,
            glass_offset_m,
            lip_m: if glass_fit == GlassFit::Flush {
                0.
            } else {
                stock.random_range(0.0006_f32..0.0024).min(d.bezel_m * 0.24)
            },
            edge_bevel_m: stock
                .random_range(0.00065_f32..0.0032)
                .min(d.bezel_m * 0.26)
                .min(d.panel_m * 0.12),
            corner_m: stock.random_range(0.0015_f32..0.012).min(d.bezel_m * 0.82),
            front_shell_fraction: stock.random_range(0.32..0.68),
            shoulder_fraction: if stock.random_bool(0.30) {
                0.
            } else {
                stock.random_range(0.12..0.94)
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Rim {
    half: Vec2,
    center_y: f32,
    z: f32,
    corner: f32,
}

impl Rim {
    fn outline(self) -> [Vec3; 8] {
        let Vec2 { x, y } = self.half;
        let k = self.corner;
        [
            (-x + k, -y),
            (x - k, -y),
            (x, -y + k),
            (x, y - k),
            (x - k, y),
            (-x + k, y),
            (-x, y - k),
            (-x, -y + k),
        ]
        .map(|(x, y)| Vec3::new(x, self.center_y + y, self.z))
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &DisplayProgram,
    d: &DisplayGeometry,
) -> PanelSection {
    let section = PanelSection::sample(o, d);
    let half = Vec2::new(d.width_m, d.height_m) * 0.5;
    let front = d.panel_m * 0.5;
    let edge = section.edge_bevel_m;
    let chin = d.height_m * p.chin_fraction * 0.5;
    let optical = Rim {
        half: half - Vec2::new(d.bezel_m, d.bezel_m + chin),
        center_y: chin,
        z: front + section.glass_offset_m,
        corner: section.corner_m.min(d.bezel_m * 0.16),
    };
    let outer = Rim {
        half,
        center_y: 0.,
        z: front - edge,
        corner: section.corner_m,
    };
    let rear = Rim {
        half: half * p.back_fraction,
        z: -front,
        ..outer
    };
    let shell_back = front - d.panel_m * section.front_shell_fraction;
    let shoulder = Rim {
        half: rear.half.lerp(half, section.shoulder_fraction),
        z: shell_back,
        ..rear
    };
    let rim = Rim {
        half: half - Vec2::splat(edge),
        z: front,
        corner: (section.corner_m - edge * 0.5).max(section.corner_m * 0.5),
        ..outer
    };
    let lip = Rim {
        half: optical.half + Vec2::splat(section.lip_m),
        corner: optical.corner + section.lip_m * 0.5,
        z: front,
        ..optical
    };
    let bend = bend::Bend::new(d, &[optical, lip, rim, outer, shoulder, rear]);
    let label = o.kind.class_name();
    let image = if o.kind == ObjectKind::Display {
        Surface::Television
    } else {
        Surface::Screen
    };
    let mut inner = optical.outline();
    bend.face(a.part(image, label), &inner, Vec3::Z, true);
    let shell = a.part(p.casing, label);
    let mut connect = |section: Rim| {
        let next = section.outline();
        for i in 0..inner.len() {
            let j = (i + 1) % inner.len();
            let q = [inner[i], next[i], next[j], inner[j]];
            let normal = (q[1] - q[0]).cross(q[3] - q[0]).normalize();
            bend.face(shell, &q, normal, false);
        }
        inner = next;
    };
    if section.glass_fit != GlassFit::Flush {
        connect(lip);
    }
    connect(rim);
    connect(outer);
    connect(Rim {
        z: shell_back,
        ..outer
    });
    connect(shoulder);
    connect(rear);
    inner.reverse();
    bend.face(shell, &inner, Vec3::NEG_Z, false);
    section
}
