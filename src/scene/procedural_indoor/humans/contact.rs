//! Static surface support. Layout uses metric IK without loading a body model;
//! the existing Anny skinning pass resolves the actual palm/cloth thickness.
use super::super::{envelope::polygon, layout::IndoorObject, objects::tables};
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorktopContact {
    pub table: usize,
    /// Horizontal top and convex outline in the person's local coordinates.
    pub height: f32,
    pub outline: Vec<Vec2>,
    /// Left, right; the other arm retains its ordinary sampled pose.
    pub palms: [bool; 2],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContactReport {
    pub arm: usize,
    pub minimum_gap_metres: f32,
    pub palm_gap_metres: f32,
    pub cloth_compression_metres: f32,
    pub skin_compression_metres: f32,
    pub contact_vertices: usize,
}

pub(super) fn palm_region(joints: &[Vec3], arm: usize, p: Vec3, s: f32) -> bool {
    let wrist = joints[7 + arm * 4];
    let axis = (joints[8 + arm * 4] - wrist)
        .with_y(0.)
        .normalize_or(Vec3::NEG_Z);
    let delta = p - wrist;
    let along = delta.dot(axis);
    (0.012 * s..0.060 * s).contains(&along) && delta.dot(Vec3::Y.cross(axis)).abs() < 0.026 * s
}

impl WorktopContact {
    /// Cuff stitching is emitted after cloth fitting and has its own thickness.
    /// Compress only the millimetre seam relief at an already supported arm.
    pub(super) fn support_stitching(&self, mesh: &mut HumanAssembly, s: f32) {
        let Some(g) = mesh.parts.get_mut(&HumanSurface::Seam) else {
            return;
        };
        let mut changed = vec![false; g.positions.len()];
        for (i, vertex) in g.positions.iter_mut().enumerate() {
            let p = Vec3::from_array(*vertex);
            let delta = self.height + 0.0005 - p.y;
            if !(0.0..=0.006).contains(&delta) || !self.contains(p, 0.) {
                continue;
            }
            for report in &mut mesh.contacts {
                let elbow = mesh.local_joints[6 + report.arm * 4];
                let wrist = mesh.local_joints[7 + report.arm * 4];
                let v = wrist - elbow;
                let q =
                    elbow + v * ((p - elbow).dot(v) / v.length_squared().max(1e-8)).clamp(0., 1.);
                if p.distance(q) < 0.11 * s {
                    vertex[1] += delta;
                    changed[i] = true;
                    report.cloth_compression_metres = report.cloth_compression_metres.max(delta);
                    break;
                }
            }
        }
        if !changed.iter().any(|&v| v) {
            return;
        }
        let mut normals = vec![Vec3::ZERO; g.positions.len()];
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|v| v as usize);
            let p = |i| Vec3::from_array(g.positions[i]);
            let n = (p(b) - p(a)).cross(p(c) - p(a));
            for i in [a, b, c] {
                normals[i] += n;
            }
        }
        for (n, value) in g.normals.iter_mut().zip(normals) {
            *n = value.normalize_or(Vec3::from_array(*n)).to_array();
        }
    }
    pub(super) fn validate(&self, h: &IndoorHuman, scene: &IndoorManifest) -> Result<(), String> {
        let table = scene
            .objects
            .get(self.table)
            .ok_or("missing worktop support")?;
        let tf = h.transform().compute_affine().inverse() * table.transform().compute_affine();
        let outline: Vec<_> = tables::outline(table)
            .iter()
            .map(|p| tf.transform_point3(Vec3::new(p.x, 0., p.y)).xz())
            .collect();
        if !h.pose.seated()
            || !matches!(table.kind, ObjectKind::Desk | ObjectKind::Table)
            || !self.height.is_finite()
            || self.outline.len() != outline.len()
            || self
                .outline
                .iter()
                .zip(&outline)
                .any(|(a, b)| !a.is_finite() || a.distance(*b) > 1e-4)
            || (self.height + h.position.y - table.position.y - table.size.y).abs() > 1e-4
            || !self.palms.iter().any(|&p| p)
            || h.chair
                .and_then(|id| scene.objects.get(id))
                .and_then(|c| c.interaction_target)
                != Some(self.table)
        {
            return Err("invalid tabletop support constraint".into());
        }
        Ok(())
    }
    pub fn contains(&self, p: Vec3, margin: f32) -> bool {
        polygon::contains(&self.outline, p.xz(), margin)
    }

    /// Preserve both bone lengths. Choose the elbow swivel nearest gravity that
    /// stays above the support, instead of bending the elbow through the top.
    pub(super) fn arm(&self, joints: &mut [Vec3], arm: usize, wrist: Vec3, s: f32) {
        let base = 5 + arm * 4;
        let root = joints[base];
        let direction = (wrist - root).normalize_or(Vec3::NEG_Z);
        let distance = root.distance(wrist).clamp(0.0301 * s, 0.5499 * s);
        let along = ((0.29 * s).powi(2) + distance.powi(2) - (0.26 * s).powi(2)) / (2. * distance);
        let center = root + direction * along;
        let radius = ((0.29 * s).powi(2) - along.powi(2)).max(0.).sqrt();
        let up = (Vec3::Y - direction * direction.y).normalize_or(Vec3::X);
        let lateral = direction.cross(up).normalize_or(Vec3::Z);
        let cosine =
            ((self.height + 0.09 * s - center.y) / (radius * up.y).max(1e-6)).clamp(-0.95, 1.);
        let side = if arm == 0 { -1. } else { 1. };
        let elbow =
            center + radius * (up * cosine + lateral * side * (1. - cosine * cosine).sqrt());
        let wrist = root + direction * distance;
        let forward = (wrist - root).with_y(0.).normalize_or(Vec3::NEG_Z);
        joints[base + 1] = elbow;
        joints[base + 2] = wrist;
        joints[base + 3] = wrist + forward * 0.075 * s;
    }
}

pub(super) fn plan(h: &mut IndoorHuman, table: &IndoorObject, proposal: u64) {
    let inverse = h.transform().compute_affine().inverse();
    let tf = inverse * table.transform().compute_affine();
    let mut rng = stream(h.seed, 1807 + proposal);
    let active = rng.random_range(0..2);
    let both = rng.random_bool(0.65);
    let mut contact = WorktopContact {
        table: table.id,
        height: table.position.y + table.size.y - h.position.y,
        outline: tables::outline(table)
            .iter()
            .map(|p| tf.transform_point3(Vec3::new(p.x, 0., p.y)).xz())
            .collect(),
        palms: [false; 2],
    };
    let s = h.stature / 1.75;
    for arm in 0..2 {
        if arm != active && !both {
            continue;
        }
        let wrist = Vec3::new(
            (if arm == 0 { -1. } else { 1. }) * rng.random_range(0.17..0.29) * s,
            contact.height + 0.045 * s,
            -rng.random_range(0.27..0.43) * s,
        );
        let mut joints = h.joints.clone();
        contact.arm(&mut joints, arm, wrist, s);
        let base = 5 + 4 * arm;
        // The whole palm must be supported. Avoid promising contact with an
        // edge that an anatomically fixed-length arm cannot reach.
        if joints[base + 2].distance(wrist) < 0.001
            && joints[base].distance(wrist) < 0.51 * s
            && contact.contains(joints[base + 2], 0.045 * s)
            && contact.contains(
                joints[base + 3] + (joints[base + 3] - joints[base + 2]),
                0.045 * s,
            )
        {
            contact.palms[arm] = true;
            h.joints = joints;
        }
    }
    if contact.palms.iter().any(|&p| p) {
        h.worktop_contact = Some(contact);
    }
}
