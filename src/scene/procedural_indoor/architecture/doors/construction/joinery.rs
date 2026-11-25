//! Stock dimensions in leaf coordinates. Opposed fittings share the same fixing line.
use super::*;
use crate::scene::procedural_indoor::materials::glass::SheetGauge;

pub(super) struct Joinery {
    pub(super) track: Option<sliding::Track>,
    vision_width: f32,
    vision_height: f32,
    vision_head: f32,
    vision_shift: f32,
    pub(super) bead: f32,
    pub(super) seal: f32,
    pub(super) glass: SheetGauge,
    pub(super) inset: f32,
    pub(super) shoulder: f32,
    pub(super) recess: f32,
    pub(super) bays: usize,
    pub(super) balance: f32,
    pub(super) pitch: f32,
    pub(super) slope: f32,
    pub(super) gauge: f32,
    pub(super) lever: f32,
    pub(super) projection: f32,
    pub(super) rose: f32,
    pub(super) round_rose: bool,
    pub(super) pull: f32,
    pub(super) pocket: Vec2,
}

impl Joinery {
    pub(super) fn stock_seed(p: &DoorProgram) -> u64 {
        // These are immutable, continuously sampled stock dimensions already
        // present in archived manifests. Changing pose or mount cannot resample a leaf.
        ((p.thickness.to_bits() as u64) << 32) | u64::from(p.stile.to_bits())
    }

    pub(super) fn sample(p: &DoorProgram) -> Self {
        let mut r = stream(Self::stock_seed(p), 0x444f_4f52_4c45_4146);
        let narrow = r.random::<f32>();
        Self {
            track: sliding::Track::for_program(p),
            vision_width: 0.16 + 0.84 * narrow * narrow,
            vision_height: r.random_range(0.22..0.73),
            vision_head: r.random_range(0.015..0.19),
            vision_shift: r.random_range(-0.30..0.85),
            bead: r.random_range(0.009..0.024),
            seal: r.random_range(0.0022..0.0048),
            glass: if !r.random_bool(0.46) || p.mechanism == DoorMechanism::Revolving {
                SheetGauge::Eight
            } else {
                SheetGauge::Six
            },
            inset: r.random_range(-0.002..0.002),
            shoulder: r.random_range(0.012..0.040),
            recess: r.random_range(0.004..0.010),
            bays: r.random_range(2..=4),
            balance: r.random_range(-0.45..0.45),
            pitch: r.random_range(0.046..0.092),
            slope: r.random_range(0.36..0.78),
            gauge: r.random_range(0.004..0.009),
            lever: r.random_range(0.095..0.143),
            projection: r.random_range(0.040..0.055),
            rose: r.random_range(0.019..0.028),
            round_rose: r.random_bool(0.56),
            pull: r.random_range(0.115..0.23),
            pocket: Vec2::new(r.random_range(0.024..0.043), r.random_range(0.095..0.158)),
        }
    }
}

pub(super) struct Fit {
    pub(super) side: f32,
    pub(super) stile: f32,
    pub(super) mount: Vec2,
    pub(super) bar_anchors: [f32; 2],
    pub(super) opening: Option<Rect>,
    pub(super) pocket: Rect,
    veneer: Option<[bevy::math::Affine2; 2]>,
}

impl Fit {
    pub(super) fn new(d: &DoorInstallation, leaf: Leaf, index: usize, j: &Joinery) -> Self {
        let p = &d.program;
        let half = leaf.size.truncate() * 0.5;
        let side = if p.mechanism == DoorMechanism::Paired {
            if index == 0 {
                -1.
            } else {
                1.
            }
        } else {
            p.handedness
        };
        // Retain enough solid timber for the shoes and the frame's stop on
        // the closed, non-swing face. A narrow lite cannot occupy a fixing.
        let stile = p
            .swing_hanging
            .map_or(p.stile, |h| {
                p.stile.max(2. * (h.stop_projection_m + j.rose + 0.009))
            })
            .min(half.x * 0.36);
        let heel_stile = if p.mechanism == DoorMechanism::Revolving {
            // At the hub two push bars meet at right angles. Place the inner
            // fixing far enough out that even the shoe clears the next wing.
            let half_shoe = j.rose.min(stile * 0.40) * 0.83;
            let clearance = leaf.size.z * 0.5 + j.projection + 0.009 + 0.006;
            stile.max(2. * (clearance - ROTOR_HEEL_RADIUS_M + half_shoe))
        } else {
            stile
        };
        let bar_anchors = [-half.x + heel_stile * 0.5, half.x - stile * 0.5];
        let framed = p.panel != DoorPanel::Flush;
        let setback = if framed {
            stile * 0.5
        } else {
            p.stile.clamp(0.05, 0.075).max(stile * 0.5)
        };
        let mount = Vec2::new(-side * (half.x - setback), p.handle_height - half.y - 0.012);
        let pocket_size = Vec2::new(j.pocket.x.min(setback * 1.48), j.pocket.y);
        let pocket = Rect::from_center_size(mount, pocket_size);
        let opening = framed.then(|| {
            let extent = half - Vec2::splat(stile);
            let field = Rect::from_corners(Vec2::new(-half.x + heel_stile, -extent.y), extent);
            if p.panel != DoorPanel::VisionLite {
                return field;
            }
            if let Some(vision) = p.vision_light {
                return vision.fit(field, -half.y, side);
            }
            let size = field.size() * Vec2::new(j.vision_width, j.vision_height);
            let free = field.size() - size;
            // Tall narrow lights may be toward the latch. Broader lights tend to
            // fill the upper field and always leave both stiles intact.
            let x = -side * j.vision_shift * free.x * 0.5;
            let y = free.y * (0.5 - j.vision_head);
            Rect::from_center_size(field.center() + Vec2::new(x, y), size)
        });
        Self {
            side,
            stile,
            mount,
            bar_anchors,
            opening,
            pocket,
            veneer: p
                .veneer
                .filter(|_| {
                    p.finish.is_cut_timber()
                        && matches!(p.panel, DoorPanel::Flush | DoorPanel::VisionLite)
                })
                .map(|v| v.charts(p, leaf, index)),
        }
    }

    pub(super) fn protective_face(
        &self,
        p: &DoorProgram,
        leaf: Leaf,
        index: usize,
        side: f32,
        j: &Joinery,
    ) -> Option<FittedKickPlate> {
        let stock = p.kick_plate?;
        if !KickPlateStock::eligible(p) {
            return None;
        }
        let push = side == -p.swing_side;
        if matches!(
            (stock.faces, push),
            (ProtectedFaces::Push, false) | (ProtectedFaces::Pull, true)
        ) {
            return None;
        }
        let half = leaf.size.truncate() * 0.5;
        let inset = if push {
            stock
                .edge_inset_m
                .max(p.swing_hanging.map_or(0., |h| h.stop_projection_m + 0.003))
        } else {
            stock.edge_inset_m
        };
        let bottom = -half.y + stock.bottom_inset_m;
        // The pull has the longest vertical reach of this hardware set.
        let pull_foot = j.pull + 0.014_f32.min(self.stile * 0.36);
        let limit = (bottom + stock.height_m).min(self.mount.y - pull_foot - 0.008);
        let top = self
            .opening
            .map_or(limit, |hole| limit.min(hole.min.y - 0.012));
        if top - bottom < 0.065 || half.x - inset < 0.045 {
            return None;
        }
        Some(FittedKickPlate {
            leaf_index: index,
            leaf_transform: leaf.transform,
            side,
            bounds: Rect::from_corners(
                Vec2::new(-half.x + inset, bottom),
                Vec2::new(half.x - inset, top),
            ),
            base_z: side * leaf.size.z * 0.5,
            gauge_m: stock.gauge_m,
            corner_clip_m: stock.corner_clip_m,
            finish: stock.finish,
        })
    }

    pub(super) fn face_chart(&self, uvs: &mut [[f32; 2]], side: f32) {
        if let Some(charts) = &self.veneer {
            let chart = charts[usize::from(side > 0.)];
            for uv in uvs {
                *uv = chart.transform_point2(Vec2::from(*uv)).to_array();
            }
        }
    }

    pub(super) fn handle(p: &DoorProgram, track: usize, face: f32) -> DoorHandle {
        // The rear of a slider faces its wall. On the wallward telescoping
        // track the outward face also passes within 14 mm of another leaf.
        if matches!(
            p.mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        ) && (face != p.swing_side || p.mechanism == DoorMechanism::Telescoping && track == 0)
        {
            return DoorHandle::Recessed;
        }
        if p.handle == DoorHandle::PanicBar
            && p.mechanism != DoorMechanism::Revolving
            && face == p.swing_side
        {
            return DoorHandle::Pull;
        }
        p.handle
    }
}
