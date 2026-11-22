//! Stock dimensions in leaf coordinates. Opposed fittings share the same fixing line.
use super::*;
use crate::scene::procedural_indoor::materials::glass::SheetGauge;

pub(super) struct Joinery {
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
    pub(super) fn sample(p: &DoorProgram) -> Self {
        // These are immutable, continuously sampled stock dimensions already
        // present in archived manifests. Changing pose or mount cannot resample a leaf.
        let stock = ((p.thickness.to_bits() as u64) << 32) | u64::from(p.stile.to_bits());
        let mut r = stream(stock, 0x444f_4f52_4c45_4146);
        let narrow = r.random::<f32>();
        Self {
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
        let stile = p.stile.min(half.x * 0.36);
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
            p.stile.clamp(0.05, 0.075)
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
