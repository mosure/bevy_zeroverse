//! Low privacy screens installed on occupied work surfaces.
use super::*;
use crate::scene::procedural_indoor::layout::IndoorManifest;

pub const INTRODUCED: u32 = 238;
pub(crate) const VARIANT: u32 = 3;

pub fn is_desktop(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::RoomDivider && o.variant == VARIANT && o.support.is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Mount {
    WeightedFoot,
    ScrewPlate,
}
impl Mount {
    pub(crate) fn sample(seed: u64) -> Self {
        if stream(seed, u64::from_le_bytes(*b"DeskFoot")).random_bool(0.49) {
            Self::WeightedFoot
        } else {
            Self::ScrewPlate
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct DeskScreen {
    pub mount: Mount,
    pub face: Surface,
    pub hardware: Surface,
    pub thickness_m: f32,
    pub cable_gap_m: f32,
    pub top_drop_m: [f32; 2],
    pub upper_corner_m: [f32; 2],
    pub lower_corner_m: f32,
    pub post_x_m: [f32; 2],
}

pub fn parameters(o: &IndoorObject) -> DeskScreen {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"DeskForm"));
    let mut finish = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"DeskText"),
    );
    let mut drop = [0.; 2];
    if r.random_bool(0.48) {
        drop[usize::from(r.random_bool(0.5))] = o.size.y * r.random_range(0.05..0.29);
    }
    let radius = o.size.x.min(o.size.y) * r.random_range(0.06..0.29);
    DeskScreen {
        mount: Mount::sample(o.seed),
        face: match finish.random_range(0..5) {
            0 | 1 => Surface::Fabric,
            2 | 3 => Surface::FabricAlt,
            _ => Surface::Wood,
        },
        hardware: if finish.random_bool(0.75) {
            Surface::Metal
        } else {
            Surface::Chrome
        },
        thickness_m: r.random_range(0.017..0.039),
        cable_gap_m: (o.size.y * r.random_range(0.10..0.22)).min(0.095),
        top_drop_m: drop,
        upper_corner_m: [radius, radius * r.random_range(0.65..1.31)],
        lower_corner_m: r.random_range(0.009..0.026),
        post_x_m: [
            -o.size.x * r.random_range(0.235..0.39),
            o.size.x * r.random_range(0.235..0.39),
        ],
    }
}

impl DeskScreen {
    /// Round the four corners of a convex trapezoid along its actual edges.
    /// The local slab uses XZ; negative Z becomes up when installed vertically.
    fn outline(&self, o: &IndoorObject, center: f32) -> Vec<Vec2> {
        let x = o.size.x * 0.5;
        let top = center - o.size.y;
        let bottom = center - self.cable_gap_m;
        let corners = [
            Vec2::new(-x, top + self.top_drop_m[0]),
            Vec2::new(x, top + self.top_drop_m[1]),
            Vec2::new(x, bottom),
            Vec2::new(-x, bottom),
        ];
        let radii = [
            self.upper_corner_m[0],
            self.upper_corner_m[1],
            self.lower_corner_m,
            self.lower_corner_m,
        ];
        let mut outline = Vec::with_capacity(20);
        for (i, &corner) in corners.iter().enumerate() {
            let from = (corners[(i + 3) % 4] - corner).normalize() * radii[i];
            let to = (corners[(i + 1) % 4] - corner).normalize() * radii[i];
            for k in 0..=4 {
                let t = k as f32 * 0.25;
                outline.push(corner + from * (1. - t).powi(2) + to * t.powi(2));
            }
        }
        outline
    }
}

fn foot_half_size(depth: f32) -> Vec2 {
    Vec2::new((depth * 0.25).min(0.047), depth * 0.5)
}

/// Conservative solids for the board and each mount. A cable or small prop can
/// pass under the raised board between the mounts even when their outer boxes overlap.
pub(crate) struct Clearance {
    bottom: f32,
    half_sheet: f32,
    post_x: [f32; 2],
}

impl Clearance {
    pub fn new(o: &IndoorObject) -> Self {
        let p = parameters(o);
        Self {
            bottom: p.cable_gap_m,
            half_sheet: p.thickness_m * 0.5,
            post_x: p.post_x_m,
        }
    }

    pub fn bounds(&self, size: Vec3) -> [(Vec3, Vec3); 3] {
        let foot = foot_half_size(size.z);
        let mount = |x: f32| {
            (
                Vec3::new(x - foot.x, 0., -foot.y),
                Vec3::new(x + foot.x, self.bottom, foot.y),
            )
        };
        [
            (
                Vec3::new(-size.x * 0.5, self.bottom, -self.half_sheet),
                Vec3::new(size.x * 0.5, size.y, self.half_sheet),
            ),
            mount(self.post_x[0]),
            mount(self.post_x[1]),
        ]
    }
}

/// Protect the entire display from the assigned seat, including oblique dual monitors.
pub(crate) fn obscures_display(
    scene: &IndoorManifest,
    board: &IndoorObject,
    display: &IndoorObject,
) -> bool {
    use crate::scene::procedural_indoor::footprint::Footprint;
    if !matches!(display.kind, ObjectKind::Monitor | ObjectKind::Laptop) {
        return false;
    }
    display
        .interaction_target
        .and_then(|id| scene.objects.get(id))
        .is_some_and(|seat| {
            seat.kind == ObjectKind::Chair
                && seat.interaction_target == display.support
                && Footprint::object(board)
                    .obstructs_view(seat.position.xz(), Footprint::object(display))
        })
}

pub(crate) fn overlaps_peer(board: &IndoorObject, peer: &IndoorObject, margin: f32) -> bool {
    use crate::scene::procedural_indoor::footprint::Footprint;
    let peer_footprint = Footprint::object(peer);
    let tf = board.transform();
    Clearance::new(board)
        .bounds(board.size)
        .into_iter()
        .any(|(lo, hi)| {
            board.position.y + lo.y < peer.position.y + peer.size.y + margin
                && board.position.y + hi.y + margin > peer.position.y
                && Footprint::oriented(
                    tf.transform_point((lo + hi) * 0.5).xz(),
                    (hi - lo).xz(),
                    board.yaw,
                )
                .overlaps(peer_footprint, margin)
        })
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let center = (o.size.y + p.cable_gap_m) * 0.5;
    let tf = Transform::from_xyz(0., center, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    a.part(p.face, label)
        .profile_slab(&p.outline(o, center), p.thickness_m, 0.003, tf);

    let g = a.part(p.hardware, label);
    for x in p.post_x_m {
        let origin = Transform::from_xyz(x, 0., 0.);
        let half = foot_half_size(o.size.z);
        let base = match p.mount {
            Mount::WeightedFoot => {
                g.housing(
                    &[
                        (0., half),
                        (0.006, half),
                        (0.017, half * Vec2::new(0.84, 0.83)),
                    ],
                    3.2,
                    origin,
                );
                0.017
            }
            Mount::ScrewPlate => {
                let height = 0.005;
                g.cuboid(
                    Vec3::new(2. * half.x, height, 2. * half.y),
                    0.001,
                    Transform::from_xyz(x, height * 0.5, 0.),
                );
                for z in [-half.y * 0.64, half.y * 0.64] {
                    g.lathe(
                        &[(0., 0.), (0.004, 0.), (0.0033, 0.002), (0., 0.002)],
                        8,
                        Transform::from_xyz(x, height, z),
                    );
                }
                height
            }
        };
        // The narrow post seats inside the board. Its end cap stays concealed.
        g.cuboid(
            Vec3::new(
                half.x * 0.84,
                p.cable_gap_m + 0.010 - base,
                p.thickness_m * 0.72,
            ),
            0.0018,
            Transform::from_xyz(x, (base + p.cable_gap_m + 0.010) * 0.5, 0.),
        );
    }
}
