//! Local +Z faces the customer. The service leaves, handles and shelf cleats
//! all fit inside the same admitted footprint, including on the staff side.
use super::*;
use crate::scene::procedural_indoor::materials::glass::display::SheetGauge;

const CLASS: &str = "display_case";

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ShelfEdge {
    pub depth_m: f32,
    pub height_m: f32,
    pub stock: Surface,
}

impl ShelfEdge {
    pub(super) fn sample(o: &IndoorObject) -> Option<Self> {
        let mut r = stream(o.seed, 0x5348454c46454447);
        if o.variant >= 16 && r.random_bool(0.44) {
            return None;
        }
        Some(Self {
            depth_m: r.random_range(0.005..0.010),
            height_m: r.random_range(0.011..0.019),
            stock: [
                Surface::Chrome,
                Surface::Chrome,
                Surface::Metal,
                Surface::Plastic,
            ][r.random_range(0..4)],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseDoor {
    Hinged,
    PairedHinged,
    Sliding,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ServiceGlazing {
    pub door: CaseDoor,
    #[serde(rename = "pane_m")]
    pub pane: SheetGauge,
    pub pull_on_left: bool,
    pub pull_span_m: f32,
    pub pull_height_fraction: f32,
}

impl ServiceGlazing {
    pub(super) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, 0x56495452494e4553);
        let door = if r.random_bool(if o.variant < 16 { 0.66 } else { 0.37 }) {
            CaseDoor::Sliding
        } else if o.size.x > 1.14 || r.random_bool(0.24) {
            CaseDoor::PairedHinged
        } else {
            CaseDoor::Hinged
        };
        Self {
            door,
            pane: [SheetGauge::Four, SheetGauge::Five, SheetGauge::Six][r.random_range(0..3)],
            pull_on_left: r.random(),
            pull_span_m: r.random_range(0.068..0.153),
            pull_height_fraction: r.random_range(0.36..0.69),
        }
    }
}

pub(super) struct Cabinet {
    half: Vec2,
    pub frame: f32,
    bottom: f32,
    pub head: f32,
    rear: f32,
    pub shelf: (Vec2, Vec2),
    pub shelf_thickness: f32,
    edge: Option<ShelfEdge>,
    apron_depth: f32,
}

impl Cabinet {
    pub fn new(size: Vec3, p: &CaseProgram, g: ServiceGlazing) -> Self {
        let half = size.xz() * 0.5;
        let frame = (size.x.min(size.z) * p.wall_fraction).clamp(0.014, 0.042);
        let rear = -half.y + frame * 0.55 + 0.025;
        let x = half.x - frame * 1.38;
        let inner_pane = rear
            + g.pane.metres() * 0.5
            + if g.door == CaseDoor::Sliding {
                g.pane.metres() + 0.003
            } else {
                0.
            };
        Self {
            half,
            frame,
            rear,
            edge: p.shelf_edge,
            shelf_thickness: if p.structure == 1 {
                SheetGauge::Eight.metres()
            } else {
                match p.frame {
                    Surface::Wood => (size.x * 0.013).clamp(0.014, 0.026),
                    Surface::Plastic => (size.x * 0.011).clamp(0.012, 0.020),
                    _ => 0.008,
                }
            },
            apron_depth: if size.x > 1.18 {
                (size.x * 0.014).min(0.029)
            } else {
                0.
            },
            bottom: size.y * p.base_fraction + frame * 0.5,
            head: size.y - frame,
            shelf: (
                Vec2::new(-x, inner_pane + 0.010),
                Vec2::new(x, half.y - frame * 1.38),
            ),
        }
    }

    pub fn shelf_height(&self, fraction: f32, height: f32) -> f32 {
        (fraction * height)
            .max(self.bottom + self.apron_depth.max(0.010) + self.shelf_thickness * 0.5 + 0.002)
    }

    /// The pane terminates at the edge extrusion. Never lay a second visible
    /// face over the glass. The folded leg returns under the cut sheet.
    pub fn shelf_front(&self, a: &mut Assembly, shelf_y: f32) -> f32 {
        let (lo, hi) = self.shelf;
        let Some(edge) = self.edge else { return hi.y };
        let end = hi.y - edge.depth_m;
        let top = shelf_y + self.shelf_thickness * 0.5;
        a.box_part(
            edge.stock,
            CLASS,
            Vec3::new(0., top - edge.height_m * 0.5, end + edge.depth_m * 0.5),
            Vec3::new(hi.x - lo.x, edge.height_m, edge.depth_m),
            0.0008,
        );
        if self.apron_depth == 0. {
            a.box_part(
                edge.stock,
                CLASS,
                Vec3::new(0., top - self.shelf_thickness - 0.001, end - 0.002),
                Vec3::new((self.half.x - self.frame * 1.84) * 2., 0.002, 0.004),
                0.0004,
            );
        }
        end
    }

    pub fn shelf_supports(&self, a: &mut Assembly, stock: Surface, shelf_y: f32, pane: f32) {
        let outer = self.half.x - self.frame * 0.5 - pane * 0.5 - 0.001;
        let inner = self.half.x - self.frame * 1.84;
        let gauge = (self.frame * 0.33).clamp(0.004, 0.010);
        for sign in [-1., 1.] {
            a.box_part(
                stock,
                CLASS,
                Vec3::new(
                    sign * (outer + inner) * 0.5,
                    shelf_y - (self.shelf_thickness + gauge) * 0.5,
                    0.,
                ),
                Vec3::new(outer - inner, gauge, self.half.y * 2. - self.frame),
                0.0006,
            );
        }
        if self.apron_depth > 0. {
            // Long spans gain a front and rear beam. Both terminate on the
            // cleats and stay within the merchandise setback at the shelf edge.
            let front_end = self.shelf.1.y - self.edge.map_or(0., |e| e.depth_m);
            for z in [self.shelf.0.y + 0.005, front_end - 0.005] {
                a.box_part(
                    stock,
                    CLASS,
                    Vec3::new(
                        0.,
                        shelf_y - (self.shelf_thickness + self.apron_depth) * 0.5,
                        z,
                    ),
                    Vec3::new(inner * 2., self.apron_depth, 0.010),
                    0.001,
                );
            }
        }
    }

    pub fn enclosure(&self, a: &mut Assembly, p: &CaseProgram, g: ServiceGlazing) {
        let f = self.frame;
        let w = self.half.x * 2.;
        let d = self.half.y * 2.;
        let h = self.head - self.bottom;
        let cy = self.bottom + h * 0.5;
        let cap_y = self.head + f * 0.5;
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                a.box_part(
                    p.frame,
                    CLASS,
                    Vec3::new(side * (w - f) * 0.5, cy, end * (d - f) * 0.5),
                    Vec3::new(f, h, f),
                    f * 0.11,
                );
            }
            a.box_part(
                Surface::ContainerGlass,
                g.pane.label(),
                Vec3::new(side * (w - f) * 0.5, cy, 0.),
                Vec3::new(g.pane.metres(), h, d - 2. * f),
                0.0005,
            );
            if p.glazed_top {
                a.box_part(
                    p.frame,
                    CLASS,
                    Vec3::new(side * (w - f) * 0.5, cap_y, 0.),
                    Vec3::new(f, f, d),
                    0.001,
                );
                a.box_part(
                    p.frame,
                    CLASS,
                    Vec3::new(0., cap_y, side * (d - f) * 0.5),
                    Vec3::new(w - 2. * f, f, f),
                    0.001,
                );
            }
        }
        a.box_part(
            Surface::ContainerGlass,
            g.pane.label(),
            Vec3::new(0., cy, (d - f) * 0.5),
            Vec3::new(w - 2. * f, h, g.pane.metres()),
            0.0005,
        );
        if p.glazed_top {
            a.box_part(
                Surface::ContainerGlass,
                g.pane.label(),
                Vec3::new(0., self.head + f * 0.72, 0.),
                Vec3::new(w - f * 2., g.pane.metres(), d - f * 2.),
                0.0005,
            );
        } else {
            a.box_part(
                p.finish,
                CLASS,
                Vec3::new(0., cap_y, 0.),
                Vec3::new(w, f, d),
                0.002,
            );
        }
        self.service_leaves(a, p.frame, g);
    }

    fn service_leaves(&self, a: &mut Assembly, frame: Surface, g: ServiceGlazing) {
        let f = self.frame;
        let inside = self.half.x - f;
        let clear = inside - f * 0.90;
        let height = self.head - self.bottom;
        let cy = self.bottom + height * 0.5;
        let track_depth = g.pane.metres() * 2. + 0.007;
        let back = -self.half.y + f * 0.45;
        let front = self.rear + track_depth;
        for sign in [-1., 1.] {
            // The rebated stile connects the service glazing to the corner post.
            a.box_part(
                frame,
                CLASS,
                Vec3::new(sign * (inside + clear) * 0.5, cy, (back + front) * 0.5),
                Vec3::new(inside - clear, height, front - back),
                0.001,
            );
        }
        let rail = (f * 0.53).clamp(0.008, 0.021);
        for y in [self.bottom + rail * 0.5, self.head - rail * 0.5] {
            a.box_part(
                frame,
                CLASS,
                Vec3::new(0., y, self.rear + g.pane.metres() * 0.5),
                Vec3::new(clear * 2., rail, track_depth),
                0.001,
            );
        }
        let leaf_height = height - rail * 2. - 0.002;
        let pull_y = self.bottom + height * g.pull_height_fraction;
        let mut leaf = |left: f32, right: f32, z: f32, pull_left: bool| {
            a.box_part(
                Surface::ContainerGlass,
                g.pane.label(),
                Vec3::new((left + right) * 0.5, cy, z),
                Vec3::new(right - left, leaf_height, g.pane.metres()),
                0.0005,
            );
            let reach = ((right - left) * 0.085).clamp(0.024, 0.058);
            let x = if pull_left {
                left + reach
            } else {
                right - reach
            };
            let face = z - g.pane.metres() * 0.5;
            if g.door == CaseDoor::Sliding {
                // A thin bonded finger strip also clears the other leaf at full travel.
                a.box_part(
                    Surface::Metal,
                    CLASS,
                    Vec3::new(x, pull_y, face - 0.0011),
                    Vec3::new(0.014, g.pull_span_m, 0.0022),
                    0.0008,
                );
            } else {
                let projection = 0.019;
                let grip = 0.007;
                for y in [pull_y - g.pull_span_m * 0.5, pull_y + g.pull_span_m * 0.5] {
                    a.box_part(
                        Surface::Metal,
                        CLASS,
                        Vec3::new(x, y, face - projection * 0.5),
                        Vec3::new(grip, grip, projection),
                        0.001,
                    );
                }
                a.box_part(
                    Surface::Metal,
                    CLASS,
                    Vec3::new(x, pull_y, face - projection),
                    Vec3::new(grip, g.pull_span_m + grip, grip),
                    0.002,
                );
                let hinge = if pull_left { right } else { left };
                for y in [cy - leaf_height * 0.32, cy + leaf_height * 0.32] {
                    a.box_part(
                        Surface::Metal,
                        CLASS,
                        Vec3::new(hinge, y, face - 0.002),
                        Vec3::new(0.027, 0.039, 0.004),
                        0.001,
                    );
                    a.part(Surface::Metal, CLASS).cylinder(
                        0.004,
                        0.042,
                        Transform::from_xyz(hinge, y, face - 0.006),
                    );
                }
            }
        };
        let limit = clear - 0.001;
        match g.door {
            CaseDoor::Hinged => leaf(-limit, limit, self.rear, g.pull_on_left),
            CaseDoor::PairedHinged => {
                leaf(-limit, -0.0012, self.rear, false);
                leaf(0.0012, limit, self.rear, true);
            }
            CaseDoor::Sliding => {
                let lap = (limit * 0.065).clamp(0.017, 0.042);
                let inner = self.rear + g.pane.metres() + 0.003;
                let (l, r) = if g.pull_on_left {
                    (self.rear, inner)
                } else {
                    (inner, self.rear)
                };
                leaf(-limit, lap, l, true);
                leaf(-lap, limit, r, false);
            }
        }
    }
}
