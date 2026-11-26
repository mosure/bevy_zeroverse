//! Open shelving. Side rails carry the decks between pairs of posts. The
//! rear members sit behind the uprights, clear of every storage compartment.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RackBracing {
    Diagonal,
    Alternating,
    Stretchers,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct RackFrame {
    pub bracing: RackBracing,
    pub post_width_m: f32,
    pub post_depth_m: f32,
    pub rail_height_m: f32,
    /// How far each end of a deck bears on its side rail.
    pub bearing_m: f32,
    /// Setback of the deck from the inside face of each post.
    pub reveal_m: f32,
    pub brace_depth_m: f32,
    pub brace_width_m: f32,
    pub low_attachment_m: f32,
    pub high_attachment_m: f32,
    /// Direction of the first diagonal when the back uses diagonal members.
    pub rises_right: bool,
}

impl RackFrame {
    pub(super) fn sample(o: &IndoorObject, bays: &mut [ShelfBay], panel: f32) -> Self {
        let mut r = stream(o.seed, 0x5241_434b_4241_434b);
        let divisions = (bays.len() + 1) as f32;
        let post_width_m = r
            .random_range(0.024_f32..0.048)
            .min(o.size.x / (6. * divisions));
        let post_depth_m = (post_width_m * r.random_range(0.72..1.25)).min(o.size.z * 0.12);
        // Keep the original distribution of clear bay widths and adjustable
        // shelf levels while fitting the actual upright stock inside the case.
        let ratio = (o.size.x - post_width_m * divisions) / (o.size.x - panel * divisions);
        let mut left = -o.size.x * 0.5 + post_width_m;
        for b in bays.iter_mut() {
            let width = (b.right - b.left) * ratio;
            b.left = left;
            b.right = left + width;
            left = b.right + post_width_m;
        }
        let bracing = match r.random_range(0..5) {
            0 | 1 => RackBracing::Stretchers,
            2 | 3 if bays.len() > 1 => RackBracing::Alternating,
            _ => RackBracing::Diagonal,
        };
        let bottom_clearance = bays[0].levels[0] - panel * 0.5;
        Self {
            bracing,
            post_width_m,
            post_depth_m,
            rail_height_m: r
                .random_range(0.022_f32..0.054)
                .min(bottom_clearance * 0.88),
            // Two independently adjustable decks can share a middle upright.
            bearing_m: post_width_m * r.random_range(0.33..0.45),
            reveal_m: r.random_range(0.001..0.004),
            brace_depth_m: r.random_range(0.006..0.015),
            brace_width_m: match bracing {
                RackBracing::Stretchers => r.random_range(0.034..0.074),
                RackBracing::Diagonal => r.random_range(0.018_f32..0.042).min(post_width_m),
                RackBracing::Alternating => post_width_m * r.random_range(0.45..0.63),
            },
            low_attachment_m: o.size.y * r.random_range(0.08..0.32),
            high_attachment_m: o.size.y * r.random_range(0.69..0.94),
            rises_right: r.random_bool(0.5),
        }
    }

    pub(super) fn posts(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &BookcaseProgram,
        cuts: &mut TimberCuts,
        x: f32,
    ) {
        let front = o.size.z * 0.5;
        for z in [
            -front + self.brace_depth_m + self.post_depth_m * 0.5,
            front - self.post_depth_m * 0.5,
        ] {
            a.board(
                cuts,
                (p.frame, o.kind.class_name()),
                Vec3::new(x, o.size.y * 0.5, z),
                Vec3::new(self.post_width_m, o.size.y, self.post_depth_m),
                0.002,
                GrainAxis::Y,
            );
        }
    }

    fn rear_members(&self, bays: &[ShelfBay], mut member: impl FnMut(Vec2, Vec2)) {
        // Alternating straps terminate on opposite halves of a shared stud.
        // Their ends stay seated on it even at steep angles without intersecting.
        let end = self.post_width_m
            * match self.bracing {
                RackBracing::Alternating => 0.16,
                RackBracing::Diagonal | RackBracing::Stretchers => 0.48,
            };
        let left = bays[0].left - end;
        let right = bays[bays.len() - 1].right + end;
        let low = self.low_attachment_m;
        let high = self.high_attachment_m;
        let mut diagonal = |l, r, rising| {
            let (y0, y1) = if rising { (low, high) } else { (high, low) };
            member(Vec2::new(l, y0), Vec2::new(r, y1));
        };
        match self.bracing {
            RackBracing::Diagonal => diagonal(left, right, self.rises_right),
            RackBracing::Alternating => {
                for (index, bay) in bays.iter().enumerate() {
                    diagonal(
                        bay.left - end,
                        bay.right + end,
                        self.rises_right == (index % 2 == 0),
                    );
                }
            }
            RackBracing::Stretchers => {
                for y in [low, high] {
                    member(Vec2::new(left, y), Vec2::new(right, y));
                }
            }
        }
    }

    pub(super) fn build_supports(&self, a: &mut Assembly, o: &IndoorObject, p: &BookcaseProgram) {
        let mut cuts = TimberCuts::new(o.seed, 0x5241_434b);
        let geometry = a.part(p.frame, o.kind.class_name());
        let (rear, front) = p.deck_limits(o.size.z);
        let rail_depth = front - rear + 2. * self.reveal_m;
        for b in &p.bays {
            for &level in &b.levels {
                let y = level - (p.panel_m + self.rail_height_m) * 0.5;
                for x in [
                    b.left - self.bearing_m * 0.5,
                    b.right + self.bearing_m * 0.5,
                ] {
                    cuts.cuboid(
                        geometry,
                        p.frame,
                        Vec3::new(self.bearing_m, self.rail_height_m, rail_depth),
                        0.,
                        Transform::from_xyz(x, y, (front + rear) * 0.5),
                        GrainAxis::Z,
                    );
                }
            }
        }
        self.rear_members(&p.bays, |a0, a1| {
            let span = a1 - a0;
            let center = ((a0 + a1) * 0.5).extend((-o.size.z + self.brace_depth_m) * 0.5);
            let tf = Transform::from_translation(center)
                .with_rotation(Quat::from_rotation_z(span.y.atan2(span.x)));
            cuts.cuboid(
                geometry,
                p.frame,
                Vec3::new(span.length(), self.brace_width_m, self.brace_depth_m),
                0.,
                tf,
                GrainAxis::X,
            );
        });
    }
}
