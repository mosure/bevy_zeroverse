use super::*;
use crate::scene::procedural_indoor::motion::RollerFeed;

pub(in super::super) fn feed_plane(tf: Transform, opening: &WindowOpening) -> RollerFeed {
    RollerFeed::new(tf, opening.max.y - 0.014 - super::RAIL_DROP)
}

impl Builder<'_> {
    pub(super) fn roller_head(&mut self) {
        let top = self.top();
        if self.p.identity.is_multiple_of(2) {
            // A cloth-covered roll, supported at both ends. Its exposed lower
            // arc hides the transition between the visible web and stored yarn.
            self.tube(
                self.p.fabric,
                &[
                    Vec3::new(self.lo.x + 0.006, top, 0.101),
                    Vec3::new(self.hi.x - 0.006, top, 0.101),
                ],
                0.042,
            );
        } else {
            self.block(
                Surface::Metal,
                Vec3::new(self.center(), top - 0.003, 0.111),
                Vec3::new(self.width(), 0.068, 0.062),
                0.003,
            );
        }
    }

    pub(super) fn roller(&mut self, moving: bool) {
        let available = self.height();
        let height = 0.044 + (available - 0.044) * self.coverage;
        let width = self.width() - 0.035;
        let center = self.center();
        let rail = self.top();
        let warp = 0.13 * self.p.fold_depth;
        let length = if moving { available } else { height };
        // Only an articulated roller needs vertices for cloth still on the roll.
        // A row crossing the feed plane fits entirely behind either head profile.
        let rows = if moving {
            (available / 0.018).ceil() as usize
        } else {
            1
        };
        let body_label = format!("{}#roller", self.cloth_label);
        let g = self.a.part(self.p.fabric, &body_label);
        let first = g.uvs.len();
        let columns = 4;
        cloth::sheet(
            g,
            self.tf,
            [columns, rows],
            Vec2::new(width, length),
            |u, v| {
                let above_rail = v * length - height;
                Vec3::new(
                    center + (u - 0.5) * width,
                    rail + RollerFeed::pack(above_rail),
                    0.112 + warp * (u * PI).sin() * (u * TAU + 0.6).sin(),
                )
            },
        );
        if !moving {
            let faces = 2 * (columns + 1) * (rows + 1);
            for uv in &mut g.uvs[first..first + faces] {
                uv[1] += available - height;
            }
        }
        self.block(
            Surface::Plastic,
            Vec3::new(center, rail - height - 0.007, 0.112),
            Vec3::new(width + 0.006, 0.014, 0.018),
            0.002,
        );
    }
}
