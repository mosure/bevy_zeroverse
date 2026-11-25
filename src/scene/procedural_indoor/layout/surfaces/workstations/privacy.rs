//! Fit privacy screens to real operator sides before installing desktop equipment.
use super::*;
use crate::scene::procedural_indoor::objects::services::divider::desktop;

#[derive(Clone, Copy)]
enum Edge {
    Center,
    Rear(f32),
    Side(f32),
}
impl Edge {
    fn placement(self, desk: &IndoorObject, size: Vec3, along: f32, inset: f32) -> (Vec3, f32) {
        let (span, cross) = match self {
            Self::Side(_) => (desk.size.z, desk.size.x),
            _ => (desk.size.x, desk.size.z),
        };
        let along = along * ((span - size.x) * 0.5 - 0.03).max(0.);
        let outer = ((cross - size.z) * 0.5 - inset).max(0.);
        let (x, z, turn) = match self {
            Self::Center => (along, inset - 0.05, 0.),
            Self::Rear(sign) => (along, sign * outer, 0.),
            Self::Side(sign) => (sign * outer, along, std::f32::consts::FRAC_PI_2),
        };
        (
            desk.transform()
                .transform_point(Vec3::new(x, desk.size.y, z)),
            desk.yaw + turn,
        )
    }
}

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn furnish_desk_screens(&mut self) {
        if self.generator_version < desktop::INTRODUCED {
            return;
        }
        let frequency = match self.layout {
            IndoorLayout::OpenOffice => 0.30..0.91,
            IndoorLayout::Coworking => 0.18..0.78,
            _ => return,
        };
        let mut suite = stream(self.seed, u64::from_le_bytes(*b"DeskFelt"));
        if suite.random_bool(0.17) {
            return;
        }
        let prevalence = suite.random_range(frequency);
        let finish_lot = suite.random();
        for id in 0..self.objects.len() {
            let desk = &self.objects[id];
            if desk.kind != ObjectKind::Desk || desk.neighbor {
                continue;
            }
            let mut r = stream(desk.seed, u64::from_le_bytes(*b"DeskEdge"));
            if r.random_bool(prevalence) {
                self.fit_desk_screen(&desk.clone(), finish_lot, &mut r);
            }
        }
    }

    fn fit_desk_screen(&mut self, desk: &IndoorObject, finish_lot: u64, r: &mut ChaCha8Rng) {
        let mut sides = [false; 2];
        let forward = desk.transform().rotation * Vec3::Z;
        for seat in self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(desk.id))
        {
            sides[usize::from((seat.position - desk.position).dot(forward) > 0.)] = true;
        }
        let side = Edge::Side(if r.random_bool(0.5) { 1. } else { -1. });
        let rear = match sides {
            [true, true] => Edge::Center,
            [false, true] => Edge::Rear(-1.),
            [true, false] => Edge::Rear(1.),
            [false, false] => return,
        };
        let edges = if !matches!(rear, Edge::Center) && r.random_bool(0.25) {
            [side, rear]
        } else {
            [rear, side]
        };
        let mut panel = self.candidate(
            ObjectKind::RoomDivider,
            desk.position,
            Vec3::ONE,
            desk.yaw,
            r,
        );
        panel.variant = desktop::VARIANT;
        panel.finish_seed = r.random_bool(0.67).then_some(finish_lot);
        let fraction = r.random_range(0.48..0.88);
        panel.size.y = r.random_range(0.29..0.57);
        panel.size.z = match desktop::Mount::sample(panel.seed) {
            desktop::Mount::ScrewPlate => r.random_range(0.067..0.119),
            desktop::Mount::WeightedFoot => r.random_range(0.143..0.221),
        };
        let yaw = r.random_range(-0.036..0.036);
        let along = r.random_range(-0.85..0.85);
        let inset = r.random_range(0.027..0.080);
        // Try the full panel, then a shorter panel on the opposite part of the same edge.
        // A lateral return can use a free end when the rear cable route is occupied.
        for edge in edges {
            let span = if matches!(edge, Edge::Side(_)) {
                desk.size.z
            } else {
                desk.size.x
            };
            for (scale, offset) in [(1., along), (0.78, -along)] {
                panel.size.x = (span * fraction * scale).min(1.70);
                if panel.size.x < 0.39 {
                    continue;
                }
                (panel.position, panel.yaw) = edge.placement(desk, panel.size, offset, inset);
                panel.yaw += yaw;
                if self.admit_supported_prop(desk, panel.clone()) {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{footprint::Footprint, validation::validate_layout};

    #[test]
    fn desk_screen_keeps_all_sightlines_to_an_assigned_display_open() {
        let eye = Vec2::new(0., 1.);
        let display = Footprint::oriented(Vec2::new(0., -0.35), Vec2::new(0.76, 0.18), -0.12);
        // A screen down the middle blocks neither outside ray but obscures the image.
        let middle = Footprint::oriented(Vec2::new(0., 0.19), Vec2::new(0.04, 0.31), 0.12);
        assert!(middle.obstructs_view(eye, display));
        for x in [-0.68, 0.68] {
            let side = Footprint::oriented(Vec2::new(x, 0.15), Vec2::new(0.10, 0.60), 0.03);
            assert!(!side.obstructs_view(eye, display));
        }
        let rear = Footprint::oriented(Vec2::new(0., -0.68), Vec2::new(1.3, 0.12), 0.02);
        assert!(!rear.obstructs_view(eye, display));
    }

    #[test]
    fn desk_screens_preserve_operator_views_clearance_and_some_open_workstations() {
        let mut positions = [0; 3];
        let mut eligible = 0;
        let mut installed = 0;
        for layout in [IndoorLayout::OpenOffice, IndoorLayout::Coworking] {
            for seed in 89300..89324 {
                let s = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.25).unwrap();
                validate_layout(&s).unwrap();
                eligible += s
                    .objects
                    .iter()
                    .filter(|o| {
                        o.kind == ObjectKind::Desk
                            && !o.neighbor
                            && s.objects.iter().any(|c| {
                                c.kind == ObjectKind::Chair && c.interaction_target == Some(o.id)
                            })
                    })
                    .count();
                for board in s.objects.iter().filter(|o| desktop::is_desktop(o)) {
                    let desk = &s.objects[board.support.unwrap()];
                    assert_eq!(desk.kind, ObjectKind::Desk);
                    assert!(!board.solid && !board.neighbor);
                    assert!((board.position.y - desk.position.y - desk.size.y).abs() < 1e-5);
                    assert!(s.prop_clear(board, desk, 0.012));
                    assert!(s
                        .objects
                        .iter()
                        .filter(|display| display.support == Some(desk.id))
                        .all(|display| !desktop::obscures_display(&s, board, display)));
                    let local =
                        desk.transform().rotation.inverse() * (board.position - desk.position);
                    let i = if (board.yaw - desk.yaw).sin().abs() > 0.9 {
                        2
                    } else if local.z.abs() < 0.06 {
                        0
                    } else {
                        1
                    };
                    positions[i] += 1;
                    installed += 1;
                }
            }
        }
        assert!(positions.iter().all(|n| *n > 0), "{positions:?}");
        assert!(
            installed > 10 && installed < eligible,
            "{installed}/{eligible}"
        );
    }
}
