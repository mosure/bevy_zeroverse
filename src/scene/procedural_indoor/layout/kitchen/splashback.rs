//! Keep mounted displays clear of the wall immediately above the burners.
use super::*;
use crate::scene::procedural_indoor::footprint::Footprint;

const INTRODUCED: u32 = 248;

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn covers_cooktop_splashback(
        &self,
        panel: &IndoorObject,
    ) -> bool {
        if self.generator_version < INTRODUCED
            || self.assembly_diagnostics.kitchens.is_empty()
            || !matches!(
                panel.kind,
                ObjectKind::NoticeBoard
                    | ObjectKind::Whiteboard
                    | ObjectKind::WallArt
                    | ObjectKind::Display
                    | ObjectKind::ProjectionScreen
            )
        {
            return false;
        }
        let face = panel.transform().rotation * Vec3::Z;
        let patch = Footprint::object(panel);
        let margin = 0.16;
        self.assembly_diagnostics
            .kitchens
            .iter()
            .flat_map(|group| group.units.iter().chain(&group.props))
            .filter_map(|&id| self.objects.get(id))
            .filter(|unit| matches!(unit.kind, ObjectKind::Range | ObjectKind::Cooktop))
            .any(|unit| {
                // The chassis encloses the slab even when the oven door extends
                // the envelope. Most wall candidates can avoid sampling the unit.
                if !patch.overlaps(Footprint::object(unit), margin) {
                    return false;
                }
                let (center, size) =
                    geometry::appliances::cooking_surface(unit, &geometry::parameters(unit));
                panel.position.y < center.y + 0.85
                    && panel.position.y + panel.size.y > center.y - 0.03
                    && (center - panel.position).dot(face) >= -0.01
                    && patch.overlaps(Footprint::oriented(center.xz(), size, unit.yaw), margin)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_wall_board_respects_the_burners_without_reserving_the_entire_wall() {
        let mut scene =
            IndoorManifest::generate_with_humans(5211, IndoorLayout::Kitchen, 0.65, 3, 0.25)
                .unwrap();
        let range = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Range)
            .unwrap();
        let (center, size) =
            geometry::appliances::cooking_surface(range, &geometry::parameters(range));
        let transform = range.transform().rotation;
        let mut board =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::NoticeBoard, 1);
        board.size = Vec3::new(0.75, 0.45, 0.035);
        board.yaw = range.yaw;
        board.position = center + transform * Vec3::new(0., 0.12, -size.y * 0.5 - 0.055);
        assert!(scene.covers_cooktop_splashback(&board));
        let pose = board.position;
        board.position += transform * Vec3::X * (size.x + board.size.x);
        assert!(!scene.covers_cooktop_splashback(&board));
        board.position = pose + Vec3::Y;
        assert!(!scene.covers_cooktop_splashback(&board));
        board.position = pose;
        board.yaw += std::f32::consts::PI;
        assert!(!scene.covers_cooktop_splashback(&board));
        board.yaw = range.yaw;
        scene.generator_version = INTRODUCED - 1;
        assert!(!scene.covers_cooktop_splashback(&board));
    }
}
