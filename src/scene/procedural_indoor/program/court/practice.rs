//! Permanently painted shooting stations in multipurpose halls. The line is set
//! out from the face of the installed backboard, independent of the volleyball centre.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::polygon,
    layout::{IndoorObject, ObjectKind},
    objects::public::basketball::BasketballGoal,
};

const ADULT_DISTANCE: f32 = 4.60;
const FULL_LINE: f32 = 3.60;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FreeThrowMark {
    pub goal_id: usize,
    /// Centre of the stripe on the room floor (world X, Z).
    pub center: Vec2,
    /// Unit floor direction from the backboard towards the shooter.
    pub forward: Vec2,
    pub line_length_m: f32,
    /// Perpendicular distance from the board face to the stripe's far edge.
    pub backboard_distance_m: f32,
    /// A semicircle extends away from the basket. Its outer diameter equals the stripe length.
    pub semicircle: bool,
}

impl FreeThrowMark {
    pub(crate) const FIRST_GENERATOR: u32 = 320;

    pub(crate) fn world(self, across: f32, along: f32) -> Vec2 {
        self.center + self.forward.perp() * across + self.forward * along
    }

    pub fn bounds(self) -> Rect {
        let r = self.line_length_m * 0.5;
        Rect::from_corners(
            self.world(-r, -LINE * 0.5),
            self.world(r, if self.semicircle { r } else { LINE * 0.5 }),
        )
    }

    fn board(goal: &IndoorObject) -> (Vec2, Vec2, f32) {
        let spec = BasketballGoal::for_object(goal);
        let pose = goal.transform();
        let normal = (pose.rotation * Vec3::Z).xz().round();
        let face = pose.transform_point(Vec3::Z * spec.face_z(goal.size)).xz();
        (face, normal, goal.position.y + spec.rim_height())
    }

    fn fits(self, scene: &IndoorManifest) -> bool {
        let Some(shell) = &scene.envelope else {
            return false;
        };
        // Keep space for a shooter behind the stripe. A deployed volleyball net,
        // fixed seating or a water station can make an otherwise in-bounds mark unusable.
        let half = self.line_length_m * 0.5;
        let far = if self.semicircle { half } else { 0. } + 0.65;
        let lane = Rect::from_corners(
            self.world(-half - 0.16, -self.backboard_distance_m + LINE * 0.5),
            self.world(half + 0.16, far),
        );
        polygon::box_inside(&shell.footprint, lane.min, lane.max, 0.08)
            && !scene.objects.iter().any(|o| {
                if o.neighbor || !(o.solid || o.kind == ObjectKind::VolleyballNet) {
                    return false;
                }
                let (lo, hi) = o.bounds();
                lo.y < 2.5
                    && hi.y > 0.03
                    && lo.xz().cmplt(lane.max).all()
                    && hi.xz().cmpgt(lane.min).all()
            })
    }

    pub(crate) fn install(scene: &mut IndoorManifest) {
        if scene.generator_version < Self::FIRST_GENERATOR {
            return;
        }
        let mut r = stream(scene.seed, u64::from_le_bytes(*b"HoopLine"));
        if !r.random_bool(0.78) {
            return;
        }
        // Opposed baskets share the contractor's set-out, including a possible
        // shorter throw for a lowered basket. Position still follows each bracket.
        let semicircle = r.random_bool(0.60);
        let full = semicircle || r.random_bool(0.50);
        let short = r.random_range(0.95..2.05);
        let youth_distance = r.random_range(3.10..4.20);
        let mut marks = Vec::<Self>::new();
        for goal in scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::BasketballGoal)
        {
            let (back, forward, height) = Self::board(goal);
            let distance = if height >= 3. {
                ADULT_DISTANCE
            } else {
                youth_distance
            };
            let mut mark = Self {
                goal_id: goal.id,
                center: back + forward * (distance - LINE * 0.5),
                forward,
                line_length_m: if full { FULL_LINE } else { short },
                backboard_distance_m: distance,
                semicircle,
            };
            if !mark.fits(scene) {
                if !full {
                    continue;
                }
                mark.line_length_m = short;
                mark.semicircle = false;
                if !mark.fits(scene) {
                    continue;
                }
            }
            if marks
                .iter()
                .all(|other| other.bounds().intersect(mark.bounds()).is_empty())
            {
                marks.push(mark);
            }
        }
        if let Some(p) = &mut scene.program {
            p.free_throw_marks = marks;
        }
    }

    pub(in crate::scene::procedural_indoor) fn validate_set(
        scene: &IndoorManifest,
        marks: &[Self],
    ) -> Result<(), String> {
        if scene.generator_version < Self::FIRST_GENERATOR
            || scene.layout != IndoorLayout::Gymnasium
            || marks.len() > 2
        {
            return Err("unexpected basketball practice markings".into());
        }
        for (i, &m) in marks.iter().enumerate() {
            let Some(goal) = scene.objects.get(m.goal_id).filter(|o| {
                o.id == m.goal_id && !o.neighbor && o.kind == ObjectKind::BasketballGoal
            }) else {
                return Err("basketball practice mark has no installed goal".into());
            };
            let (face, forward, height) = Self::board(goal);
            let full = (m.line_length_m - FULL_LINE).abs() < 0.0001;
            let distance_matches = if height >= 3. {
                (m.backboard_distance_m - ADULT_DISTANCE).abs() < 0.0001
            } else {
                (3.10..=4.20).contains(&m.backboard_distance_m)
            };
            if !m.center.is_finite()
                || m.forward != forward
                || forward.length_squared() != 1.
                || !(full || (!m.semicircle && (0.95..=2.05).contains(&m.line_length_m)))
                || !distance_matches
                || !m.center.abs_diff_eq(
                    face + forward * (m.backboard_distance_m - LINE * 0.5),
                    0.0002,
                )
                || !m.fits(scene)
                || marks[..i]
                    .iter()
                    .any(|o| o.goal_id == m.goal_id || !o.bounds().intersect(m.bounds()).is_empty())
            {
                return Err(
                    "basketball practice mark misses its backboard or clear shooting lane".into(),
                );
            }
        }
        Ok(())
    }
}
