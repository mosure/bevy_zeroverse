//! Sightlines along the playing surface, interleaved with free room proposals.
use crate::scene::procedural_indoor::{
    layout::{IndoorObject, ObjectKind},
    program::court::VolleyballCourt,
};
use bevy::prelude::*;
use rand::Rng;

#[derive(Clone, Copy)]
pub(super) struct Proposal {
    pub eye: Vec2,
    pub target: Vec3,
    pub person: Option<Vec3>,
}

/// At most one installed bank per sideline. Cache only its transform and size
/// for the proposal loop, without allocating or binding a particular camera.
#[derive(Clone, Copy, Default)]
pub(super) struct Sidelines {
    first: Option<(Transform, Vec3)>,
    second: Option<(Transform, Vec3)>,
}

pub(super) fn sidelines(objects: &[IndoorObject]) -> Sidelines {
    let mut banks = objects
        .iter()
        .filter(|o| !o.neighbor && o.kind == ObjectKind::BleacherBank)
        .map(|o| (o.transform(), o.size));
    Sidelines {
        first: banks.next(),
        second: banks.next(),
    }
}

pub(super) fn propose(
    c: VolleyballCourt,
    stands: Sidelines,
    person: Option<Vec3>,
    rng: &mut impl Rng,
) -> Proposal {
    if let Some(first) = stands
        .first
        .filter(|_| person.is_none() && rng.random_bool(0.35))
    {
        let (bank, size) = stands
            .second
            .filter(|_| rng.random_bool(0.5))
            .unwrap_or(first);
        let eye = c.world(Vec2::new(
            c.width * rng.random_range(-0.44..0.44),
            c.length * rng.random_range(-0.46..0.46),
        ));
        // The point of aim is in the open front aisle. Oblique views of the
        // seating and court still undergo the usual path and overlap tests.
        let target = bank.transform_point(Vec3::new(
            size.x * rng.random_range(-0.44..0.44),
            size.y * rng.random_range(0.27..0.67),
            size.z * 0.5 + rng.random_range(0.12..0.64),
        ));
        return Proposal {
            eye,
            target,
            person,
        };
    }
    let relative = person.map(|p| {
        if c.long_x {
            p.x - c.center.x
        } else {
            p.z - c.center.y
        }
    });
    let side = if let Some(along) = relative.filter(|v| v.abs() > 0.35) {
        -along.signum()
    } else if rng.random_bool(0.5) {
        1.
    } else {
        -1.
    };
    let eye = c.world(Vec2::new(
        c.width * rng.random_range(-0.70..0.70),
        side * c.length * rng.random_range(0.20..0.54),
    ));
    let aim = c.world(Vec2::new(
        c.width * rng.random_range(-0.43..0.43),
        c.length * rng.random_range(-0.12..0.12),
    ));
    Proposal {
        eye,
        target: Vec3::new(aim.x, rng.random_range(0.7..c.net_height), aim.y),
        person,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::stream, program::court::room_size};

    #[test]
    fn gymnasium_court_and_both_spectator_sidelines_share_the_proposal_distribution() {
        for seed in [12, 19, 36, 59] {
            let c = VolleyballCourt::fit(seed, room_size(seed)).unwrap();
            let size = Vec3::new(6.1, 1.83, 1.92);
            let banks = [-1., 1.].map(|side| {
                let pos = c.world(Vec2::new(side * (c.width * 0.5 + c.free_zone + 1.7), 0.));
                let inward = (c.center - pos).normalize();
                let bank = Transform::from_xyz(pos.x, 0., pos.y)
                    .with_rotation(Quat::from_rotation_y(inward.x.atan2(inward.y)));
                (bank, size)
            });
            let stands = Sidelines {
                first: Some(banks[0]),
                second: Some(banks[1]),
            };
            let mut rng = stream(seed, 91);
            let mut hits = [0; 2];
            for _ in 0..128 {
                let v = propose(c, stands, None, &mut rng);
                for (i, (bank, _)) in banks.iter().enumerate() {
                    let t = bank.rotation.inverse() * (v.target - bank.translation);
                    if (size.z * 0.5 + 0.11..size.z * 0.5 + 0.65).contains(&t.z) {
                        assert!(
                            t.x.abs() < size.x * 0.45 && (0.25..0.68).contains(&(t.y / size.y))
                        );
                        let eye =
                            bank.rotation.inverse() * (v.eye.extend(0.).xzy() - bank.translation);
                        assert!(eye.z > size.z * 0.5 + c.free_zone);
                        hits[i] += 1;
                    }
                }
            }
            assert!(hits.into_iter().all(|n| n >= 6));
            assert!(hits.iter().sum::<usize>() < 85);
            let mut other = rng.clone();
            let person = c.center.extend(1.6).xzy();
            let v = propose(c, stands, Some(person), &mut rng);
            let control = propose(c, Sidelines::default(), Some(person), &mut other);
            assert_eq!(
                (v.eye, v.target, v.person),
                (control.eye, control.target, Some(person))
            );
        }
    }

    #[test]
    fn gymnasium_people_are_framed_from_both_sides_through_the_playing_area() {
        for seed in 0..24 {
            let c = VolleyballCourt::fit(seed, room_size(seed)).unwrap();
            for side in [-1., 1.] {
                let p = c.world(Vec2::new(c.width * 0.58, c.length * side * 0.28));
                let person = Vec3::new(p.x, 1.45, p.y);
                let mut rng = stream(seed, 34);
                let mut eyes = Vec::new();
                for _ in 0..12 {
                    let view = propose(c, Sidelines::default(), Some(person), &mut rng);
                    let along = if c.long_x {
                        view.eye.x - c.center.x
                    } else {
                        view.eye.y - c.center.y
                    };
                    assert!(along * side < 0.);
                    let half = room_size(seed).xz() * 0.5;
                    assert!(view.eye.abs().cmplt(half - 0.5).all());
                    assert_eq!(view.person, Some(person));
                    eyes.push(view.eye);
                }
                assert!(eyes.iter().any(|p| p.distance(eyes[0]) > 2.));
            }
        }
    }
}
