use super::*;
impl Assembly {
    pub(crate) fn motion_selected(&self, kind: MotionKind, seed: u64) -> bool {
        self.motion_policy.as_ref().is_some_and(|p| {
            p.fraction(kind) > 0.
                && (kind != MotionKind::Chair
                    || p.chair_travel_m > 0.
                    || p.chair_swivel_degrees > 0.)
                && stream(seed, 0x4d4f544e).random_bool(p.fraction(kind) as f64)
        })
    }
    pub(crate) fn append_assembly(&mut self, mut other: Assembly) {
        if other.joints.is_empty() {
            for ((s, l), g) in other.parts {
                self.part(s, &l).append(g);
            }
            return;
        }
        let offset = self.joints.len();
        for joint in &mut other.joints {
            for (_, label) in &mut joint.parts {
                if let Some(id) = tag(label) {
                    *label =
                        label.replace(&format!("#joint{id}"), &format!("#joint{}", id + offset));
                }
            }
            joint.morph = std::mem::take(&mut joint.morph)
                .into_iter()
                .map(|((s, l), g)| {
                    let l = tag(&l).map_or(l.clone(), |id| {
                        l.replace(&format!("#joint{id}"), &format!("#joint{}", id + offset))
                    });
                    ((s, l), g)
                })
                .collect();
        }
        for ((s, l), g) in other.parts {
            let l = tag(&l).map_or(l.clone(), |id| {
                l.replace(&format!("#joint{id}"), &format!("#joint{}", id + offset))
            });
            self.part(s, &l).append(g);
        }
        self.joints.extend(other.joints);
    }
    pub(crate) fn articulated(&mut self, other: Assembly, mut joint: Joint) {
        let id = self.joints.len();
        for ((s, l), g) in other.parts {
            let key = (s, tagged(&l, id));
            if let Some(end) = joint.morph.remove(&(s, l)) {
                joint.morph.insert(key.clone(), end);
            }
            joint.parts.push(key.clone());
            self.parts.insert(key, g);
        }
        self.joints.push(joint);
    }
}

pub(crate) fn rigid(
    kind: MotionKind,
    seed: u64,
    pivot: Vec3,
    axis: Vec3,
    angle: f32,
    translation: Vec3,
) -> Joint {
    Joint {
        pivot,
        axis,
        angle,
        translation,
        ..Joint::new(kind, seed)
    }
}

pub(crate) fn endpoint(seed: u64, start: f32, max: f32) -> f32 {
    let mut r = stream(seed, 0x454e4454);
    if start > max * 0.45 {
        r.random_range(0.0..0.18) * max
    } else {
        r.random_range(0.65..1.0) * max
    }
}

pub(crate) fn morph(
    a: &mut Assembly,
    mut start: Assembly,
    mut end: Assembly,
    seed: u64,
    frame: Transform,
) {
    // A topology mismatch is never converted into an invalid flow correspondence.
    let compatible = start.parts.len() == end.parts.len()
        && start.parts.iter().all(|(k, g)| {
            end.parts
                .get(k)
                .is_some_and(|e| e.indices == g.indices && e.positions.len() == g.positions.len())
        });
    if !compatible {
        a.append_assembly(start);
        return;
    }
    let fixed: Vec<_> = start
        .parts
        .iter()
        .filter(|(k, g)| g.positions == end.parts[*k].positions)
        .map(|(k, _)| k.clone())
        .collect();
    for key in fixed {
        a.part(key.0, &key.1)
            .append(start.parts.remove(&key).unwrap());
        end.parts.remove(&key);
    }
    if start.parts.is_empty() {
        return;
    }
    let mut joint = Joint::new(MotionKind::Covering, seed);
    joint.frame = frame;
    joint.morph = end.parts;
    a.articulated(start, joint);
}
