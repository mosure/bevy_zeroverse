//! Mounts and service links are explicit scene relationships, not decorative wires.
use super::*;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MountedUnit {
    pub object: usize,
    pub host: usize,
    pub ceiling_height_metres: f32,
}
impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn finish_program_equipment(&mut self) {
        if !self.layout.is_public() && !self.layout.is_kitchen() {
            return;
        }
        let mut r = stream(self.seed, 8270);
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor && matches!(o.kind, ObjectKind::Range | ObjectKind::ExhibitPlinth)
            })
            .cloned()
            .collect();
        for host in anchors {
            let ceiling = self.ceiling_height(host.position.xz());
            let (kind, size, bottom) = if host.kind == ObjectKind::Range {
                let bottom = host.position.y + host.size.y + 0.72;
                if ceiling - bottom < 0.30 {
                    continue;
                }
                (
                    ObjectKind::ExtractionHood,
                    Vec3::new(host.size.x * 1.06, ceiling - bottom - 0.004, 0.62),
                    bottom,
                )
            } else {
                let size = Vec3::new(0.21, 0.46, 0.21);
                (ObjectKind::CeilingPendant, size, ceiling - size.y - 0.004)
            };
            let mut o = self.candidate(kind, host.position.with_y(bottom), size, host.yaw, &mut r);
            o.solid = false;
            let (lo, hi) = o.bounds();
            let clear = if kind == ObjectKind::CeilingPendant {
                super::super::equipment::ceiling_fixture_clear(self, &o)
            } else {
                [
                    lo.xz(),
                    hi.xz(),
                    Vec2::new(lo.x, hi.z),
                    Vec2::new(hi.x, lo.z),
                ]
                .iter()
                .all(|&p| self.ceiling_height(p) >= hi.y + 0.001)
                    && self.objects.iter().all(|other| {
                        let (a, b) = other.bounds();
                        !lo.cmplt(b + Vec3::splat(0.012)).all()
                            || !hi.cmpgt(a - Vec3::splat(0.012)).all()
                    })
                    && self
                        .columns()
                        .iter()
                        .all(|&(a, b)| !lo.cmplt(b).all() || !hi.cmpgt(a).all())
            };
            if clear {
                self.assembly_diagnostics.mounts.push(MountedUnit {
                    object: o.id,
                    host: host.id,
                    ceiling_height_metres: ceiling,
                });
                self.objects.push(o);
            }
        }
    }
    pub(in crate::scene::procedural_indoor::layout) fn record_program_services(&mut self) {
        for p in &mut self.assembly_diagnostics.programs {
            p.services.clear();
            for mount in &self.assembly_diagnostics.mounts {
                if !p.members.contains(&mount.host) {
                    continue;
                }
                let object = &self.objects[mount.object];
                let host = &self.objects[mount.host];
                p.services.push(ServiceConnection {
                    kind: if object.kind == ObjectKind::ExtractionHood {
                        ServiceKind::Ventilation
                    } else {
                        ServiceKind::Display
                    },
                    source: object.id,
                    target: host.id,
                    source_point: object.position,
                    target_point: host.position + Vec3::Y * host.size.y,
                });
            }
            for cable in self.objects.iter().filter(|o| o.kind == ObjectKind::Cable) {
                let (Some(source), Some(target), Some(path)) = (
                    cable.interaction_target,
                    cable.cable_target,
                    cable.cable_path.as_ref(),
                ) else {
                    continue;
                };
                if !p.members.contains(&source)
                    && !self.objects[source]
                        .support
                        .is_some_and(|s| p.members.contains(&s))
                {
                    continue;
                }
                p.services.push(ServiceConnection {
                    kind: ServiceKind::Power,
                    source,
                    target,
                    source_point: cable.transform().transform_point(path[0]),
                    target_point: cable.transform().transform_point(*path.last().unwrap()),
                });
            }
        }
    }
    pub(crate) fn validate_program_mounts(&self) -> Result<(), String> {
        for m in &self.assembly_diagnostics.mounts {
            let o = self.objects.get(m.object).ok_or("missing mounted object")?;
            let host = self.objects.get(m.host).ok_or("missing mount host")?;
            if o.solid
                || o.support.is_some()
                || o.neighbor
                || !m.ceiling_height_metres.is_finite()
                || (o.position.y + o.size.y + 0.004 - self.ceiling_height(o.position.xz())).abs()
                    > 0.002
                || o.position.xz().distance(host.position.xz()) > 0.001
                || !matches!(
                    (o.kind, host.kind),
                    (ObjectKind::ExtractionHood, ObjectKind::Range)
                        | (ObjectKind::CeilingPendant, ObjectKind::ExhibitPlinth)
                )
            {
                return Err("invalid ceiling/service mount relationship".into());
            }
        }
        Ok(())
    }
}
