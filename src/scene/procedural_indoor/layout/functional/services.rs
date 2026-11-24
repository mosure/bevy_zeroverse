//! Mounts and service links are explicit scene relationships, not decorative wires.
use super::*;
use crate::scene::procedural_indoor::objects::kitchen;
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
        for index in 0..self.objects.len() {
            let host = &self.objects[index];
            if host.neighbor {
                continue;
            }
            let mounted = match host.kind {
                ObjectKind::Range | ObjectKind::Cooktop => self.hood_over(host),
                ObjectKind::ExhibitPlinth => self.pendant_over(host),
                _ => continue,
            };
            if let Some(o) = mounted {
                self.assembly_diagnostics.mounts.push(MountedUnit {
                    object: o.id,
                    host: host.id,
                    ceiling_height_metres: self.mount_height(&o),
                });
                self.objects.push(o);
            }
        }
    }

    pub(in crate::scene::procedural_indoor::layout) fn record_program_services(&mut self) {
        for p in &mut self.assembly_diagnostics.programs {
            p.services.clear();
            for mount in &self.assembly_diagnostics.mounts {
                if !p.members.contains(&mount.host)
                    && !self.objects[mount.host]
                        .support
                        .is_some_and(|s| p.members.contains(&s))
                {
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
                    target_point: if object.kind == ObjectKind::ExtractionHood {
                        kitchen::appliances::cooking_surface(host, &kitchen::parameters(host)).0
                    } else {
                        host.position + Vec3::Y * host.size.y
                    },
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
                    kind: if super::super::super::objects::equipment::wires::data_variant(
                        self.objects[source].kind,
                        self.objects[target].kind,
                    )
                    .is_some()
                    {
                        ServiceKind::Data
                    } else {
                        ServiceKind::Power
                    },
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
            let ceiling = self.mount_height(o);
            let gap = if o.kind == ObjectKind::CeilingPendant && self.generator_version >= 184 {
                0.
            } else {
                0.004
            };
            let over_host = match (o.kind, host.kind) {
                (ObjectKind::ExtractionHood, ObjectKind::Range | ObjectKind::Cooktop) => {
                    let center = if self.generator_version >= 123 {
                        kitchen::appliances::cooking_surface(host, &kitchen::parameters(host)).0
                    } else {
                        host.position
                    };
                    o.position.xz().distance(center.xz()) <= 0.001
                }
                (ObjectKind::CeilingPendant, ObjectKind::ExhibitPlinth) => {
                    let p = host
                        .transform()
                        .compute_affine()
                        .inverse()
                        .transform_point3(o.position);
                    p.x.abs() <= host.size.x * 0.30 + 0.001
                        && p.z.abs() <= host.size.z * 0.30 + 0.001
                }
                _ => false,
            };
            if o.solid
                || o.support.is_some()
                || o.neighbor
                || !ceiling.is_finite()
                || !m.ceiling_height_metres.is_finite()
                || (m.ceiling_height_metres - ceiling).abs() > 0.002
                || (o.position.y + o.size.y + gap - ceiling).abs() > 0.002
                || !over_host
            {
                return Err("invalid ceiling/service mount relationship".into());
            }
        }
        Ok(())
    }
}
