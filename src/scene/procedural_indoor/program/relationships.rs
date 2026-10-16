//! Accepted functional graph. Edges come from committed construction/placement
//! receipts; proximity alone never invents a support, socket or service link.
use super::super::layout::{IndoorManifest, ObjectKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", content = "index", rename_all = "snake_case")]
pub enum Participant {
    Object(usize),
    Person(usize),
    Region(usize),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relationship {
    Supports,
    SeatsAt,
    Powers,
    Serves,
    MountedAbove,
    Occupies,
    UsesSurface,
    SharesCirculation,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalEdge {
    pub source: Participant,
    pub relation: Relationship,
    pub target: Participant,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalCoverage {
    pub region: usize,
    pub program: String,
    pub requested_capacity: usize,
    pub accepted_capacity: usize,
    pub required_kinds: Vec<ObjectKind>,
    pub missing_required: Vec<ObjectKind>,
    pub rejected_candidates: usize,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalGraph {
    pub edges: Vec<FunctionalEdge>,
    pub coverage: Vec<FunctionalCoverage>,
}
impl IndoorManifest {
    /// Opt-in annotation/analysis API; no extra per-frame or generation work.
    pub fn functional_graph(&self) -> Result<FunctionalGraph, String> {
        use Participant::*;
        use Relationship::*;
        let mut edges = BTreeSet::new();
        let mut add = |source, relation, target| {
            edges.insert(FunctionalEdge {
                source,
                relation,
                target,
            });
        };
        for o in &self.objects {
            if let Some(s) = o.support {
                add(Object(s), Supports, Object(o.id));
            }
            if o.kind == ObjectKind::Chair {
                if let Some(s) = o.interaction_target {
                    add(Object(o.id), SeatsAt, Object(s));
                }
            }
            if o.kind == ObjectKind::Cable {
                if let (Some(source), Some(target)) = (o.interaction_target, o.cable_target) {
                    // Cable routing stores device -> outlet; power is supplied
                    // in the opposite direction, not by the cable instance.
                    add(Object(target), Powers, Object(source));
                }
            }
        }
        for h in &self.humans {
            if let Some(chair) = h.chair {
                add(Person(h.id), Occupies, Object(chair));
                if h.worktop_contact.is_some() {
                    if let Some(surface) =
                        self.objects.get(chair).and_then(|o| o.interaction_target)
                    {
                        add(Person(h.id), UsesSurface, Object(surface));
                    }
                }
            }
        }
        for mount in &self.assembly_diagnostics.mounts {
            add(Object(mount.object), MountedAbove, Object(mount.host));
        }
        let mut coverage = Vec::new();
        for p in &self.assembly_diagnostics.programs {
            for &id in &p.members {
                add(Object(id), Serves, Region(p.zone));
            }
            coverage.push(FunctionalCoverage {
                region: p.zone,
                program: format!("{:?}", p.activity),
                requested_capacity: p.requested_capacity,
                accepted_capacity: p.realized_capacity(&self.objects),
                required_kinds: p.required.clone(),
                missing_required: p
                    .required
                    .iter()
                    .filter(|&&kind| {
                        !p.members
                            .iter()
                            .any(|&id| self.objects.get(id).is_some_and(|o| o.kind == kind))
                    })
                    .copied()
                    .collect(),
                rejected_candidates: p.rejected_candidates,
            });
        }
        for k in &self.assembly_diagnostics.kitchens {
            for &id in &k.units {
                add(Object(id), Serves, Region(k.zone));
            }
            let required = vec![ObjectKind::KitchenSink, ObjectKind::KitchenCounter];
            let missing_required = required
                .iter()
                .filter(|&&kind| {
                    !k.units
                        .iter()
                        .any(|&id| self.objects.get(id).is_some_and(|o| o.kind == kind))
                })
                .copied()
                .collect();
            coverage.push(FunctionalCoverage {
                region: k.zone,
                program: "kitchen-service-run".into(),
                requested_capacity: k.requested.len(),
                accepted_capacity: k.units.len(),
                required_kinds: required,
                missing_required,
                rejected_candidates: k.rejected_optional,
            });
        }
        for connection in self.room_topology().open_connections {
            add(
                Region(connection.regions[0]),
                SharesCirculation,
                Region(connection.regions[1]),
            );
        }
        let graph = FunctionalGraph {
            edges: edges.into_iter().collect(),
            coverage,
        };
        graph.validate(self)?;
        Ok(graph)
    }
}
impl FunctionalGraph {
    pub fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        let exists = |p: Participant| match p {
            Participant::Object(i) => scene.objects.get(i).is_some_and(|o| o.id == i),
            Participant::Person(i) => scene.humans.iter().any(|h| h.id == i),
            Participant::Region(i) => scene.program.as_ref().is_some_and(|p| i < p.zones.len()),
        };
        for e in &self.edges {
            if !exists(e.source) || !exists(e.target) || e.source == e.target {
                return Err("dangling/self functional relationship".into());
            }
        }
        // A support chain must reach the floor/architecture; it cannot form a
        // cycle, even if every referenced instance happens to exist.
        for o in &scene.objects {
            let mut seen = BTreeSet::new();
            let mut current = Some(o.id);
            while let Some(id) = current {
                if !seen.insert(id) {
                    return Err("cyclic object support chain".into());
                }
                current = scene.objects.get(id).ok_or("dangling support")?.support;
            }
        }
        if self.coverage.iter().any(|c| !c.missing_required.is_empty()) {
            return Err("missing required functional role".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    #[test]
    fn committed_graph_is_complete_across_every_program() {
        for profile in IndoorLayout::PROFILES {
            let scene = IndoorManifest::generate_with_humans(37, profile, 0.65, 0, 0.4).unwrap();
            let graph = scene.functional_graph().unwrap();
            let replay: FunctionalGraph =
                serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
            assert_eq!(graph, replay);
            for o in &scene.objects {
                if let Some(s) = o.support {
                    assert!(graph.edges.contains(&FunctionalEdge {
                        source: Participant::Object(s),
                        relation: Relationship::Supports,
                        target: Participant::Object(o.id)
                    }));
                }
            }
        }
    }
    #[test]
    fn corrupt_support_cycle_cannot_be_reported_as_valid() {
        let mut scene =
            IndoorManifest::generate_with_humans(3, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
        scene.objects[0].support = Some(1);
        scene.objects[1].support = Some(0);
        assert!(scene.functional_graph().is_err());
    }
}
