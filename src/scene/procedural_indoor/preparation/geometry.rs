//! Immutable assembly construction and consumed-material selection.
use super::*;

/// One construction of each assembly, shared by diffuse transport and rendering.
/// Ownership passes into render meshes after the baker has copied its triangles.
pub(crate) struct SceneGeometry {
    pub architecture: objects::Assembly,
    pub objects: Vec<objects::Assembly>,
    pub humans: Vec<humans::HumanAssembly>,
}
impl SceneGeometry {
    /// Geometry, rather than probabilistic object kinds, decides which texture
    /// programs are needed. Finish parent handles remain available even when a
    /// nonzero structure replaces every map and no part uses the base slot.
    pub(crate) fn material_selection(&self, manifest: &IndoorManifest) -> MaterialSelection {
        let keys = self
            .architecture
            .parts
            .iter()
            .chain(self.objects.iter().flat_map(|assembly| &assembly.parts))
            .filter(|(_, geometry)| !geometry.indices.is_empty())
            .map(|(key, _)| key);
        let mut selection = MaterialSelection {
            surfaces: std::collections::BTreeSet::new(),
            finishes: std::collections::BTreeSet::new(),
            direct_surfaces: std::collections::BTreeSet::new(),
        };
        for (surface, label) in keys {
            // Rug programs build their own bounded maps; they do not consume
            // the upholstery parent unless a separate binding/seat uses it.
            if manifest.program.is_some()
                && materials::rugs::key(label).is_some_and(|(_, binding)| !binding)
            {
                continue;
            }
            selection.surfaces.insert(*surface);
            if let Some(slot) = label
                .rsplit_once("#finish")
                .and_then(|(_, slot)| slot.parse::<usize>().ok())
            {
                if materials::variants::supports(*surface) && slot < materials::variants::COUNT {
                    selection.finishes.insert((*surface, slot));
                } else {
                    selection.direct_surfaces.insert(*surface);
                }
            } else {
                selection.direct_surfaces.insert(*surface);
            }
        }
        if !manifest.humans.is_empty() {
            // Human material selection also uses neutral cloth, shoe leather
            // and sole rubber templates, separate from assembly surface keys.
            selection
                .surfaces
                .extend([Surface::Fabric, Surface::Leather, Surface::Rubber]);
            selection
                .direct_surfaces
                .extend([Surface::Fabric, Surface::Leather, Surface::Rubber]);
        }
        for person in &manifest.humans {
            for surface in [humans::HumanSurface::Top, humans::HumanSurface::Trousers] {
                if surface == humans::HumanSurface::Top
                    && person.outfit.knitted()
                    && !person.outfit.open_front()
                {
                    continue;
                }
                let finish = humans::cloth_finish(person, surface);
                selection.surfaces.insert(finish.0);
                selection.finishes.insert(finish);
            }
        }
        selection
    }

    pub(super) async fn build(scene: &IndoorManifest) -> (Self, timings::GeometryJobs) {
        cooperate().await;
        #[cfg(not(target_arch = "wasm32"))]
        {
            enum Part {
                Architecture(objects::Assembly),
                Object(objects::Assembly),
                Human(humans::HumanAssembly),
            }
            // All jobs are spawned directly by the scope: Bevy preserves this
            // order independently of completion order. Each builder owns its RNG.
            let parts = workers::pool().scope(|scope| {
                scope.spawn(async move {
                    let start = bevy::platform::time::Instant::now();
                    let part = Part::Architecture(architecture::architecture(scene));
                    (part, start.elapsed().as_secs_f64())
                });
                for object in &scene.objects {
                    scope.spawn(async move {
                        let start = bevy::platform::time::Instant::now();
                        let part = Part::Object(objects::build_object(object));
                        (part, start.elapsed().as_secs_f64())
                    });
                }
                for person in &scene.humans {
                    scope.spawn(async move {
                        let start = bevy::platform::time::Instant::now();
                        let part = Part::Human(humans::build_human(person));
                        (part, start.elapsed().as_secs_f64())
                    });
                }
            });
            let mut result = Self {
                architecture: default(),
                objects: Vec::with_capacity(scene.objects.len()),
                humans: Vec::with_capacity(scene.humans.len()),
            };
            let mut timings = timings::GeometryJobs::default();
            for (part, seconds) in parts {
                match part {
                    Part::Architecture(a) => {
                        timings.architecture.add(seconds);
                        result.architecture = a;
                    }
                    Part::Object(a) => {
                        timings.objects.add(seconds);
                        result.objects.push(a);
                    }
                    Part::Human(a) => {
                        timings.humans.add(seconds);
                        result.humans.push(a);
                    }
                }
            }
            (result, timings)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut timings = timings::GeometryJobs::default();
            let start = bevy::platform::time::Instant::now();
            let architecture = architecture::architecture(scene);
            timings.architecture.add(start.elapsed().as_secs_f64());
            let mut objects = Vec::with_capacity(scene.objects.len());
            for object in &scene.objects {
                cooperate().await;
                let start = bevy::platform::time::Instant::now();
                objects.push(objects::build_object(object));
                timings.objects.add(start.elapsed().as_secs_f64());
            }
            let mut humans = Vec::with_capacity(scene.humans.len());
            for person in &scene.humans {
                cooperate().await;
                let start = bevy::platform::time::Instant::now();
                humans.push(humans::build_human(person));
                timings.humans.add(start.elapsed().as_secs_f64());
            }
            (
                Self {
                    architecture,
                    objects,
                    humans,
                },
                timings,
            )
        }
    }
}
