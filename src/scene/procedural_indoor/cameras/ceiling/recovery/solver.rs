//! Budgeted reference-star constraint search over reusable qualified paths.
use super::*;

pub(super) struct Solver {
    visibility_left: usize,
    search_left: usize,
    // Pools only append immutable candidates. A calibrated pair's admission is
    // therefore reusable across domain widths and subsequent recovery passes.
    visibility: std::collections::BTreeMap<[usize; 3], bool>,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            visibility_left: 8192,
            search_left: 8192,
            visibility: Default::default(),
        }
    }
}

impl Solver {
    /// Rank reference domains by the number of supported destination slots,
    /// not by raw edge count (one easy ceiling slot may have many edges).
    pub(super) fn ranked_references(&self, count: usize, slots: usize) -> Vec<usize> {
        let mut supported = vec![vec![false; slots]; count];
        for (&[reference, slot, _], &accepted) in &self.visibility {
            if accepted {
                supported[reference][slot] = true;
            }
        }
        let mut order: Vec<_> = (0..count).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(supported[i].iter().filter(|&&yes| yes).count()));
        order
    }
    pub(super) fn compatible_candidates(&self, reference: usize) -> Vec<(usize, usize)> {
        self.visibility
            .iter()
            .filter_map(|(&[r, slot, candidate], &accepted)| {
                (r == reference && accepted).then_some((slot, candidate))
            })
            .collect()
    }
    fn cached_visibility(
        &mut self,
        key: [usize; 3],
        reserve: usize,
        evaluate: impl FnOnce() -> bool,
    ) -> Option<bool> {
        if let Some(&accepted) = self.visibility.get(&key) {
            return Some(accepted);
        }
        if self.visibility_left <= reserve {
            return None;
        }
        self.visibility_left -= 1;
        let accepted = evaluate();
        self.visibility.insert(key, accepted);
        Some(accepted)
    }

    pub(super) fn solve(
        &mut self,
        scene: &mut IndoorManifest,
        coverage: &Coverage,
        pools: &[Vec<Candidate>],
        contract: MixedRigContract,
        reserve: usize,
    ) -> bool {
        if pools.iter().any(Vec::is_empty) {
            return false;
        }
        let policy = scene.camera_settings.multiview.as_ref().unwrap();
        let mut order: Vec<_> = (1..pools.len()).collect();
        order.sort_by_key(|i| pools[*i].len());
        // Try small viable domains first. Enumerating every possible peer before
        // trying one assignment spends most ray tests on candidates never needed.
        // Widen on failure; cached pair decisions keep this bounded and exact.
        for width in [1, 4, 16, usize::MAX] {
            for reference_index in self.ranked_references(pools[0].len(), pools.len()) {
                let reference = &pools[0][reference_index];
                let mut domains: Vec<Vec<&Candidate>> =
                    (0..pools.len()).map(|_| Vec::new()).collect();
                for &index in &order {
                    // Recovery can contain several pinhole calibrations for one slot.
                    // Never reuse a footprint cap from a different realized FOV.
                    let mut floors = std::collections::BTreeMap::new();
                    for (candidate_index, candidate) in pools[index].iter().enumerate() {
                        if super::bodies_overlap(&reference.camera, &candidate.camera)
                            || !contract.accepts_geometry(
                                policy,
                                &[&reference.camera, &candidate.camera],
                                &[&reference.track, &candidate.track],
                                false,
                            )
                        {
                            continue;
                        }
                        let floor = *floors
                            .entry(candidate.camera.fov_degrees.to_bits())
                            .or_insert_with(|| {
                                contract.pair_overlap_floor(
                                    &reference.camera,
                                    &candidate.camera,
                                    scene.camera_aspect_ratio,
                                )
                            });
                        let Some(accepted) = self.cached_visibility(
                            [reference_index, index, candidate_index],
                            reserve,
                            || {
                                contract.accepts_visibility(
                                    policy,
                                    &reference.camera,
                                    &candidate.camera,
                                    &pair_samples_cached(
                                        coverage,
                                        &reference.camera,
                                        &reference.views,
                                        &candidate.camera,
                                        &candidate.views,
                                    ),
                                    floor,
                                )
                            },
                        ) else {
                            // Exhausting new ray tests cannot invalidate cached
                            // compatible paths later in this or another domain.
                            continue;
                        };
                        if accepted {
                            domains[index].push(candidate);
                            if domains[index].len() == width {
                                break;
                            }
                        }
                    }
                    if domains[index].is_empty() {
                        break;
                    }
                }
                if domains.iter().skip(1).any(Vec::is_empty) {
                    continue;
                }
                order.sort_by_key(|i| domains[*i].len());
                let mut selected = vec![reference];
                // A difficult early reference must not consume all assignment work and
                // starve later references or the next domain-expansion pass.
                let allowance = self.search_left.min(512);
                let mut budget = allowance;
                let found = assign(
                    &domains,
                    &order,
                    &mut selected,
                    policy,
                    contract,
                    &mut budget,
                );
                self.search_left -= allowance - budget;
                if found {
                    let mut result: Vec<_> = selected
                        .into_iter()
                        .zip(std::iter::once(0).chain(order.iter().copied()))
                        .collect();
                    result.sort_by_key(|(_, index)| *index);
                    scene.cameras = result.into_iter().map(|(c, _)| c.camera.clone()).collect();
                    return true;
                }
                if self.search_left == 0 {
                    return false;
                }
            }
        }

        false
    }
}

fn assign<'a>(
    domains: &[Vec<&'a Candidate>],
    order: &[usize],
    selected: &mut Vec<&'a Candidate>,
    policy: &super::super::super::multiview::MultiViewSettings,
    contract: MixedRigContract,
    budget: &mut usize,
) -> bool {
    if selected.len() == domains.len() {
        return true;
    }
    let slot = order[selected.len() - 1];
    for &candidate in &domains[slot] {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if selected
            .iter()
            .any(|c| super::bodies_overlap(&c.camera, &candidate.camera))
        {
            continue;
        }
        selected.push(candidate);
        let cameras: Vec<_> = selected.iter().map(|c| &c.camera).collect();
        let tracks: Vec<_> = selected.iter().map(|c| &c.track).collect();
        if contract.accepts_geometry(policy, &cameras, &tracks, selected.len() == domains.len())
            && assign(domains, order, selected, policy, contract, budget)
        {
            return true;
        }
        selected.pop();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::cameras::multiview::MultiViewSettings;

    #[test]
    fn reference_priority_counts_supported_slots_and_keeps_stable_ties() {
        let mut solver = Solver::default();
        for candidate in 0..32 {
            solver.visibility.insert([0, 1, candidate], true);
        }
        for slot in 1..4 {
            solver.visibility.insert([1, slot, 0], true);
        }
        for slot in 1..3 {
            solver.visibility.insert([2, slot, 0], true);
        }
        solver.visibility.insert([3, 1, 0], false);
        assert_eq!(solver.ranked_references(5, 4), [1, 2, 0, 3, 4]);
        assert_eq!(solver.compatible_candidates(1), [(1, 0), (2, 0), (3, 0)]);
        assert!(solver.compatible_candidates(3).is_empty());
    }

    #[test]
    fn visibility_cache_reuses_decisions_and_reserves_later_passes() {
        let mut solver = Solver {
            visibility_left: 2,
            ..Default::default()
        };
        assert_eq!(
            solver.cached_visibility([0, 1, 0], 1, || false),
            Some(false)
        );
        assert_eq!(solver.visibility_left, 1);
        assert_eq!(
            solver.cached_visibility([0, 1, 0], 1, || panic!("repeated ray test")),
            Some(false)
        );
        assert_eq!(
            solver.cached_visibility([0, 1, 1], 1, || panic!("spent reserved work")),
            None
        );
        assert_eq!(solver.cached_visibility([0, 1, 1], 0, || true), Some(true));
        assert_eq!(solver.visibility_left, 0);
        assert_eq!(
            solver.cached_visibility([0, 1, 1], 0, || panic!("cached after exhaustion")),
            Some(true)
        );
        assert_eq!(
            solver.cached_visibility([1, 1, 1], 0, || panic!("different reference")),
            None
        );
        assert_eq!(solver.visibility.len(), 2);
    }

    fn candidate(x: f32, z: f32) -> Candidate {
        let start = Vec3::new(x, 1.5, z);
        let camera = IndoorCamera {
            ceiling_mount: None,
            fisheye: None,
            principal_point_offset: None,
            start,
            end: start,
            target: start + Vec3::NEG_Z * 2.,
            fov_degrees: 60.,
            motion: None,
        };
        Candidate {
            track: Track::new(&camera),
            camera,
            views: Vec::new(),
        }
    }

    #[test]
    fn exhausted_visibility_budget_still_assembles_previously_qualified_paths() {
        use crate::scene::procedural_indoor::layout::IndoorLayout;
        let mut scene =
            IndoorManifest::generate_with_humans(12, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let policy = MultiViewSettings::from_baseline(0.5).unwrap();
        let contract = MixedRigContract::new(&policy).unwrap();
        scene.camera_settings.multiview = Some(policy);
        let coverage = Coverage::for_ceiling_search(&scene);
        let pools = vec![
            vec![candidate(0., 0.)],
            vec![candidate(-1.5, 0.), candidate(1.5, 0.)],
            vec![candidate(0., -1.5), candidate(0., 1.5)],
        ];
        let mut solver = Solver {
            visibility_left: 0,
            ..Default::default()
        };
        solver.visibility.insert([0, 1, 1], true);
        solver.visibility.insert([0, 2, 1], true);
        assert!(solver.solve(&mut scene, &coverage, &pools, contract, 0));
        assert_eq!(scene.cameras.len(), 3);
        assert_eq!(scene.cameras[1], pools[1][1].camera);
        assert_eq!(scene.cameras[2], pools[2][1].camera);
        assert_eq!(solver.visibility_left, 0);
    }

    #[test]
    fn assignment_backtracks_instead_of_accepting_a_colliding_or_partial_rig() {
        let policy = MultiViewSettings::from_baseline(0.5).unwrap();
        let contract = MixedRigContract::new(&policy).unwrap();
        let reference = candidate(0., 0.);
        let a = candidate(1.5, 0.);
        let b = candidate(0., 1.5);
        let domains = vec![vec![], vec![&a, &b], vec![&a]];
        let mut selected = vec![&reference];
        assert!(assign(
            &domains,
            &[1, 2],
            &mut selected,
            &policy,
            contract,
            &mut 16
        ));
        assert!(std::ptr::eq(selected[1], &b));
        assert!(std::ptr::eq(selected[2], &a));
        let mut selected = vec![&reference];
        assert!(!assign(
            &domains,
            &[1, 2],
            &mut selected,
            &policy,
            contract,
            &mut 0
        ));
        assert_eq!(selected.len(), 1);
        let impossible = vec![vec![], vec![&a], vec![&a]];
        assert!(!assign(
            &impossible,
            &[1, 2],
            &mut selected,
            &policy,
            contract,
            &mut 16
        ));
        assert_eq!(selected.len(), 1);
    }
}
