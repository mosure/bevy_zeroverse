//! Configurable multi-view sets for reconstruction: camera zero is the reference.
//! Visibility uses first-surface proxy ray casts. It is a proposal constraint,
//! not a guarantee about rendered pixels; the depth audit measures actual overlap.
use super::coverage::{Coverage, VisibilityView};
use super::diversity::{geometry, Track};
use crate::scene::procedural_indoor::layout::{stream, IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod connected;
mod recovery;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlapGraph {
    #[default]
    Reference,
    Connected,
}
impl OverlapGraph {
    fn is_reference(&self) -> bool {
        *self == Self::Reference
    }
}

pub const CHECK_TIMES: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const GROUP_ATTEMPTS: usize = 64;
const VIEW_ATTEMPTS: usize = 384;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MultiViewSettings {
    /// Reference preserves archived rigs. Connected requires a stable spanning
    /// graph of bidirectionally overlapping pairs, not visibility to camera zero.
    #[serde(default, skip_serializing_if = "OverlapGraph::is_reference")]
    pub graph: OverlapGraph,
    /// Minimum fraction in BOTH directions, at every CHECK_TIMES sample.
    pub min_overlap: f32,
    /// Minimum Euclidean separation between EVERY pair, throughout sampled paths.
    pub min_baseline: f32,
    /// Minimum reference-to-view distance. Zero retains legacy custom policies.
    #[serde(default)]
    pub min_reference_baseline: f32,
    /// Maximum Euclidean reference-to-view separation, throughout sampled paths.
    pub max_baseline: f32,
    /// Horizontal minor/major standard-deviation ratio for groups of 3+ cameras.
    /// Zero permits collinear rigs; one requests an isotropic footprint.
    pub min_spread: f32,
    /// Independent heading, travel and curvature variation; zero permits rigid rigs.
    pub trajectory_variation: f32,
    /// Visible-region gaze and count-aware viewpoint selection; zero retains
    /// shared-target rigs. Archived policies omitting this field retain zero.
    #[serde(default)]
    pub target_variation: f32,
}
impl Default for MultiViewSettings {
    fn default() -> Self {
        Self::from_baseline(0.5).unwrap()
    }
}
impl MultiViewSettings {
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_overlap.is_finite()
            || !(0.0..=1.0).contains(&self.min_overlap)
            || !self.min_baseline.is_finite()
            || !self.max_baseline.is_finite()
            || self.min_baseline <= 0.0
            || self.max_baseline < self.min_baseline
            || self.max_baseline > 100.0
            || !self.min_reference_baseline.is_finite()
            || self.min_reference_baseline < 0.0
            || self.min_reference_baseline > self.max_baseline
            || !(0.0..=1.0).contains(&self.min_spread)
            || !(0.0..=1.0).contains(&self.trajectory_variation)
            || !(0.0..=1.0).contains(&self.target_variation)
        {
            return Err("indoor_camera.multiview requires min_overlap, min_spread, trajectory_variation and target_variation in [0,1] and 0 < min_baseline <= max_baseline <= 100 metres, with 0 <= min_reference_baseline <= max_baseline".into());
        }
        Ok(())
    }
    pub fn accepts(&self, samples: &[OverlapSample]) -> bool {
        samples.iter().all(|s| {
            s.reference_to_view.min(s.view_to_reference) >= self.min_overlap
                && s.baseline_m + 1e-5 >= self.min_baseline.max(self.min_reference_baseline)
                && s.baseline_m <= self.max_baseline + 1e-5
        })
    }
}

// Stored cameras from older generators were translated copies and did not have a
// spread contract. New requests get the new defaults; archives keep their policy.
pub(super) fn deserialize_archived_policy<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<MultiViewSettings>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    value
        .map(|mut value| {
            if let Some(object) = value.as_object_mut() {
                object.entry("min_spread").or_insert(0.0.into());
                object.entry("trajectory_variation").or_insert(0.0.into());
            }
            serde_json::from_value(value).map_err(serde::de::Error::custom)
        })
        .transpose()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapSample {
    pub time: f32,
    pub reference_to_view: f32,
    pub view_to_reference: f32,
    pub baseline_m: f32,
    pub mean_triangulation_degrees: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPairOverlap {
    pub reference: usize,
    pub camera: usize,
    pub samples: Vec<OverlapSample>,
}

fn pair_samples(
    coverage: &Coverage,
    reference: &IndoorCamera,
    reference_views: &[VisibilityView],
    camera: &IndoorCamera,
    aspect: f32,
) -> Vec<OverlapSample> {
    let camera_views: Vec<_> = CHECK_TIMES
        .into_iter()
        .map(|t| coverage.view(camera, aspect, t))
        .collect();
    pair_samples_cached(coverage, reference, reference_views, camera, &camera_views)
}

pub(super) fn pair_samples_cached(
    coverage: &Coverage,
    reference: &IndoorCamera,
    reference_views: &[VisibilityView],
    camera: &IndoorCamera,
    camera_views: &[VisibilityView],
) -> Vec<OverlapSample> {
    CHECK_TIMES
        .into_iter()
        .zip(reference_views)
        .zip(camera_views)
        .map(|((time, a), b)| {
            let (ab, angle_a) = coverage.shared(a, b);
            let (ba, angle_b) = coverage.shared(b, a);
            OverlapSample {
                time,
                reference_to_view: ab,
                view_to_reference: ba,
                baseline_m: reference
                    .transform_at(time)
                    .translation
                    .distance(camera.transform_at(time).translation),
                mean_triangulation_degrees: (angle_a + angle_b) * 0.5,
            }
        })
        .collect()
}

impl IndoorManifest {
    /// O(cameras) reference edges, including when the placement policy is off.
    pub fn camera_overlap(&self) -> Vec<CameraPairOverlap> {
        let Some(reference) = self.cameras.first() else {
            return Vec::new();
        };
        if self.cameras.len() < 2 {
            return Vec::new();
        }
        if self
            .camera_settings
            .multiview
            .as_ref()
            .is_some_and(|p| p.graph == OverlapGraph::Connected)
        {
            return connected::all_pairs(self);
        }
        let coverage = Coverage::new(self);
        let reference_views: Vec<_> = CHECK_TIMES
            .into_iter()
            .map(|t| coverage.view(reference, self.camera_aspect_ratio, t))
            .collect();
        self.cameras
            .iter()
            .enumerate()
            .skip(1)
            .map(|(camera, view)| CameraPairOverlap {
                reference: 0,
                camera,
                samples: pair_samples(
                    &coverage,
                    reference,
                    &reference_views,
                    view,
                    self.camera_aspect_ratio,
                ),
            })
            .collect()
    }

    pub(crate) fn sample_cameras(&mut self, count: usize) -> Result<(), String> {
        self.camera_diagnostics = default();
        if (0..count).any(|i| self.camera_settings.ceiling_mounted(self.seed, i)) {
            super::ceiling::sample(self, count)?;
        } else {
            self.sample_camera_set(count, true, self.seed)?;
        }
        if self.camera_settings.principal_point_jitter != Vec2::ZERO {
            let coverage = Coverage::new(self);
            for camera in &mut self.cameras {
                camera.principal_point_offset = Some(coverage.optical_center(camera));
            }
        }
        Ok(())
    }

    fn sample_camera_set(
        &mut self,
        count: usize,
        allow_overcomplete: bool,
        proposal_seed: u64,
    ) -> Result<(), String> {
        self.camera_settings.validate()?;
        let mut rng = stream(proposal_seed, 3);
        let coverage = Coverage::new(self);
        self.cameras.clear();
        let Some(policy) = self.camera_settings.multiview.clone().filter(|_| count > 1) else {
            return self.sample_independent_cameras(count, &mut rng, &coverage);
        };
        if policy.graph == OverlapGraph::Connected {
            return match connected::sample(self, count, &policy, &mut rng, &coverage) {
                Ok(()) => Ok(()),
                Err(connected_error) => {
                    // A reference star is a valid connected graph. Reuse its
                    // bounded room-space recovery rather than weaken a failed
                    // connected request's clearance, path or overlap contract.
                    self.camera_diagnostics.connected_reference_recoveries += 1;
                    self.camera_settings.multiview.as_mut().unwrap().graph =
                        OverlapGraph::Reference;
                    let result = self.sample_camera_set(count, allow_overcomplete, proposal_seed);
                    self.camera_settings.multiview.as_mut().unwrap().graph =
                        OverlapGraph::Connected;
                    result.map_err(|e| format!("{connected_error}; reference recovery: {e}"))
                }
            };
        }
        // Select strata once for this seed. A failed difficult/negative stratum
        // must not be redrawn as an easier one during anchor or view retries.
        let targets = self
            .camera_settings
            .overlap_mixture
            .as_ref()
            .map_or_else(Vec::new, |m| m.targets(self.seed, count));
        // Reserve constrained shared-content views before placing negatives.
        // Class assignments and final camera indices remain fixed for this seed.
        let placement_order = super::mixture::placement_order(&targets, count);
        // Retry the whole set if an anchor lies in a cramped cul-de-sac. The
        // budget is bounded and thresholds are never silently relaxed.
        let mut most_placed = 0;
        let mut rejected = [0usize; 6];
        let mut partial_groups = Vec::new();
        let viewpoint = super::viewpoint::ViewpointProgram::sample(self.seed);
        for _ in 0..GROUP_ATTEMPTS {
            self.cameras.clear();
            if self
                .sample_independent_cameras(1, &mut rng, &coverage)
                .is_err()
            {
                // One exhausted anchor must not abort the bounded group search.
                // Settings were validated before entering this loop.
                continue;
            }
            let reference = self.cameras[0].clone();
            let (lo, hi) = if self.camera_settings.primary_room {
                self.primary_room_bounds()
            } else {
                let half = self.room_size.xz() * 0.5;
                (-half, half)
            };
            // Bound proposals by reachable room space without weakening policy.
            let reachable = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                .into_iter()
                .map(|p| p.distance(reference.start.xz()))
                .fold(0.0_f32, f32::max)
                .hypot(0.9)
                .min(policy.max_baseline);
            if reachable < policy.min_baseline.max(policy.min_reference_baseline) {
                continue;
            }
            let ceiling = (self.room_size.y
                - self.program.as_ref().map_or(0.5, |p| p.light_drop)
                - 0.04
                - super::super::layout::CAMERA_CLEARANCE)
                .min(3.25);
            let heights = Vec2::new(0.70 - reference.start.y, ceiling - reference.start.y);
            let extent = reference
                .start
                .distance(reference.end)
                .max(reference.path_length() * 0.5)
                .max(0.001);
            let mut tracks = vec![Track::new(&reference)];
            let views: Vec<_> = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&reference, self.camera_aspect_ratio, t))
                .collect();
            let mut summaries = vec![super::program::ViewSummary::new(
                &reference, &views[0], self,
            )];
            for &camera_index in &placement_order {
                let target = targets.get(camera_index - 1);
                let mut found = None;
                let mut first_valid = None;
                let mut accepted = 0;
                let pool = super::program::pool_size(policy.target_variation, count);
                for attempt in 0..VIEW_ATTEMPTS {
                    self.camera_diagnostics.peer_proposals += 1;
                    if first_valid.is_some_and(|first| attempt >= first + 12) || accepted >= pool {
                        break;
                    }

                    // Larger rigs in concave rooms benefit from proposals near
                    // already verified free corridors. Keep the same bounded
                    // budget and all overlap/spread/separation constraints.
                    let mut candidate = if count > 4
                        && attempt >= VIEW_ATTEMPTS / 2
                        && self.cameras.len() >= 4
                        && rng.random_bool(0.5)
                    {
                        let template = &self.cameras[rng.random_range(1..self.cameras.len())];
                        let local_policy = MultiViewSettings {
                            min_reference_baseline: 0.,
                            ..policy.clone()
                        };
                        let mut camera = propose(
                            template,
                            template.path_length().max(0.001),
                            reachable.min(1.2).max(policy.min_baseline),
                            Vec2::new(0.70 - template.start.y, ceiling - template.start.y),
                            &local_policy,
                            &self.camera_settings.intrinsics,
                            &mut rng,
                        );
                        // A sloping roof leaves a different usable height at
                        // each origin. Sample that interval directly instead of
                        // repeatedly proposing above the local ceiling. Move
                        // the entire path; all swept and rig checks still apply.
                        let low = self.floor_height(camera.start.xz()) + 0.70;
                        let high = (self.ceiling_height(camera.start.xz())
                            - self.program.as_ref().map_or(0.5, |p| p.light_drop)
                            - 0.04
                            - super::super::layout::CAMERA_CLEARANCE)
                            .min(3.25);
                        if low > high {
                            continue;
                        }
                        let offset = Vec3::Y * (rng.random_range(low..=high) - camera.start.y);
                        camera.start += offset;
                        camera.end += offset;
                        if let Some(motion) = &mut camera.motion {
                            for point in motion.control.iter_mut().chain(&mut motion.route) {
                                *point += offset;
                            }
                        }
                        camera
                    } else {
                        propose(
                            &reference,
                            extent,
                            reachable,
                            heights,
                            &policy,
                            &self.camera_settings.intrinsics,
                            &mut rng,
                        )
                    };
                    if target.is_some() && attempt % 2 == 1 {
                        // A translated reference path over-proposes outside a
                        // compact/concave room. Mix in independent origins in
                        // its actual floor-to-roof interval, then retain all
                        // clearance, rig and requested overlap tests below.
                        let xz = Vec2::new(
                            rng.random_range(lo.x + 0.5..hi.x - 0.5),
                            rng.random_range(lo.y + 0.5..hi.y - 0.5),
                        );
                        let low = self.floor_height(xz) + 0.70;
                        let high = (self.ceiling_height(xz)
                            - self.program.as_ref().map_or(0.5, |p| p.light_drop)
                            - 0.04
                            - super::super::layout::CAMERA_CLEARANCE)
                            .min(3.25);
                        if low > high {
                            continue;
                        }
                        let origin = Vec3::new(xz.x, rng.random_range(low..=high), xz.y);
                        let offset = origin - candidate.start;
                        candidate.start += offset;
                        candidate.end += offset;
                        if let Some(motion) = &mut candidate.motion {
                            for point in motion.control.iter_mut().chain(&mut motion.route) {
                                *point += offset;
                            }
                        }
                    }
                    if target.is_some_and(|t| t.class != super::mixture::OverlapClass::High) {
                        // Low/negative pairs need independent headings. Translating
                        // a shared-target rig cannot cover those requested strata.
                        let yaw = rng.random_range(0.0..std::f32::consts::TAU);
                        let aim = candidate.start
                            + Vec3::new(
                                4.0 * yaw.cos(),
                                rng.random_range(-0.6..0.6),
                                4.0 * yaw.sin(),
                            );
                        let offset = aim - candidate.target;
                        candidate.target = aim;
                        if let Some(motion) = &mut candidate.motion {
                            motion.target_end += offset;
                        }
                    }
                    // Shared-surface retargeting would undo the independent
                    // heading requested for a low/negative pair. Keep the same
                    // oracle below, but propose from the requested stratum.
                    if target.is_none_or(|t| t.class == super::mixture::OverlapClass::High) {
                        super::program::retarget(
                            self,
                            &mut candidate,
                            &views,
                            policy.target_variation,
                            count,
                            &mut rng,
                        );
                    }
                    if policy.target_variation > 0. {
                        viewpoint.level_gaze(&mut candidate, &mut rng);
                    }
                    self.camera_settings.prepare_motion(
                        self.seed,
                        camera_index,
                        &mut candidate,
                        &mut rng,
                    );
                    let track = Track::new(&candidate);
                    // Cheap rejection before casts: swept collision checks still
                    // cover the complete paths, including route segments/curves.
                    let rejection = if !self.camera_clear(candidate.start) {
                        Some(0)
                    } else if !geometry(&tracks, Some(&track)).unwrap().accepts(&policy) {
                        Some(1)
                    } else if !self.camera_settings.accepts_motion(
                        self.seed,
                        camera_index,
                        &candidate,
                    ) {
                        Some(2)
                    } else if !self.camera_curve_clear(&candidate) {
                        Some(3)
                    } else if !coverage.suitable(&candidate, false) {
                        Some(4)
                    } else {
                        None
                    };
                    if let Some(reason) = rejection {
                        rejected[reason] += 1;
                        self.camera_diagnostics.peer_rejections[reason] += 1;
                        continue;
                    }
                    let candidate_views: Vec<_> = CHECK_TIMES
                        .into_iter()
                        .map(|t| coverage.view(&candidate, self.camera_aspect_ratio, t))
                        .collect();
                    let samples = pair_samples_cached(
                        &coverage,
                        &reference,
                        &views,
                        &candidate,
                        &candidate_views,
                    );
                    if target.map_or_else(
                        || policy.accepts(&samples),
                        |t| t.accepts(&policy, &samples),
                    ) {
                        let summary =
                            super::program::ViewSummary::new(&candidate, &candidate_views[0], self);
                        let score = summary.score(&summaries, count, self);
                        accepted += 1;
                        first_valid.get_or_insert(attempt);
                        if found.as_ref().is_none_or(|(_, _, _, best)| score > *best) {
                            found = Some((candidate, track, summary, score));
                        }
                        if pool == 1 || score >= 0.85 {
                            break;
                        }
                        continue;
                    }
                    rejected[5] += 1;
                    self.camera_diagnostics.peer_rejections[5] += 1;
                }
                if let Some((camera, track, summary, _)) = found {
                    self.cameras.push(camera);
                    tracks.push(track);
                    summaries.push(summary);
                } else {
                    break;
                }
            }
            if self.cameras.len() == count {
                if !targets.is_empty() {
                    restore_camera_order(&mut self.cameras, &placement_order);
                }
                return Ok(());
            }
            // Retain the best partial group and diverse references. Mixtures
            // keep this same placement order throughout bounded recovery.
            // Retention consumes no RNG and leaves successful searches untouched.
            recovery::retain_diverse(&mut partial_groups, &self.cameras);
            most_placed = most_placed.max(self.cameras.len());
            self.camera_diagnostics.group_restarts += 1;
        }
        let recovered = recovery::complete(
            self,
            &partial_groups,
            count,
            &policy,
            &mut rng,
            &coverage,
            &mut rejected,
        );
        most_placed = most_placed.max(recovered);
        if recovered == count {
            if !targets.is_empty() {
                restore_camera_order(&mut self.cameras, &placement_order);
            }
            return Ok(());
        }
        // With the shared-target policy, rigs above four views use the same
        // proposal sequence regardless of count. Asking for one more camera just
        // repeats the exhausted search. Spend that existing second-pass budget
        // on an independent placement stream for the requested count instead.
        // The manifest seed, viewpoint prior, motion roles, geometry and all
        // requested constraints stay fixed. Ordinary successful rigs are untouched.
        if allow_overcomplete
            && targets.is_empty()
            && (5..12).contains(&count)
            && policy.target_variation == 0.
        {
            return self.sample_camera_set(count, false, self.seed.wrapping_add(2));
        }
        // Greedy packing is not monotonic in camera count. One bounded
        // overcomplete search can expose a feasible set missed by the requested
        // count's proposal sequence. Every retained subset rechecks the same
        // synchronized geometry; reference overlap and swept clearance remain
        // valid because removing a peer cannot invalidate those checks.
        if allow_overcomplete
            && targets.is_empty()
            && (2..12).contains(&count)
            && self
                .sample_camera_set(count + 1, false, proposal_seed)
                .is_ok()
        {
            let complete = self.cameras.clone();
            for omit in (1..complete.len()).rev() {
                let subset: Vec<_> = complete
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != omit)
                    .map(|(_, c)| c.clone())
                    .collect();
                // Removing a view must preserve both motion and the complete
                // calibration of shifted slots, not merely their lens model.
                if !(omit..count).all(|i| {
                    self.camera_settings.camera_moves(self.seed, i)
                        == self.camera_settings.camera_moves(self.seed, i + 1)
                        && self.camera_settings.lenses.as_ref().is_none_or(|prior| {
                            prior.sample(self.seed, i) == prior.sample(self.seed, i + 1)
                        })
                }) {
                    continue;
                }
                let subset_tracks: Vec<_> = subset.iter().map(Track::new).collect();
                if geometry(&subset_tracks, None).unwrap().accepts(&policy) {
                    self.cameras = subset;
                    return Ok(());
                }
            }
        }
        self.cameras.clear();
        Err(format!("seed {} ({:?}): unable to sample {count} multi-view cameras after {GROUP_ATTEMPTS} anchors x {VIEW_ATTEMPTS} proposals/view and bounded room-space recovery; most_placed={most_placed}, rejections clearance/rig/length/curve/coverage/overlap={rejected:?}; requested_pairs={targets:?}; min_overlap={}, pair_min={}m, reference={}..{}m, min_spread={}, trajectory_variation={}. Lower overlap/separation/spread bounds, reduce trajectory variation, or shorten paths; no unconstrained fallback was used", self.seed, self.layout, policy.min_overlap, policy.min_baseline, policy.min_reference_baseline, policy.max_baseline, policy.min_spread, policy.trajectory_variation))
    }
}

fn restore_camera_order(cameras: &mut Vec<IndoorCamera>, placement_order: &[usize]) {
    let indices = std::iter::once(0).chain(placement_order.iter().copied());
    let mut indexed: Vec<_> = indices.zip(std::mem::take(cameras)).collect();
    indexed.sort_by_key(|(index, _)| *index);
    *cameras = indexed.into_iter().map(|(_, camera)| camera).collect();
}

fn propose(
    reference: &IndoorCamera,
    extent: f32,
    reachable: f32,
    heights: Vec2,
    policy: &MultiViewSettings,
    intrinsics: &super::intrinsics::IntrinsicsSettings,
    rng: &mut impl Rng,
) -> IndoorCamera {
    // Uniform metric baselines avoid the former logarithmic bias toward an
    // almost coincident cluster around the anchor.
    let radius =
        rng.random_range(policy.min_baseline.max(policy.min_reference_baseline)..=reachable);
    let azimuth = rng.random_range(0.0..std::f32::consts::TAU);
    // Sample usable heights directly; wide rigs may look over furniture that
    // occludes a low reference camera. The complete path is still swept below.
    let vertical_limit = (radius * 0.4).min(0.9);
    let (low, high) = (
        heights.x.max(-vertical_limit),
        heights.y.min(vertical_limit),
    );
    if low > high {
        // No feasible vertical interval at this radius. The unchanged reference
        // fails pair separation and is rejected by the caller, without panicking.
        return reference.clone();
    }
    let vertical = rng.random_range(low..=high);
    let planar = (radius * radius - vertical * vertical).sqrt();
    let offset = Vec3::new(planar * azimuth.cos(), vertical, planar * azimuth.sin());
    let horizontal_jitter = 0.03 + 0.32 * policy.target_variation;
    let vertical_jitter = 0.015 + 0.135 * policy.target_variation;
    let jitter = Vec3::new(
        rng.random_range(-horizontal_jitter..horizontal_jitter),
        rng.random_range(-vertical_jitter..vertical_jitter),
        rng.random_range(-horizontal_jitter..horizontal_jitter),
    );
    // Deform in the reference path's local frame, preserving its navigated
    // topology while independently varying heading, distance, bend and height.
    // Long paths have bounded deformation so they can still share scene content.
    let budget = (reachable * 0.65).min(extent) * policy.trajectory_variation;
    let angle = rng.random_range(-1.0..=1.0) * (budget / extent).min(std::f32::consts::PI);
    let rotation = Quat::from_rotation_y(angle);
    let scale = 1.0 + rng.random_range(-1.0..=1.0) * (budget / extent).min(0.65);
    let deform = |p| reference.start + offset + rotation * (p - reference.start) * scale;
    let mut camera = reference.clone();
    camera.start = deform(reference.start);
    camera.end = deform(reference.end);
    camera.target += jitter;
    // Intrinsics remain independently sampled; overlap rejects extreme mismatches.
    camera.fov_degrees = intrinsics.sample(rng);
    if let Some(motion) = &mut camera.motion {
        let bend = Vec3::new(
            rng.random_range(-1.0..=1.0),
            rng.random_range(-0.25..=0.25),
            rng.random_range(-1.0..=1.0),
        ) * budget.min(extent)
            * 0.25;
        for p in &mut motion.control {
            *p = deform(*p) + bend;
        }
        for p in &mut motion.route {
            *p = deform(*p);
        }
        motion.target_end += jitter;
        motion.roll = [rng.random_range(-0.09..0.09), rng.random_range(-0.09..0.09)];
    }
    camera
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::CameraSettings, layout::IndoorLayout, validation::validate_layout,
    };

    #[test]
    fn fisheye_recovery_keeps_each_slots_calibration_and_all_admission_constraints() {
        // Production training rejection: recovery reused a saved pinhole path
        // for fisheye slot 2 before testing coverage and rendered overlap.
        let mut scene =
            IndoorManifest::generate_with_humans(64_989_645, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        let original = scene.clone();
        let settings = CameraSettings::parse(r#"{"primary_room":false,"path_length_min":0.8,"path_length_max":1.6,"long_path_fraction":0.0,"lenses":{"probability":0.25},"multiview":{"max_baseline":6.0,"min_baseline":0.4,"min_overlap":0.15,"min_reference_baseline":0.8,"min_spread":0.25,"trajectory_variation":0.3}}"#).unwrap();
        scene.resample_cameras(8, settings.clone(), 1.).unwrap();
        validate_layout(&scene).unwrap();
        assert_eq!(scene.objects, original.objects);
        assert_eq!(scene.envelope, original.envelope);
        assert_eq!(scene.humans, original.humans);
        assert_eq!(scene.camera_settings, settings);
        assert_eq!(scene.cameras.len(), 8);
        assert!(scene.cameras[2].fisheye.is_some());
        let policy = settings.multiview.as_ref().unwrap();
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
        let coverage = Coverage::new(&scene);
        for (slot, camera) in scene.cameras.iter().enumerate() {
            assert_eq!(
                camera.fisheye,
                settings.lenses.as_ref().unwrap().sample(scene.seed, slot)
            );
            assert!(settings.accepts_motion(scene.seed, slot, camera));
            assert!(scene.camera_curve_clear(camera));
            assert!(coverage.suitable(camera, false));
            for step in 0..=128 {
                assert!(scene.camera_clear(camera.transform_at(step as f32 / 128.).translation));
            }
        }
        let cameras = scene.cameras.clone();
        scene.resample_cameras(8, settings, 1.).unwrap();
        assert_eq!(scene.cameras, cameras);
    }

    #[test]
    fn rejected_training_library_recovers_eight_cameras_without_relaxation() {
        let mut scene =
            IndoorManifest::generate_with_humans(63_343_058, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        let original = scene.clone();
        let settings = CameraSettings::parse(r#"{"primary_room":false,"path_length_min":0.8,"path_length_max":1.6,"long_path_fraction":0.0,"multiview":{"max_baseline":6.0,"min_baseline":0.4,"min_overlap":0.15,"min_reference_baseline":0.8,"min_spread":0.25,"trajectory_variation":0.3}}"#).unwrap();
        scene.resample_cameras(8, settings.clone(), 1.).unwrap();
        validate_layout(&scene).unwrap();
        assert_eq!(scene.objects, original.objects);
        assert_eq!(scene.envelope, original.envelope);
        assert_eq!(scene.humans, original.humans);
        assert_eq!(scene.camera_settings, settings);
        assert_eq!(scene.cameras.len(), 8);
        let policy = settings.multiview.as_ref().unwrap();
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
        let coverage = Coverage::new(&scene);
        for (index, camera) in scene.cameras.iter().enumerate() {
            assert!(settings.accepts_motion(scene.seed, index, camera));
            assert!(scene.camera_curve_clear(camera));
            assert!(coverage.suitable(camera, false));
            for i in 0..=128 {
                assert!(scene.camera_clear(camera.transform_at(i as f32 / 128.).translation));
            }
        }
        let cameras = scene.cameras.clone();
        scene.resample_cameras(8, settings, 1.).unwrap();
        assert_eq!(scene.cameras, cameras);
    }

    #[test]
    fn dense_sloped_room_keeps_eight_connected_paths() {
        let scene = IndoorManifest::generate(4, IndoorLayout::Reception, 1., 8).unwrap();
        validate_layout(&scene).unwrap();
        assert_eq!(scene.cameras.len(), 8);
        let policy = scene.camera_settings.multiview.as_ref().unwrap();
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
    }

    #[test]
    fn compact_full_occupancy_group_recovers_without_relaxing_policy() {
        let mut scene =
            IndoorManifest::generate_with_humans(158, IndoorLayout::Mixed, 0., 0, 1.).unwrap();
        let original = scene.clone();
        let settings = CameraSettings::default();
        let policy = settings.multiview.as_ref().unwrap();
        scene.resample_cameras(4, settings.clone(), 1.).unwrap();
        validate_layout(&scene).unwrap();
        assert_eq!(scene.objects, original.objects);
        assert_eq!(scene.humans, original.humans);
        assert_eq!(scene.cameras.len(), 4);
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
        let coverage = Coverage::new(&scene);
        for (index, camera) in scene.cameras.iter().enumerate() {
            assert!(scene.camera_clear(camera.start));
            assert!(scene.camera_curve_clear(camera));
            assert!(coverage.suitable(camera, index == 0));
            assert!(camera.path_length() + 1e-4 >= settings.path_length_min);
            assert!(camera.path_length() <= settings.path_length_max + 1e-4);
        }
        let cameras = scene.cameras.clone();
        scene.resample_cameras(4, settings, 1.).unwrap();
        assert_eq!(scene.cameras, cameras);
    }

    fn assert_recovery_group(seed: u64, density: f32, human_density: f32) {
        let mut scene = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            density,
            0,
            human_density,
        )
        .unwrap();
        let objects = scene.objects.clone();
        let humans = scene.humans.clone();
        scene
            .resample_cameras(4, CameraSettings::default(), 1.)
            .unwrap();
        validate_layout(&scene).unwrap();
        assert_eq!(scene.objects, objects);
        assert_eq!(scene.humans, humans);
        let policy = scene.camera_settings.multiview.as_ref().unwrap();
        assert_eq!(scene.cameras.len(), 4);
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
        let coverage = Coverage::new(&scene);
        let primary_person = scene
            .humans
            .iter()
            .any(|h| !h.neighbor && scene.in_primary_room(h.position, 0.));
        for (index, camera) in scene.cameras.iter().enumerate() {
            assert!(scene.camera_clear(camera.start));
            assert!(scene.camera_curve_clear(camera));
            assert!(coverage.suitable(camera, index == 0 && primary_person));
            assert!(camera.path_length() + 1e-4 >= scene.camera_settings.path_length_min);
            assert!(camera.path_length() <= scene.camera_settings.path_length_max + 1e-4);
        }
        let replay = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            density,
            4,
            human_density,
        )
        .unwrap();
        assert_eq!(
            scene.cameras, replay.cameras,
            "seed={seed} density={density}"
        );
    }

    #[test]
    fn sparse_person_groups_complete_with_every_original_constraint() {
        for seed in [881, 2342, 3779, 5608] {
            assert_recovery_group(seed, 0., 0.25);
        }
    }

    #[test]
    fn incomplete_reception_pair_restarts_without_relaxing_group_constraints() {
        assert_recovery_group(3779, 0.35, 0.25);
    }

    #[test]
    fn dense_full_occupancy_workshop_completes_valid_camera_group() {
        assert_recovery_group(670, 0.65, 1.);
    }

    #[test]
    fn furnished_lounge_group_completes_valid_camera_group() {
        assert_recovery_group(3445, 0.65, 0.25);
    }

    #[test]
    fn constrained_lounge_seed_49309985_supports_landscape_views() {
        let mut scene =
            IndoorManifest::generate_with_humans(49_309_985, IndoorLayout::Mixed, 0.65, 0, 0.25)
                .unwrap();
        scene
            .resample_cameras(4, CameraSettings::default(), 4. / 3.)
            .unwrap();
        validate_layout(&scene).unwrap();
        let policy = scene.camera_settings.multiview.as_ref().unwrap();
        assert!(scene.camera_group_geometry().unwrap().accepts(policy));
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|p| policy.accepts(&p.samples)));
        let cameras = scene.cameras.clone();
        scene
            .resample_cameras(4, CameraSettings::default(), 4. / 3.)
            .unwrap();
        assert_eq!(scene.cameras, cameras);
    }

    #[test]
    fn dense_full_occupancy_studio_completes_valid_camera_group() {
        assert_recovery_group(682, 0.65, 1.);
    }

    #[test]
    fn policy_validation_and_legacy_json() {
        assert!(CameraSettings::parse("{}").unwrap().multiview.is_some());
        assert!(CameraSettings::parse(r#"{"multiview":null}"#)
            .unwrap()
            .multiview
            .is_none());
        for json in [
            r#"{"multiview":{"min_overlap":1.01}}"#,
            r#"{"multiview":{"min_overlap":-0.1}}"#,
            r#"{"multiview":{"min_baseline":0}}"#,
            r#"{"multiview":{"min_baseline":3,"max_baseline":2}}"#,
            r#"{"multiview":{"minimum_overlap":0.3}}"#,
            r#"{"multiview":{"min_spread":1.01}}"#,
            r#"{"multiview":{"trajectory_variation":-0.1}}"#,
            r#"{"multiview":{"target_variation":1.01}}"#,
            r#"{"multiview":{"target_variation":-0.1}}"#,
        ] {
            assert!(CameraSettings::parse(json).is_err(), "{json}");
        }
        let policy = CameraSettings::parse(r#"{"multiview":{}}"#).unwrap();
        assert_eq!(policy.multiview.unwrap().min_reference_baseline, 0.0);
        assert_eq!(
            CameraSettings::parse(r#"{"multiview":{}}"#)
                .unwrap()
                .multiview
                .unwrap()
                .target_variation,
            0.0
        );
    }

    #[test]
    fn connected_sets_preserve_layout_intrinsics_and_trajectories() {
        let mut translated = 0;
        for seed in 0..32 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            let before = scene.clone();
            let settings = CameraSettings {
                multiview: Some(MultiViewSettings::default()),
                ..default()
            };
            let count = if seed < 8 { 4 } else { 2 };
            let aspect = if seed % 2 == 0 { 4.0 / 3.0 } else { 0.75 };
            scene
                .resample_cameras(count, settings.clone(), aspect)
                .unwrap();
            validate_layout(&scene).unwrap();
            assert_eq!(scene.objects, before.objects);
            assert_eq!(scene.humans, before.humans);
            for pair in scene.camera_overlap() {
                assert!(settings.multiview.as_ref().unwrap().accepts(&pair.samples));
                assert_eq!(pair.reference, 0);
            }
            translated += usize::from(scene.cameras[0].path_length() > 0.1);
            let cameras = scene.cameras.clone();
            scene.resample_cameras(count, settings, aspect).unwrap();
            assert_eq!(scene.cameras, cameras);
            let restored: IndoorManifest =
                serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
            assert_eq!(restored, scene);
        }
        assert!(
            translated > 8,
            "overlap must not freeze all camera trajectories"
        );
    }

    #[test]
    fn default_policy_is_deterministic_and_impossible_policy_is_atomic() {
        let mut scene =
            IndoorManifest::generate_with_humans(1, IndoorLayout::Mixed, 0.65, 2, 0.0).unwrap();
        let original = scene.cameras.clone();
        let aspect = scene.camera_aspect_ratio;
        scene
            .resample_cameras(2, CameraSettings::default(), aspect)
            .unwrap();
        assert_eq!(scene.cameras, original);
        let original = scene.clone();
        let settings = CameraSettings {
            multiview: Some(MultiViewSettings {
                min_baseline: 99.0,
                max_baseline: 100.0,
                ..default()
            }),
            ..default()
        };
        assert!(scene
            .resample_cameras(2, settings, 1.0)
            .unwrap_err()
            .contains("no unconstrained fallback"));
        assert_eq!(scene, original);
        assert!(scene
            .resample_cameras(257, CameraSettings::default(), 1.0)
            .is_err());
        assert!(scene
            .resample_cameras(2, CameraSettings::default(), f32::NAN)
            .is_err());
    }

    #[test]
    fn archive_without_camera_policy_retains_independent_semantics() {
        let scene =
            IndoorManifest::generate_with_humans(9, IndoorLayout::Mixed, 0.65, 2, 0.0).unwrap();
        let mut json = serde_json::to_value(scene).unwrap();
        json["camera_settings"]["multiview"]
            .as_object_mut()
            .unwrap()
            .remove("min_spread");
        json["camera_settings"]["multiview"]
            .as_object_mut()
            .unwrap()
            .remove("trajectory_variation");
        json["camera_settings"]["multiview"]
            .as_object_mut()
            .unwrap()
            .remove("target_variation");
        let old_group: IndoorManifest = serde_json::from_value(json.clone()).unwrap();
        let old_policy = old_group.camera_settings.multiview.unwrap();
        assert_eq!(old_policy.min_spread, 0.0);
        assert_eq!(old_policy.trajectory_variation, 0.0);
        assert_eq!(old_policy.target_variation, 0.0);
        json["camera_settings"]
            .as_object_mut()
            .unwrap()
            .remove("multiview");
        let older: IndoorManifest = serde_json::from_value(json.clone()).unwrap();
        assert!(older.camera_settings.multiview.is_none());
        json.as_object_mut().unwrap().remove("camera_settings");
        let archive: IndoorManifest = serde_json::from_value(json).unwrap();
        assert!(archive.camera_settings.multiview.is_none());
    }

    #[test]
    fn project_room_has_spread_and_distinct_paths_and_rigid_rigs_are_explicit() {
        let mut scene =
            IndoorManifest::generate_with_humans(24005, IndoorLayout::Mixed, 0.65, 0, 0.25)
                .unwrap();
        scene
            .resample_cameras(4, CameraSettings::default(), 4.0 / 3.0)
            .unwrap();
        validate_layout(&scene).unwrap();
        let geometry = scene.camera_group_geometry().unwrap();
        assert!(geometry.min_horizontal_spread.unwrap() >= 0.25 - 1e-5);
        assert!(geometry.accepts(&MultiViewSettings::default()));

        let mut settings = CameraSettings::default();
        let policy = settings.multiview.as_mut().unwrap();
        policy.min_spread = 0.0;
        policy.trajectory_variation = 0.0;
        scene.resample_cameras(4, settings, 4.0 / 3.0).unwrap();
        validate_layout(&scene).unwrap();
        assert!(
            scene
                .camera_group_geometry()
                .unwrap()
                .min_relative_motion
                .unwrap()
                < 1e-4
        );
    }

    #[test]
    fn fixed_small_stereo_baselines_and_single_views_work() {
        let mut scene =
            IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
        let settings = CameraSettings {
            path_length_min: 0.0,
            path_length_max: 0.0,
            multiview: Some(MultiViewSettings {
                min_baseline: 0.1,
                min_reference_baseline: 0.0,
                max_baseline: 0.1,
                min_overlap: 0.6,
                ..default()
            }),
            ..default()
        };
        scene.resample_cameras(2, settings.clone(), 1.0).unwrap();
        for camera in &scene.cameras {
            assert_eq!(camera.transform_at(0.0), camera.transform_at(1.0));
        }
        for pair in scene.camera_overlap() {
            assert!(settings.multiview.as_ref().unwrap().accepts(&pair.samples));
        }
        scene.resample_cameras(1, settings, 1.0).unwrap();
        assert_eq!(scene.cameras.len(), 1);
        assert!(scene.camera_overlap().is_empty());
    }
}
