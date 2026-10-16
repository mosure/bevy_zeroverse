//! Versioned, renderer-free contracts for configuration-conditioned diversity.
//!
//! A descriptor defines what counts as distinct. These contracts never interpret
//! seed bits, unused recipes, views or unobserved motion as independent samples.
//! Qualification is explicit; reading an estimate starts no work or model load.
mod validation;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub use validation::{validate_plan, validate_receipt, validate_record};

pub const SCHEMA_VERSION: u32 = 1;
pub const ESTIMATOR_VERSION: &str = "occupancy-disjoint-binomial-v1";

/// Versioned independent counter stream; worker scheduling never changes seeds.
#[cfg(feature = "provenance")]
pub fn seed_at(policy: &SeedPolicy, index: u32) -> u64 {
    match policy {
        SeedPolicy::UniformWithReplacement { audit_seed } => {
            let mut input = b"zeroverse-capacity-seed-v1".to_vec();
            input.extend_from_slice(&audit_seed.to_le_bytes());
            input.extend_from_slice(&index.to_le_bytes());
            u64::from_str_radix(&crate::provenance::sha(&input)[..16], 16)
                .expect("SHA256 hex prefix")
        }
        SeedPolicy::Consecutive { first } => first.wrapping_add(u64::from(index)),
        SeedPolicy::Fixed { seed } => *seed,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleUnit {
    DressedScene,
    MultiViewRoom,
    TemporalClip,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SeedPolicy {
    /// Independent counter-hashed u64 draws; repetitions are retained.
    UniformWithReplacement {
        audit_seed: u64,
    },
    /// Diagnostic sweep. It does not justify IID confidence intervals.
    Consecutive {
        first: u64,
    },
    Fixed {
        seed: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditBudget {
    pub manifests: u32,
    pub rendered_rooms: u32,
    pub mesh_checks: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRequest {
    pub schema_version: u32,
    pub unit: SampleUnit,
    pub definitions: Vec<DiversityDefinition>,
    /// Number of future successful samples in the conditional population.
    /// Audit attempt/failure counts are reported separately; this is not a
    /// promise that the same number of generation attempts will all succeed.
    pub requested_draws: u64,
    pub seed_policy: SeedPolicy,
    pub confidence: f64,
    pub budget: AuditBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorKind {
    /// Joint accepted construction. Appearance fields are recipes, not pixels.
    AcceptedProgram,
    /// Joint observed RGB/metric geometry/semantic grids from completed captures.
    RenderedGrid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiversityDefinition {
    pub id: String,
    pub descriptor: DescriptorKind,
    pub version: u32,
    /// Frozen, origin-anchored bins. These are resolutions, not perceptual claims.
    pub metres: f64,
    pub radians: f64,
    pub appearance_step: f64,
    pub log2_light_step: f64,
    /// Rendered descriptors resample into this square grid using area averages.
    pub grid_side: u32,
    pub primary_only: bool,
}

impl Default for CapacityRequest {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            unit: SampleUnit::MultiViewRoom,
            definitions: vec![DiversityDefinition {
                id: "accepted-joint-10cm".into(),
                descriptor: DescriptorKind::AcceptedProgram,
                version: 1,
                metres: 0.1,
                radians: 0.1,
                appearance_step: 0.05,
                log2_light_step: 0.5,
                grid_side: 16,
                primary_only: true,
            }],
            requested_draws: 10_000_000,
            seed_policy: SeedPolicy::UniformWithReplacement { audit_seed: 17 },
            confidence: 0.95,
            budget: AuditBudget {
                manifests: 1024,
                rendered_rooms: 0,
                mesh_checks: 64,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePlan {
    pub image_size: [u32; 2],
    pub cameras: u32,
    /// Distinct normalized progress values, in delivery order.
    pub progress: Vec<f32>,
    /// Explicit physical times or None when the caller has no physical clock.
    pub seconds: Option<Vec<f64>>,
    pub modes: Vec<String>,
    pub annotation_glass: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedPopulation {
    pub generator: crate::GeneratorIdentity,
    pub capture_engine: String,
    pub platform: String,
    pub config_sha256: String,
    /// Canonical resolved JSON, including defaults and all preparation settings.
    pub resolved_config_json: String,
    pub capture: CapturePlan,
    pub failure_policy: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturedFactorState {
    Observed,
    AdmissionOnly,
    Inactive,
    Unqualified,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorInventory {
    pub id: String,
    pub parents: Vec<String>,
    pub state: CapturedFactorState,
    pub distribution: String,
    pub evidence_scope: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityPlan {
    pub schema_version: u32,
    pub population: ResolvedPopulation,
    pub request: CapacityRequest,
    pub factors: Vec<FactorInventory>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub class_sha256: String,
    /// Enables paired interventions without multiplying marginal cardinalities.
    pub blocks: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum SampleOutcome {
    Accepted {
        family_sha256: String,
        descriptors: BTreeMap<String, Descriptor>,
        diagnostics: BTreeMap<String, f64>,
        /// All delivered views/times, not merely requested renderer settings.
        delivered_views: u32,
        delivered_progress: Vec<f32>,
        /// Hash of the verified capture artifacts. None for manifest-only work.
        capture_sha256: Option<String>,
    },
    Failed {
        stage: String,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRecord {
    pub draw: u32,
    pub seed: u64,
    pub mesh_checked: bool,
    pub outcome: SampleOutcome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum EstimateStatus {
    ObservedOnly,
    Bounded,
    ModelConditional { model: String },
    NotQualified { reasons: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarEstimate {
    pub point: Option<f64>,
    pub lower: Option<f64>,
    /// None is unbounded/unknown, never infinity in a JSON number.
    pub upper: Option<f64>,
    pub confidence: Option<f64>,
    pub method: String,
    pub scope: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccupancyPoint {
    pub draws: u64,
    pub distinct: ScalarEstimate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoveltyValidation {
    pub discovery_rooms: u32,
    pub holdout_rooms: u32,
    pub discovery_classes: u32,
    pub holdout_new_classes: u32,
    pub holdout_unseen_draws: u32,
    /// Good-Turing unseen-mass diagnostic. Not an extrapolated support estimate.
    pub predicted_unseen_fraction: f64,
    pub observed_unseen_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionEstimate {
    pub definition: DiversityDefinition,
    pub status: EstimateStatus,
    pub rooms: u32,
    pub observed_classes: u32,
    pub singletons: u32,
    pub doubletons: u32,
    /// Marginal diagnostics only; never multiplied into a capacity estimate.
    pub block_distinct_counts: BTreeMap<String, u32>,
    pub disjoint_pairs: u32,
    pub matching_disjoint_pairs: u32,
    pub collision_probability: ScalarEstimate,
    pub collision_diversity: ScalarEstimate,
    pub empirical_entropy_diversity: f64,
    pub rarefaction: Vec<OccupancyPoint>,
    pub requested_occupancy: OccupancyPoint,
    pub max_qualified_forecast_draws: u64,
    pub nested_validation: Vec<NoveltyValidation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityEstimate {
    pub estimator: String,
    pub requested_rooms: u32,
    pub accepted_rooms: u32,
    pub failed_rooms: u32,
    pub mesh_checked_rooms: u32,
    pub rendered_rooms: u32,
    pub unique_families: u32,
    pub definitions: Vec<DefinitionEstimate>,
    /// Opportunity counts and per-room summaries, including zero observations.
    pub diagnostics: BTreeMap<String, DiagnosticSummary>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticSummary {
    pub rooms: u32,
    pub nonzero_rooms: u32,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityReceipt {
    pub schema_version: u32,
    pub plan: CapacityPlan,
    pub records_sha256: String,
    pub complete: bool,
    pub estimate: CapacityEstimate,
    pub qualification_seconds: f64,
}
