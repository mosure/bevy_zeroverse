//! Planning-only diversity audit. Optional tokenization never loads model weights.
use anyhow::{Context, Result};
use bevy_zeroverse::human_motion::MotionPlan;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub struct PromptAudit {
    count: usize,
    texts: BTreeSet<String>,
    programs: BTreeSet<String>,
    families: BTreeMap<String, usize>,
    gaits: BTreeMap<String, usize>,
    actions: BTreeMap<String, usize>,
    action_counts: BTreeMap<usize, usize>,
    tokens: BTreeMap<usize, usize>,
    examples: BTreeMap<String, String>,
}
impl PromptAudit {
    pub fn observe(
        &mut self,
        plans: &[MotionPlan],
        tokenizer: Option<&burn_llama::tokenizer::PromptTokenizer>,
    ) -> Result<()> {
        for plan in plans {
            self.count += 1;
            self.texts.insert(plan.request.prompt.clone());
            if let Some(tokenizer) = tokenizer {
                let encoded = tokenizer.encode(&plan.request.prompt).with_context(|| {
                    format!("actor {} prompt: {}", plan.actor_id, plan.request.prompt)
                })?;
                *self
                    .tokens
                    .entry(encoded.attention.into_iter().filter(|&v| v).count())
                    .or_default() += 1;
            }
            if let Some(r) = &plan.prompt_recipe {
                *self.families.entry(r.family.name().into()).or_default() += 1;
                if let Some(gait) = &r.gait {
                    *self.gaits.entry(gait.clone()).or_default() += 1;
                }
                *self.action_counts.entry(r.actions.len()).or_default() += 1;
                for a in &r.actions {
                    *self.actions.entry(a.action.clone()).or_default() += 1;
                }
                let signature = format!(
                    "{}:{}:{}",
                    r.family.name(),
                    r.gait.as_deref().unwrap_or(&plan.behavior),
                    r.actions
                        .iter()
                        .map(|a| a.action.as_str())
                        .collect::<Vec<_>>()
                        .join("/")
                );
                self.programs.insert(signature.clone());
                self.examples
                    .entry(signature)
                    .or_insert_with(|| plan.request.prompt.clone());
            }
        }
        Ok(())
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"prompt_program_version":bevy_zeroverse::human_motion::PROMPT_PROGRAM_VERSION,
            "requests":self.count,"unique_texts":self.texts.len(),"unique_action_programs":self.programs.len(),
            "family_counts":self.families,"gait_counts":self.gaits,"action_counts":self.actions,
            "actions_per_request":self.action_counts,"tokens_including_header_histogram":self.tokens,
            "examples_by_program":self.examples,
            "policy":"Feasible plans after geometry filtering, before model admission. Program signatures omit wording, handedness, repetitions, amplitude and timing. Text diversity does not establish motion fidelity."})
    }
}
