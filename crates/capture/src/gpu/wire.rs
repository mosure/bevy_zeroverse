//! Versioned descriptors and lease state, testable without a GPU or OS handles.
use serde::{Deserialize, Serialize};
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_MESSAGE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId {
    pub nonce: [u8; 16],
    pub lane: u32,
    pub incarnation: [u8; 16],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestId {
    pub sequence: u64,
    pub seed: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseId {
    pub session: SessionId,
    pub generation: u64,
    pub request: RequestId,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub bytes: u64,
    pub allocation_bytes: u64,
    pub dedicated: bool,
}
impl Allocation {
    pub fn validate(&self, budget: u64) -> Result<(), &'static str> {
        if self.bytes == 0
            || !self.bytes.is_multiple_of(16)
            || self.allocation_bytes < self.bytes
            || self.allocation_bytes > budget
            || !self.dedicated
        {
            Err("invalid or over-budget dedicated allocation")
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewRange {
    pub camera: u32,
    pub timestep: u32,
    pub offset: u64,
    pub bytes: u64,
}
/// Validate exact ordered view identities, alignment and non-overlapping ranges.
pub fn validate_views(
    views: &[ViewRange],
    cameras: usize,
    steps: usize,
    view_bytes: u64,
    allocation: u64,
) -> Result<(), &'static str> {
    if cameras.checked_mul(steps) != Some(views.len()) || views.is_empty() {
        return Err("view count mismatch");
    }
    let mut ranges = Vec::with_capacity(views.len());
    for (index, view) in views.iter().enumerate() {
        if view.camera as usize != index % cameras
            || view.timestep as usize != index / cameras
            || view.bytes != view_bytes
            || !view.offset.is_multiple_of(16)
        {
            return Err("view identity/stride mismatch");
        }
        let end = view
            .offset
            .checked_add(view.bytes)
            .ok_or("view range overflow")?;
        if end > allocation {
            return Err("view outside allocation");
        }
        ranges.push((view.offset, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err("overlapping view ranges");
    }
    Ok(())
}
/// One renderer-owned arena, one wait per completed signal. A replacement must
/// use a fresh session identity. Generations cannot be recycled after failure.
#[derive(Debug)]
pub struct LeaseState {
    session: SessionId,
    generation: u64,
    active: Option<LeaseId>,
    last_sequence: Option<u64>,
    poisoned: bool,
}
impl LeaseState {
    pub fn new(session: SessionId) -> Self {
        Self {
            session,
            generation: 0,
            active: None,
            last_sequence: None,
            poisoned: false,
        }
    }
    pub fn begin(&mut self, request: RequestId) -> Result<LeaseId, &'static str> {
        if self.poisoned
            || self.active.is_some()
            || self.last_sequence.is_some_and(|s| request.sequence <= s)
        {
            return Err("busy, stale or poisoned slot");
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or("generation overflow")?;
        let lease = LeaseId {
            session: self.session,
            generation: self.generation,
            request,
        };
        self.active = Some(lease);
        self.last_sequence = Some(request.sequence);
        Ok(lease)
    }
    pub fn retire(&mut self, lease: LeaseId) -> Result<(), &'static str> {
        if self.poisoned || self.active != Some(lease) {
            return Err("stale or duplicate release");
        }
        self.active = None;
        Ok(())
    }
    pub fn poison(&mut self) {
        self.poisoned = true;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replacement_duplicates_and_aba_are_rejected() {
        let session = SessionId {
            nonce: [1; 16],
            lane: 0,
            incarnation: [2; 16],
        };
        let mut state = LeaseState::new(session);
        let first = state
            .begin(RequestId {
                sequence: 4,
                seed: 777,
            })
            .unwrap();
        assert!(state.begin(first.request).is_err());
        let mut wrong = first;
        wrong.session.incarnation[0] += 1;
        assert!(state.retire(wrong).is_err());
        state.retire(first).unwrap();
        assert!(state.retire(first).is_err());
        assert!(state.begin(first.request).is_err());
        let next = state
            .begin(RequestId {
                sequence: 5,
                seed: 777,
            })
            .unwrap();
        assert_ne!(first.generation, next.generation);
        assert!(state.retire(first).is_err());
        state.poison();
        assert!(state.retire(next).is_err());
    }
    #[test]
    fn production_and_boundary_ranges() {
        for (cameras, steps, side) in [(8, 5, 256), (8, 3, 512), (1, 1, 257), (12, 3, 256)] {
            let bytes = ((side as u64).div_ceil(16) * 16) * side as u64 * 16;
            let views: Vec<_> = (0..cameras * steps)
                .map(|i| ViewRange {
                    camera: (i % cameras) as u32,
                    timestep: (i / cameras) as u32,
                    offset: i as u64 * bytes,
                    bytes,
                })
                .collect();
            let size = views.len() as u64 * bytes;
            assert!(validate_views(&views, cameras, steps, bytes, size).is_ok());
            let mut bad = views.clone();
            bad[0].offset = u64::MAX;
            assert!(validate_views(&bad, cameras, steps, bytes, size).is_err());
            if views.len() > 1 {
                let mut bad = views.clone();
                bad[1].offset = 0;
                assert!(validate_views(&bad, cameras, steps, bytes, size).is_err());
            }
        }
    }
}
