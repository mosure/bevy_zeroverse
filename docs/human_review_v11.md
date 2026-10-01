# Human appearance, controls and motion — local v11 review

> **Archived evaluation.** This report describes its recorded build. Use the
> [documentation index](README.md) for current capabilities and defaults.

This revision fixes concrete rendering and control defects while retaining the
AnnyBody phenotype mesh and optional ARDY motion. The [portrait/body gallery](evidence/human_review_v11/review.html)
and [machine-readable report](evidence/human_review_v11/summary.json) show the
bounded evaluation. These are synthetic people with approximate garment and hair
surfaces; these checks do not establish photographic realism.

- Palms, metacarpals, wrists and fingers remain skin, without garment displacement.
  Clothing uses a rounded hanging torso envelope instead of following every skin
  depression. Lip pigmentation follows a small mask on the pinned Anny mouth
  topology, independently of the broader `oris` deformation weights.
- Hair is an opaque fitted volume with a tapered, sealed hairline, continuous
  length/part/curl parameters and mipmapped fibre relief. Overlapping narrow
  ribbons were removed. Brows follow the facial surface; eyewear has fitted rims,
  a nose bridge, temples and clear lenses. Lenses do not cast opaque shadows or
  enter the diffuse GI proxy as opaque surfaces.
- Scene and motion edits apply only on **Regenerate / R**. Preparation retains an
  immutable settings snapshot. The active scene's motion policy cannot be changed
  by an unapplied slider edit. Camera gizmos, bounding boxes and pose joints have
  independent controls. Instance roots also carry their semantic labels.
- Moving-fraction requests use a deterministic quota. The navigation fraction
  stages selected people in free space after furnishing and before camera/mesh/GI
  preparation. Routes consider furniture, walls, other actors at matching times,
  and camera trajectories. Walking conditions include standing pelvis height and
  stable approach headings. A rejected clip gets at most two deterministic model
  samples by default; models and prompt embeddings remain cached. Unselected and
  rejected people remain static. The inspector exposes rejection reasons.

The appearance code is separated into `humans/anatomy.rs`, `garments.rs`, `hair.rs`
and `face.rs`. Generation version is 11; capture identity is v16 so old and new
samples cannot silently share a resume identity.

## Validation

| Check | Result |
| --- | --- |
| Library suite with `human_motion` | 106 passed; 3 existing long qualifications ignored |
| Native default viewer | Compiles without motion dependencies being initialized |
| Strict Clippy, root package/all targets, motion feature | Passed |
| Wasm viewer, `web,human_motion` | Compile check passed; browser runtime not requalified |
| Consecutive CPU planning seeds 0–63 | 398 people; 347 feasible plans; 296 navigation plans |
| Real ARDY, seeds 0, 1, 13, 34 | 14 of 17 people admitted; 3 rejected for overlapping generated bounds |
| Accepted root endpoint displacement | 1.23–4.38 metres over 120 frames / 6 seconds |
| RGB/depth/normal/semantic/flow captures | 40 views, five times, two cameras per scene |
| Optical flow | 55,743 moving-person observations; zero missing expected static correspondences |
| Object and person boxes | All 150 expected boxes exported, including all 17 person labels |
| Appearance review | Six phenotypes; six portraits and six torso/hand views |

All real-model motion captures use `fraction=1`, `locomotion_fraction=1`,
`max_actors=16`, `frames=120`, `batch_size=4`, and the default two attempts. The
64-room planning audit is broader than the four-room inference sample: feasibility
is not the same as successful model admission. It found 51 people without an
admissible plan. A high requested fraction is not a promise to force colliding
clips into crowded rooms.

The admission checks remain conservative, including body-bound overlap; they are
not a continuous physical contact solver. Static-model initialization regressions,
settings/epoch isolation, palm classification, geometry finiteness, population
identity, timed path reservation and semantic/OBB export were tested. The normal
and flow checks use actual captured buffers. Browser execution and a large human
motion distribution still need separate qualification.

Reproduce the captures with the command recorded in the JSON report, and run
`cargo run --example review_humans --features human_motion -- out/human_review/verified`
for the studio views. Raw scene reports and accepted clips are retained in
`out/human_review/verified_motion`. The motion validator now encodes its linear
RGB previews as sRGB; geometric/flow checks use the original float attachments.
All work is local; no commit, push, CI run or publication was performed.
