# Renderer optimization follow-up

The downstream `burn_gekko` study reports two distinct
costs: roughly 12 seconds to restart a renderer group after a rejected room,
and slower optimizer updates during live rendering despite no input starvation.
The isolated trainer result excludes generation and is not a live-throughput
target. These observations motivate the following ordered work.

1. **Retain the renderer after explicitly permitted scene rejections.** Add
   typed rejection categories at their source, transaction retirement and an
   opt-in bounded policy. Keep configuration, manifest-integrity, render,
   readback, device, archive and unknown failures fatal. The default remains
   fail-fast. Never identify a recoverable failure by matching log text.
2. **Make admission/export identity unambiguous.** Each attempted seed receives
   exactly one terminal outcome. Attempt sequence and archive index are separate;
   rejected seeds are recorded and never recycled. Cumulative grants remain
   monotonic; rejection returns the unpublished reservation. Bound both total
   and consecutive rejections and preserve ordering against delayed exports.
3. **Measure the actual render work.** Reuse CPU preparation/admission telemetry;
   make GPU phase measurements optional, bounded and nonblocking. Distinguish
   scene installation on the CPU, material/GI GPU execution, camera rendering,
   copies, and queue/preemption latency. GPU intervals can overlap and do not
   establish a CPU/context-switch diagnosis on their own.
4. **Optimize only evidenced redundant work.** Compare persistent rendering and
   preparation budgets with equal seeds, views, times, SSAA and annotations.
   Preserve CPU-only controlled waits and bounded lookahead. Do not multiply
   devices or alter the downstream trainer to make a capture-only number larger.

Qualification covers rejected/valid/rejected sequences in a single process,
exact accepted-room replay, identity/calibration/annotation alignment, cancellation,
credit exhaustion, rejection bounds, delayed/failed writers and bounded residency.
The two reported seeds are 62231318 (missing elevator) and 62236796 (bounded
long-path camera search). Actual seed/config failures must be reproduced; known
generation defects should be repaired rather than hidden by skip policy.

Performance receipts must include completed physical rooms and views, rejection
and recovery latency, CPU preparation and GPU phase work, RSS/VRAM, and the
sampling contract. A mixed-workload qualification also needs optimizer updates,
input blocking and energy, under the downstream owner's existing checkpoint and
experiment budget. Do not interrupt production training to manufacture isolated
measurements. Local matched captures and reports remain under ignored `out/`.

## Implemented safeguards and remaining qualification

Controlled recovery uses the same geometry and camera acceptance checks. Missing
functional anchors, insufficient capacity and bounded camera-search failures have
typed sources; public string diagnostics remain compatible. Integrity validation
wins over a recoverable rejection, including corrupt functional receipts and later
renderer errors. Initialization failures are delivered to the first ordinary
capture instead of being cleared while no usable scene exists.

Retirement cancels owned preparation and staging, drops the rejected transaction's
scene and capture attachments, preserves monotonically increasing request epochs,
and advances exactly once. A writer rendezvous orders rejected receipts after
previous atomic exports. Published receipts carry archive identity separately
from capture sequence. Total/consecutive bounds, cancellation and writer failures
are exercised by focused regressions. See [the API contract](capture_admission.md).

The identified rendering inefficiency was the settling fence's assumption that
its render target must equal its readback image. SSAA intentionally uses different
images. The fence now qualifies the exact source/output/scale contract and retains
all existing readiness and readback guards. Optional diagnostics include the SSAA
box-resolve span; the default path adds no timestamp queries or device waits.

Further context/CPU-budget changes need an owner-controlled mixed-workload trial:

1. Journal existing completed-capture telemetry through a bounded observer queue;
   retain requested, published and rejected room counts separately.
2. Compare one persistent renderer against the existing multi-context producer,
   using identical room/view/time/SSAA/shadow/GI/annotation settings and a fixed
   total CPU preparation budget. Then vary that budget in the single renderer.
3. Trace CUDA and Vulkan with CPU scheduling in the same interval. Correlate
   graphics submission gaps, material/GI spans, readback, and optimizer updates;
   overlapping intervals must not be summed into a fictitious GPU total.
4. Accept a new production default only if completed optimizer updates improve,
   input-starvation and memory bounds hold, and matched accepted captures retain
   their geometry, calibration and annotations. A faster isolated capture run
   alone is insufficient.

Matched images, timing runs and finite recovery residency measurements are local
qualification artifacts under ignored `out/renderer_followup/`. They are not
publication assets or a claim of unlimited-process stability. Existing rejected
layouts remain rejected; recovery does not repair or silently accept them.
