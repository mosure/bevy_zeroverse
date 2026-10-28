# Procedural exit signs

Exit signs sample European white running-person/door pictograms on green and
American red or green `EXIT` lettering independently of their mounting style.
Four supports are available: wall, ceiling canopy, paired rods and paired cables.
Ceiling installations have independently readable front and rear faces. Housing
width, height, depth, bevel, edge-lit rail construction, support spacing/radius,
drop, manufactured finish, face contrast and emission strength vary continuously.
Suspended cables use coated wire with a physical minimum diameter of 2.4 mm
within the supported production dimensions; rods/canopies retain metal finishes.

The vector pictogram is original procedural geometry. Visual references are the
[HSE safety-sign guide](https://www.hse.gov.uk/pubns/priced/l64.pdf) and
[Sure-Lites red/green wall/ceiling fixtures](https://www.cooperlighting.com/global/brands/sure-lites/10630480/apxrg-series/apx6rgbk).
These synthetic designs are not a certification of installed safety signage.

## Placement and annotation contract

- Place signs at the actual primary doorway. Pictogram arrows point down toward
  that exit; they are never randomly directed through walls or glazing.
- Check all canopy corners against the sampled ceiling, plus room containment,
  doorway/header clearance, raised-floor headroom, people and other installations.
- Try at most three candidates, falling back to a wall sign when the preferred
  ceiling attachment cannot be supported. Skip instead of accepting a collision.
- Preserve the `ExitSign` object category and `exit_sign` semantic label on all
  housing, support, background and foreground geometry. Bounding boxes include
  the fixture and its supports. All surfaces remain opaque geometric targets.
- Export realized legend/mount/double-sided/rail categories and metric panel
  width, height, depth, suspension drop and emission scale in content metrics.
- Reuse the existing finish palette. European panels and pictograms have two
  bounded emission channels; American lettering has one. Add no images, new
  rendering passes or shadow-casting lights.

## Qualification

Local evidence is in `out/exit_signs_101_final`. It is not a project-page publication.
The source identity is generator **101**, capture **v126**. Existing scene caches
are not bitwise equivalent to this distribution.

- 512 exit meshes cover 64 seeds × four mounts × two dimension extremes, checking
  actual triangles, normals, envelope containment and semantic labels.
- Two-sided glyph geometry, mount-independent color and actual source-to-material
  bindings are tested. A studio-review bug that bound unrelated source IDs has
  been fixed; fixtures now use their own emission colors.
- 20 focused tests pass. A 128-room services sweep admits 109 signs: 84 wall,
  10 canopy, nine rod and six cable mounts; 29 European, 43 red American and 37
  green American legends. These are observed acceptance counts, not a promised
  uniform mixture after ceiling/clearance constraints.
- All 6,848 meshes in the 107-category/64-seed audit pass. Another 128 consecutive
  Mixed rooms and 16 dense eight-camera rooms pass full layout/mesh validation,
  including regression seed 66369950.
- Workspace Clippy with CUDA IPC/WGPU fusion features, WASM with human motion,
  formatting and diff checks pass. This does not constitute a live browser test.
- For the first 32 fixed-envelope exit fixtures, average triangle count is 845;
  the complete 107-category audit gains only 1,192 triangles out of approximately
  9.69 million. Median exit-constructor time was 47 microseconds in a diagnostic
  run. These measurements are not an isolated render-throughput benchmark.

Rendered RGB/normal/semantic verification and its input manifest are recorded in
`out/exit_signs_101_final/renders`. No local hardware-GPU or concurrent-training
performance claim is made from the software-rendered review.

The final 12-view matrix contains 36 matched RGB/normal/semantic captures.
Legibility, front/rear orientation, ceiling contacts and semantic supports were
visually reviewed. Reproducible input objects and source provenance accompany
the renders; `qualification.json` records the complete local checks.
