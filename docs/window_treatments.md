# Procedural window coverings

Exterior facades sample uncovered windows, Venetian blinds, roller shades,
vertical blinds, Roman shades, pleated shades, sliding panel tracks, and curtains. Each facade records
its `shade`, normalized `shade_coverage`, slat `shade_tilt`, and optional
`treatment` program in the room manifest. Openings on one facade share an
installation and finish. The dimensions adapt to each aperture.

Some banks move together. Others have individually adjustable openings that
share a typical draw while varying around it. `Facade::shade_coverage_at` gives
the realized draw for a particular opening. Both ends of the control still
fully open or close the bank. The population export records each opening's draw
in `windows.csv` under `shade_coverage`, alongside the bank setting in
`facade_shade_coverage`. Archived banks retain their recorded shared draw.

| Covering | Geometric controls |
| --- | --- |
| Venetian | Metric slat pitch, curved cross section, tilt, compressed raised stack; wood, plastic or metal |
| Roller | Unrolled length, exposed roll or cassette, weighted hem, control loop |
| Vertical | Vane pitch, rotation, one-sided or split stacking, hangers and control wand |
| Roman | Drop, broad horizontal folds, gathered height and hem sag |
| Pleated | Drop, fold pitch/depth and compressed stack |
| Sliding panels | Overlapping rigid panels, parked width, reed spacing and open weave |
| Curtain | Wave, pinch-pleat or gathered heading, one or two panels, fullness, draw, hem flare and optional jamb-mounted tiebacks |

The curtain heading and lining determine how much width the open stack occupies.
Wave headings attach to regularly spaced carriers. Pinched and gathered headings
have narrower folds near the track that relax toward the hem. Each panel has
its own fabric cut, folds and tie height. On tied panels the hem hangs below
the jamb fixing, with its lateral flare bounded by the free length of cloth.
The open stack and fold depths are
procedural approximations informed by commercial track dimensions. [Silent Gliss Wave workroom guide](https://www.silentglissglobal.com/fileadmin/redaktion/contentserv/Images/Web_Partner_Downloads/SGGB/Various_Documents/Wave_-_Curtain_Workroom_Guide.pdf)

The daylight meter estimates how much glass is visible past the cloth in a
frontal projection, accounting for the track, frame, mullions and overlapping
frit. RGB uses the PBR fabric finish and its lining-dependent diffuse
transmission. The closed fabric shell supplies the visible surface for depth,
normals, semantics and co-visibility. Free curtains can draw along their track
when covering motion is enabled. Their triangle correspondence and fabric
texture coordinates persist across the draw. A tied curtain stays on its jamb
fixing. All finishes reuse the indoor PBR map pool.

Curtain cloth and hardware use the `curtain` semantic class. Other coverings
and their hardware use `blinds`.

Geometry remains within the aperture horizontally and vertically, and within
20 cm of the interior wall face, matching camera clearance reservations. Tracks
prevent inward sash articulation even when a new installation is retracted.
Manifests without `treatment` retain the original Venetian/roller implementation.
Curtains without `treatment.curtain_heading` replay the earlier panel geometry.
Population metrics record the installed family, curtain heading, parked width
fraction, and the sampled controls for each family.

## Local review

```sh
cargo run --example review_window_treatments -- out/window_treatments/renders
cargo test --lib architecture::facade
```

The review renders seven production assemblies at three matched coverage settings,
plus a second material/shape seed with an explicitly paired, tied-back curtain.
It saves RGB, normals, depth, semantics,
visible covering pixel counts, construction timings and generator provenance.
It does not install project-page assets or publish a release.
