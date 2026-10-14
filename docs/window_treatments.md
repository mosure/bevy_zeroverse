# Procedural window coverings

Exterior facades sample uncovered windows, Venetian blinds, roller shades,
vertical blinds, Roman shades, pleated shades, and curtains. Each facade records
its `shade`, normalized `shade_coverage`, slat `shade_tilt`, and optional
`treatment` program in the room manifest. Openings on one facade share a coherent
installation and finish; the dimensions adapt to each aperture.

| Covering | Geometric controls |
| --- | --- |
| Venetian | Metric slat pitch, curved cross section, tilt, compressed raised stack; wood, plastic or metal |
| Roller | Unrolled length, exposed roll or cassette, weighted hem, control loop |
| Vertical | Vane pitch, rotation, one-sided or split stacking, hangers and control wand |
| Roman | Drop, broad horizontal folds, gathered height and hem sag |
| Pleated | Drop, fold pitch/depth and compressed stack |
| Curtain | One or paired panels, split, fullness, unequal folds, hem sag, opening and optional tieback gathering |

Cloth, slats, tracks, brackets, cords and rods use distinct material roles. Seeded
finish slots reuse the bounded PBR map pools already used by indoor objects.
There are no per-window texture uploads or per-frame simulation systems.
Coverings use opaque, closed geometry with separated front and back surfaces;
sheer fabric transmission is not modeled. Depth, normals, semantic labels and
co-visibility therefore see the same covering surfaces as RGB. Curtain parts use
the `curtain` class; other coverings and their hardware use `blinds`.

Geometry remains within the aperture horizontally and vertically, and within
20 cm of the interior wall face, matching camera clearance reservations. Tracks
prevent inward sash articulation even when a new installation is retracted.
Manifests without `treatment` retain the original Venetian/roller implementation.
Population metrics record the covering family and only the continuous controls
used by that family; parameter counts are not independent diversity estimates.

## Local review

```sh
cargo run --example review_window_treatments -- out/window_treatments/renders
cargo test --lib architecture::facade
```

The review renders six production assemblies at three matched coverage settings,
plus a second material/shape seed with an explicitly paired, tied-back curtain.
It saves RGB, normals, depth, semantics,
visible covering pixel counts, construction timings and generator provenance.
It does not install project-page assets or publish a release.
