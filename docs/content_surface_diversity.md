# Artwork, display and whiteboard sampling

The generator samples content, physical dimensions, installation layout and surface finish independently. The aim is a broad distribution of plausible prints, interfaces and working boards, rather than pixel noise or a catalogue of fixed templates.

## Content and material ownership

Art, desktop screens, phone screens, televisions and whiteboards each have 120 content slots per room. Furniture finishes retain their coordinated twelve-slot palettes. Content slots depend on the object seed even when furniture shares a finish seed. Room material seeds resample the content of every slot. Slot collisions remain possible: for independently uniform assignments, the expected collision probability of a pair is 1/120. This capacity is not an estimate of semantic dataset size or proof that a model cannot learn generator biases.

The normal preparation path synthesizes only the slots consumed by geometry. Native synthesis uses the existing bounded worker pool; WebAssembly retains cooperative preparation. `IndoorMaterials::build_for_parts` provides the same selection behavior for isolated material/object studies. The full compatibility `build` API intentionally prepares the complete palette, and is more expensive. `materials::registered_finish` exposes the authoritative finish namespace, including specialized safety, exterior and kitchen slots.

## Composition programs

- Screens use device-aware status bars/navigation, independently sampled margins, sidebar side/width, font scale, pane counts and unequal recursive subdivisions. Documents combine charts, tables, maps, technical sections, covariance-driven scatter plots, correlated heatmaps, organization trees, calendars, task notes, media and editorial typography. Phones favor vertical arrangements; TV media and presentations retain their own sampling. Dormant panels remain non-emissive.
- Whiteboards vary handwritten glyph scale, line spacing, heading frequency, margin, ink palette, occupied clusters and local erasure. Connected sketches, architectural plans, product drawings, polynomial traces, planning notes and variable-term equations supplement text and tables. Bounds clip every drawing operation; empty or lightly used boards remain valid samples.
- Art combines unequal cropped regions, coordinated palettes, focal position, negative space, mark count/orientation, landscapes, botanical/still-life studies, architectural compositions and layered abstractions. Low-frequency seeded fields drive pigment washes, cut-color prints, smooth contours and streamlines. Posters sample their graphic, hierarchy, alignment and captions independently. Field interpolation avoids rectangular derivative discontinuities and clipped contour plateaus.
- Wall installations sample area and aspect separately, with portrait/signage, panoramic boards, varied frame depths, unequal columns, one-to-three-piece side stacks, and staggered salon hangs. Intentional aligned series remain possible. Complete groups pass the existing wall-opening, ceiling, floor and occupied-space checks before installation.

Content does not become geometry: art/enamel finish maps remain dielectric and non-emissive, display emission uses the existing photometric contract, and all maps retain their full mip chains. Material labels and semantic categories remain attached to the same objects.

## Qualification

`audit_display_materials --categories screen,phone,television,whiteboard,art` records consecutive-seed programs, map hashes, low-frequency color and mean-removed luminance distances, active/dormant counts and CPU synthesis times. These distances are inexpensive repetition diagnostics, not perceptual embedding volume or downstream generalization scores. Count dormant and blank examples in the overall distribution, and report active-content results separately.

`qualify_materials --category Art,Screen,PhoneScreen,Television,Whiteboard` checks actual room material usage and renders StandardMaterial color/world-normal swatches. The wall-composition tests cover generated-room clearance and assembled frame meshes; presentation tests independently sweep physical sizes/aspects. Content tests check nested clipping, high-bit seeds, repeatability, consumed-slot selection, exact PBR binding and mip completeness. Generator version 75 identifies this changed seeded distribution.

Local matched studies belong under `out/`; they are not publication inputs. A tenfold content-slot increase does not establish tenfold perceptual diversity, photographic realism or immunity to model memorization. Those claims require rendered representation and held-out training evaluations.

## Lecture seating

`ProgramParameters::lecture_seating` records a seeded `LectureSeating` policy. Roughly 20% of sampled halls retain regular rows. Other halls mix linked seating banks with loose tablet chairs, variable row depth, independently perturbed seat pitch, bank offsets and chair yaw/position. Facing direction remains biased toward the teaching area. The shared row-boundary plan drives both floor tiers and furniture. Usable tier depth limits chair offsets smoothly; ordinary support, headroom, overlap and circulation checks still admit each seat. Classroom sampling retains its existing path.

The seating qualification checks 512 policies and 64 complete halls, including accepted pose/spacing variance, support, route clearance, capacity and deterministic replay. Local chair-only diagrams distinguish regular from irregular outcomes. Capture identity 99 and generator version 75 invalidate reuse of the previous distribution.
